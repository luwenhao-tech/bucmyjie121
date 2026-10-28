"""论文对照实验：不同底座模型 × 不同领域化配置，在评测集上的表现。"""
import asyncio, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llm_client import client, build_system_prompt, classify_intent, resolve_intent_extra, is_off_topic, _GENERAL_METHODOLOGY_KEYS
from rag_engine import search_async, format_context_for_prompt

ITEMS = json.loads((ROOT / "scoring/paper_eval_items_v2.json").read_text())
OUT = ROOT / "scoring/paper_eval_answers_v3.jsonl"
REPS = 3
SEM = None

GENERIC_SYS = "你是一名高校中药鉴定学教师，请准确、专业地回答学生的问题。"
RAG_ONLY_SYS = "你是一名高校中药鉴定学教师，请参考下列资料准确、专业地回答学生的问题。\n\n{ctx}"

# (条件名, 底座, 是否检索, 是否学科约束)
CONDS = [
    ("flash_base", "deepseek-flash", False, False),
    ("pro_base", "deepseek-v4-pro", False, False),
    ("flash_rag", "deepseek-flash", True, False),
    ("flash_cons", "deepseek-flash", False, True),
    ("flash_full", "deepseek-flash", True, True),
    ("pro_full", "deepseek-v4-pro", True, True),
    ("pro_rag", "deepseek-v4-pro", True, False),
    ("pro_cons", "deepseek-v4-pro", False, True),
]

_ctx_cache = {}
_gate = {}
_intent = {}


async def precompute(q):
    """串行预算每题的主题闸门与意图，保证同一题在各配置、各重复下一致。"""
    _gate[q] = await is_off_topic(q)
    _intent[q] = await classify_intent(q)


def is_general(q):
    return any(k in (q or "") for k in _GENERAL_METHODOLOGY_KEYS)


async def get_ctx(q):
    if q not in _ctx_cache:
        if _gate[q]:
            _ctx_cache[q] = "__NO_RESULTS__"
        else:
            res = await search_async(q, top_k=5)
            if res and res[0]["score"] >= 20:
                _ctx_cache[q] = format_context_for_prompt(res)
            elif is_general(q):
                _ctx_cache[q] = "__GENERAL_THEORY__"
            else:
                _ctx_cache[q] = "__IN_SCOPE_NO_EVIDENCE__"
    return _ctx_cache[q]


async def answer(cond, item):
    name, model, use_rag, use_cons = cond
    q = item["q"]
    if use_cons:
        if use_rag:
            ctx = await get_ctx(q)
        else:
            ctx = "__NO_RESULTS__" if _gate[q] else ""
        intent = _intent[q]
        sys_p = build_system_prompt("", resolve_intent_extra(intent), ctx)
    elif use_rag:
        res = await search_async(q, top_k=5)
        sys_p = RAG_ONLY_SYS.format(ctx=format_context_for_prompt(res) if res else "（无）")
    else:
        sys_p = GENERIC_SYS
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": q}]
    for attempt in range(4):
        try:
            async with SEM:
                t0 = time.time()
                r = await client.chat.completions.create(model=model, messages=msgs, temperature=0.6)
                return r.choices[0].message.content or "", time.time() - t0
        except Exception as e:
            print("retry", name, item["id"], e)
            await asyncio.sleep(5 * (attempt + 1))
    return "", 0.0


async def main():
    global SEM
    SEM = asyncio.Semaphore(4)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            d = json.loads(line)
            done.add((d["cond"], d["id"], d["rep"]))
    tasks = [(c, it, r) for c in CONDS for it in ITEMS for r in range(REPS)
             if (c[0], it["id"], r) not in done]
    print("todo", len(tasks), flush=True)

    async def one(cond, item, rep):
        a, dt = await answer(cond, item)
        with OUT.open("a") as f:
            f.write(json.dumps({"cond": cond[0], "id": item["id"], "rep": rep, "answer": a, "sec": round(dt, 1)}, ensure_ascii=False) + "\n")

    for item in ITEMS:
        await precompute(item["q"])
        await get_ctx(item["q"])
    json.dump({it["id"]: [_gate[it["q"]], _intent[it["q"]], _ctx_cache[it["q"]][:40]] for it in ITEMS},
              open(ROOT / "scoring/paper_eval_precompute_v3.json", "w"), ensure_ascii=False, indent=1)
    await asyncio.gather(*(one(*t) for t in tasks))


if __name__ == "__main__":
    asyncio.run(main())
