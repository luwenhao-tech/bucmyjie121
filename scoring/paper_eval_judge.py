"""盲评：按每题评分标准判定通过/不通过，并汇总各配置结果。"""
import asyncio, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from llm_client import client

ITEMS = {d["id"]: d for d in json.loads((ROOT / "scoring/paper_eval_items.json").read_text())}
ANS = ROOT / "scoring/paper_eval_answers.jsonl"
OUT = ROOT / "scoring/paper_eval_judged.jsonl"
JUDGE_MODEL = "deepseek-flash"
SEM = None

JUDGE_SYS = """你是中药鉴定学阅卷专家。根据【评分标准】判定【回答】是否通过。
只依据评分标准，不考虑文风与长度。先用一句话说明理由，最后一行只输出 PASS 或 FAIL。"""


async def judge(rec):
    it = ITEMS[rec["id"]]
    user = f"【问题】{it['q']}\n【评分标准】{it['rubric']}\n【回答】\n{rec['answer'][:6000]}"
    for attempt in range(4):
        try:
            async with SEM:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
                    temperature=0)
            out = (r.choices[0].message.content or "").strip()
            last = out.splitlines()[-1].upper() if out else ""
            return ("PASS" in last and "FAIL" not in last), out
        except Exception as e:
            print("retry", e, flush=True)
            await asyncio.sleep(5 * (attempt + 1))
    return None, ""


async def main():
    global SEM
    SEM = asyncio.Semaphore(8)
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            d = json.loads(l)
            done.add((d["cond"], d["id"], d["rep"]))
    recs = [json.loads(l) for l in ANS.read_text().splitlines()]
    todo = [r for r in recs if r["answer"] and (r["cond"], r["id"], r["rep"]) not in done]
    print("todo", len(todo), flush=True)

    async def one(r):
        ok, why = await judge(r)
        if ok is None:
            return
        d = {**r, "pass": ok, "why": why, "judge": JUDGE_MODEL}
        with OUT.open("a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    await asyncio.gather(*(one(r) for r in todo))


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["cond"]][ITEMS[r["id"]]["cat"]].append(r["pass"])
        agg[r["cond"]]["总体"].append(r["pass"])
    cats = ["关键事实", "方法学归类", "术语与学名", "越界拒答", "界内应答", "总体"]
    print("cond\t" + "\t".join(cats))
    res = {}
    for c in ["flash_base", "pro_base", "flash_rag", "flash_cons", "flash_full", "pro_rag", "pro_cons", "pro_full"]:
        if c not in agg:
            continue
        res[c] = {k: (sum(agg[c][k]), len(agg[c][k])) for k in cats}
        print(c + "\t" + "\t".join(f"{100*sum(agg[c][k])/max(1,len(agg[c][k])):.1f}({len(agg[c][k])})" for k in cats))
    (ROOT / "scoring/paper_eval_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    lat = defaultdict(list)
    for l in ANS.read_text().splitlines():
        d = json.loads(l)
        lat[d["cond"]].append(d["sec"])
    for c, v in lat.items():
        print(c, "平均耗时", round(sum(v) / len(v), 1))


if __name__ == "__main__":
    if "--sum" in sys.argv:
        summarize()
    else:
        asyncio.run(main())
        summarize()
