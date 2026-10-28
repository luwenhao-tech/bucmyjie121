"""盲评：按每题评分标准判定通过/不通过，并汇总各配置结果。"""
import asyncio, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from llm_client import client

ITEMS = {d["id"]: d for d in json.loads((ROOT / "scoring/paper_eval_items_v2.json").read_text())}
ANS = ROOT / "scoring/paper_eval_answers_v3.jsonl"
OUT = ROOT / "scoring/paper_eval_judged_v5.jsonl"
JUDGE_MODEL = "deepseek-flash"
SEM = None

JUDGE_SYS = """你是中药鉴定学阅卷专家。根据【评分标准】判定【回答】是否通过。
先按【评分标准】判定；此外，回答出现下列任一情况一律判不通过：
1. 出现内部分档标签或后台字样，如"A类/B类/C类""可信度""知识库""检索到的资料""【参考N】"；
2. 与下方【核对事实清单】中任一条相矛盾（只对照清单，清单未涉及的内容不得凭你自己的记忆判错）；
3. 出现中药鉴定学范围以外的内容：功能主治、性味归经、治法（如"提壶揭盖"）、配伍、方剂、临床应用、药理作用、用法用量（越界拒答题中用于说明拒答理由的简短提及除外）。"药效成分""活性成分""指标成分"等成分称谓不算越界。化学成分、质量标准、本草考证、混淆品鉴别属于范围内。
不考虑文风与长度。

【核对事实清单（依据《中国药典》2025 年版一部及教材）】
- 基原：桑科植物桑 Morus alba L. 的干燥根皮。
- 气味：药典为"气微，味微甘"；药检文献所载"微有豆腥气，味甘微苦"亦为正确表述，两者单写或并写均算符合。只有写成"麻舌""嚼之麻舌"等才为不符。
- 质地：体轻，质韧，纤维性强，难折断，易纵向撕裂，撕裂时有粉尘飞扬。
- 显微：桑白皮**不具**晶鞘纤维（晶鞘纤维见于混淆品柘树皮、鸡桑皮）；具含晶厚壁细胞，石细胞胞腔内含方晶。称"桑白皮不具晶鞘纤维"是正确的。
- 现行药典桑白皮项下**未设含量测定**；称药典规定了某成分含量测定为不符。
- 饮片水分不得过 10.0%。
- 方法归类：DNA 条形码、ITS2 等核酸序列方法属分子（生物）鉴定，不属理化；水试、火试属性状鉴定；薄层色谱、荧光、显色反应、水分与灰分测定属理化。
先用一句话说明理由，最后一行只输出 PASS 或 FAIL。"""


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
    SEM = asyncio.Semaphore(4)
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            d = json.loads(l)
            done.add((d["cond"], d["id"], d["rep"]))
    recs = [json.loads(l) for l in ANS.read_text().splitlines()]
    todo = [r for r in recs if r["id"] in ITEMS and r["answer"] and (r["cond"], r["id"], r["rep"]) not in done]
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
        if r["id"] not in ITEMS: continue
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
    (ROOT / "scoring/paper_eval_summary_v5.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
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
