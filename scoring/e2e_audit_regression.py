"""审稿模式回归测试：跨三类典型场景验证 tier 分层规则是否稳定。

场景：
  1. 性状鉴别 —— A 类药典 + C 类科普混合，看 C 是否被抬太高
  2. 药理机制 —— 多为 B 类 SCI，看 🟡/🟢 判定是否稳
  3. 混伪品鉴别 —— 跨 tier 综合，看 Step 6 走法选择是否合理
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import search_async, format_context_for_prompt
from llm_client import client, MODEL, build_system_prompt

QUERIES = [
    ("性状鉴别", "桑白皮的性状鉴别要点，如何和伪品区分？"),
    ("药理机制", "桑白皮抗炎的作用机制和主要活性成分有哪些？"),
    ("混伪品鉴别", "桑白皮和构树皮、白桑皮的显微/理化鉴别差异？"),
]


async def run_one(tag: str, q: str):
    print("\n" + "=" * 88)
    print(f"【场景:{tag}】{q}")
    print("=" * 88)

    results = await search_async(q, top_k=15)
    print(f"\n>>> 检索命中 {len(results)} 条：")
    tier_count = {"A": 0, "B": 0, "C": 0}
    for r in results:
        tier_count[r["tier"]] = tier_count.get(r["tier"], 0) + 1
        print(f"    [{r['tier']}] cred={r['credibility']:.2f}  {r['filename'][:70]}")
    print(f"\n>>> Tier 分布：A={tier_count.get('A',0)}  B={tier_count.get('B',0)}  C={tier_count.get('C',0)}")

    ctx = format_context_for_prompt(results)
    system_prompt = build_system_prompt(rag_context=ctx, mode="audit")

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        temperature=0.3,
        max_tokens=3500,
    )
    print(f"\n>>> 审稿报告：\n")
    print(resp.choices[0].message.content)


async def main():
    for tag, q in QUERIES:
        try:
            await run_one(tag, q)
        except Exception as e:
            print(f"[错误] {tag} / {q}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
