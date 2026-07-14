"""审稿模式端到端测试：直接调 llm_client 走 audit mode，看真实审稿报告。"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import search_async, format_context_for_prompt
from llm_client import client, MODEL, build_system_prompt

QUERY = "桑白皮桑皮苷 A 的 HPLC 测定方法"


async def main():
    print("=" * 80)
    print(f"【审稿模式测试】{QUERY}")
    print("=" * 80)

    results = await search_async(QUERY, top_k=15)
    print(f"\n>>> 检索命中 {len(results)} 条：")
    for r in results:
        print(f"    [{r['tier']}] cred={r['credibility']:.2f}  {r['filename'][:70]}")

    ctx = format_context_for_prompt(results)
    system_prompt = build_system_prompt(rag_context=ctx, mode="audit")

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUERY},
        ],
        temperature=0.3,
        max_tokens=3500,
    )
    print("\n>>> 审稿报告：\n")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
