"""端到端测试：跑 3 个典型 query，看真实回答里证据分层是否生效。"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import search_async, format_context_for_prompt
from llm_client import client, MODEL, build_system_prompt

QUERIES = [
    "桑白皮的性状鉴别要点是什么？",
    "桑白皮桑皮苷 A 的 HPLC 测定方法",
    "桑白皮和构树皮怎么区分",
]


async def run_one(q: str):
    print("=" * 80)
    print(f"【问题】{q}")
    print("=" * 80)

    # 1) 检索
    results = await search_async(q, top_k=6)
    print(f"\n>>> 检索命中 {len(results)} 条：")
    for r in results:
        print(f"    [{r['tier']}] cred={r['credibility']:.2f}  score={r['score']:.1f}  {r['filename'][:60]}")

    # 2) 拼 context 看 tier tag
    ctx = format_context_for_prompt(results)
    # 只显示【参考N】那几行
    import re
    ref_headers = re.findall(r'【参考\d+】.*', ctx)
    print(f"\n>>> Context 参考标注：")
    for h in ref_headers:
        print(f"    {h[:110]}")

    # 3) 调 LLM 出真实回答
    system_prompt = build_system_prompt(rag_context=ctx)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    answer = resp.choices[0].message.content
    print(f"\n>>> 模型回答：\n{answer}\n")


async def main():
    for q in QUERIES:
        try:
            await run_one(q)
        except Exception as e:
            print(f"[错误] {q}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
