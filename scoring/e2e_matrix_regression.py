"""E2E · 结构化抽取 + 成分×药理矩阵 + 选题候选清单

对齐汇报稿第②点：把 130 篇论文批量跑一遍，得到：
1. 每篇的六字段结构化 JSON（缓存到 scoring/extracted/）
2. 全库成分×药理矩阵
3. 选题候选清单 + 结论矛盾

用法：
    # 全量（慢，耗 LLM 额度）
    python3 -m scoring.e2e_matrix_regression

    # 小样测试
    python3 -m scoring.e2e_matrix_regression --limit 10

    # 只重跑聚合、复用缓存
    python3 -m scoring.e2e_matrix_regression --aggregate-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag_engine import load_index                                       # noqa: E402
from llm_client import client, MODEL                                    # noqa: E402
from scoring.extract_prompt import EXTRACT_SYSTEM_PROMPT, parse_extract_output  # noqa: E402
from scoring.matrix import build_matrix, find_conflicts, format_report  # noqa: E402


CACHE_DIR = ROOT / "scoring" / "extracted"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "scoring" / "matrix_report.md"

MAX_CHARS = 8000  # 单篇喂给 LLM 的正文截断长度


def _safe_name(fn: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in fn)[:150]


def _collect_by_filename() -> dict:
    """把 chunks 按 filename 合并成一份正文。"""
    idx = load_index()
    bucket: dict = {}
    for ch in idx.get("chunks", []):
        fn = ch.get("filename") or "?"
        bucket.setdefault(fn, []).append(ch.get("text") or "")
    return {fn: "\n".join(parts)[:MAX_CHARS] for fn, parts in bucket.items()}


async def _extract_one(fn: str, body: str) -> dict | None:
    """调 LLM 抽一篇的六字段 JSON。"""
    user_msg = f"【文件名】{fn}\n\n【正文（可能已截断）】\n{body}"
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [LLM 错误] {fn[:60]}: {e}")
        return None
    obj = parse_extract_output(text)
    if obj:
        obj.setdefault("filename", fn)
    return obj


async def run_extract(limit: int | None) -> list:
    bodies = _collect_by_filename()
    files = sorted(bodies.keys())
    if limit:
        files = files[:limit]
    print(f">>> 待抽取论文数：{len(files)}")
    results = []
    for i, fn in enumerate(files, 1):
        cache = CACHE_DIR / (_safe_name(fn) + ".json")
        if cache.exists():
            try:
                obj = json.loads(cache.read_text(encoding="utf-8"))
                results.append(obj)
                print(f"  [{i:>3}/{len(files)}] ✓ 缓存命中 {fn[:60]}")
                continue
            except Exception:
                pass
        print(f"  [{i:>3}/{len(files)}] ⏳ 抽取 {fn[:60]}")
        obj = await _extract_one(fn, bodies[fn])
        if obj:
            cache.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(obj)
    return results


def load_cached() -> list:
    out = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 篇（省额度）")
    ap.add_argument("--aggregate-only", action="store_true", help="跳过抽取，只用缓存聚合")
    args = ap.parse_args()

    if args.aggregate_only:
        docs = load_cached()
        print(f">>> 从缓存加载 {len(docs)} 篇")
    else:
        docs = await run_extract(args.limit)

    if not docs:
        print("没有可聚合的抽取结果。先跑抽取。")
        return

    mat = build_matrix(docs)
    conflicts = find_conflicts(docs)
    report = format_report(mat, conflicts)

    print("\n" + "=" * 88)
    print(report)
    print("=" * 88)

    header = (
        f"# 桑白皮论文库 · 选题矩阵回归报告\n\n"
        f"- 抽取论文数：{len(docs)}\n"
        f"- 化合物覆盖：{len(mat['compounds'])}\n"
        f"- 空缺格：{len(mat['gaps'])}\n"
        f"- 饱和格：{len(mat['saturated'])}\n"
        f"- 结论矛盾：{len(conflicts)}\n\n---\n\n"
    )
    REPORT_PATH.write_text(header + report, encoding="utf-8")
    print(f"\n✅ 报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
