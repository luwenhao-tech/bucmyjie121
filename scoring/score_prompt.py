"""LLM 打分 prompt + 单篇打分函数。

用法：
    python -m scoring.score_prompt --sample     # 跑 5 篇样本
    python -m scoring.score_prompt --all        # 全量跑 111 篇
    python -m scoring.score_prompt --files "a.pdf,b.pdf"

产出：scoring/credibility_scores.json
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# 允许直接 python scoring/score_prompt.py 也能 import 到项目根的 llm_client
sys.path.insert(0, str(Path(__file__).parent.parent))

INDEX_PATH = Path(__file__).parent.parent / "papers_index.json"
OUT_PATH = Path(__file__).parent / "credibility_scores.json"
REPORT_PATH = Path(__file__).parent / "credibility_report.md"

MAX_TEXT_CHARS = 3000

# ============ 期刊分区小码表（桑白皮/中药高频期刊）============
# 覆盖 111 篇里大约 80% 的常见期刊，避免 LLM 保守拍脑袋。
# 表外期刊按 LLM 自行判断。
JOURNAL_TIER_TABLE = """
【期刊分区参考表】遇到下列期刊直接按此判 journal_tier_score：
== SCI ==
- Journal of Ethnopharmacology = 0.85 (SCI Q1)
- Phytochemistry = 0.85 (Q1)
- Journal of Natural Products = 0.85 (Q1)
- Journal of Agricultural and Food Chemistry = 0.85 (Q1)
- Fitoterapia = 0.75 (Q2)
- Phytochemistry Letters = 0.70 (Q3)
- Biochemical Systematics and Ecology = 0.75 (Q2)
- Bioorganic & Medicinal Chemistry Letters = 0.75 (Q2)
- Molecules = 0.70 (Q2/Q3)
- Chemistry & Biodiversity = 0.70 (Q3)
- Natural Product Research = 0.65 (Q3)
- Chemical & Pharmaceutical Bulletin = 0.70 (Q3)
- Food Chemistry = 1.0 (Q1 top)
- Journal of Chromatography A/B = 0.85 (Q1)
- European Journal of Medicinal Chemistry = 0.90 (Q1)
- Bioorganic Chemistry = 0.85 (Q1)
== 中文核心 / CSCD ==
- 中国中药杂志 = 0.75 (CSCD 核心)
- 中草药 = 0.75 (CSCD 核心)
- 药学学报 = 0.80 (CSCD 顶级)
- 中国药学杂志 = 0.70 (北大核心)
- 中成药 = 0.70 (北大核心)
- 中国药房 = 0.60 (北大核心)
- 中国实验方剂学杂志 = 0.65 (北大核心)
- 中药材 = 0.65 (北大核心)
- 时珍国医国药 = 0.55 (普通期刊)
- 中国农学通报 = 0.60 (北大核心)
- 蚕业科学 = 0.55 (普通期刊)
- 现代中药研究与实践 = 0.50 (普刊)
== 官方标准 ==
- 中国药典 = 1.0
- 团体标准 (T/CNHFA 等) = 1.0
- 地方标准 = 0.90
表外期刊按 LLM 自行判断。
"""

SCORING_PROMPT = f"""你是中医药文献方法学评审专家。请对下面这篇中药（桑白皮相关）文献做**证据分层打分**，用于 RAG 检索加权。

严格按以下 5 个维度打分，每个维度 0.0–1.0：

1. **study_type_score** 文献类型（权重 0.30）
   - 官方药典 / 团体标准 = 1.0
   - Meta 分析 / 系统综述 = 0.90
   - 随机对照试验 (RCT) = 0.85
   - 原始实验（HPLC/UPLC/药理/化学成分/农学品质等） = 0.70
   - 一般综述 = 0.50
   - 本草考证 = 0.55（考证类文献价值稳定，不能过低）
   - 市场/产业分析 / 科普 = 0.40
   - 学位论文 = 0.55，会议摘要 = 0.30

2. **journal_tier_score** 期刊层级（权重 0.25）
{JOURNAL_TIER_TABLE}
   表外规则：
   - SCI Q1 = 0.95，Q2 = 0.80，Q3/Q4 = 0.65
   - 北大中文核心 = 0.60，CSCD 核心 = 0.70
   - 普通期刊 = 0.40，学位论文 = 0.55
   - 无法判断 = 0.55

3. **methodology_score** 方法学完整度（权重 0.20）
   **重要：先判 doc_kind，再选对应 checklist 打勾算命中率。别把 HPLC 清单硬套到植物学论文上。**

   **A. 质量评价/含量测定/HPLC/UPLC 类**（doc_kind = 质量评价原始）checklist：
      样品来源(产地/批号) / 对照品来源与纯度 / 色谱条件完整(柱/流动相/波长/柱温/流速/进样量) /
      系统适用性(理论塔板数/分离度) / 线性范围 / 精密度 / 重复性 / 加样回收率 / RSD报告 / 样本≥3批

   **B. 药理/活性/化学成分分离类**（doc_kind = SCI原始/药理原始）checklist：
      提取分离流程 / 结构鉴定手段(NMR/MS) / 阴阳性对照 / 动物细胞来源与伦理 /
      剂量-效应关系 / 生物学重复n≥3 / 统计方法 / 机制讨论

   **C. 农学品质/种质资源/植物学分析类**（doc_kind = 农学品质原始）checklist：
      样本来源(品种/产地/采集时间) / 样本量≥30 / 表型指标定义清晰 / 测定方法标准化 /
      重复次数≥3 / 统计方法(方差分析/主成分/聚类) / 数据可视化 / 结论有生态或育种意义

   **D. 本草考证/文献综述类**（doc_kind = 本草考证/综述）：
      不套 checklist，按信息深度给 0.5–0.7；若引证详实、有独立见解可到 0.75

   **E. 药典/团标**：直接给 1.0

   **F. 市场/产业分析/科普类**：不套 checklist，按数据来源与分析深度给 0.4–0.6

4. **recency_score** 时效性（权重 0.15）
   - 出版年份 ≤ 5 年前（相对 2026）= 1.0
   - 5–10 年 = 0.8
   - 10–20 年 = 0.6
   - > 20 年 = 0.4
   - 本草考证 / 药典历史类不降权 = 1.0

5. **consensus_score** 共识度（权重 0.10）
   - 结论是否与主流一致（药典/多篇高质量文献）
   - 无法判断给 0.6

**输出严格 JSON**（不加任何解释、不用 markdown 代码块），字段完全按下例：
{{
  "doc_kind": "药典|团标|SCI原始|药理原始|质量评价原始|农学品质原始|中文核心原始|普刊原始|综述|本草考证|市场分析|科普|学位|会议|其他",
  "study_type_score": 0.0,
  "journal_tier_score": 0.0,
  "methodology_score": 0.0,
  "recency_score": 0.0,
  "consensus_score": 0.0,
  "meta": {{"title_cn": "", "journal": "", "year": 0, "language": "zh|en"}},
  "checklist_used": "A|B|C|D|E|F",
  "checklist_hits": [],
  "checklist_miss": [],
  "notes": "一句话说明主要扣分点"
}}
"""


def load_chunks_by_file() -> Dict[str, List[dict]]:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for c in data.get("chunks", []):
        fn = c.get("filename", "")
        if fn:
            grouped[fn].append(c)
    return grouped


def build_text_for_scoring(filename: str, chunks: List[dict]) -> str:
    """按 filename 拼一段送 LLM 的文本，优先中文摘要 chunk，再拼后续 chunk 各截前 500 字。"""
    zh_chunk = next((c for c in chunks if c.get("text", "").startswith("【中文摘要】")), None)
    parts = [f"[文件名] {filename}"]
    if zh_chunk:
        parts.append(zh_chunk["text"])
    for c in chunks:
        if c is zh_chunk:
            continue
        snippet = c.get("text", "")[:500]
        if snippet:
            parts.append(snippet)
        if sum(len(p) for p in parts) > MAX_TEXT_CHARS:
            break
    text = "\n\n".join(parts)
    return text[:MAX_TEXT_CHARS]


def compute_credibility(scores: dict) -> float:
    return round(
        0.30 * scores.get("study_type_score", 0)
        + 0.25 * scores.get("journal_tier_score", 0)
        + 0.20 * scores.get("methodology_score", 0)
        + 0.15 * scores.get("recency_score", 0)
        + 0.10 * scores.get("consensus_score", 0),
        3,
    )


def tier_of(cred: float) -> str:
    if cred >= 0.85:
        return "A"
    if cred >= 0.60:
        return "B"
    return "C"


async def score_one(filename: str, chunks: List[dict]) -> dict:
    from llm_client import client, MODEL
    text = build_text_for_scoring(filename, chunks)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"filename": filename, "error": f"JSON parse fail: {e}", "raw": raw[:500]}
    parsed["filename"] = filename
    parsed["credibility"] = compute_credibility(parsed)
    parsed["tier"] = tier_of(parsed["credibility"])
    return parsed


# ============ 样本挑选 ============
SAMPLE_KEYWORDS = [
    ("药典", "2025版中国药典—桑白皮.docx"),
    ("团标", "保健食品用原料桑白皮团体标准"),
    ("SCI 原始",  "Morus alba"),
    ("中文核心 HPLC", "HPLC法同时测定"),
    ("农学品质", "廖源源"),
]


def pick_samples(grouped: Dict[str, List[dict]]) -> List[str]:
    picks = []
    files = list(grouped.keys())
    for label, kw in SAMPLE_KEYWORDS:
        hit = next((f for f in files if kw in f), None)
        if hit and hit not in picks:
            picks.append(hit)
    return picks


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="只跑 5 篇样本")
    parser.add_argument("--all", action="store_true", help="全量")
    parser.add_argument("--files", type=str, default="", help="逗号分隔文件名（支持关键词模糊匹配）")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rescore", action="store_true", help="强制重新打分（默认跳过已有）")
    args = parser.parse_args()

    grouped = load_chunks_by_file()
    all_files = sorted(grouped.keys())
    print(f"[加载] 索引中共 {len(all_files)} 个源文件")

    if args.files:
        keywords = [f.strip() for f in args.files.split(",") if f.strip()]
        targets = []
        for kw in keywords:
            hits = [f for f in all_files if kw in f]
            targets.extend(hits)
        # 去重保序
        seen = set()
        targets = [f for f in targets if not (f in seen or seen.add(f))]
    elif args.sample:
        targets = pick_samples(grouped)
        print(f"[样本] {len(targets)} 篇:")
        for t in targets:
            print(f"   - {t}")
    elif args.all:
        targets = all_files
    else:
        print("必须指定 --sample / --all / --files")
        return

    # 增量：跳过已有（除非 --rescore）
    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    if not args.rescore:
        skip = [t for t in targets if t in existing and "error" not in existing[t]]
        if skip:
            print(f"[跳过已有] {len(skip)} 篇（加 --rescore 可强制重跑）")
        targets = [t for t in targets if t not in existing or "error" in existing.get(t, {})]

    if not targets:
        print("[无待处理]")
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def bound(fn):
        async with sem:
            print(f"  [打分中] {fn}")
            try:
                r = await score_one(fn, grouped[fn])
                print(f"  [完成] {fn} → tier={r.get('tier','?')} cred={r.get('credibility','?')}")
                return r
            except Exception as e:
                print(f"  [错误] {fn}: {e}")
                return {"filename": fn, "error": str(e)}

    results = await asyncio.gather(*[bound(fn) for fn in targets])

    for r in results:
        existing[r["filename"]] = r
    OUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {OUT_PATH}  共 {len(existing)} 条")

    tiers = defaultdict(int)
    for r in results:
        tiers[r.get("tier", "?")] += 1
    print(f"\n[本次分布] A={tiers['A']}  B={tiers['B']}  C={tiers['C']}  错误={tiers['?']}")


if __name__ == "__main__":
    asyncio.run(main())
