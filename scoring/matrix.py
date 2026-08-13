"""审稿模式 · 成分×药理矩阵 + 结论矛盾检测

对齐汇报稿第②点"证据缺口识别"：
1. 读一批 extract_prompt.py 抽出的结构化 JSON
2. 聚合成"成分 × 药理"矩阵：密集格=做烂了，空缺格=选题机会
3. 结论矛盾检测：同化合物+同药理+不同论文 IC50 差 3 倍以上 → 标记为"证据不一致，可作为选题"

用法：
    from scoring.matrix import build_matrix, find_conflicts, format_report
    docs = [json.load(open(p)) for p in extracted_json_paths]
    mat = build_matrix(docs)
    conflicts = find_conflicts(docs)
    print(format_report(mat, conflicts))
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Tuple

from .extract_prompt import normalize_action, normalize_compound


# 规范药理动词全集（用于矩阵列头）
STANDARD_ACTIONS = [
    "降糖", "降脂", "抗炎", "抗肿瘤", "抗氧化",
    "神经保护", "抗菌", "抗病毒", "免疫调节", "保肝",
]


# ============ 矩阵构建 ============

def build_matrix(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把结构化抽取结果聚合成矩阵。

    返回：
      {
        "compounds": [c1, c2, ...],           # 行
        "actions":   [a1, a2, ...],           # 列
        "counts":    {(c,a): n_papers},       # 单元格论文数
        "papers":    {(c,a): [filename,...]}, # 单元格覆盖的论文
        "gaps":      [(c,a), ...],            # 空缺格（选题机会）
        "saturated": [(c,a), ...],            # 密集格（≥5 篇，做烂了）
      }
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    papers: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    compounds_seen: set = set()

    for d in docs:
        if not d or d.get("doc_kind") != "原始实验":
            continue
        fn = d.get("filename") or "?"
        # 论文里的所有化合物 × 所有药理动词做笛卡尔
        cps = [normalize_compound(c.get("name", "")) for c in (d.get("compounds") or [])]
        cps = [c for c in cps if c]
        acts = []
        for ph in (d.get("pharmacology") or []):
            a = normalize_action(ph.get("action", ""))
            if a:
                acts.append(a)
                # 如果药理条目里明确指定了 compound，走精确配对而不是笛卡尔
                cc = normalize_compound(ph.get("compound") or "")
                if cc:
                    key = (cc, a)
                    counts[key] += 1
                    if fn not in papers[key]:
                        papers[key].append(fn)
                    compounds_seen.add(cc)
        if not any((ph.get("compound") for ph in (d.get("pharmacology") or []))):
            # 没有明确成分对应，退化为笛卡尔
            for c in cps:
                compounds_seen.add(c)
                for a in acts:
                    key = (c, a)
                    counts[key] += 1
                    if fn not in papers[key]:
                        papers[key].append(fn)

    compounds = sorted(compounds_seen)
    actions = STANDARD_ACTIONS[:]

    gaps: List[Tuple[str, str]] = []
    saturated: List[Tuple[str, str]] = []
    for c in compounds:
        for a in actions:
            n = counts.get((c, a), 0)
            if n == 0:
                gaps.append((c, a))
            elif n >= 5:
                saturated.append((c, a))

    return {
        "compounds": compounds,
        "actions": actions,
        "counts": dict(counts),
        "papers": {k: v for k, v in papers.items()},
        "gaps": gaps,
        "saturated": saturated,
    }


# ============ 结论矛盾检测 ============

_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:±\s*[\d.]+)?\s*(nM|μM|uM|mM|mg/kg|μg/mL|ug/mL|mg/mL|%)?", re.I)


def _parse_ic50(endpoint: str) -> List[Tuple[float, str]]:
    """从 endpoint 字符串里粗抽 IC50 / EC50 数值 + 单位。"""
    if not endpoint:
        return []
    out = []
    for m in _NUM_RE.finditer(endpoint):
        try:
            v = float(m.group(1))
        except Exception:
            continue
        u = (m.group(2) or "").lower().replace("um", "μM").replace("ug", "μg")
        if u:
            out.append((v, u))
    return out


def _to_um(value: float, unit: str) -> float | None:
    """尝试把浓度换算到 μM（只处理常见几档）。"""
    u = unit.lower()
    if u == "μm":
        return value
    if u == "nm":
        return value / 1000.0
    if u == "mm":
        return value * 1000.0
    return None


def find_conflicts(docs: List[Dict[str, Any]], fold: float = 3.0) -> List[Dict[str, Any]]:
    """同化合物 + 同药理动词，不同论文 IC50 差 fold 倍以上 → 标记为矛盾。"""
    bucket: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for d in docs:
        if not d or d.get("doc_kind") != "原始实验":
            continue
        fn = d.get("filename") or "?"
        for ph in (d.get("pharmacology") or []):
            cc = normalize_compound(ph.get("compound") or "")
            a = normalize_action(ph.get("action") or "")
            if not (cc and a):
                continue
            for v, u in _parse_ic50(ph.get("endpoint") or ""):
                um = _to_um(v, u)
                if um is None:
                    continue
                bucket[(cc, a)].append((fn, um))

    conflicts = []
    for (c, a), items in bucket.items():
        if len(items) < 2:
            continue
        vals = [x[1] for x in items]
        lo, hi = min(vals), max(vals)
        if lo <= 0:
            continue
        if hi / lo >= fold:
            conflicts.append({
                "compound": c,
                "action": a,
                "min_uM": round(lo, 3),
                "max_uM": round(hi, 3),
                "fold": round(hi / lo, 2),
                "median_uM": round(median(vals), 3),
                "papers": items,
                "suggestion": f"同 {c} × {a} 在不同论文中 IC50 差 {round(hi/lo,1)} 倍（{lo}–{hi} μM），"
                              f"建议以更严谨的方法学重复实验，澄清剂量-效应关系。",
            })
    return conflicts


# ============ Markdown 报告 ============

def format_report(mat: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> str:
    """把矩阵 + 矛盾输出成 Markdown。"""
    lines: List[str] = []
    compounds = mat["compounds"]
    actions = mat["actions"]
    counts = mat["counts"]

    lines.append("### 🧭 成分 × 药理矩阵（论文覆盖数）")
    lines.append("")
    if not compounds:
        lines.append("> 本轮结构化抽取未产出任何『原始实验』论文。")
    else:
        header = "| 化合物 \\ 药理 | " + " | ".join(actions) + " |"
        sep = "|" + "|".join(["---"] * (len(actions) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for c in compounds:
            row = [c]
            for a in actions:
                n = counts.get((c, a), 0)
                if n == 0:
                    row.append("·")
                elif n >= 5:
                    row.append(f"🔥{n}")
                else:
                    row.append(f"{n}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(f"> `·` 空缺（潜在选题）｜数字=论文数｜🔥≥5 篇（已饱和）")

    if mat.get("gaps"):
        lines.append("")
        lines.append("### 🎯 选题候选（空缺格 Top 10）")
        for c, a in mat["gaps"][:10]:
            lines.append(f"- **{c} × {a}**：库内 0 篇，可作为潜在切入点")

    if mat.get("saturated"):
        lines.append("")
        lines.append("### 🔥 已做烂（≥5 篇）")
        for c, a in mat["saturated"][:10]:
            n = counts.get((c, a), 0)
            lines.append(f"- {c} × {a}：{n} 篇，不建议再做重复研究")

    if conflicts:
        lines.append("")
        lines.append("### ⚠️ 结论矛盾（同成分×同药理，跨论文 IC50 差 ≥ 3 倍）")
        for cf in conflicts:
            lines.append(
                f"- **{cf['compound']} × {cf['action']}**："
                f"{cf['min_uM']}–{cf['max_uM']} μM（{cf['fold']}×，中位数 {cf['median_uM']} μM）"
            )
            lines.append(f"  · 建议：{cf['suggestion']}")

    return "\n".join(lines)
