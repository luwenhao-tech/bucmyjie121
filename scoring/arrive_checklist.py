"""审稿模式 · ARRIVE 2.0 方案合规性预审

对齐汇报稿第③点：不做"AI 生成实验方案"，改做"AI 审查用户提交的方案"。

清单来源：
- ARRIVE 2.0 完整 21 条（Essential 10 + Recommended 11-21），带官方页码追溯
- 桑白皮领域惯例（从 130 篇论文经验倒推）
- 对照组硬约束
- 期刊要求（Chinese Medicine / JEP 等常投期刊）

规则数据以 YAML 单独维护（scoring/arrive_rules.yaml），支持热更新。
如果 PyYAML 未安装或 YAML 缺失，退化到内置精简版清单，保证服务不挂。

用法：
    from scoring.arrive_checklist import CHECKLIST, format_checklist_for_llm
    prompt = format_checklist_for_llm()
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_RULES_PATH = Path(__file__).parent / "arrive_rules.yaml"


def _load_yaml() -> Dict[str, Any] | None:
    """尝试从 arrive_rules.yaml 加载；失败返回 None。"""
    if not _RULES_PATH.exists():
        return None
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with _RULES_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


# ============ 内置精简 fallback（保证 YAML 缺失时可用）============
_FALLBACK_ESSENTIAL = [
    {"id": "E1", "item": "研究设计", "must": ["每组样本数 n", "对照组划分"], "hint": "n<5 → 🟡"},
    {"id": "E7", "item": "统计方法", "must": ["检验方法名", "显著性阈值"], "hint": "推荐 ANOVA + Tukey"},
    {"id": "E8", "item": "实验对象", "must": ["物种/细胞系", "STR 鉴定"], "hint": "细胞系未 STR → 🔴"},
    {"id": "E10", "item": "干预/给药", "must": ["剂量梯度", "溶媒"], "hint": "剂量梯度 <3 → 🔴"},
]

_FALLBACK_CONTROLS = [
    {"id": "C1", "item": "阴性对照", "hint": "缺 → 🔴"},
    {"id": "C2", "item": "阳性对照", "hint": "抗肿瘤推荐顺铂/紫杉醇"},
    {"id": "C3", "item": "溶媒对照", "hint": "DMSO 溶解必须设 DMSO 对照，缺 → 🔴"},
    {"id": "C4", "item": "空白对照", "hint": "含量测定/HPLC 必须设空白，缺 → 🔴"},
]


def _build_checklist() -> Dict[str, List[Dict[str, Any]]]:
    """构造 CHECKLIST 结构。优先 YAML，失败退回 fallback。"""
    y = _load_yaml()
    if y and isinstance(y, dict):
        return {
            "ARRIVE Essential 10": y.get("essential") or _FALLBACK_ESSENTIAL,
            "ARRIVE Recommended 11-21": y.get("recommended") or [],
            "对照组硬约束": y.get("controls") or _FALLBACK_CONTROLS,
            "桑白皮领域惯例": y.get("domain_sangbaipi") or [],
            "期刊硬要求": y.get("journal") or [],
        }
    # fallback：YAML 不可用时的精简清单
    return {
        "ARRIVE Essential 10": _FALLBACK_ESSENTIAL,
        "对照组硬约束": _FALLBACK_CONTROLS,
    }


CHECKLIST: Dict[str, List[Dict[str, Any]]] = _build_checklist()


# 便于外部导入的分组变量（向后兼容原 API）
ARRIVE_ESSENTIAL = CHECKLIST.get("ARRIVE Essential 10", [])
ARRIVE_RECOMMENDED = CHECKLIST.get("ARRIVE Recommended 11-21", [])
CONTROL_GROUPS = CHECKLIST.get("对照组硬约束", [])
SANGBAIPI_CONVENTIONS = CHECKLIST.get("桑白皮领域惯例", [])
JOURNAL_REQUIREMENTS = CHECKLIST.get("期刊硬要求", [])


def _fmt_item(it: Dict[str, Any]) -> List[str]:
    """把单条规则渲染成 2-3 行 Markdown/plain text。"""
    lines: List[str] = []
    head = f"[{it.get('id','?')}] {it.get('item','?')}"
    # 追溯到 ARRIVE 官方 PDF 页码
    if it.get("page"):
        head += f" · ARRIVE 2.0 p.{it['page']}"
    must = it.get("must") or []
    if must:
        head += f"（应含：{'、'.join(must)}）"
    lines.append(f"  · {head}")
    if it.get("official"):
        lines.append(f"    · 官方原文：{it['official']}")
    if it.get("hint"):
        lines.append(f"    ↳ 经验判定：{it['hint']}")
    return lines


def format_checklist_for_llm() -> str:
    """渲染成 LLM 可读的表格，追加进 protocol 模式的 system prompt。"""
    lines = [
        "【★ 预审清单 · 逐项对照，不许跳项，每条建议必须带 ID 号 ★】",
        "【★ 输出规范：每条判定用 ✅ / 🟡 / 🔴 三色 + 引用条目号（如 [E7] 或 [S4]）★】",
    ]
    for section, items in CHECKLIST.items():
        if not items:
            continue
        lines.append(f"\n▎{section}")
        for it in items:
            lines.extend(_fmt_item(it))
    return "\n".join(lines)


def get_rule_by_id(rule_id: str) -> Dict[str, Any] | None:
    """根据条目号取单条规则（供 API 追溯用）。"""
    for items in CHECKLIST.values():
        for it in items:
            if it.get("id") == rule_id:
                return it
    return None


def rules_summary() -> Dict[str, Any]:
    """给 /api/protocol/rules 用的元信息摘要。"""
    return {
        "sections": [
            {"name": name, "count": len(items), "ids": [i.get("id") for i in items]}
            for name, items in CHECKLIST.items()
        ],
        "total": sum(len(items) for items in CHECKLIST.values()),
        "source": "arrive_rules.yaml" if _load_yaml() else "fallback (built-in)",
    }
