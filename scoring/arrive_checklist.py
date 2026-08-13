"""审稿模式 · ARRIVE 2.0 方案合规性预审

对齐汇报稿第③点：不做"AI 生成实验方案"，改做"AI 审查用户提交的方案"。

清单来源：
- ARRIVE 2.0 Essential 10 + Recommended Set（动物实验报告规范）
- 桑白皮领域惯例（从 130 篇论文经验倒推）
- 期刊要求（Chinese Medicine / JEP 等常投期刊）

用法：
    from scoring.arrive_checklist import CHECKLIST, format_checklist_for_llm
    prompt = format_checklist_for_llm()
    # 附到 LLM system prompt 里，让 LLM 按清单逐项审 user 方案
"""
from __future__ import annotations

from typing import Any, Dict, List


# ============ ARRIVE 2.0 Essential 10（细胞/动物实验通用）============
ARRIVE_ESSENTIAL: List[Dict[str, Any]] = [
    {
        "id": "E1", "item": "研究设计",
        "must": ["每组样本数 n", "实验/对照组划分", "生物学重复次数"],
        "hint": "n≥5 起报；仅生物学重复 n=3 → 🟡；无重复 → 🔴",
    },
    {
        "id": "E2", "item": "样本量说明",
        "must": ["样本量确定依据"],
        "hint": "缺乏 power analysis 或经验依据 → 🟡",
    },
    {
        "id": "E3", "item": "入组 / 排除标准",
        "must": ["纳入标准", "排除标准"],
        "hint": "细胞实验也需说明代次范围与状态判据",
    },
    {
        "id": "E4", "item": "随机化",
        "must": ["分组随机方法（随机数表/软件）"],
        "hint": "未提及随机化 → 🟡",
    },
    {
        "id": "E5", "item": "盲法",
        "must": ["测量/评估是否盲法"],
        "hint": "关键定量指标未盲评 → 🟡",
    },
    {
        "id": "E6", "item": "结局指标",
        "must": ["主要指标", "次要指标", "测量方法"],
        "hint": "指标定义不清 → 🔴",
    },
    {
        "id": "E7", "item": "统计方法",
        "must": ["检验方法名", "显著性阈值"],
        "hint": "推荐：单因素 ANOVA + Tukey / Dunnett；两组 t 检验；非正态 → Mann-Whitney",
    },
    {
        "id": "E8", "item": "实验对象",
        "must": ["物种/细胞系", "性别", "年龄/代次", "来源"],
        "hint": "细胞系未写 STR 鉴定 → 🔴（多数期刊已强制）",
    },
    {
        "id": "E9", "item": "实验条件",
        "must": ["温度", "湿度", "饲养条件/培养基"],
        "hint": "细胞：培养基+血清批号+CO₂；动物：昼夜循环+饲料",
    },
    {
        "id": "E10", "item": "干预 / 给药",
        "must": ["剂量", "浓度", "剂量梯度", "溶媒", "给药方式", "时长"],
        "hint": "剂量梯度 <3 → 🔴；≤4 → 🟡；≥5 用于剂量-效应曲线才合格",
    },
]


# ============ 对照组硬约束（药理学实验通用）============
CONTROL_GROUPS: List[Dict[str, Any]] = [
    {"id": "C1", "item": "阴性对照", "hint": "抗肿瘤/抗炎实验缺阴性对照 → 🔴"},
    {"id": "C2", "item": "阳性对照", "hint": "抗肿瘤推荐顺铂/紫杉醇；抗炎推荐地塞米松；缺 → 🟡"},
    {"id": "C3", "item": "溶媒对照", "hint": "DMSO 溶解的化合物必须设最高浓度 DMSO 对照，缺 → 🔴"},
    {"id": "C4", "item": "空白对照", "hint": "含量测定/HPLC 必须设空白，缺 → 🔴"},
]


# ============ 桑白皮领域惯例（从 130 篇经验倒推）============
SANGBAIPI_CONVENTIONS: List[Dict[str, Any]] = [
    {
        "id": "S1", "item": "样品来源",
        "hint": "必须说明产地（亳州/安徽/川黔）+ 采收年份 + 药用部位（根皮内层 vs 粗皮）+ 生用/蜜炙",
    },
    {
        "id": "S2", "item": "指标成分",
        "hint": "推荐至少测桑皮苷A / 桑根酮C / 氧化白藜芦醇 三者之一；仅测总黄酮 → 🟡",
    },
    {
        "id": "S3", "item": "色谱条件（含量测定）",
        "hint": "HPLC-DAD 需报柱型号 + 流动相 + 检测波长；HPLC 单一检测器建议升级到 UPLC-MS/MS",
    },
    {
        "id": "S4", "item": "方法学验证",
        "hint": "必须报 n / RSD / 加样回收率 / R² / 检测限；n<3 → 🔴，RSD>5% → 🟡",
    },
    {
        "id": "S5", "item": "抗肿瘤模型选择",
        "hint": "桑白皮 Morusin 类抗胃癌常用 HGC-27 / MKN-45；抗肝癌 HepG2；抗乳腺癌 MCF-7；孤证单一细胞系 → 🟡",
    },
    {
        "id": "S6", "item": "剂量范围合理性",
        "hint": "异戊烯基黄酮体外 IC50 常见 5–50 μM 区间；超出 100 μM 提示非特异性毒性",
    },
    {
        "id": "S7", "item": "近缘种/伪品对照",
        "hint": "鉴定研究应对照蒙桑/鸡桑/构树皮；缺伪品对照 → 🟡",
    },
]


# ============ 期刊硬要求（常投期刊）============
JOURNAL_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "J1", "item": "STR 鉴定", "hint": "Chinese Medicine / JEP 等强制细胞系 STR 鉴定报告"},
    {"id": "J2", "item": "支原体检测", "hint": "细胞培养需报支原体阴性"},
    {"id": "J3", "item": "伦理批号", "hint": "动物实验必须提供 IACUC/伦理委员会批号"},
    {"id": "J4", "item": "数据可及性", "hint": "原始数据 supplementary 或公共仓库存档"},
]


CHECKLIST = {
    "ARRIVE Essential 10": ARRIVE_ESSENTIAL,
    "对照组硬约束": CONTROL_GROUPS,
    "桑白皮领域惯例": SANGBAIPI_CONVENTIONS,
    "期刊硬要求": JOURNAL_REQUIREMENTS,
}


def format_checklist_for_llm() -> str:
    """把清单渲染成 LLM 可读的表格，追加进 system prompt。"""
    lines = ["【★ 预审清单 · 逐项对照，不许跳项 ★】"]
    for section, items in CHECKLIST.items():
        lines.append(f"\n▎{section}")
        for it in items:
            must = "、".join(it.get("must", [])) if it.get("must") else ""
            head = f"[{it['id']}] {it['item']}"
            if must:
                head += f"（应含：{must}）"
            lines.append(f"  · {head}")
            if it.get("hint"):
                lines.append(f"    ↳ {it['hint']}")
    return "\n".join(lines)
