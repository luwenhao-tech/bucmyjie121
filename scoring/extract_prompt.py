"""审稿模式 · 文献结构化抽取

把每篇【原始实验】论文抽成六字段结构化 JSON，供后续矩阵图 / 矛盾检测使用。
对齐汇报稿第②点："基于统计的证据缺口识别"。

字段：
- compounds: 涉及的化合物（尽量落到具体亚型：桑皮苷A、桑根酮C、Morusin、Moracin M/N ...）
- pharmacology: 药理方向（降糖 / 抗炎 / 抗肿瘤 / 神经保护 / 抗菌 / 抗氧化 / 免疫调节 ...）
- model: 实验模型（细胞系名 / 动物种属 / 体外/体内）
- dose: 剂量或浓度（含单位）
- conclusion: 一句话主要结论
- methodology: 方法学关键点（n / RSD / 对照组 / 统计方法）

调用方式：把 EXTRACT_SYSTEM_PROMPT + 论文全文喂给 LLM，要求只输出 JSON。
本模块不直接调 LLM，只提供 prompt + 解析工具，交给上层流程编排。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


EXTRACT_SYSTEM_PROMPT = """你是中药论文结构化抽取器。

任务：把用户给的论文正文抽成一个 JSON 对象，字段固定如下。
凡是原文没出现的信息，一律填 null 或 []，**严禁编造**。

【输出 schema】
{
  "filename": "论文文件名，用户会告诉你",
  "doc_kind": "原始实验 | 综述 | 药典/团标 | 其他",
  "herb": "桑白皮",
  "compounds": [
    {"name": "桑皮苷A", "aliases": ["Mulberroside A"], "assay": "HPLC-DAD", "content": "0.42%-1.87%"}
  ],
  "pharmacology": [
    {"action": "抗炎", "model": "LPS 诱导 RAW264.7", "in_vivo": false,
     "compound": "Morusin", "dose": "10-40 μM",
     "endpoint": "IL-6 抑制 IC50=12.3 μM",
     "conclusion": "剂量依赖抑制 IL-6 释放"}
  ],
  "methodology": {
    "n": 6, "RSD_pct": 1.8, "recovery_pct": 99.2, "R2": 0.9998,
    "controls": ["空白","阴性"], "stats_method": "单因素 ANOVA + Tukey",
    "issues": []
  },
  "citation_year": 2023,
  "journal": "中草药"
}

【硬约束】
1. 只输出 JSON，前后不许有任何解释文字。
2. 原文没写的数字字段填 null，不许写 0 冒充。
3. compounds / pharmacology 允许多条；一条论文报多个药理作用就展开成多条。
4. action 用规范术语：降糖 / 降脂 / 抗炎 / 抗肿瘤 / 抗氧化 / 神经保护 / 抗菌 / 抗病毒 / 免疫调节 / 保肝 / 利尿 / 平喘 / 其他。
5. model 尽量落到细胞系名（HGC-27 / RAW264.7 / HepG2 / SH-SY5Y / MCF-7 ...）或动物种属（SD 大鼠 / C57 小鼠 / db/db 小鼠 ...）+ 造模方式。
6. 综述 / 药典 / 其他文档类型：只填 filename + doc_kind + herb，其余字段留空。

【★ pharmacology 逐条强制字段（关系到冲突检测，必须尽全力填写）★】
7. compound 字段：**必须**归属到具体化合物名。如果原文这段活性归属不到具体化合物（"总提取物"/"总黄酮"），才填 null；否则一律必填。
8. dose 字段：原文写了浓度/剂量就**必填**（如 "10 μM"、"50 mg/kg"），带单位。原文没写才 null。
9. endpoint 字段：**必须包含具体检测数值**。凡是原文里出现 IC50 / EC50 / CC50 / MIC / IC₅₀ 数值，一律照搬进 endpoint 字符串（例："IL-6 IC50 = 12.3 μM"）。只有原文完全没报数值，才写"XX 抑制活性"这种定性描述。

【★ 化合物名归一化词表（在 compounds.name 和 pharmacology.compound 里都用规范名）★】
- Mulberroside A / mulberroside A → 桑皮苷A
- Sanggenon C / sanggenon C → 桑根酮C
- Oxyresveratrol → 氧化白藜芦醇
- Moracin M / moracin M → 桑辛素M
- Moracin N / moracin N → 桑辛素N
- Morusin → Morusin（保留英文）
- Kuwanon G / kuwanon G → 桑黄酮G
- Sanggenol A → 桑根醇A
- 遇到 IUPAC 系统名（如"2,6,8,10-四甲基…"这种冗长化学式）：**优先使用论文正文里同义给出的俗名/英文商品名**；两者都没有才落 IUPAC 名，并把 IUPAC 全名放进 aliases 里。
"""


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_extract_output(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出里稳健抽取 JSON 对象。"""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # 尝试修一下常见问题：单引号、尾逗号
        raw = m.group(0)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(raw)
        except Exception:
            return None


def normalize_action(action: str) -> str:
    """把常见同义词归一到规范药理动词。"""
    if not action:
        return "其他"
    a = action.strip()
    table = {
        "降血糖": "降糖", "抗糖尿病": "降糖", "hypoglycemic": "降糖",
        "抗炎症": "抗炎", "抗炎作用": "抗炎", "anti-inflammatory": "抗炎",
        "抗癌": "抗肿瘤", "抗癌作用": "抗肿瘤", "抗癌活性": "抗肿瘤",
        "anti-tumor": "抗肿瘤", "anticancer": "抗肿瘤",
        "抗氧化作用": "抗氧化", "自由基清除": "抗氧化", "antioxidant": "抗氧化",
        "神经保护作用": "神经保护", "neuroprotection": "神经保护",
        "抗菌作用": "抗菌", "抑菌": "抗菌", "antibacterial": "抗菌",
        "免疫增强": "免疫调节", "免疫调节作用": "免疫调节",
        "肝保护": "保肝", "护肝": "保肝",
    }
    return table.get(a, table.get(a.lower(), a))


def normalize_compound(name: str) -> str:
    """化合物名归一（中英同义）。"""
    if not name:
        return ""
    n = name.strip()
    table = {
        "Mulberroside A": "桑皮苷A", "mulberroside a": "桑皮苷A",
        "Sanggenon C": "桑根酮C", "sanggenon c": "桑根酮C",
        "Sanggenon D": "桑根酮D", "sanggenon d": "桑根酮D",
        "Oxyresveratrol": "氧化白藜芦醇", "oxyresveratrol": "氧化白藜芦醇",
        "Moracin M": "桑辛素M", "moracin m": "桑辛素M",
        "Moracin N": "桑辛素N", "moracin n": "桑辛素N",
        "Kuwanon G": "桑黄酮G", "kuwanon g": "桑黄酮G",
        "Kuwanon H": "桑黄酮H", "kuwanon h": "桑黄酮H",
        "Sanggenol A": "桑根醇A", "sanggenol a": "桑根醇A",
        "Morusin": "Morusin",
        "morusin": "Morusin",
        "桑辛素": "桑辛素",
    }
    return table.get(n, table.get(n.lower(), n))
