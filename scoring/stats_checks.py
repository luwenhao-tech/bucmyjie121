"""审稿模式 · 统计学硬检验模块

对齐"给导师的汇报稿"里第一点：把 LLM 目测升级为 scipy 真跑 p 值。
LLM 端在审稿模式第二步会为每篇【原始实验】输出一段结构化 JSON：
    {"filename": "...", "compound": "...", "values": [...], "unit": "%",
     "n": 4, "R2": 0.9998, "checks": ["chisquare_lastdigit","benford",
                                       "runlength","shapiro"]}
本模块解析这些 JSON、跑 scipy 检验、返回每项 p 值 + 判定。

依赖：scipy。若环境未安装 scipy，退化为"未检验"状态，不阻塞主流程。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy import stats  # type: ignore
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# ============ 单项检验 ============

def chisquare_lastdigit(values: List[float]) -> Optional[Dict[str, Any]]:
    """末位数字 χ² 均匀性检验。
    真实数据的末位应在 0-9 均匀分布；造假常扎堆 0/5。
    返回：{"method","n","p","stat","zero_five_ratio","verdict"}
    """
    if not _SCIPY_OK or not values or len(values) < 8:
        return None
    last = []
    for v in values:
        # 用字符串取真正的末位（避免 float 精度）
        s = f"{v}"
        digits = [c for c in s if c.isdigit()]
        if digits:
            last.append(int(digits[-1]))
    if len(last) < 8:
        return None
    obs = [last.count(d) for d in range(10)]
    exp = [len(last) / 10.0] * 10
    try:
        chi2, p = stats.chisquare(obs, exp)
    except Exception:
        return None
    zf = (last.count(0) + last.count(5)) / len(last)
    verdict = "🔴 造假嫌疑" if p < 0.001 else ("🟡 可疑" if zf > 0.5 else "✅ 正常")
    return {
        "method": "chisquare_lastdigit",
        "n": len(last),
        "stat": round(chi2, 4),
        "p": round(p, 6),
        "zero_five_ratio": round(zf, 3),
        "verdict": verdict,
    }


def benford_firstdigit(values: List[float]) -> Optional[Dict[str, Any]]:
    """Benford 首位定律 χ² 检验（1-9）。适合大批含量测定。"""
    if not _SCIPY_OK or not values or len(values) < 30:
        return None
    first = []
    for v in values:
        if v is None or v == 0:
            continue
        s = f"{abs(v):.10g}".lstrip("0.")
        for c in s:
            if c.isdigit() and c != "0":
                first.append(int(c))
                break
    if len(first) < 30:
        return None
    obs = [first.count(d) for d in range(1, 10)]
    exp = [len(first) * math.log10(1 + 1 / d) for d in range(1, 10)]
    try:
        chi2, p = stats.chisquare(obs, exp)
    except Exception:
        return None
    verdict = "🔴 偏离 Benford" if p < 0.01 else ("🟡 边缘" if p < 0.05 else "✅ 符合")
    return {"method": "benford", "n": len(first), "stat": round(chi2, 4),
            "p": round(p, 6), "verdict": verdict}


def runlength_variance(values: List[float]) -> Optional[Dict[str, Any]]:
    """完美等差 / 步长方差检验。步长方差 ≈ 0 → 造假嫌疑。
    同时给出线性 R²（值 vs 序号），R² ≥ 0.9999 且 n ≤ 5 也可疑。
    """
    if not values or len(values) < 3:
        return None
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if not steps:
        return None
    mean_step = sum(steps) / len(steps)
    var = sum((s - mean_step) ** 2 for s in steps) / len(steps)
    # 相对方差（无量纲）
    rel = var / (abs(mean_step) ** 2 + 1e-12)

    r2 = None
    if _SCIPY_OK and len(values) >= 3:
        try:
            xs = list(range(len(values)))
            res = stats.linregress(xs, values)
            r2 = res.rvalue ** 2
        except Exception:
            r2 = None

    if rel < 1e-6:
        verdict = "🔴 步长恒定，疑似完美等差"
    elif r2 is not None and r2 >= 0.9999 and len(values) <= 5:
        verdict = "🟡 校准点过少却近乎完美线性"
    else:
        verdict = "✅ 步长有正常波动"
    return {"method": "runlength", "n": len(values),
            "step_var_rel": round(rel, 8),
            "R2": (round(r2, 6) if r2 is not None else None),
            "verdict": verdict}


def shapiro_normality(values: List[float]) -> Optional[Dict[str, Any]]:
    """Shapiro-Wilk 残差正态性——'太整齐'的数据往往非自然。"""
    if not _SCIPY_OK or not values or len(values) < 5 or len(values) > 5000:
        return None
    try:
        w, p = stats.shapiro(values)
    except Exception:
        return None
    verdict = "🟡 非正态（提示非自然分布）" if p < 0.05 else "✅ 正态"
    return {"method": "shapiro", "n": len(values), "stat": round(w, 4),
            "p": round(p, 6), "verdict": verdict}


def cross_table_constant(a: List[float], b: List[float]) -> Optional[Dict[str, Any]]:
    """跨表互推：表A - 表B ≡ 常数 → 疑似同一原始数据经算术变换重复使用。"""
    if not a or not b or len(a) != len(b) or len(a) < 3:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    rel = var / (abs(mean) ** 2 + 1e-12)
    if rel < 1e-6:
        verdict = f"🔴 差值≈{round(mean, 4)} 恒定，疑似算术变换复用"
    elif rel < 1e-3:
        verdict = "🟡 差值近乎恒定，建议核查"
    else:
        verdict = "✅ 差值有正常波动"
    return {"method": "cross_table", "n": len(diffs),
            "mean_diff": round(mean, 4), "var_rel": round(rel, 8),
            "verdict": verdict}


# ============ JSON 抓取 & 分派 ============

# 从 LLM 输出里抠出 ```{...}``` 或裸 {...} 结构块
_JSON_BLOCK_RE = re.compile(
    r"\{[^{}]*?\"filename\"\s*:[^{}]*?\"values\"\s*:\s*\[[^\]]*\][^{}]*?\}",
    re.DOTALL,
)


def extract_blocks(text: str) -> List[Dict[str, Any]]:
    """从审稿模式回答里抽取所有结构化 JSON 抓取块。"""
    out: List[Dict[str, Any]] = []
    for m in _JSON_BLOCK_RE.finditer(text or ""):
        raw = m.group(0)
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and "values" in obj:
            out.append(obj)
    return out


DISPATCH = {
    "chisquare_lastdigit": chisquare_lastdigit,
    "benford": benford_firstdigit,
    "runlength": runlength_variance,
    "shapiro": shapiro_normality,
}


def run_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """对单个抓取块跑全部指定的检验，返回结果字典。"""
    values = [float(v) for v in block.get("values", []) if isinstance(v, (int, float))]
    checks = block.get("checks") or list(DISPATCH.keys())
    results: List[Dict[str, Any]] = []
    for name in checks:
        fn = DISPATCH.get(name)
        if not fn:
            continue
        r = fn(values)
        if r:
            results.append(r)
    return {
        "filename": block.get("filename"),
        "compound": block.get("compound"),
        "unit": block.get("unit"),
        "n": len(values),
        "results": results,
        "scipy_available": _SCIPY_OK,
    }


def format_report(reports: List[Dict[str, Any]]) -> str:
    """把 scipy 回填的 p 值格式化成 Markdown 追加块，拼到 LLM 报告末尾。"""
    if not reports:
        return ""
    lines = ["", "---", "", "### 📊 统计学硬检验回填（scipy）", ""]
    if not _SCIPY_OK:
        lines.append("> ⚠️ 服务器未安装 scipy，本轮未执行硬检验。请 `pip install scipy` 后重跑。")
        return "\n".join(lines)
    for rep in reports:
        head = f"**{rep.get('filename','?')}** · {rep.get('compound','?')}"
        u = rep.get("unit")
        if u:
            head += f"（{u}）"
        head += f" · n={rep['n']}"
        lines.append(f"- {head}")
        if not rep["results"]:
            lines.append("  · 样本量不足或指标缺失，本项未检验")
            continue
        for r in rep["results"]:
            m = r["method"]
            if m == "chisquare_lastdigit":
                lines.append(f"  · **末位 χ²**：p={r['p']}｜0/5 占比 {r['zero_five_ratio']} → {r['verdict']}")
            elif m == "benford":
                lines.append(f"  · **Benford 首位**：p={r['p']} → {r['verdict']}")
            elif m == "runlength":
                extra = f"，R²={r['R2']}" if r.get("R2") is not None else ""
                lines.append(f"  · **步长方差**：rel={r['step_var_rel']}{extra} → {r['verdict']}")
            elif m == "shapiro":
                lines.append(f"  · **Shapiro-Wilk**：W={r['stat']}, p={r['p']} → {r['verdict']}")
    return "\n".join(lines)


def audit_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """入口：解析 LLM 审稿文本 → 跑 scipy → 返回 (追加 Markdown, 原始报告列表)。"""
    blocks = extract_blocks(text)
    reports = [run_block(b) for b in blocks]
    return format_report(reports), reports
