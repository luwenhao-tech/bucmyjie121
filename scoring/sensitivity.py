"""可信度权重敏感性分析：单维±0.05、五维随机±0.05×1000、等权重。"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIMS = ["study_type_score", "journal_tier_score", "methodology_score", "recency_score", "consensus_score"]
W0 = [0.30, 0.25, 0.20, 0.15, 0.10]


def tier(c):
    return "A" if c >= 0.80 else ("B" if c >= 0.60 else "C")


def tiers(docs, w):
    s = sum(w)
    w = [x / s for x in w]
    return [tier(round(sum(wi * d.get(k, 0) for wi, k in zip(w, DIMS)), 3)) for d in docs]


def run(docs):
    n = len(docs)
    base = tiers(docs, W0)
    from collections import Counter
    print("文献数", n, "基线分档", dict(Counter(base)))
    single = []
    for i in range(5):
        for dlt in (0.05, -0.05):
            w = W0[:]; w[i] += dlt
            single.append(sum(a != b for a, b in zip(base, tiers(docs, w))))
    print(f"单维扰动改档 {min(single)}~{max(single)} 份，比例 {100*min(single)/n:.1f}%~{100*max(single)/n:.1f}%")
    rnd = random.Random(42)
    ch = []
    for _ in range(1000):
        w = [x + rnd.uniform(-0.05, 0.05) for x in W0]
        ch.append(sum(a != b for a, b in zip(base, tiers(docs, w))))
    m = sum(ch) / len(ch)
    print(f"随机1000次 平均改档 {m:.2f} 份({100*m/n:.2f}%)，最大 {max(ch)} 份({100*max(ch)/n:.1f}%)")
    eq = tiers(docs, [0.2] * 5)
    k = sum(a != b for a, b in zip(base, eq))
    a_changed = sum(1 for a, b in zip(base, eq) if a == "A" and b != "A")
    print(f"等权重改档 {k} 份({100*k/n:.1f}%)，A类改档 {a_changed}")


if __name__ == "__main__":
    d = json.loads((ROOT / "scoring/credibility_scores.json").read_text())
    if "--all" not in sys.argv:
        d.pop("桑白皮.xlsx", None)
    run(list(d.values()))
