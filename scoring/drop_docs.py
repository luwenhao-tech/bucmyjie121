"""从知识库索引和可信度评分表中移除指定文件，其余知识单元原样保留（不重新切分、不重算页码）。
用法：在 tcm-liuchunsheng 目录下运行  python3 scoring/drop_docs.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from rag_engine import _chunk_tokens  # 与建库时相同的分词方式

DROP = {
    "桑白皮鉴别.pptx",
    "国内桑属物种DNA序列数据清洗与分析报告 (1).pptx",
}

idx_path = ROOT / "papers_index.json"
d = json.loads(idx_path.read_text(encoding="utf-8"))
old_n = len(d["chunks"])
chunks = [c for c in d["chunks"] if c.get("filename") not in DROP]
doc_freq = Counter()
for c in chunks:
    for t in set(_chunk_tokens(c)):
        doc_freq[t] += 1
idx_path.write_text(
    json.dumps({"chunks": chunks, "doc_freq": dict(doc_freq), "total_docs": len(chunks)}, ensure_ascii=False),
    encoding="utf-8",
)
files = {c.get("filename") for c in chunks}
print(f"知识单元：{old_n} -> {len(chunks)}；来源文件：{len(files)} 个")

cred_path = ROOT / "scoring/credibility_scores.json"
cred = json.loads(cred_path.read_text(encoding="utf-8"))
old_c = len(cred)
cred = {k: v for k, v in cred.items() if k not in DROP}
cred_path.write_text(json.dumps(cred, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"可信度评分表：{old_c} -> {len(cred)} 条")

missing = files - set(cred)
print("索引中有、评分表中缺的文件：", missing or "无")
