"""RAG 引擎（轻量版）：PDF/Excel 论文解析 + 关键词检索。

设计目标：在低内存服务器（< 1GB）上也能运行。
- 索引构建：在本地运行，生成 JSON 索引文件
- 检索：加载 JSON，用 jieba 分词 + BM25 评分，内存占用极低

使用方式：
1. 本地：把论文放入 papers/ 目录，运行 python build_index.py
2. 推送 papers_index.json 到 GitHub
3. 服务器 git pull 后重启即可，无需在服务器上构建索引
"""
import os
import json
import math
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

# ============ 配置 ============
PAPERS_DIR = os.getenv("PAPERS_DIR", "papers")
INDEX_FILE = os.getenv("RAG_INDEX_FILE", "papers_index.json")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def ensure_dirs():
    os.makedirs(PAPERS_DIR, exist_ok=True)


# ============ PDF 解析 ============
def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取全部文本"""
    import fitz
    doc = fitz.open(pdf_path)
    text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text)
    doc.close()
    return "\n".join(text_parts)


def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, str]:
    import fitz
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}
    doc.close()
    return {
        "title": metadata.get("title", "") or Path(pdf_path).stem,
        "author": metadata.get("author", ""),
        "filename": Path(pdf_path).name,
    }


# ============ Word (.docx) 解析 ============
def extract_text_from_docx(docx_path: str) -> str:
    """从 .docx 提取全部文本（段落 + 表格单元格）"""
    from docx import Document
    doc = Document(docx_path)
    parts = []
    # 正文段落
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    # 表格内容
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    parts.append(cell_text)
    return "\n".join(parts)


# ============ Excel 解析 ============
def extract_papers_from_excel(xlsx_path: str) -> List[Dict[str, str]]:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    papers = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            continue
        header = [str(cell or "").strip() for cell in rows[0]]
        col_map = {}
        for i, h in enumerate(header):
            h_lower = h.lower()
            if "论文" in h or "名称" in h or "标题" in h or "title" in h_lower:
                col_map["title"] = i
            elif "作者" in h or "author" in h_lower:
                col_map["author"] = i
            elif "方法" in h or "method" in h_lower:
                col_map["method"] = i
            elif "关键" in h or "keyword" in h_lower:
                col_map["keywords"] = i
            elif "摘要" in h or "abstract" in h_lower:
                col_map["abstract"] = i
            elif "期刊" in h or "journal" in h_lower:
                col_map["journal"] = i
            elif "年" in h or "year" in h_lower:
                col_map["year"] = i
        if "title" not in col_map:
            continue
        for row in rows[1:]:
            if not row or not row[col_map.get("title", 0)]:
                continue
            paper = {
                "title": str(row[col_map["title"]] or "").strip(),
                "author": str(row[col_map.get("author", -1)] or "").strip() if "author" in col_map and col_map["author"] < len(row) else "",
                "method": str(row[col_map.get("method", -1)] or "").strip() if "method" in col_map and col_map["method"] < len(row) else "",
                "keywords": str(row[col_map.get("keywords", -1)] or "").strip() if "keywords" in col_map and col_map["keywords"] < len(row) else "",
                "abstract": str(row[col_map.get("abstract", -1)] or "").strip() if "abstract" in col_map and col_map["abstract"] < len(row) else "",
                "journal": str(row[col_map.get("journal", -1)] or "").strip() if "journal" in col_map and col_map["journal"] < len(row) else "",
                "year": str(row[col_map.get("year", -1)] or "").strip() if "year" in col_map and col_map["year"] < len(row) else "",
            }
            if paper["title"]:
                papers.append(paper)
    wb.close()
    return papers


def paper_to_text(paper: Dict[str, str]) -> str:
    parts = []
    parts.append(f"论文标题：{paper['title']}")
    if paper.get("author"):
        parts.append(f"作者：{paper['author']}")
    if paper.get("journal") or paper.get("year"):
        journal_info = paper.get("journal", "")
        if paper.get("year"):
            journal_info += f"（{paper['year']}）"
        parts.append(f"发表于：{journal_info}")
    if paper.get("keywords"):
        parts.append(f"关键词：{paper['keywords']}")
    if paper.get("method"):
        parts.append(f"研究方法：{paper['method']}")
    if paper.get("abstract"):
        parts.append(f"研究内容：{paper['abstract']}")
    return "\n".join(parts)


# ============ 文本切块 ============
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(current_chunk) + len(sent) <= chunk_size:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
        elif len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += ("\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped_chunks.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped_chunks
    chunks = [c for c in chunks if len(c.strip()) > 30]
    return chunks


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[。；;！!？?])|(?<=\. )', text)
    return [p for p in parts if p.strip()]


# ============ 轻量分词（不依赖 jieba，用简单正则）============
def tokenize(text: str) -> List[str]:
    """简单中英文分词：中文按字/常用双字，英文按单词"""
    text = text.lower()
    # 提取英文单词
    en_words = re.findall(r'[a-z]{2,}', text)
    # 提取中文：按 2-4 字的滑动窗口生成 n-gram
    cn_chars = re.findall(r'[\u4e00-\u9fff]+', text)
    cn_tokens = []
    for seg in cn_chars:
        # 生成 bigram 和 trigram
        for i in range(len(seg)):
            if i + 2 <= len(seg):
                cn_tokens.append(seg[i:i+2])
            if i + 3 <= len(seg):
                cn_tokens.append(seg[i:i+3])
        # 也加入单字（但权重低，靠频率自然调节）
        for ch in seg:
            cn_tokens.append(ch)
    return en_words + cn_tokens


# ============ 索引数据（内存中）============
_index_data: Optional[Dict] = None


def _get_index_path() -> str:
    """获取索引文件的完整路径"""
    # 先检查项目根目录
    root_path = Path(__file__).parent / INDEX_FILE
    if root_path.exists():
        return str(root_path)
    # 再检查当前工作目录
    cwd_path = Path(INDEX_FILE)
    if cwd_path.exists():
        return str(cwd_path)
    return str(root_path)


def load_index() -> Dict:
    """加载索引文件到内存"""
    global _index_data
    if _index_data is not None:
        return _index_data

    index_path = _get_index_path()
    if not Path(index_path).exists():
        _index_data = {"chunks": [], "doc_freq": {}, "total_docs": 0}
        return _index_data

    with open(index_path, "r", encoding="utf-8") as f:
        _index_data = json.load(f)
    return _index_data


def save_index(data: Dict):
    """保存索引到 JSON 文件"""
    global _index_data
    index_path = str(Path(__file__).parent / INDEX_FILE)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    _index_data = data
    print(f"  索引已保存到: {index_path}")


# ============ 索引构建 ============
def build_index(papers_dir: str = PAPERS_DIR, force: bool = False) -> Dict[str, int]:
    """扫描目录下所有 PDF 和 Excel 并构建索引，保存为 JSON。"""
    ensure_dirs()
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        print(f"[错误] 论文目录不存在: {papers_dir}")
        return {}

    pdf_files = list(papers_path.glob("*.pdf")) + list(papers_path.glob("*.PDF"))
    xlsx_files = list(papers_path.glob("*.xlsx")) + list(papers_path.glob("*.XLSX"))
    docx_files = list(papers_path.glob("*.docx")) + list(papers_path.glob("*.DOCX"))
    txt_files = list(papers_path.glob("*_ocr.txt"))

    if not pdf_files and not xlsx_files and not docx_files and not txt_files:
        print(f"[提示] {papers_dir}/ 目录下没有可处理的文件")
        return {}

    all_chunks = []  # [{"text": ..., "title": ..., "filename": ...}, ...]
    results = {}

    # 处理 PDF
    if pdf_files:
        print(f"\n{'='*50}")
        print(f"处理 PDF：{len(pdf_files)} 个文件")
        print(f"{'='*50}\n")
        for pdf_file in sorted(pdf_files):
            try:
                text = extract_text_from_pdf(str(pdf_file))
                if not text.strip():
                    print(f"  [警告] {pdf_file.name} 文本为空，跳过")
                    results[pdf_file.name] = 0
                    continue
                metadata = extract_metadata_from_pdf(str(pdf_file))
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "title": metadata["title"],
                        "filename": metadata["filename"],
                    })
                print(f"  [完成] {pdf_file.name} → {len(chunks)} 个文本块")
                results[pdf_file.name] = len(chunks)
            except Exception as e:
                print(f"  [错误] {pdf_file.name}: {e}")
                results[pdf_file.name] = -1

    # 处理 OCR 文本
    if txt_files:
        print(f"\n{'='*50}")
        print(f"处理 OCR 文本：{len(txt_files)} 个文件")
        print(f"{'='*50}\n")
        for txt_file in sorted(txt_files):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    continue
                chunks = chunk_text(text)
                title = txt_file.stem.replace("_ocr", "")
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "title": title,
                        "filename": txt_file.name,
                    })
                print(f"  [完成] {txt_file.name} → {len(chunks)} 个文本块")
                results[txt_file.name] = len(chunks)
            except Exception as e:
                print(f"  [错误] {txt_file.name}: {e}")
                results[txt_file.name] = -1

    # 处理 Word (.docx)
    if docx_files:
        print(f"\n{'='*50}")
        print(f"处理 Word：{len(docx_files)} 个文件")
        print(f"{'='*50}\n")
        for docx_file in sorted(docx_files):
            try:
                text = extract_text_from_docx(str(docx_file))
                if not text.strip():
                    print(f"  [警告] {docx_file.name} 文本为空，跳过")
                    results[docx_file.name] = 0
                    continue
                title = docx_file.stem
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "title": title,
                        "filename": docx_file.name,
                    })
                print(f"  [完成] {docx_file.name} → {len(chunks)} 个文本块")
                results[docx_file.name] = len(chunks)
            except Exception as e:
                print(f"  [错误] {docx_file.name}: {e}")
                results[docx_file.name] = -1

    # 处理 Excel
    if xlsx_files:
        print(f"\n{'='*50}")
        print(f"处理 Excel：{len(xlsx_files)} 个文件")
        print(f"{'='*50}\n")
        for xlsx_file in sorted(xlsx_files):
            try:
                papers = extract_papers_from_excel(str(xlsx_file))
                for paper in papers:
                    text = paper_to_text(paper)
                    if len(text.strip()) > 30:
                        all_chunks.append({
                            "text": text,
                            "title": paper.get("title", ""),
                            "filename": xlsx_file.name,
                        })
                print(f"  [完成] {xlsx_file.name} → {len(papers)} 篇论文摘要")
                results[xlsx_file.name] = len(papers)
            except Exception as e:
                print(f"  [错误] {xlsx_file.name}: {e}")
                results[xlsx_file.name] = -1

    # 计算文档频率（IDF 用）
    doc_freq: Dict[str, int] = Counter()
    for chunk in all_chunks:
        tokens = set(tokenize(chunk["text"]))
        for token in tokens:
            doc_freq[token] += 1

    # 保存索引
    index_data = {
        "chunks": all_chunks,
        "doc_freq": dict(doc_freq),
        "total_docs": len(all_chunks),
    }
    save_index(index_data)

    print(f"\n{'='*50}")
    print(f"索引完成：{len(all_chunks)} 个文本块，来自 {len(results)} 个文件")
    print(f"{'='*50}\n")

    return results


# ============ BM25 检索 ============
def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """BM25 关键词检索，内存友好。"""
    index = load_index()
    chunks = index.get("chunks", [])
    doc_freq = index.get("doc_freq", {})
    total_docs = index.get("total_docs", 0)

    if not chunks or total_docs == 0:
        return []

    # BM25 参数
    k1 = 1.5
    b = 0.75

    # 计算平均文档长度
    avg_dl = sum(len(c["text"]) for c in chunks) / total_docs

    # 对 query 分词
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # 计算每个文档的 BM25 分数
    scores = []
    for i, chunk in enumerate(chunks):
        doc_text = chunk["text"]
        doc_len = len(doc_text)
        doc_tokens = tokenize(doc_text)
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for token in query_tokens:
            if token not in tf_counter:
                continue
            tf = tf_counter[token]
            df = doc_freq.get(token, 0)
            if df == 0:
                continue
            # IDF
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            # TF 归一化
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
            score += idf * tf_norm

        if score > 0:
            scores.append((score, i))

    # 排序取 top_k
    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "title": chunk.get("title", "未知"),
            "filename": chunk.get("filename", ""),
            "score": round(score, 4),
            "chunk_index": idx,
        })

    return results


def format_context_for_prompt(search_results: List[Dict], max_chars: int = 6000) -> str:
    """将检索结果格式化为可注入 system prompt 的参考文本。

    max_chars 默认 6000：deepseek 64K context 完全够用，3000 在 top_k=10 时
    截断太狠会让"深度思考多检索"白费。
    """
    if not search_results:
        return ""

    parts = []
    total_len = 0
    for i, item in enumerate(search_results, 1):
        entry = f"【参考{i}】（来源：{item['title']}）\n{item['text']}\n"
        if total_len + len(entry) > max_chars:
            break
        parts.append(entry)
        total_len += len(entry)

    if not parts:
        return ""

    header = "以下是从刘春生教授团队发表的论文中检索到的相关内容，回答时必须以这些内容为依据，不得编造论文中没有的信息：\n\n"
    return header + "\n".join(parts)


# ============ 状态查询 ============
def get_index_stats() -> Dict:
    """获取索引状态统计"""
    try:
        index = load_index()
        chunks = index.get("chunks", [])
        if chunks:
            filenames = set(c.get("filename", "") for c in chunks)
            return {
                "total_chunks": len(chunks),
                "total_files": len(filenames),
                "files": sorted(filenames),
                "status": "ready",
            }
        else:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "files": [],
                "status": "empty",
            }
    except Exception as e:
        return {
            "total_chunks": 0,
            "total_files": 0,
            "files": [],
            "status": f"error: {e}",
        }
