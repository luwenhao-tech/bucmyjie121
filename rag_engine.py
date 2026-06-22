"""RAG 引擎：PDF/Excel 论文解析 + 向量检索，让 AI 只基于论文内容回答。

使用方式：
1. 把论文 PDF 或 Excel（含论文摘要信息）放入 papers/ 目录
2. 运行 python build_index.py 构建索引
3. 启动服务后，聊天时自动从论文中检索相关内容
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============ 配置 ============
PAPERS_DIR = os.getenv("PAPERS_DIR", "papers")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))  # 每个文本块的目标字符数
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))  # 相邻块重叠字符数
TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # 检索返回的最相关片段数
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))  # 相似度阈值（越小越严格）


def ensure_dirs():
    """确保必要目录存在"""
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)


# ============ PDF 解析 ============
def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取全部文本"""
    import fitz  # PyMuPDF

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
    """提取 PDF 元数据（标题、作者等）"""
    import fitz

    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}
    doc.close()

    return {
        "title": metadata.get("title", "") or Path(pdf_path).stem,
        "author": metadata.get("author", ""),
        "filename": Path(pdf_path).name,
        "filepath": pdf_path,
    }


# ============ Excel 解析（论文摘要表格）============
def extract_papers_from_excel(xlsx_path: str) -> List[Dict[str, str]]:
    """从 Excel 表格中提取论文信息。

    期望的列结构：论文名称、作者、研究方法、关键词、摘要、期刊、年份
    返回列表，每项是一篇论文的完整信息字典。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    papers = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            continue

        # 读取表头，自动匹配列
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
            continue  # 没找到标题列，跳过此 sheet

        # 逐行读取论文数据
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
    """将一篇论文的信息组合成可检索的文本段落"""
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
    """将长文本切成小块，带重叠窗口。

    策略：
    1. 优先按段落（双换行）切分
    2. 单段过长时按句号/分号切分
    3. 相邻块有 overlap 字符重叠，保证上下文连贯
    """
    if not text or not text.strip():
        return []

    # 先按段落切
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果当前段落本身就超长，按句子再拆
        if len(para) > chunk_size:
            # 先把之前积累的 chunk 存下
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 按句号/分号/问号拆这个长段落
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

    # 添加重叠：把前一个块的尾部拼到后一个块的头部
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped_chunks.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped_chunks

    # 过滤太短的块（可能是噪音）
    chunks = [c for c in chunks if len(c.strip()) > 30]

    return chunks


def _split_sentences(text: str) -> List[str]:
    """按中英文句号/分号/问号切分句子"""
    import re
    # 匹配中文句号、英文句号+空格、分号、问号、感叹号
    parts = re.split(r'(?<=[。；;！!？?])|(?<=\. )', text)
    return [p for p in parts if p.strip()]


# ============ 向量数据库 ============
_collection = None


def get_collection():
    """获取/初始化 ChromaDB collection（懒加载单例）"""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.config import Settings

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # 使用 ChromaDB 内置的 embedding（multilingual-MiniLM，支持中英文）
    _collection = chroma_client.get_or_create_collection(
        name="tcm_papers",
        metadata={"hnsw:space": "cosine"},  # 用余弦相似度
    )
    return _collection


def file_hash(filepath: str) -> str:
    """计算文件 MD5，用于检测是否需要重新索引"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============ 索引构建 ============
def index_pdf(pdf_path: str, force: bool = False) -> int:
    """索引单个 PDF 文件，返回新增的 chunk 数。

    如果文件已经索引过（MD5 没变）且 force=False，跳过。
    """
    collection = get_collection()
    fpath = str(Path(pdf_path).resolve())
    fhash = file_hash(fpath)
    doc_id_prefix = f"doc_{fhash}"

    # 检查是否已索引
    if not force:
        existing = collection.get(where={"file_hash": fhash}, limit=1)
        if existing and existing["ids"]:
            print(f"  [跳过] {Path(pdf_path).name} (已索引，MD5 未变)")
            return 0

    # 如果是强制重建，先删除旧数据
    old_data = collection.get(where={"filepath": fpath})
    if old_data and old_data["ids"]:
        collection.delete(ids=old_data["ids"])

    # 解析 PDF
    text = extract_text_from_pdf(fpath)
    if not text.strip():
        print(f"  [警告] {Path(pdf_path).name} 提取文本为空，跳过")
        return 0

    metadata = extract_metadata_from_pdf(fpath)

    # 切块
    chunks = chunk_text(text)
    if not chunks:
        print(f"  [警告] {Path(pdf_path).name} 切块后为空，跳过")
        return 0

    # 批量写入 ChromaDB
    ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "title": metadata["title"],
            "author": metadata["author"],
            "filename": metadata["filename"],
            "filepath": fpath,
            "file_hash": fhash,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # ChromaDB 批量上限约 5000，分批处理
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"  [完成] {metadata['filename']} → {len(chunks)} 个文本块")
    return len(chunks)


def index_excel(xlsx_path: str, force: bool = False) -> int:
    """索引 Excel 表格中的论文摘要信息，返回新增的 chunk 数。"""
    collection = get_collection()
    fpath = str(Path(xlsx_path).resolve())
    fhash = file_hash(fpath)
    doc_id_prefix = f"xlsx_{fhash}"

    # 检查是否已索引
    if not force:
        existing = collection.get(where={"file_hash": fhash}, limit=1)
        if existing and existing["ids"]:
            print(f"  [跳过] {Path(xlsx_path).name} (已索引，MD5 未变)")
            return 0

    # 如果是强制重建，先删除旧数据
    old_data = collection.get(where={"filepath": fpath})
    if old_data and old_data["ids"]:
        collection.delete(ids=old_data["ids"])

    # 解析 Excel
    papers = extract_papers_from_excel(fpath)
    if not papers:
        print(f"  [警告] {Path(xlsx_path).name} 未提取到论文数据，跳过")
        return 0

    # 每篇论文生成一个完整文本块
    chunks = []
    metadatas_list = []
    for i, paper in enumerate(papers):
        text = paper_to_text(paper)
        if len(text.strip()) < 30:
            continue
        chunks.append(text)
        metadatas_list.append({
            "title": paper.get("title", ""),
            "author": paper.get("author", ""),
            "filename": Path(xlsx_path).name,
            "filepath": fpath,
            "file_hash": fhash,
            "chunk_index": i,
            "total_chunks": len(papers),
            "source_type": "excel",
        })

    if not chunks:
        print(f"  [警告] {Path(xlsx_path).name} 切块后为空，跳过")
        return 0

    # 批量写入
    ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas_list[start:end],
        )

    print(f"  [完成] {Path(xlsx_path).name} → {len(chunks)} 篇论文摘要")
    return len(chunks)


def build_index(papers_dir: str = PAPERS_DIR, force: bool = False) -> Dict[str, int]:
    """扫描目录下所有 PDF 和 Excel 并构建索引。

    返回 {文件名: chunk数} 的字典。
    """
    ensure_dirs()
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        print(f"[错误] 论文目录不存在: {papers_dir}")
        return {}

    pdf_files = list(papers_path.glob("*.pdf")) + list(papers_path.glob("*.PDF"))
    xlsx_files = list(papers_path.glob("*.xlsx")) + list(papers_path.glob("*.XLSX"))

    if not pdf_files and not xlsx_files:
        print(f"[提示] {papers_dir}/ 目录下没有 PDF 或 Excel 文件")
        print("  请将论文文件放入该目录后重新运行")
        return {}

    results = {}
    total_chunks = 0

    # 处理 PDF 文件
    if pdf_files:
        print(f"\n{'='*50}")
        print(f"开始构建索引：找到 {len(pdf_files)} 个 PDF 文件")
        print(f"{'='*50}\n")

        for pdf_file in sorted(pdf_files):
            try:
                n = index_pdf(str(pdf_file), force=force)
                results[pdf_file.name] = n
                total_chunks += n
            except Exception as e:
                print(f"  [错误] {pdf_file.name}: {e}")
                results[pdf_file.name] = -1

    # 处理 Excel 文件（论文摘要表格）
    if xlsx_files:
        print(f"\n{'='*50}")
        print(f"开始处理 Excel：找到 {len(xlsx_files)} 个表格文件")
        print(f"{'='*50}\n")

        for xlsx_file in sorted(xlsx_files):
            try:
                n = index_excel(str(xlsx_file), force=force)
                results[xlsx_file.name] = n
                total_chunks += n
            except Exception as e:
                print(f"  [错误] {xlsx_file.name}: {e}")
                results[xlsx_file.name] = -1

    print(f"\n{'='*50}")
    print(f"索引完成：{len(results)} 个文件，{total_chunks} 个新文本块")
    collection = get_collection()
    print(f"数据库总量：{collection.count()} 个文本块")
    print(f"{'='*50}\n")

    return results


# ============ 检索 ============
def search(query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """根据用户问题检索最相关的论文片段。

    返回列表，每项包含：
    - text: 文本内容
    - title: 来源论文标题
    - filename: 文件名
    - score: 相似度分数（0~1，越高越相关）
    - chunk_index: 在论文中的位置
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = 完全相同, 2 = 完全相反
        # 转换为相似度: 1 - distance/2
        similarity = 1 - dist / 2
        if similarity < (1 - threshold):  # threshold 越小越严格
            continue
        items.append({
            "text": doc,
            "title": meta.get("title", "未知"),
            "filename": meta.get("filename", ""),
            "score": round(similarity, 4),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return items


def format_context_for_prompt(search_results: List[Dict], max_chars: int = 3000) -> str:
    """将检索结果格式化为可注入 system prompt 的参考文本。

    限制总字符数避免 prompt 过长。
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
        collection = get_collection()
        count = collection.count()

        # 获取已索引文件列表
        if count > 0:
            all_data = collection.get(include=["metadatas"], limit=1)
            # 获取所有唯一文件名
            all_meta = collection.get(include=["metadatas"])
            filenames = set()
            for meta in (all_meta["metadatas"] or []):
                if meta and meta.get("filename"):
                    filenames.add(meta["filename"])
            return {
                "total_chunks": count,
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
