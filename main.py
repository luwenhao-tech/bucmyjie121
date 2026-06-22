"""FastAPI 入口：中药鉴定学 - 刘春生教授 AI 助教。
支持多轮对话历史，流式 SSE 输出。
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_client import generate_stream, generate, vision_client, generate_followups, resolve_intent_extra, classify_intent

# RAG 检索引擎（论文知识库）
try:
    from rag_engine import search as rag_search, format_context_for_prompt, get_index_stats
    _rag_available = True
except ImportError:
    _rag_available = False
    print("[INFO] RAG 模块未加载（缺少依赖），将不使用论文检索")

app = FastAPI(title="中药鉴定学 - 刘春生教授 AI 助教")

# CORS：默认收紧到自有域名，可通过 ALLOWED_ORIGINS 环境变量覆盖（逗号分隔）
_default_origins = "https://lcsbucm.tech,https://www.lcsbucm.tech,http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============ 日志数据库 ============
DB_PATH = Path(__file__).parent / "chat_logs.db"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    print("[WARN] 未设置 ADMIN_PASSWORD 环境变量，/admin 接口将禁用以防弱口令暴露")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ip TEXT,
            ua TEXT,
            user_name TEXT,
            user_id TEXT,
            prompt TEXT,
            answer TEXT,
            think INTEGER DEFAULT 0,
            duration_ms INTEGER
        )
    """)
    # 给历史库补字段（已存在则忽略错误）
    for col in ("user_name TEXT", "user_id TEXT"):
        try:
            conn.execute(f"ALTER TABLE chat_log ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # 登录 token 持久化
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_token (
            token TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            name TEXT,
            expires REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ============ 账号系统 ============
ACCOUNTS_PATH = Path(__file__).parent / "accounts.json"
TOKEN_TTL = 7 * 24 * 3600  # 7 天
PASSWORD_SALT = os.getenv("PASSWORD_SALT", "lcsbucm-tcm-2025")


def _hash_password(raw: str) -> str:
    """sha256(salt + password)，加 'sha256:' 前缀以兼容明文存量。"""
    h = hashlib.sha256((PASSWORD_SALT + raw).encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def _verify_password(raw: str, stored: str) -> bool:
    """支持新格式（sha256:xxx）和遗留明文。"""
    if not stored:
        return False
    if stored.startswith("sha256:"):
        return secrets.compare_digest(stored, _hash_password(raw))
    # 遗留明文：比对成功后由调用方负责升级
    return secrets.compare_digest(stored, raw)


def load_accounts() -> Dict[str, str]:
    """读取 accounts.json：{ "account": "password_or_hash", ... }"""
    if not ACCOUNTS_PATH.exists():
        return {}
    try:
        with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k.strip(): str(v) for k, v in data.items() if not k.startswith("_")}
        if isinstance(data, list):
            return {item["account"].strip(): str(item["password"]) for item in data if "account" in item}
        return {}
    except Exception as e:
        print(f"[load_accounts error] {e}")
        return {}


def issue_token(account: str, name: str) -> str:
    tk = secrets.token_urlsafe(24)
    expires = time.time() + TOKEN_TTL
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO auth_token (token, account, name, expires) VALUES (?, ?, ?, ?)",
            (tk, account, name, expires),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[issue_token error] {e}")
    return tk


def verify_token(token: Optional[str]) -> Optional[Dict]:
    if not token:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT account, name, expires FROM auth_token WHERE token=?", (token,)
        ).fetchone()
        if not row:
            conn.close()
            return None
        if row["expires"] < time.time():
            conn.execute("DELETE FROM auth_token WHERE token=?", (token,))
            conn.commit()
            conn.close()
            return None
        conn.close()
        return {"account": row["account"], "name": row["name"], "expires": row["expires"]}
    except Exception as e:
        print(f"[verify_token error] {e}")
        return None


def revoke_account_tokens(account: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM auth_token WHERE account=?", (account,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[revoke_account_tokens error] {e}")


# ============ 接口限流（同一账号每分钟最多 N 次聊天）============
RATE_LIMIT_WINDOW = 60  # 秒
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))
_rate_buckets: Dict[str, deque] = defaultdict(deque)


def rate_limit_check(account: str):
    now = time.time()
    bucket = _rate_buckets[account]
    while bucket and bucket[0] < now - RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX:
        retry = int(RATE_LIMIT_WINDOW - (now - bucket[0])) + 1
        raise HTTPException(429, f"问得太快了，{retry} 秒后再试。")
    bucket.append(now)


class LoginRequest(BaseModel):
    account: str
    password: str
    name: Optional[str] = ""


@app.post("/api/login")
async def api_login(req: LoginRequest):
    accounts = load_accounts()
    acc = req.account.strip()
    if not acc or acc not in accounts:
        raise HTTPException(401, "账号不存在")
    stored = accounts[acc]
    if not _verify_password(req.password, stored):
        raise HTTPException(401, "密码错误")
    # 遗留明文密码 → 登录成功后自动升级为哈希
    if not stored.startswith("sha256:"):
        accounts[acc] = _hash_password(req.password)
        try:
            save_accounts(accounts)
        except Exception as e:
            print(f"[upgrade hash error] {e}")
    name = (req.name or "").strip()[:32] or acc
    token = issue_token(acc, name)
    return {"token": token, "account": acc, "name": name}


def require_user(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization:
        raise HTTPException(401, "请先登录")
    token = authorization.replace("Bearer ", "").strip()
    info = verify_token(token)
    if not info:
        raise HTTPException(401, "登录已过期，请重新登录")
    return info


def log_chat(ip: str, ua: str, user_name: str, user_id: str, prompt: str, answer: str, think: bool, duration_ms: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO chat_log (ts, ip, ua, user_name, user_id, prompt, answer, think, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ip, ua, user_name, user_id, prompt, answer, 1 if think else 0, duration_ms),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[log_chat error] {e}")


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


_FOLLOWUP_LINE_RE = re.compile(r"(?:^|\n)\s*💬[^\n]*\s*$")


def strip_followup(text: str) -> str:
    """剥掉模型可能在末尾追加的『💬 …』反问行（含其前面的空行）。"""
    if not text:
        return text
    # 反复剥，防止模型连发多行 💬
    out = text
    while True:
        new = _FOLLOWUP_LINE_RE.sub("", out).rstrip()
        if new == out:
            break
        out = new
    return out


class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Message]] = None
    temperature: float = 0.6
    stream: bool = True
    think: bool = False
    image: Optional[str] = None  # base64 data URL: "data:image/jpeg;base64,..."
    user_name: Optional[str] = None
    user_id: Optional[str] = None
    intent: Optional[str] = None  # identify | concept | exam | compare


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request, user: Dict = Depends(require_user)):
    if not req.prompt.strip() and not req.image:
        raise HTTPException(400, "问题不能为空")

    # 后端兜底：图片大小校验（base64 字符串长度 * 0.75 ≈ 字节数）
    if req.image:
        if not req.image.startswith("data:image/"):
            raise HTTPException(400, "图片格式不合法")
        # 6MB base64 ≈ 4.5MB 原图，超出直接拒
        if len(req.image) > 6 * 1024 * 1024:
            raise HTTPException(413, "图片过大，请压缩后重传（建议 ≤4MB）")
        allowed_prefixes = ("data:image/jpeg", "data:image/png", "data:image/webp")
        if not req.image.startswith(allowed_prefixes):
            raise HTTPException(400, "仅支持 JPG / PNG / WebP 格式")

    # 限流：同一账号每分钟最多 RATE_LIMIT_MAX 次
    rate_limit_check(user["account"])

    history_dicts: List[Dict[str, str]] = (
        [m.model_dump() for m in req.history] if req.history else []
    )

    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "")
    client_ip = client_ip.split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "")
    # 用 token 里的 account/name 而不是前端自报，杜绝伪造
    user_name = user["name"][:32]
    user_id = user["account"][:32]
    started = time.time()

    # 日志里的 prompt 标记是否带图
    prompt_for_log = req.prompt + ("  [📷 含图片]" if req.image else "")
    # 自动识别意图（前端不再传，全部由后端判别）
    detected_intent = await classify_intent(req.prompt, has_image=bool(req.image))
    intent_extra = resolve_intent_extra(detected_intent)
    if detected_intent:
        prompt_for_log = f"[意图:{detected_intent}] " + prompt_for_log

    # RAG 检索：从论文库中查找相关内容
    rag_context = ""
    if _rag_available and req.prompt.strip() and not req.image:
        try:
            results = rag_search(req.prompt)
            if results:
                rag_context = format_context_for_prompt(results)
        except Exception as e:
            print(f"[RAG search error] {e}")

    if not req.stream:
        text = await generate(
            req.prompt, history=history_dicts,
            temperature=req.temperature, think=req.think,
            image_data=req.image, user_name=user_name,
            extra_system=intent_extra,
            rag_context=rag_context,
        )
        text = strip_followup(text)
        log_chat(client_ip, user_agent, user_name, user_id, prompt_for_log, text, req.think, int((time.time() - started) * 1000))
        return {"content": text}

    async def event_stream():
        full_answer = ""
        emitted_len = 0  # 已经流给前端的 full_answer 前缀长度
        try:
            async for token in generate_stream(
                req.prompt, history=history_dicts,
                temperature=req.temperature, think=req.think,
                image_data=req.image, user_name=user_name,
                extra_system=intent_extra,
                rag_context=rag_context,
            ):
                full_answer += token
                # 检查累计文本里是否已经出现了 💬 行的开头：一旦出现，停止往前端流
                idx = full_answer.find("💬")
                if idx == -1:
                    safe_until = len(full_answer)
                else:
                    safe_until = idx
                if safe_until > emitted_len:
                    delta = full_answer[emitted_len:safe_until]
                    emitted_len = safe_until
                    yield f"data: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
                # 出现了 💬 之后的内容直接丢弃，不再下发
            # 流结束：剥掉 💬 行后落库
            full_answer = strip_followup(full_answer)
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            log_chat(client_ip, user_agent, user_name, user_id, prompt_for_log, full_answer, req.think, int((time.time() - started) * 1000))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/features")
async def api_features():
    """前端启动时查询哪些可选功能（如视觉、RAG）可用。"""
    rag_stats = get_index_stats() if _rag_available else {"status": "unavailable"}
    return {
        "vision": vision_client is not None,
        "rag": _rag_available and rag_stats.get("total_chunks", 0) > 0,
        "rag_stats": rag_stats,
    }


class FollowupReq(BaseModel):
    prompt: str
    answer: str
    existing: Optional[List[str]] = None


@app.post("/api/followups")
async def api_followups(req: FollowupReq, user: Dict = Depends(require_user)):
    """根据本次问答 + 当前页面已出现的追问，生成 3 条全新追问。"""
    if not req.answer.strip():
        return {"followups": []}
    qs = await generate_followups(req.prompt or "", req.answer, req.existing or [])
    # 二次保险：服务端再做一次去重（按归一化键）
    seen = set()
    norm = lambda s: re.sub(r"[\s。．\.,，、!！?？:：;；~～\-—_…\"'《》()（）\[\]【】]", "", (s or "")).lower()
    for e in (req.existing or []):
        seen.add(norm(e))
    out = []
    for q in qs:
        k = norm(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return {"followups": out}


# ============ 管理后台 ============
def check_admin(password: Optional[str]) -> bool:
    # 未设置环境变量则一律拒绝，防止默认 admin123 被滥用
    if not ADMIN_PASSWORD:
        return False
    return password == ADMIN_PASSWORD


@app.get("/api/admin/logs")
async def admin_logs(password: str = "", limit: int = 200, offset: int = 0):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM chat_log ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
    conn.close()
    return {"total": total, "logs": [dict(r) for r in rows]}


@app.get("/api/admin/stats")
async def admin_stats(password: str = ""):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = cur.execute("SELECT COUNT(*) FROM chat_log WHERE ts LIKE ?", (today + "%",)).fetchone()[0]
    unique_ips = cur.execute("SELECT COUNT(DISTINCT ip) FROM chat_log").fetchone()[0]
    today_ips = cur.execute("SELECT COUNT(DISTINCT ip) FROM chat_log WHERE ts LIKE ?", (today + "%",)).fetchone()[0]
    think_count = cur.execute("SELECT COUNT(*) FROM chat_log WHERE think=1").fetchone()[0]
    think_rate = round(think_count / total * 100, 1) if total else 0

    # 最近 7 天每日问答数
    daily = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        c = cur.execute("SELECT COUNT(*) FROM chat_log WHERE ts LIKE ?", (d + "%",)).fetchone()[0]
        daily.append({"date": d[5:], "count": c})

    # 热门关键词（按 prompt 出现的常见药材/术语统计）
    hot_keywords = ["人参", "黄连", "大黄", "甘草", "当归", "何首乌", "三七", "天麻",
                    "川芎", "白芍", "黄芪", "防风", "党参", "川贝母", "浙贝母",
                    "菊花心", "朱砂点", "起霜", "车轮纹", "鹦哥嘴", "过桥",
                    "狮子盘头", "怀中抱月", "云锦花纹", "金井玉栏", "蚯蚓头", "星点", "珍珠疙瘩"]
    hot = []
    for kw in hot_keywords:
        c = cur.execute("SELECT COUNT(*) FROM chat_log WHERE prompt LIKE ?", (f"%{kw}%",)).fetchone()[0]
        if c > 0:
            hot.append({"keyword": kw, "count": c})
    hot.sort(key=lambda x: -x["count"])

    # 热门 IP
    top_ips = cur.execute(
        "SELECT ip, COUNT(*) AS c FROM chat_log GROUP BY ip ORDER BY c DESC LIMIT 10"
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "today_count": today_count,
        "unique_ips": unique_ips,
        "today_ips": today_ips,
        "think_count": think_count,
        "think_rate": think_rate,
        "daily": daily,
        "hot_keywords": hot[:20],
        "top_ips": [{"ip": r[0], "count": r[1]} for r in top_ips],
    }


# ============ 账号管理 API（管理员）============
def save_accounts(accounts: Dict[str, str]):
    """写回 accounts.json，保留下划线开头字段。"""
    existing = {}
    if ACCOUNTS_PATH.exists():
        try:
            with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    existing = {k: v for k, v in raw.items() if k.startswith("_")}
        except Exception:
            pass
    merged = {**existing, **accounts}
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


class AccountAddReq(BaseModel):
    password: str
    account: str
    user_password: str


class AccountDelReq(BaseModel):
    password: str
    account: str


@app.get("/api/admin/accounts")
async def admin_accounts_list(password: str = ""):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    accs = load_accounts()
    # 不回传密码/哈希，避免泄露
    return {"accounts": [{"account": k, "has_password": bool(v)} for k, v in accs.items()]}


@app.post("/api/admin/accounts/add")
async def admin_accounts_add(req: AccountAddReq):
    if not check_admin(req.password):
        raise HTTPException(401, "密码错误")
    acc = req.account.strip()
    if not acc or acc.startswith("_"):
        raise HTTPException(400, "账号不合法（不能为空，不能以下划线开头）")
    if len(acc) > 32:
        raise HTTPException(400, "账号过长（最多 32 字符）")
    if not req.user_password:
        raise HTTPException(400, "密码不能为空")
    accs = load_accounts()
    accs[acc] = _hash_password(req.user_password)
    save_accounts(accs)
    return {"ok": True}


@app.post("/api/admin/accounts/delete")
async def admin_accounts_delete(req: AccountDelReq):
    if not check_admin(req.password):
        raise HTTPException(401, "密码错误")
    accs = load_accounts()
    if req.account in accs:
        accs.pop(req.account)
        save_accounts(accs)
        # 立即使该账号当前所有 token 失效
        revoke_account_tokens(req.account)
    return {"ok": True}


# 静态资源
static_dir = Path(__file__).parent / "static"

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html", headers=NO_CACHE_HEADERS)

    @app.get("/admin")
    async def admin_page():
        return FileResponse(static_dir / "admin.html", headers=NO_CACHE_HEADERS)
