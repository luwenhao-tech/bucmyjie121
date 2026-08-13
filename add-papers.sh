#!/usr/bin/env bash
# 本地：加论文 → 重建索引 → 推 GitHub → 同步服务器
# 用法：./add-papers.sh "feat: 新增 X 篇论文"
set -e

MSG="${1:-feat: 新增论文并重建索引}"
SERVER="root@47.82.64.166"
REMOTE_DIR="/root/tcm-liuchunsheng"
SERVICE="tcm"

echo "==> [1/4] 本地重建 RAG 索引..."
python3 build_index.py

echo "==> [2/4] 提交到 Git..."
git add papers/ papers_index.json chroma_db/ scoring/credibility_scores.json 2>/dev/null || true
git add -A
git commit -m "$MSG" || echo "   (没有新变更)"

echo "==> [3/4] 推送 GitHub..."
git push origin main

echo "==> [4/4] 服务器拉取并重启..."
ssh "$SERVER" "cd $REMOTE_DIR && git pull && pip install -r requirements.txt -q && systemctl restart $SERVICE && systemctl status $SERVICE --no-pager | head -5"

echo "✅ 完成。访问 https://lcsbucm.tech/ 验证。"
