#!/usr/bin/env bash
# 一键部署到阿里云：本地 push -> 服务器 pull -> 重启 tcm.service -> 验证
# 用法：./deploy.sh          （需要先 git push 完）
#      ./deploy.sh --push   （连带先帮您 push）

set -euo pipefail

SERVER="root@47.82.64.166"
PROJECT_DIR="/root/tcm-liuchunsheng"
SERVICE="tcm.service"

# 选项：--push 时先在本地推一次
if [[ "${1:-}" == "--push" ]]; then
  echo "→ 本地 git push…"
  git push
fi

LOCAL_HEAD=$(git rev-parse --short HEAD)
echo "→ 本地 HEAD: ${LOCAL_HEAD}"

echo "→ 服务器 git pull…"
ssh "$SERVER" "cd ${PROJECT_DIR} && git pull --ff-only && git log --oneline -1"

echo "→ 重启 ${SERVICE}…"
ssh "$SERVER" "systemctl restart ${SERVICE} && sleep 2 && systemctl is-active ${SERVICE}"

echo "→ 检查服务最近日志…"
ssh "$SERVER" "journalctl -u ${SERVICE} -n 5 --no-pager"

echo ""
echo "✅ 部署完成。打开 https://lcsbucm.tech/ 验证。"
