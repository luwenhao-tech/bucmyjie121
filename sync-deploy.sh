#!/usr/bin/env bash
# 只做部署：不加论文，只把最新代码同步到服务器
# 用法：./sync-deploy.sh
set -e

SERVER="root@47.82.64.166"
REMOTE_DIR="/root/tcm-liuchunsheng"
SERVICE="tcm"

echo "==> 推送 GitHub..."
git push origin main

echo "==> 服务器拉取并重启..."
ssh "$SERVER" "cd $REMOTE_DIR && git pull && pip install -r requirements.txt -q && systemctl restart $SERVICE && systemctl status $SERVICE --no-pager | head -5"

echo "✅ 部署完成：https://lcsbucm.tech/"
