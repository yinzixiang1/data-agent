#!/bin/bash
# Lumen RAG 引擎部署脚本
# 用法: ./deploy-rag.sh [--rebuild] [--sync-deps]
#   --rebuild     启动后全量重建向量索引（修改了语义层数据时使用）
#   --sync-deps   同步依赖（pyproject.toml 变更时使用，uv pip install）
#
# 流程: rsync 源码 → (可选)安装依赖 → 停止旧进程 → 启动新进程
#
# 环境: node2 使用 uv + .venv 管理 Python 依赖

set -e

# ── 配置 ──
BASTION_KEY="$HOME/company/BI/uat-infra-msk-bastion.pem"
NODE_KEY="$HOME/company/BI/nl2sql/sg-sandbox-private.pem"
BASTION="rocky@13.228.165.213"
NODE="ec2-user@10.2.2.16"
REMOTE_DIR="data-agen"
PROXY_CMD="ssh -o StrictHostKeyChecking=no -i $BASTION_KEY -W %h:%p $BASTION"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 解析参数
REBUILD=""
SYNC_DEPS=false
for arg in "$@"; do
  case $arg in
    --rebuild) REBUILD="REBUILD_INDEX_ON_STARTUP=true" ;;
    --sync-deps) SYNC_DEPS=true ;;
  esac
done

do_ssh() {
  ssh -o StrictHostKeyChecking=no -o "ProxyCommand=$PROXY_CMD" -i "$NODE_KEY" "$NODE" "$@"
}

do_rsync() {
  rsync -avz --delete \
    -e "ssh -o StrictHostKeyChecking=no -o 'ProxyCommand=$PROXY_CMD' -i $NODE_KEY" \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    --exclude '.DS_Store' \
    --exclude '.env' \
    --exclude '.env.*' \
    --exclude 'milvus/' \
    --exclude 'index_store/' \
    --exclude 'config/' \
    --exclude '*.egg-info/' \
    --exclude 'dist/' \
    --exclude 'build/' \
    --exclude '*.bin' \
    --exclude '*.safetensors' \
    --exclude 'logs/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude 'models/' \
    "$@"
}

echo "=== 1. 同步源码到 node2 ==="
do_rsync "$PROJECT_DIR/" "$NODE:~/$REMOTE_DIR/"

# 可选: 同步依赖
if [ "$SYNC_DEPS" = true ]; then
  echo ""
  echo "=== 2.5 安装/更新依赖 (uv) ==="
  do_ssh "export PATH=\$HOME/.local/bin:\$PATH && cd ~/$REMOTE_DIR && uv pip install --python .venv/bin/python . pymilvus langchain langchain-openai langchain-anthropic anthropic"
fi

echo ""
echo "=== 3. 停止旧进程 ==="
do_ssh "PID=\$(lsof -ti :9090 2>/dev/null); if [ -n \"\$PID\" ]; then kill \$PID && echo \"已停止旧进程 (PID: \$PID)\"; else echo '无运行中的进程'; fi" || true
sleep 3

echo ""
echo "=== 4. 启动 RAG 引擎 ==="
# nohup + setsid + stdin 重定向 /dev/null，确保 SSH 退出后进程不受影响
do_ssh "bash -c 'mkdir -p ~/$REMOTE_DIR/logs && cd ~/$REMOTE_DIR && export NL2SQL_ENV=prod CONFIG_SOURCE=mysql CONFIG_PROFILE=1 $REBUILD && nohup setsid .venv/bin/uvicorn app:app --host 0.0.0.0 --port 9090 > logs/app.log 2>&1 < /dev/null & disown'"
sleep 5

echo ""
echo "=== 5. 验证启动 ==="
do_ssh "lsof -i :9090 2>/dev/null | head -3 && echo '' && tail -10 ~/$REMOTE_DIR/logs/app.log 2>/dev/null" || echo "启动可能失败，请检查日志"

if [ -n "$REBUILD" ]; then
  echo ""
  echo "[INFO] 已启用索引重建 (REBUILD_INDEX_ON_STARTUP=true)"
  echo "[INFO] 首次启动较慢，可通过以下命令查看日志："
else
  echo ""
  echo "[INFO] 未重建索引，复用已有 Collection"
fi

echo "  ssh → tail -f ~/$REMOTE_DIR/logs/app.log"
echo ""
echo "=== 部署完成 ==="
