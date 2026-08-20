#!/bin/bash
# Lumen Agent 沙盒部署：拉取代码、构建镜像并直接重建容器。

set -euo pipefail

AGENT_HOST="${AGENT_HOST:-agent}"
SSH_CONFIG="${DEPLOY_SSH_CONFIG:-$HOME/.ssh/config}"
REMOTE_DIR="data-agen"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}"
IMAGE_TAG="data-agen:latest-$(date +%Y%m%d%H%M%S)"

do_ssh() {
  ssh -F "$SSH_CONFIG" "$AGENT_HOST" "$@"
}

echo "拉取沙盒 Agent 最新代码"
do_ssh "cd ~/$REMOTE_DIR && \
  git fetch origin $DEPLOY_BRANCH && \
  git checkout $DEPLOY_BRANCH && \
  git pull --ff-only origin $DEPLOY_BRANCH"

echo "构建镜像"
do_ssh "cd ~/$REMOTE_DIR && \
  docker build -t $IMAGE_TAG . && \
  docker tag $IMAGE_TAG data-agen:latest"

echo "启动最新容器"
do_ssh "cd ~/$REMOTE_DIR && \
  docker rm -f data-agen >/dev/null 2>&1 || true; \
  docker run -d \
    --name data-agen \
    --restart always \
    --gpus all \
    -p 9090:9090 \
    --env-file .env \
    -e NL2SQL_ENV=prod \
    -e DENSE_DEVICE=cuda \
    -e CONFIG_SOURCE=mysql \
    -e CONFIG_PROFILE=1 \
    -e REBUILD_INDEX_ON_STARTUP=false \
    -e CODEX_HOME=/var/lib/lumen-codex \
    -v data-agen_model-cache:/root/.cache:z \
    -v data-agen_codex-home:/var/lib/lumen-codex:z \
    data-agen:latest"

echo "等待服务就绪"
for attempt in $(seq 1 18); do
  if do_ssh "curl -fsS http://127.0.0.1:9090/health"; then
    echo
    echo "部署完成: $IMAGE_TAG"
    exit 0
  fi
  if [ "$attempt" -eq 18 ]; then
    break
  fi
  sleep 10
done

echo "服务未在预期时间内就绪，最近日志如下"
do_ssh "docker logs --tail 100 data-agen"
exit 1
