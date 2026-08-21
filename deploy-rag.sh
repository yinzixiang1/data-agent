#!/usr/bin/env bash
# Lumen Agent 沙盒部署：从本地工作区发布，构建成功后重建容器。
# 服务器、用户、密钥和跳板机仅由《环境资料.md》及 SSH config 管理。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_HOST="${AGENT_HOST:-agent}"
SSH_CONFIG="${DEPLOY_SSH_CONFIG:-$HOME/.ssh/config}"
REMOTE_DIR="${REMOTE_DIR:-data-agen}"
CONFIG_PROFILE="${CONFIG_PROFILE:-1}"
EXPECTED_ADMIN_DATABASE="${EXPECTED_ADMIN_DATABASE:-uqpay_infra_lumenv10}"
RELEASE_ID="$(date +%Y%m%d%H%M%S)"
IMAGE_TAG="data-agen:release-$RELEASE_ID"
LOCAL_ARCHIVE="$(mktemp -t lumen-agent-release)"
LOCAL_MANIFEST="$(mktemp -t lumen-agent-manifest)"
REMOTE_ARCHIVE="/tmp/lumen-agent-$RELEASE_ID.tar.gz"
REMOTE_STAGE="/tmp/lumen-agent-$RELEASE_ID"

cleanup() {
  rm -f "$LOCAL_ARCHIVE" "$LOCAL_MANIFEST"
}
trap cleanup EXIT

do_ssh() {
  ssh -F "$SSH_CONFIG" "$AGENT_HOST" "$@"
}

check_ssh_alias() {
  local resolved_host proxy_jump
  resolved_host="$(ssh -F "$SSH_CONFIG" -G "$AGENT_HOST" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }' || true)"
  proxy_jump="$(ssh -F "$SSH_CONFIG" -G "$AGENT_HOST" 2>/dev/null | awk '$1 == "proxyjump" { print $2; exit }' || true)"
  if [ -z "$resolved_host" ] || [ "$resolved_host" = "$AGENT_HOST" ]; then
    echo "SSH 别名 '$AGENT_HOST' 未解析到目标主机；请按《环境资料.md》配置 SSH config。" >&2
    exit 1
  fi
  if [ -z "$proxy_jump" ] || [ "$proxy_jump" = "none" ]; then
    echo "SSH 别名 '$AGENT_HOST' 未配置 ProxyJump；请按《环境资料.md》补全跳板关系。" >&2
    exit 1
  fi
}

check_ssh_alias

echo "打包本地 Agent 工作区"
: > "$LOCAL_MANIFEST"

while IFS= read -r -d '' relative_path; do
  if [ ! -e "$SCRIPT_DIR/$relative_path" ] && [ ! -L "$SCRIPT_DIR/$relative_path" ]; then
    continue
  fi
  normalized_path="$(printf '%s' "/$relative_path" | tr '[:upper:]' '[:lower:]')"
  base_name="$(basename "$normalized_path")"
  if [ -L "$SCRIPT_DIR/$relative_path" ]; then
    echo "发布中止：文件清单包含符号链接 $relative_path" >&2
    exit 1
  fi
  case "$normalized_path" in
    */.env|*/.env.*|*/.codex/*|*/.ssh/*|*/auth.json|*/credentials/*|*/secrets/*)
      echo "发布中止：文件清单包含敏感路径 $relative_path" >&2
      exit 1
      ;;
  esac
  case "$base_name" in
    credentials|credentials.*|secrets|secrets.*|client_secret*.json|service_account*.json|id_rsa|id_ed25519|*.pem|*.key|*.p12|*.pfx|*.jks)
      echo "发布中止：文件清单包含敏感文件 $relative_path" >&2
      exit 1
      ;;
  esac
  if grep -IqE -- '-----BEGIN ([A-Z0-9]+ )?PRIVATE KEY-----' "$SCRIPT_DIR/$relative_path"; then
    echo "发布中止：文件内容疑似包含私钥 $relative_path" >&2
    exit 1
  fi
  printf '%s\0' "$relative_path" >> "$LOCAL_MANIFEST"
done < <(git -C "$SCRIPT_DIR" ls-files --cached --others --exclude-standard -z)

if [ ! -s "$LOCAL_MANIFEST" ]; then
  echo "发布中止：本地工作区文件清单为空" >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar --null -czf "$LOCAL_ARCHIVE" \
  -C "$SCRIPT_DIR" \
  -T "$LOCAL_MANIFEST"

echo "上传 Agent 发布包"
scp -F "$SSH_CONFIG" "$LOCAL_ARCHIVE" "$AGENT_HOST:$REMOTE_ARCHIVE"

echo "构建并重建 Agent"
do_ssh bash -s -- \
  "$REMOTE_ARCHIVE" \
  "$REMOTE_STAGE" \
  "$REMOTE_DIR" \
  "$IMAGE_TAG" \
  "$CONFIG_PROFILE" \
  "$EXPECTED_ADMIN_DATABASE" <<'REMOTE'
set -euo pipefail

remote_archive="$1"
remote_stage="$2"
remote_dir="$HOME/$3"
image_tag="$4"
config_profile="$5"
expected_admin_database="$6"
keep_candidate_image=0

cleanup_remote() {
  rm -f -- "$remote_archive"
  case "$remote_stage" in
    /tmp/lumen-agent-*) rm -rf -- "$remote_stage" ;;
  esac
  if [ "$keep_candidate_image" -ne 1 ]; then
    docker image rm "$image_tag" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote EXIT

test -f "$remote_dir/.env" || {
  echo "缺少服务器运行配置: $remote_dir/.env" >&2
  exit 1
}
env_checksum_before="$(sha256sum "$remote_dir/.env" | awk '{print $1}')"

rm -rf "$remote_stage"
mkdir -p "$remote_stage"
tar -xzf "$remote_archive" -C "$remote_stage"

cd "$remote_stage"
docker build -t "$image_tag" .

echo "使用服务器运行配置预检候选镜像"
docker run --rm \
  --env-file "$remote_dir/.env" \
  -e NL2SQL_ENV=prod \
  -e CONFIG_SOURCE=mysql \
  -e CONFIG_PROFILE="$config_profile" \
  -e EXPECTED_ADMIN_DATABASE="$expected_admin_database" \
  "$image_tag" \
  python -m src.retrieval.deployment_preflight

# 发布包不含真实配置；仅更新服务器上的代码副本。
mkdir -p "$remote_dir"
tar -cf - . | tar -xf - -C "$remote_dir"
env_checksum_after="$(sha256sum "$remote_dir/.env" | awk '{print $1}')"
if [ "$env_checksum_before" != "$env_checksum_after" ]; then
  echo "发布中止：服务器 .env 在代码同步期间发生变化" >&2
  exit 1
fi

docker tag "$image_tag" data-agen:latest
docker rm -f data-agen >/dev/null 2>&1 || true
docker run -d \
  --name data-agen \
  --restart always \
  --gpus all \
  -p 9090:9090 \
  --env-file "$remote_dir/.env" \
  -e NL2SQL_ENV=prod \
  -e DENSE_DEVICE=cuda \
  -e CONFIG_SOURCE=mysql \
  -e CONFIG_PROFILE="$config_profile" \
  -e REBUILD_INDEX_ON_STARTUP=false \
  -e CODEX_HOME=/var/lib/lumen-codex \
  -v data-agen_model-cache:/root/.cache:z \
  -v data-agen_codex-home:/var/lib/lumen-codex:z \
  data-agen:latest
keep_candidate_image=1
REMOTE

echo "等待 Agent 就绪"
for attempt in $(seq 1 18); do
  if do_ssh "curl -fsS http://127.0.0.1:9090/health"; then
    echo
    echo "Agent 部署完成: $IMAGE_TAG"
    exit 0
  fi
  if [ "$attempt" -lt 18 ]; then
    sleep 10
  fi
done

echo "Agent 未在预期时间内就绪，最近日志如下" >&2
do_ssh "docker logs --tail 100 data-agen"
exit 1
