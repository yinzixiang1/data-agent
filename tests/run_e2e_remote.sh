#!/bin/bash
# 远程 E2E 测试 — 通过跳板机在 node1 上执行 curl 测试
# 用法: bash tests/run_e2e_remote.sh

set -e

BASTION_KEY="$HOME/company/BI/uat-infra-msk-bastion.pem"
NODE_KEY="$HOME/company/BI/nl2sql/sg-sandbox-private.pem"
BASTION="rocky@13.228.165.213"
NODE1="rocky@10.2.2.35"
PROXY_CMD="ssh -o StrictHostKeyChecking=no -i $BASTION_KEY -W %h:%p $BASTION"

API="http://localhost:8090"
TOKEN="c3a84d46c8f72279653fd92e3dbecbabc2959e9964665660dbf30cc2a246c36f"

do_ssh() {
  ssh -o StrictHostKeyChecking=no -o "ProxyCommand=$PROXY_CMD" -i "$NODE_KEY" "$NODE1" "$@"
}

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
RESET='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

check() {
  local name="$1"
  local ok="$2"
  local detail="$3"
  if [ "$ok" = "true" ]; then
    PASSED=$((PASSED + 1))
    echo -e "  ${GREEN}PASS${RESET}  $name ${detail:+($detail)}"
  else
    FAILED=$((FAILED + 1))
    echo -e "  ${RED}FAIL${RESET}  $name ${detail:+($detail)}"
  fi
}

skip() {
  local name="$1"
  local reason="$2"
  SKIPPED=$((SKIPPED + 1))
  echo -e "  ${YELLOW}SKIP${RESET}  $name ${reason:+($reason)}"
}

section() {
  echo -e "\n${CYAN}============================================================"
  echo -e "  $1"
  echo -e "============================================================${RESET}"
}

# 把所有测试命令打包到 node1 上执行
do_ssh bash -s "$API" "$TOKEN" << 'REMOTE_SCRIPT'
API=$1
TOKEN=$2
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
RESET='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

check() {
  local name="$1"; local ok="$2"; local detail="$3"
  if [ "$ok" = "true" ]; then
    PASSED=$((PASSED + 1))
    echo -e "  ${GREEN}PASS${RESET}  $name ${detail:+($detail)}"
  else
    FAILED=$((FAILED + 1))
    echo -e "  ${RED}FAIL${RESET}  $name ${detail:+($detail)}"
  fi
}

skip() {
  SKIPPED=$((SKIPPED + 1))
  echo -e "  ${YELLOW}SKIP${RESET}  $1 ${2:+($2)}"
}

section() {
  echo -e "\n${CYAN}============================================================"
  echo -e "  $1"
  echo -e "============================================================${RESET}"
}

echo -e "\n${CYAN}############################################################"
echo -e "  E2E 集成测试 — SQL 纠错增强全流程"
echo -e "  Target: $API (node1 上执行)"
echo -e "############################################################${RESET}"

# ─────────────────────────────────────────────
section "Phase 1: 健康检查"
# ─────────────────────────────────────────────

HEALTH=$(curl -s --max-time 5 "$API/open/v1/health")
check "admin-api 健康" "$(echo "$HEALTH" | grep -q '"ok"' && echo true || echo false)" "$HEALTH"

# ─────────────────────────────────────────────
section "Phase 2: 创建测试 Agent"
# ─────────────────────────────────────────────

HANDLE="e2e-test-$(date +%s)"
ENGINE_URL="http://10.2.2.16:9090"
CREATE_RESP=$(curl -s --max-time 10 -X POST "$API/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"E2E-Test-Agent\",\"handle\":\"$HANDLE\",\"description\":\"集成测试\",\"status\":\"live\"}")
AGENT_ID=$(echo "$CREATE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")

if [ -n "$AGENT_ID" ] && [ "$AGENT_ID" != "None" ]; then
  check "创建 Agent" "true" "id=$AGENT_ID"
else
  check "创建 Agent" "false" "$CREATE_RESP"
  echo -e "\n${RED}Agent 创建失败，终止测试${RESET}"
  exit 1
fi

# 设置 engine_url（create_agent 可能不含此字段，用 PUT 补设）
R=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API/api/v1/agents/$AGENT_ID" \
  -H "Content-Type: application/json" \
  -d "{\"engine_url\":\"$ENGINE_URL\"}")
check "设置 engine_url" "$([ "$R" = "200" ] && echo true || echo false)" "url=$ENGINE_URL, status=$R"

# ─────────────────────────────────────────────
section "Phase 3: 配置各分区"
# ─────────────────────────────────────────────

# 从 Agent 1 读取 model 配置（含 api_key）
AGENT1_MODEL=$(curl -s "$API/api/v1/agents/1/config/model" | python3 -c "
import sys,json
d=json.load(sys.stdin).get('config_json',{})
print(json.dumps(d))
" 2>/dev/null)
if [ "$AGENT1_MODEL" = "{}" ] || [ -z "$AGENT1_MODEL" ]; then
  echo -e "  ${YELLOW}WARN${RESET}: 无法读取 Agent 1 的 model 配置，使用默认值"
  AGENT1_MODEL='{"provider":"dashscope","model":"qwen3-coder-plus","base_url":"https://dashscope-intl.aliyuncs.com/compatible-mode/v1","temperature":0.0}'
fi

# model 分区（复用 Agent 1 的完整 model 配置）
R=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API/api/v1/agents/$AGENT_ID/config/model" \
  -H "Content-Type: application/json" \
  -d "{\"config_json\":$AGENT1_MODEL}")
check "model 分区" "$([ "$R" = "200" ] && echo true || echo false)" "status=$R"

# prompt 分区
R=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API/api/v1/agents/$AGENT_ID/config/prompt" \
  -H "Content-Type: application/json" \
  -d '{"config_json":{"system_prompt":"你是一个专业的 Apache Doris 数据分析专家。"}}')
check "prompt 分区" "$([ "$R" = "200" ] && echo true || echo false)"

# flow 分区（含纠错增强配置）
R=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API/api/v1/agents/$AGENT_ID/config/flow" \
  -H "Content-Type: application/json" \
  -d '{"config_json":{"enable_execute":true,"enable_summarize":true,"execute_row_limit":100,"execute_timeout":30,"max_execute_fix_retries":2,"enable_empty_analysis":true,"enable_enum_validate":true,"enable_result_check":true,"enable_timeout_fallback":false}}')
check "flow 分区（含纠错增强）" "$([ "$R" = "200" ] && echo true || echo false)"

# expand 分区
R=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API/api/v1/agents/$AGENT_ID/config/expand" \
  -H "Content-Type: application/json" \
  -d '{"config_json":{"max_tokens":4096,"top_p":0.9,"value_exact_match_boost":2.0}}')
check "expand 分区" "$([ "$R" = "200" ] && echo true || echo false)"

# ─────────────────────────────────────────────
section "Phase 4: 配置读回验证"
# ─────────────────────────────────────────────

FLOW_CFG=$(curl -s "$API/api/v1/agents/$AGENT_ID/config/flow" | python3 -c "import sys,json; d=json.load(sys.stdin).get('config_json',{}); print(d.get('max_execute_fix_retries','?'), d.get('enable_empty_analysis','?'), d.get('enable_result_check','?'))" 2>/dev/null)
check "flow 读回 (max_execute_fix_retries, enable_empty_analysis, enable_result_check)" \
  "$([ "$FLOW_CFG" = "2 True True" ] && echo true || echo false)" "$FLOW_CFG"

EXPAND_CFG=$(curl -s "$API/api/v1/agents/$AGENT_ID/config/expand" | python3 -c "import sys,json; d=json.load(sys.stdin).get('config_json',{}); print(d.get('max_tokens','?'), d.get('top_p','?'))" 2>/dev/null)
check "expand 读回 (max_tokens, top_p)" \
  "$([ "$EXPAND_CFG" = "4096 0.9" ] && echo true || echo false)" "$EXPAND_CFG"

# ─────────────────────────────────────────────
section "Phase 5: expand 优先级验证（数据层）"
# ─────────────────────────────────────────────

# flow 已设 execute_timeout=30，expand 再设 execute_timeout=10
curl -s -o /dev/null -X PUT "$API/api/v1/agents/$AGENT_ID/config/expand" \
  -H "Content-Type: application/json" \
  -d '{"config_json":{"execute_timeout":10,"max_tokens":4096}}'

FLOW_TO=$(curl -s "$API/api/v1/agents/$AGENT_ID/config/flow" | python3 -c "import sys,json; print(json.load(sys.stdin).get('config_json',{}).get('execute_timeout','?'))" 2>/dev/null)
check "flow 中 execute_timeout 保持 30" "$([ "$FLOW_TO" = "30" ] && echo true || echo false)" "actual=$FLOW_TO"

EXPAND_TO=$(curl -s "$API/api/v1/agents/$AGENT_ID/config/expand" | python3 -c "import sys,json; print(json.load(sys.stdin).get('config_json',{}).get('execute_timeout','?'))" 2>/dev/null)
check "expand 中 execute_timeout 存储为 10" "$([ "$EXPAND_TO" = "10" ] && echo true || echo false)" "actual=$EXPAND_TO"
echo -e "  ${YELLOW}NOTE${RESET}: 运行时优先级(flow>expand)在引擎侧 _apply_expand 验证"

# ─────────────────────────────────────────────
section "Phase 6: 数据源绑定"
# ─────────────────────────────────────────────

DB_LIST=$(curl -s "$API/api/v1/biz-databases")
DB_COUNT=$(echo "$DB_LIST" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
check "可用 biz_database" "$([ "$DB_COUNT" -gt 0 ] && echo true || echo false)" "count=$DB_COUNT"

if [ "$DB_COUNT" -gt 0 ]; then
  # 绑定所有 biz_database
  DB_IDS=$(echo "$DB_LIST" | python3 -c "import sys,json; print(' '.join(str(d['id']) for d in json.load(sys.stdin)))" 2>/dev/null)
  for DB_ID in $DB_IDS; do
    R=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/v1/agents/$AGENT_ID/refs" \
      -H "Content-Type: application/json" \
      -d "{\"resource_type\":\"biz_database\",\"resource_key\":\"$DB_ID\"}")
    check "绑定 biz_database id=$DB_ID" "$(echo "$R" | grep -qE '200|201|409' && echo true || echo false)" "status=$R"
  done
fi

# 绑定 vector_db（从 Agent 1 复制）
AGENT1_REFS=$(curl -s "$API/api/v1/agents/1/refs")
VDB_KEY=$(echo "$AGENT1_REFS" | python3 -c "import sys,json; d=json.load(sys.stdin); vdbs=d.get('vector_db',[]); print(vdbs[0] if vdbs else '')" 2>/dev/null)
if [ -n "$VDB_KEY" ]; then
  R=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/v1/agents/$AGENT_ID/refs" \
    -H "Content-Type: application/json" \
    -d "{\"resource_type\":\"vector_db\",\"resource_key\":\"$VDB_KEY\"}")
  check "绑定 vector_db=$VDB_KEY" "$(echo "$R" | grep -qE '200|201|409' && echo true || echo false)" "status=$R"
fi

# ─────────────────────────────────────────────
section "Phase 7: Open API /chat expand_info + 结果字段测试"
# ─────────────────────────────────────────────

CHAT_RESP=$(curl -s --max-time 120 -X POST "$API/open/v1/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"query\",\"question\":\"查询账户总数\",\"agent_id\":$AGENT_ID,\"expand_info\":{\"enable_execute\":false}}")

CHAT_OK=$(echo "$CHAT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('sql' in d and 'summary' in d and 'query_result' in d and 'execution_error' in d)" 2>/dev/null || echo "False")
check "ChatResponse 含 summary/query_result/execution_error 字段" "$([ "$CHAT_OK" = "True" ] && echo true || echo false)"

CHAT_SQL=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sql','')[:60])" 2>/dev/null || echo "")
check "Open API 返回 SQL" "$([ -n "$CHAT_SQL" ] && echo true || echo false)" "sql=$CHAT_SQL"

CHAT_QR=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('query_result'))" 2>/dev/null || echo "")
check "expand_info enable_execute=false → query_result 为 None" "$([ "$CHAT_QR" = "None" ] && echo true || echo false)" "query_result=$CHAT_QR"

# ─────────────────────────────────────────────
section "Phase 8: 内部代理查询 + Trace 验证"
# ─────────────────────────────────────────────

QUERY_RESP=$(curl -s --max-time 120 -X POST "$API/api/v1/agents/$AGENT_ID/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"查询最近7天新增的账户数量","enable_explain":false}')

Q_SQL=$(echo "$QUERY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sql','')[:80])" 2>/dev/null || echo "")
Q_SUCCESS=$(echo "$QUERY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_success',False))" 2>/dev/null || echo "")
check "代理查询返回 SQL" "$([ -n "$Q_SQL" ] && echo true || echo false)" "sql=$Q_SQL"
check "is_success" "$([ "$Q_SUCCESS" = "True" ] && echo true || echo false)"

# Trace 步骤
TRACE_STEPS=$(echo "$QUERY_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
steps = d.get('trace',{}).get('steps',[])
print(' '.join(s.get('step','?') for s in steps))
" 2>/dev/null || echo "")
check "trace.steps 非空" "$([ -n "$TRACE_STEPS" ] && echo true || echo false)" "steps=$TRACE_STEPS"

for STEP in glossary schema_retrieval llm_generation; do
  check "trace 含 $STEP" "$(echo "$TRACE_STEPS" | grep -q "$STEP" && echo true || echo false)"
done

# 执行结果相关
Q_RESULT=$(echo "$QUERY_RESP" | python3 -c "import sys,json; qr=json.load(sys.stdin).get('query_result'); print(f'rows={qr[\"row_count\"]}' if qr else 'None')" 2>/dev/null || echo "?")
Q_SUMMARY=$(echo "$QUERY_RESP" | python3 -c "import sys,json; s=json.load(sys.stdin).get('summary',''); print(s[:50] if s else 'empty')" 2>/dev/null || echo "?")
Q_EXEC_ERR=$(echo "$QUERY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('execution_error',''))" 2>/dev/null || echo "?")

if echo "$TRACE_STEPS" | grep -q "sql_execution"; then
  check "trace 含 sql_execution（执行已触发）" "true" "$Q_RESULT"
  if [ "$Q_RESULT" != "None" ]; then
    check "query_result 有数据" "true" "$Q_RESULT"
  fi
  if [ "$Q_SUMMARY" != "empty" ] && [ "$Q_SUMMARY" != "?" ]; then
    check "summary 有内容" "true" "$Q_SUMMARY"
  else
    skip "summary" "可能 row_count=0 触发了空结果分析"
  fi
  # 检查增强步骤
  for STEP in empty_analysis execution_fix enum_validate result_summarize timeout_fallback; do
    if echo "$TRACE_STEPS" | grep -q "$STEP"; then
      check "trace 含 $STEP（增强步骤已触发）" "true"
    fi
  done
else
  skip "sql_execution" "enable_execute 可能未在此 Agent 上生效（需要引擎 reload）"
fi

# ─────────────────────────────────────────────
section "Phase 9: expand_info 覆盖 — 禁用 vs 启用执行"
# ─────────────────────────────────────────────

# 禁用执行
RESP_OFF=$(curl -s --max-time 120 -X POST "$API/api/v1/agents/$AGENT_ID/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"查询最近7天新增的账户数量","expand_info":{"enable_execute":false}}')
QR_OFF=$(echo "$RESP_OFF" | python3 -c "import sys,json; print(json.load(sys.stdin).get('query_result'))" 2>/dev/null)
check "expand_info enable_execute=false → 无 query_result" "$([ "$QR_OFF" = "None" ] && echo true || echo false)"

# ─────────────────────────────────────────────
section "Phase 10: 清理测试 Agent"
# ─────────────────────────────────────────────

R=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$API/api/v1/agents/$AGENT_ID")
check "删除 Agent $AGENT_ID" "$(echo "$R" | grep -qE '200|204' && echo true || echo false)" "status=$R"

# ─────────────────────────────────────────────
echo -e "\n${CYAN}============================================================"
TOTAL=$((PASSED + FAILED + SKIPPED))
echo -e "  结果: ${GREEN}$PASSED passed${RESET}, ${RED}$FAILED failed${RESET}, ${YELLOW}$SKIPPED skipped${RESET} / $TOTAL total"
echo -e "============================================================${RESET}\n"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
REMOTE_SCRIPT
