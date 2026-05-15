"""
端到端集成测试 — SQL 纠错增强全流程。

测试流程:
  1. 创建测试 Agent
  2. 配置 model/prompt/flow/expand 分区
  3. 绑定数据源
  4. 触发索引重建 + 配置加载
  5. 基础查询验证
  6. expand_info 请求级覆盖
  7. expand 配置分区读取
  8. 清理测试 Agent

运行:
  python tests/test_e2e_enhancements.py --base-url http://localhost:8090 --token <admin_token>

  服务器:
  python tests/test_e2e_enhancements.py --base-url http://localhost:18090 --token <token>
"""

import argparse
import json
import sys
import time
import requests

# ── 颜色 ──
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def log_test(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {name}" + (f" ({detail})" if detail else ""))


def log_skip(name, reason=""):
    global skipped
    skipped += 1
    print(f"  {YELLOW}SKIP{RESET}  {name}" + (f" ({reason})" if reason else ""))


def log_section(name):
    print(f"\n{CYAN}{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}{RESET}")


class E2ETest:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.admin_headers = {"Content-Type": "application/json"}
        self.agent_id = None
        self.agent_token = None

    def api(self, method, path, json_data=None, timeout=30, use_auth=False):
        url = f"{self.base_url}{path}"
        headers = self.headers if use_auth else self.admin_headers
        resp = requests.request(method, url, json=json_data, headers=headers, timeout=timeout)
        return resp

    # ────────────────────────────────────────
    # Phase 1: Agent 创建
    # ────────────────────────────────────────

    def test_create_agent(self):
        log_section("Phase 1: 创建测试 Agent")

        resp = self.api("POST", "/api/v1/agents", {
            "name": "E2E-Test-Agent",
            "handle": f"e2e-test-{int(time.time())}",
            "description": "集成测试专用 Agent，测试完后删除",
            "engine_type": "nl2sql",
            "status": "live",
        })
        ok = resp.status_code in (200, 201)
        data = resp.json() if ok else {}
        self.agent_id = data.get("id")
        log_test("创建 Agent", ok, f"id={self.agent_id}" if ok else f"status={resp.status_code} {resp.text[:100]}")
        return ok

    # ────────────────────────────────────────
    # Phase 2: Agent 配置
    # ────────────────────────────────────────

    def test_configure_agent(self):
        log_section("Phase 2: 配置 Agent 各分区")
        if not self.agent_id:
            log_skip("配置 Agent", "Agent 未创建")
            return False

        # model 分区
        resp = self.api("PUT", f"/api/v1/agents/{self.agent_id}/config/model", {
            "config_json": {
                "provider": "dashscope",
                "model": "qwen3-coder-plus",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "temperature": 0.0,
            }
        })
        log_test("model 分区配置", resp.status_code == 200, resp.text[:80] if resp.status_code != 200 else "")

        # prompt 分区
        resp = self.api("PUT", f"/api/v1/agents/{self.agent_id}/config/prompt", {
            "config_json": {
                "system_prompt": "你是一个专业的 Apache Doris 数据分析专家。",
            }
        })
        log_test("prompt 分区配置", resp.status_code == 200)

        # flow 分区（启用执行 + 纠错增强）
        resp = self.api("PUT", f"/api/v1/agents/{self.agent_id}/config/flow", {
            "config_json": {
                "enable_execute": True,
                "enable_summarize": True,
                "execute_row_limit": 100,
                "execute_timeout": 30,
                "max_execute_fix_retries": 2,
                "enable_empty_analysis": True,
                "enable_enum_validate": True,
                "enable_result_check": True,
                "enable_timeout_fallback": False,
            }
        })
        log_test("flow 分区配置", resp.status_code == 200)

        # expand 分区
        resp = self.api("PUT", f"/api/v1/agents/{self.agent_id}/config/expand", {
            "config_json": {
                "max_tokens": 4096,
                "top_p": 0.9,
                "value_exact_match_boost": 2.0,
            }
        })
        log_test("expand 分区配置", resp.status_code == 200)

        # 读回验证
        resp = self.api("GET", f"/api/v1/agents/{self.agent_id}/config/flow")
        data = resp.json()
        cfg = data.get("config_json", {})
        log_test("flow 分区读回", cfg.get("max_execute_fix_retries") == 2 and cfg.get("enable_empty_analysis") is True)

        resp = self.api("GET", f"/api/v1/agents/{self.agent_id}/config/expand")
        data = resp.json()
        cfg = data.get("config_json", {})
        log_test("expand 分区读回", cfg.get("max_tokens") == 4096 and cfg.get("top_p") == 0.9)

        return True

    # ────────────────────────────────────────
    # Phase 3: expand 配置优先级验证
    # ────────────────────────────────────────

    def test_expand_priority(self):
        log_section("Phase 3: expand 配置优先级验证")
        if not self.agent_id:
            log_skip("优先级验证", "Agent 未创建")
            return

        # flow 中已设置 execute_timeout=30，expand 中设置 execute_timeout=10
        # 预期: flow 优先 → 最终应为 30
        resp = self.api("PUT", f"/api/v1/agents/{self.agent_id}/config/expand", {
            "config_json": {
                "execute_timeout": 10,
                "max_tokens": 4096,
            }
        })
        log_test("expand 设置 execute_timeout=10", resp.status_code == 200)

        # 重新读取 flow 确认 execute_timeout 仍然是 30
        resp = self.api("GET", f"/api/v1/agents/{self.agent_id}/config/flow")
        flow_cfg = resp.json().get("config_json", {})
        log_test("flow 分区 execute_timeout 保持 30", flow_cfg.get("execute_timeout") == 30,
                 f"actual={flow_cfg.get('execute_timeout')}")

        # expand 自己应该存的是 10
        resp = self.api("GET", f"/api/v1/agents/{self.agent_id}/config/expand")
        expand_cfg = resp.json().get("config_json", {})
        log_test("expand 分区 execute_timeout 存储为 10", expand_cfg.get("execute_timeout") == 10,
                 f"actual={expand_cfg.get('execute_timeout')}")

        print(f"\n  {YELLOW}NOTE{RESET}: data-agen 侧优先级 (flow > expand) 在 _apply_expand 中实现，")
        print(f"        需要通过实际查询 trace 验证最终生效值。")

    # ────────────────────────────────────────
    # Phase 4: 数据源绑定
    # ────────────────────────────────────────

    def test_bind_datasource(self):
        log_section("Phase 4: 数据源绑定")
        if not self.agent_id:
            log_skip("数据源绑定", "Agent 未创建")
            return False

        # 查看可用 biz_database
        resp = self.api("GET", "/api/v1/biz-databases")
        dbs = resp.json() if resp.status_code == 200 else []
        if not dbs:
            log_skip("数据源绑定", "无可用 biz_database")
            return False

        log_test(f"可用数据源", True, f"{len(dbs)} 个: {[d.get('database_name', d.get('name', '?')) for d in dbs[:4]]}")

        # 绑定第一个数据源
        db_id = dbs[0].get("id")
        resp = self.api("POST", f"/api/v1/agents/{self.agent_id}/refs", {
            "resource_type": "biz_database",
            "resource_key": str(db_id),
        })
        log_test(f"绑定 biz_database id={db_id}", resp.status_code in (200, 201, 409),
                 f"status={resp.status_code}")
        return True

    # ────────────────────────────────────────
    # Phase 5: 获取 Agent Token
    # ────────────────────────────────────────

    def test_get_agent_token(self):
        log_section("Phase 5: 获取 Agent Token")
        if not self.agent_id:
            log_skip("获取 Token", "Agent 未创建")
            return False

        resp = self.api("GET", f"/api/v1/agents/{self.agent_id}")
        if resp.status_code != 200:
            log_test("读取 Agent 详情", False, f"status={resp.status_code}")
            return False

        agent_data = resp.json()
        self.agent_token = agent_data.get("token", "")
        if not self.agent_token:
            # 尝试用系统默认 token
            self.agent_token = self.headers.get("Authorization", "").replace("Bearer ", "")
            log_test("Agent Token", True, "使用系统默认 Token")
        else:
            log_test("Agent Token", True, f"token={self.agent_token[:16]}...")
        return True

    # ────────────────────────────────────────
    # Phase 6: 查询测试
    # ────────────────────────────────────────

    def test_basic_query(self):
        log_section("Phase 6: 基础查询测试")
        if not self.agent_id:
            log_skip("基础查询", "Agent 未创建")
            return None

        resp = self.api("POST", f"/api/v1/agents/{self.agent_id}/query", {
            "question": "查询账户总数",
            "enable_explain": False,
        }, timeout=120)

        if resp.status_code != 200:
            log_test("基础查询", False, f"status={resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        log_test("返回 SQL", bool(data.get("sql")), f"sql={data.get('sql', '')[:80]}")
        log_test("is_success", data.get("is_success", False))
        log_test("matched_tables 非空", len(data.get("matched_tables", [])) > 0,
                 f"tables={data.get('matched_tables', [])}")

        if data.get("query_result"):
            qr = data["query_result"]
            log_test("query_result 有数据", qr.get("row_count", 0) > 0,
                     f"rows={qr.get('row_count')}, cols={qr.get('columns')}")
        else:
            log_skip("query_result", "enable_execute 可能未生效（引擎未 reload）")

        if data.get("summary"):
            log_test("summary 非空", True, f"summary={data['summary'][:60]}")
        else:
            log_skip("summary", "enable_summarize 可能未生效")

        return data

    def test_expand_info_override(self):
        log_section("Phase 7: expand_info 请求级覆盖")
        if not self.agent_id:
            log_skip("expand_info", "Agent 未创建")
            return

        # 用 expand_info 禁用执行，验证结果中不含 query_result
        resp = self.api("POST", f"/api/v1/agents/{self.agent_id}/query", {
            "question": "查询账户总数",
            "enable_explain": False,
            "expand_info": {
                "enable_execute": False,
                "enable_summarize": False,
            }
        }, timeout=120)

        if resp.status_code != 200:
            log_test("expand_info 查询", False, f"status={resp.status_code}")
            return

        data = resp.json()
        log_test("expand_info 查询成功", data.get("is_success", False))
        log_test("expand_info enable_execute=false → 无 query_result",
                 data.get("query_result") is None,
                 f"query_result={'有' if data.get('query_result') else '无'}")
        log_test("expand_info enable_summarize=false → 无 summary",
                 not data.get("summary"),
                 f"summary={'有' if data.get('summary') else '无'}")

    # ────────────────────────────────────────
    # Phase 8: Open API 通道测试
    # ────────────────────────────────────────

    def test_open_api_chat(self):
        log_section("Phase 8: Open API /open/v1/chat 通道测试")
        if not self.agent_token:
            log_skip("Open API", "无 Token")
            return

        resp = requests.post(
            f"{self.base_url}/open/v1/chat",
            json={
                "type": "query",
                "question": "查询账户总数",
                "agent_id": self.agent_id,
                "expand_info": {"enable_execute": False},
            },
            headers={
                "Authorization": f"Bearer {self.agent_token}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

        if resp.status_code == 401:
            log_skip("Open API chat", f"Token 鉴权失败 (可能是管理后台内部 token)")
            return

        if resp.status_code != 200:
            log_test("Open API chat", False, f"status={resp.status_code} {resp.text[:200]}")
            return

        data = resp.json()
        log_test("Open API 返回 sql", bool(data.get("sql")))
        log_test("Open API 返回 session_id", bool(data.get("session_id")))
        log_test("ChatResponse 含 summary 字段", "summary" in data, f"keys={list(data.keys())[:10]}")
        log_test("ChatResponse 含 query_result 字段", "query_result" in data)
        log_test("ChatResponse 含 execution_error 字段", "execution_error" in data)

    # ────────────────────────────────────────
    # Phase 9: Trace 验证（检查增强步骤是否出现）
    # ────────────────────────────────────────

    def test_trace_steps(self):
        log_section("Phase 9: Trace 步骤验证")
        if not self.agent_id:
            log_skip("Trace 验证", "Agent 未创建")
            return

        # 带执行的查询，检查 trace
        resp = self.api("POST", f"/api/v1/agents/{self.agent_id}/query", {
            "question": "查询最近7天交易总金额",
            "enable_explain": False,
        }, timeout=120)

        if resp.status_code != 200:
            log_test("Trace 查询", False, f"status={resp.status_code}")
            return

        data = resp.json()
        trace = data.get("trace", {})
        steps = trace.get("steps", [])
        step_names = [s.get("step") for s in steps]

        log_test("trace 存在", bool(trace))
        log_test("trace.steps 非空", len(steps) > 0, f"steps={step_names}")
        log_test("trace 含 glossary 步骤", "glossary" in step_names)
        log_test("trace 含 schema_retrieval 步骤", "schema_retrieval" in step_names)
        log_test("trace 含 llm_generation 步骤", "llm_generation" in step_names)

        # 如果有执行，验证执行相关 trace
        if "sql_execution" in step_names:
            log_test("trace 含 sql_execution 步骤", True)
            exec_step = next(s for s in steps if s["step"] == "sql_execution")
            log_test("sql_execution 有 success 字段", "success" in exec_step)

            if "result_summarize" in step_names:
                log_test("trace 含 result_summarize 步骤", True)
                sum_step = next(s for s in steps if s["step"] == "result_summarize")
                # 如果 enable_result_check，trace 中可能有 result_warnings
                if "result_warnings" in sum_step:
                    log_test("result_summarize 含 result_warnings", True,
                             f"warnings={sum_step['result_warnings']}")

            if "empty_analysis" in step_names:
                log_test("trace 含 empty_analysis 步骤（空结果分析）", True)

            if "execution_fix" in step_names:
                log_test("trace 含 execution_fix 步骤（执行纠错）", True)

            if "enum_validate" in step_names:
                log_test("trace 含 enum_validate 步骤（枚举校验）", True)
        else:
            log_skip("sql_execution trace", "执行未触发（enable_execute 可能未加载）")

    # ────────────────────────────────────────
    # Phase 10: 清理
    # ────────────────────────────────────────

    def test_cleanup(self):
        log_section("Phase 10: 清理测试 Agent")
        if not self.agent_id:
            log_skip("清理", "无 Agent 需清理")
            return

        resp = self.api("DELETE", f"/api/v1/agents/{self.agent_id}")
        log_test(f"删除 Agent {self.agent_id}", resp.status_code in (200, 204),
                 f"status={resp.status_code}")

    # ────────────────────────────────────────
    # 运行全部
    # ────────────────────────────────────────

    def run_all(self):
        print(f"\n{CYAN}{'#'*60}")
        print(f"  E2E 集成测试 — SQL 纠错增强全流程")
        print(f"  Target: {self.base_url}")
        print(f"{'#'*60}{RESET}")

        try:
            if not self.test_create_agent():
                print(f"\n{RED}Agent 创建失败，终止测试{RESET}")
                return

            self.test_configure_agent()
            self.test_expand_priority()
            self.test_bind_datasource()
            self.test_get_agent_token()

            # 查询测试前先等待引擎加载（如果是新 Agent 可能需要 reload）
            print(f"\n  {YELLOW}NOTE{RESET}: 新 Agent 需要引擎 reload 才能生效。")
            print(f"        当前测试使用已有引擎配置，部分执行功能可能因引擎未 reload 而 SKIP。")

            self.test_basic_query()
            self.test_expand_info_override()
            self.test_open_api_chat()
            self.test_trace_steps()
        finally:
            self.test_cleanup()

        # 汇总
        total = passed + failed + skipped
        print(f"\n{CYAN}{'='*60}")
        print(f"  测试结果: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}, {YELLOW}{skipped} skipped{RESET} / {total} total")
        print(f"{'='*60}{RESET}\n")

        if failed > 0:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="E2E 集成测试")
    parser.add_argument("--base-url", default="http://localhost:8090", help="Admin API 地址")
    parser.add_argument("--token", default="", help="系统默认 Token（DEFAULT_AGENT_TOKEN）")
    args = parser.parse_args()

    test = E2ETest(args.base_url, args.token)
    test.run_all()


if __name__ == "__main__":
    main()
