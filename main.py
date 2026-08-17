"""
NL2SQL RAG 检索系统 — 交互式入口（CLI 模式）。

启动后进入 REPL 交互循环: 输入自然语言问题 → RAG 检索 → LLM 生成 SQL → EXPLAIN 校验。

运行方式::

    python main.py              # 使用默认配置启动
    python main.py --agent 1    # 使用指定 Agent 配置
    python main.py --debug      # 开启 DEBUG 日志

交互命令::

    /quit    — 退出
    /debug   — 切换调试模式
    /prompt  — 切换 Prompt 显示
    /config  — 显示当前 Agent 配置
"""

import sys
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.retrieval.retriever import SchemaRetriever
from src.retrieval.config import DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD_URL
from src.retrieval.agent_config import AgentConfigLoader
from src.retrieval.query_logger import QueryLogger
from src.retrieval.llm_factory import create_chat_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_doris_engine():
    from sqlalchemy import create_engine

    url = f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD_URL}@{DORIS_HOST}:{DORIS_PORT}/information_schema?charset=utf8mb4"
    return create_engine(url, pool_size=2, pool_recycle=3600)


def main():
    debug_mode = "--debug" in sys.argv

    # 解析 --agent 参数
    agent_id = None
    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        if idx + 1 < len(sys.argv):
            try:
                agent_id = int(sys.argv[idx + 1])
            except ValueError:
                print(f"无效的 agent_id: {sys.argv[idx + 1]}")
                sys.exit(1)

    # 加载 Agent 配置
    config_loader = AgentConfigLoader()
    config = config_loader.load(agent_id=agent_id)
    config_loader.print_config(config)

    # 连接 Doris
    print(f"\n连接 Doris ({DORIS_HOST}:{DORIS_PORT})...")
    try:
        from sqlalchemy import create_engine, text

        url = f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD_URL}@{DORIS_HOST}:{DORIS_PORT}/information_schema?charset=utf8mb4"
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        print("Doris 连接成功")
    except Exception as e:
        print(f"Doris 连接失败，无法启动: {e}")
        sys.exit(1)

    # 初始化 RAG
    retriever = SchemaRetriever()
    retriever.initialize()

    # 初始化 LLM
    llm = create_chat_model(config)
    print(f"LLM 已就绪 (provider={config.llm_provider}, model={config.llm_model})")

    # EXPLAIN 校验器
    from src.retrieval.sql_validator import SQLValidator

    validator = SQLValidator(create_doris_engine())
    print("EXPLAIN 校验已启用")

    # 查询日志
    query_logger = QueryLogger()

    # 交互阶段降低日志级别
    if not debug_mode:
        logging.getLogger().setLevel(logging.WARNING)

    print("\n" + "=" * 60)
    print("NL2SQL 系统已就绪")
    if config.agent_id:
        print(f"Agent: {config.agent_name} (ID={config.agent_id})")
    print(f"LLM: {config.llm_provider}/{config.llm_model}")
    print("输入自然语言问题，自动生成 SQL")
    print("命令: /quit 退出, /debug 调试, /prompt Prompt, /config 配置")
    print("=" * 60)

    show_prompt = True

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query == "/quit":
            print("再见!")
            break
        if query == "/debug":
            debug_mode = not debug_mode
            level = logging.DEBUG if debug_mode else logging.WARNING
            logging.getLogger().setLevel(level)
            print(f"调试模式: {'开启' if debug_mode else '关闭'}")
            continue
        if query == "/prompt":
            show_prompt = not show_prompt
            print(f"Prompt 显示: {'开启' if show_prompt else '关闭'}")
            continue
        if query == "/config":
            config_loader.print_config(config)
            continue

        import time

        start_time = time.time()

        # RAG 检索
        result = retriever.retrieve(
            query,
            top_k=config.table_search_top_k,
            fewshot_k=config.fewshot_top_k,
        )

        if result.matched_terms:
            print(f"\n[术语匹配] {', '.join(result.matched_terms)}")
        print(
            f"\n[命中表] {', '.join(t['table_name'] for t in result.relevant_tables)}"
        )

        if show_prompt:
            print(f"\n[Prompt] ({len(result.prompt_text)} 字符)")
            print("-" * 60)
            print(result.prompt_text)
            print("-" * 60)

        # 构建多轮对话
        messages = [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": f"## 用户问题\n{query}\n\n{result.prompt_text}",
            },
        ]

        def _to_lc(msgs):
            _map = {
                "system": SystemMessage,
                "user": HumanMessage,
                "assistant": AIMessage,
            }
            return [
                _map.get(m["role"], HumanMessage)(content=m["content"]) for m in msgs
            ]

        print("\n[生成 SQL]")
        try:
            resp = llm.invoke(_to_lc(messages))
            answer = resp.content
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            continue

        # EXPLAIN 校验循环
        extracted_sql = SQLValidator.extract_sql(answer)
        retry_count = 0
        is_success = True

        if not extracted_sql:
            print("[EXPLAIN] LLM 未生成 SQL，重新请求...")
            messages.append(
                {
                    "role": "user",
                    "content": "你没有生成 SQL，请根据上面的表结构生成可执行的 SQL，用 ```sql ``` 包裹。",
                }
            )
            try:
                resp = llm.invoke(_to_lc(messages))
                answer = resp.content
                messages.append({"role": "assistant", "content": answer})
                extracted_sql = SQLValidator.extract_sql(answer)
            except Exception as e:
                print(f"LLM 重新调用失败: {e}")

        syntax_ok = False
        check = None
        if extracted_sql and config.enable_explain:
            for attempt in range(config.max_fix_retries):
                check = validator.validate(answer)

                if check["valid"]:
                    if attempt > 0:
                        print(f"[EXPLAIN] 第 {attempt + 1} 次生成，语法通过 ✓")
                    else:
                        print("[EXPLAIN] 语法通过 ✓")
                    syntax_ok = True
                    break

                retry_count = attempt + 1
                print(
                    f"[EXPLAIN] 语法失败 (第 {retry_count}/{config.max_fix_retries} 次): {check['error']}"
                )
                if attempt < config.max_fix_retries - 1:
                    print("[修复中] 分析错误原因并重新生成...")
                    messages.append(
                        {
                            "role": "user",
                            "content": f"你生成的 SQL 执行 EXPLAIN 校验失败。\n\n"
                            f"## EXPLAIN 报错\n{check['error']}\n\n"
                            f"请分析错误原因（1-2句），然后输出修复后的 SQL，用 ```sql ``` 包裹。",
                        }
                    )
                    try:
                        resp = llm.invoke(_to_lc(messages))
                        answer = resp.content
                        messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        print(f"LLM 修复调用失败: {e}")
                        break
                else:
                    is_success = False
                    print(
                        f"[EXPLAIN] 已达最大重试次数 ({config.max_fix_retries})，输出最后结果"
                    )

        if syntax_ok and check and check["plan"]:
            print("[执行计划分析] 将 EXPLAIN 结果交给 LLM 分析...")
            messages.append(
                {
                    "role": "user",
                    "content": f"请分析这条 SQL 的 EXPLAIN 执行计划，判断是否有明显性能问题。\n\n"
                    f"## EXPLAIN 执行计划\n```\n{check['plan']}\n```\n\n"
                    f"关注：笛卡尔积、扫描行数过大、缺少分区裁剪、JOIN 顺序。\n"
                    f"如果有优化空间，输出优化后的 SQL，用 ```sql ``` 包裹。如果没问题，只回复：LGTM",
                }
            )
            try:
                resp = llm.invoke(_to_lc(messages))
                review_result = resp.content
                if "LGTM" in review_result.upper():
                    print("[执行计划分析] 无明显性能问题 ✓")
                else:
                    print("[执行计划分析] 发现优化空间，校验优化后 SQL...")
                    recheck = validator.validate(review_result)
                    if recheck["valid"]:
                        answer = review_result
                        print("[执行计划分析] 优化后校验通过 ✓")
                    else:
                        print("[执行计划分析] 优化后语法异常，使用原始版本")
            except Exception as e:
                print(f"执行计划分析失败: {e}")

        elapsed_ms = int((time.time() - start_time) * 1000)

        # 记录查询日志
        final_sql = SQLValidator.extract_sql(answer) or ""
        query_logger.log(
            user_query=query,
            matched_tables=[t["table_name"] for t in result.relevant_tables],
            matched_terms=result.matched_terms,
            generated_sql=final_sql,
            is_success=is_success,
            execution_time_ms=elapsed_ms,
            retry_count=retry_count,
        )

        print(answer)


if __name__ == "__main__":
    main()
