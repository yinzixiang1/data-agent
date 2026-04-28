"""
NL2SQL RAG 检索系统 — 交互式入口。

启动后进入 REPL 交互循环: 输入自然语言问题 → RAG 检索 → LLM 生成 SQL → EXPLAIN 校验。

运行方式::

    python main.py              # 连接 Doris 启动（连不上则失败）
    python main.py --debug      # 开启 DEBUG 日志

交互命令::

    /quit    — 退出
    /debug   — 切换调试模式（显示/隐藏详细日志）
    /prompt  — 切换 Prompt 显示（显示/隐藏发给 LLM 的完整 Prompt）
"""

import sys
import logging
from openai import OpenAI
from src.retrieval.retriever import SchemaRetriever
from src.retrieval.config import DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD, DORIS_DATABASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的 SQL 专家，专门将用户的自然语言问题转化为 Apache Doris SQL。

规则：
1. 必须输出一条可直接执行的 SQL，用 ```sql ``` 包裹
2. 严格使用提供的表和列，不要编造不存在的表或列
3. 列别名使用中文双引号，如 COUNT(*) AS "数量"
4. 注意参考业务上下文中的过滤条件提示
5. 参考 Few-shot 示例的 SQL 模式
6. 即使信息不完整，也要基于已有表结构尽力生成最合理的 SQL

Doris 日期函数注意：
- DATE_TRUNC(datetime, 'unit')，不是 DATE_TRUNC('unit', datetime)
- 本月: complete_time >= DATE_TRUNC(NOW(), 'month') AND complete_time < DATE_ADD(DATE_TRUNC(NOW(), 'month'), INTERVAL 1 MONTH)
- 近30天: create_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
- 按月分组: DATE_FORMAT(create_time, '%Y-%m')"""

MAX_FIX_RETRIES = 5


def create_doris_engine():
    """
    创建 Doris SQLAlchemy 连接引擎（连接池大小 2，回收周期 1 小时）。

    Returns:
        sqlalchemy.Engine: 可用于执行 SQL 的引擎实例
    """
    from sqlalchemy import create_engine
    url = f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD}@{DORIS_HOST}:{DORIS_PORT}/{DORIS_DATABASE}?charset=utf8mb4"
    return create_engine(url, pool_size=2, pool_recycle=3600)


def init_llm() -> OpenAI:
    """
    初始化 DeepSeek LLM 客户端。

    从环境变量读取 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL。

    Returns:
        OpenAI: 兼容 OpenAI 协议的客户端实例
    """
    from src.retrieval.config import os
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def llm_chat(llm: OpenAI, messages: list[dict]) -> str:
    """
    调用 LLM 生成回复，并将 assistant 回复追加到 messages（维持多轮上下文）。

    Args:
        llm: OpenAI 兼容客户端实例
        messages: 对话历史列表，格式为 [{"role": "system"|"user"|"assistant", "content": str}, ...]
            调用后会在末尾追加 {"role": "assistant", "content": response}

    Returns:
        str: LLM 生成的回复文本
    """
    resp = llm.chat.completions.create(
        model="deepseek-chat", temperature=0, messages=messages,
    )
    content = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    return content


def main():
    """
    主函数 — 初始化系统并进入交互式 REPL 循环。

    流程: 连接 Doris → 初始化 RAG（强制重建索引） → 初始化 LLM → EXPLAIN 校验器 → 交互循环
    """
    debug_mode = "--debug" in sys.argv

    # 连接 Doris（失败则直接退出）
    print(f"连接 Doris ({DORIS_HOST}:{DORIS_PORT})...")
    try:
        from sqlalchemy import create_engine, text
        url = f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD}@{DORIS_HOST}:{DORIS_PORT}/{DORIS_DATABASE}?charset=utf8mb4"
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        print("Doris 连接成功")
    except Exception as e:
        print(f"Doris 连接失败，无法启动: {e}")
        sys.exit(1)

    # 初始化 RAG（每次启动强制重建索引）
    retriever = SchemaRetriever()
    retriever.initialize()

    llm = init_llm()
    print("LLM (DeepSeek) 已就绪")

    # EXPLAIN 校验器
    from src.retrieval.sql_validator import SQLValidator
    validator = SQLValidator(create_doris_engine())
    print("EXPLAIN 校验已启用")

    # 交互阶段降低日志级别
    if not debug_mode:
        logging.getLogger().setLevel(logging.WARNING)

    print("\n" + "=" * 60)
    print("NL2SQL 系统已就绪")
    print("输入自然语言问题，自动生成 SQL")
    print("命令: /quit 退出, /debug 切换详细模式, /prompt 显示完整 Prompt")
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

        # RAG 检索
        result = retriever.retrieve(query)

        # 术语匹配
        if result.matched_terms:
            print(f"\n[术语匹配] {', '.join(result.matched_terms)}")

        # 命中表
        print(f"\n[命中表] {', '.join(t['table_name'] for t in result.relevant_tables)}")

        # 显示 Prompt（可选）
        if show_prompt:
            print(f"\n[Prompt] ({len(result.prompt_text)} 字符)")
            print("-" * 60)
            print(result.prompt_text)
            print("-" * 60)

        # 构建多轮对话（LLM 在修复时能看到完整上下文）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"## 用户问题\n{query}\n\n{result.prompt_text}"},
        ]

        # 调用 LLM 生成 SQL
        print("\n[生成 SQL]")
        try:
            answer = llm_chat(llm, messages)
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            continue

        # EXPLAIN 校验循环
        extracted_sql = SQLValidator.extract_sql(answer)

        if not extracted_sql:
            print("[EXPLAIN] LLM 未生成 SQL，重新请求...")
            messages.append({"role": "user", "content": "你没有生成 SQL，请根据上面的表结构生成可执行的 SQL，用 ```sql ``` 包裹。"})
            try:
                answer = llm_chat(llm, messages)
                extracted_sql = SQLValidator.extract_sql(answer)
            except Exception as e:
                print(f"LLM 重新调用失败: {e}")

        syntax_ok = False
        check = None
        if extracted_sql:
            for attempt in range(MAX_FIX_RETRIES):
                check = validator.validate(answer)

                if check["valid"]:
                    if attempt > 0:
                        print(f"[EXPLAIN] 第 {attempt + 1} 次生成，语法通过 ✓")
                    else:
                        print("[EXPLAIN] 语法通过 ✓")
                    syntax_ok = True
                    break

                print(f"[EXPLAIN] 语法失败 (第 {attempt + 1}/{MAX_FIX_RETRIES} 次): {check['error']}")
                if attempt < MAX_FIX_RETRIES - 1:
                    print("[修复中] 分析错误原因并重新生成...")
                    messages.append({"role": "user", "content":
                        f"你生成的 SQL 执行 EXPLAIN 校验失败。\n\n"
                        f"## EXPLAIN 报错\n{check['error']}\n\n"
                        f"请分析错误原因（1-2句），然后输出修复后的 SQL，用 ```sql ``` 包裹。"
                    })
                    try:
                        answer = llm_chat(llm, messages)
                    except Exception as e:
                        print(f"LLM 修复调用失败: {e}")
                        break
                else:
                    print(f"[EXPLAIN] 已达最大重试次数 ({MAX_FIX_RETRIES})，输出最后结果")

        if syntax_ok and check and check["plan"]:
            print("[执行计划分析] 将 EXPLAIN 结果交给 LLM 分析...")
            messages.append({"role": "user", "content":
                f"请分析这条 SQL 的 EXPLAIN 执行计划，判断是否有明显性能问题。\n\n"
                f"## EXPLAIN 执行计划\n```\n{check['plan']}\n```\n\n"
                f"关注：笛卡尔积、扫描行数过大、缺少分区裁剪、JOIN 顺序。\n"
                f"如果有优化空间，输出优化后的 SQL，用 ```sql ``` 包裹。如果没问题，只回复：LGTM"
            })
            try:
                review_result = llm_chat(llm, messages)
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

        print(answer)


if __name__ == "__main__":
    main()
