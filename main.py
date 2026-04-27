"""
NL2SQL RAG 检索系统 — 交互式入口

运行: python main.py
      python main.py --offline  # 强制离线模式（仅用语义层 YAML）
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
1. 只输出一条可直接执行的 SQL，用 ```sql ``` 包裹
2. 严格使用提供的表和列，不要编造不存在的表或列
3. 列别名使用中文双引号，如 COUNT(*) AS "数量"
4. 注意参考业务上下文中的过滤条件提示
5. 参考 Few-shot 示例的 SQL 模式
6. 如果无法生成 SQL，说明原因"""


def check_doris_connection() -> bool:
    """检测 Doris 是否可连接"""
    try:
        from sqlalchemy import create_engine, text
        url = f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD}@{DORIS_HOST}:{DORIS_PORT}/{DORIS_DATABASE}?charset=utf8mb4"
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


def init_llm() -> OpenAI:
    """初始化 DeepSeek LLM"""
    from src.retrieval.config import os
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def generate_sql(llm: OpenAI, query: str, prompt_text: str) -> str:
    """调用 LLM 生成 SQL"""
    response = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{prompt_text}\n\n用户问题：{query}"},
        ],
    )
    return response.choices[0].message.content


def main():
    force_offline = "--offline" in sys.argv
    debug_mode = "--debug" in sys.argv

    if force_offline:
        offline = True
        print("强制离线模式: 仅从 semantic_layer/ YAML 加载 Schema")
    else:
        print(f"检测 Doris 连接 ({DORIS_HOST}:{DORIS_PORT})...")
        if check_doris_connection():
            offline = False
            print("Doris 连接成功, 使用在线模式")
        else:
            offline = True
            print("Doris 不可达, 自动切换为离线模式 (仅从 semantic_layer/ YAML 加载)")

    retriever = SchemaRetriever(offline=offline)
    retriever.initialize()

    llm = init_llm()
    print("LLM (DeepSeek) 已就绪")

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

        # 调用 LLM 生成 SQL
        print("\n[生成 SQL]")
        try:
            answer = generate_sql(llm, query, result.prompt_text)
            print(answer)
        except Exception as e:
            print(f"LLM 调用失败: {e}")


if __name__ == "__main__":
    main()
