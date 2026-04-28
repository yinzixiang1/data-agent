"""
SQL 校验 — 通过 Doris EXPLAIN 预执行验证 SQL 语法。

EXPLAIN 不会真正执行 SQL，只检查语法和生成执行计划，因此是安全的校验方式。
校验通过后可将执行计划交给 LLM 分析，检测笛卡尔积、全表扫描等性能问题。

使用示例::

    from sqlalchemy import create_engine
    from src.retrieval.sql_validator import SQLValidator

    engine = create_engine("mysql+pymysql://root:@localhost:9030/dwd_banking")
    validator = SQLValidator(engine)

    # 从 LLM 回复中提取并校验 SQL
    result = validator.validate("这是 SQL:\\n```sql\\nSELECT COUNT(*) FROM pmt_account\\n```")
    # result: {"valid": True, "sql": "SELECT COUNT(*) ...", "error": None, "plan": "..."}

    # 也可以直接校验 SQL 字符串
    result = validator.explain("SELECT 1")
    # result: {"valid": True, "error": None, "plan": "..."}
"""

import re
import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class SQLValidator:
    """
    通过 Doris EXPLAIN 校验 SQL 语法，返回执行计划供 LLM 分析。

    Attributes:
        engine: SQLAlchemy Engine 实例（连接到 Doris）
    """

    def __init__(self, engine: Engine):
        """
        Args:
            engine: SQLAlchemy Engine 实例，需要有 Doris 的连接权限
        """
        self.engine = engine

    @staticmethod
    def extract_sql(llm_response: str) -> str | None:
        """
        从 LLM 回复中提取 SQL（匹配 ```sql ... ``` 代码块）。

        Args:
            llm_response: LLM 的完整回复文本

        Returns:
            str: 提取到的 SQL 字符串（已 strip），未找到代码块时返回 None
        """
        match = re.search(r"```sql\s*\n?(.*?)```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def explain(self, sql: str) -> dict:
        """
        执行 EXPLAIN 校验（不会真正执行 SQL）。

        Args:
            sql: 待校验的 SQL 语句

        Returns:
            dict，包含:
                - "valid" (bool): 语法是否正确
                - "error" (str | None): 语法错误信息（valid=True 时为 None）
                - "plan" (str | None): 完整执行计划文本（valid=False 时为 None）
        """
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(f"EXPLAIN {sql}")).fetchall()

            plan_lines = [str(row[0]) if len(row) == 1 else str(row) for row in rows]
            plan_text = "\n".join(plan_lines)

            logger.info(f"EXPLAIN 通过, 执行计划 {len(plan_lines)} 行")
            return {
                "valid": True,
                "error": None,
                "plan": plan_text,
            }

        except Exception as e:
            error_msg = str(e)
            # 提取核心错误信息（去掉 SQLAlchemy 包装）
            inner = re.search(r"\(pymysql\.err\.\w+\)\s*\((\d+),\s*[\"'](.+?)[\"']\)", error_msg)
            if inner:
                error_msg = f"Error {inner.group(1)}: {inner.group(2)}"

            logger.warning(f"EXPLAIN 失败: {error_msg}")
            return {
                "valid": False,
                "error": error_msg,
                "plan": None,
            }

    def validate(self, llm_response: str) -> dict:
        """
        从 LLM 回复中提取 SQL 并执行 EXPLAIN 校验（extract_sql + explain 的组合）。

        Args:
            llm_response: LLM 的完整回复文本（需包含 ```sql ... ``` 代码块）

        Returns:
            dict，包含:
                - "valid" (bool): SQL 是否通过校验
                - "sql" (str | None): 提取到的 SQL（未找到时为 None）
                - "error" (str | None): 错误信息
                - "plan" (str | None): 执行计划文本
        """
        sql = self.extract_sql(llm_response)
        if not sql:
            return {
                "valid": False,
                "sql": None,
                "error": "无法从 LLM 回复中提取 SQL",
                "plan": None,
            }

        print(f"\n[EXPLAIN 校验] 待校验 SQL:")
        print("-" * 40)
        print(sql)
        print("-" * 40)

        result = self.explain(sql)
        result["sql"] = sql
        return result
