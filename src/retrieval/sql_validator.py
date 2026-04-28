"""SQL 校验 — 通过 EXPLAIN 预执行验证 SQL 语法"""

import re
import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class SQLValidator:
    """通过 Doris EXPLAIN 校验 SQL 语法，返回执行计划供 LLM 分析"""

    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def extract_sql(llm_response: str) -> str | None:
        """从 LLM 回复中提取 SQL（```sql ``` 代码块）"""
        match = re.search(r"```sql\s*\n?(.*?)```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def explain(self, sql: str) -> dict:
        """
        执行 EXPLAIN 校验。

        Returns:
            {
                "valid": bool,
                "error": str | None,     # 语法错误信息
                "plan": str | None,      # 完整执行计划文本
            }
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
        从 LLM 回复中提取 SQL 并校验。

        Returns:
            {
                "valid": bool,
                "sql": str | None,
                "error": str | None,
                "plan": str | None,
            }
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
