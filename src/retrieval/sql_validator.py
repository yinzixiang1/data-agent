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

    # NEED_CLARIFY 检测
    _CLARIFY_RE = re.compile(r"NEED_CLARIFY\s*[:：]\s*(.+)", re.DOTALL)

    @staticmethod
    def extract_sql(llm_response: str) -> str | None:
        """
        从 LLM 回复中提取 SQL。

        提取优先级:
            1. ```sql ... ``` 代码块（最可靠）
            2. 裸 SQL: 以 SELECT/WITH/INSERT/UPDATE/DELETE 开头的语句（兜底）

        Args:
            llm_response: LLM 的完整回复文本

        Returns:
            str: 提取到的 SQL 字符串（已 strip），未找到时返回 None
        """
        # 优先: ```sql ``` 代码块
        match = re.search(r"```sql\s*\n?(.*?)```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 兜底: 裸 SQL（以关键词开头，取到文本末尾或空行）
        match = re.search(
            r"(?:^|\n)\s*((?:SELECT|WITH|INSERT|UPDATE|DELETE)\b.+)",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            sql = match.group(1).strip()
            # 去掉末尾可能的非 SQL 文本（以中文或 NEED_CLARIFY 开头的行）
            lines = []
            for line in sql.split("\n"):
                if re.match(r"^(NEED_CLARIFY|注意|说明|解释|备注)", line.strip()):
                    break
                lines.append(line)
            return "\n".join(lines).strip() or None

        return None

    @classmethod
    def extract_clarify(cls, llm_response: str) -> str | None:
        """
        检测 NEED_CLARIFY 回复。

        Returns:
            str: 澄清问题文本，未检测到时返回 None
        """
        match = cls._CLARIFY_RE.search(llm_response)
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

    @staticmethod
    def extract_databases(sql: str) -> set[str]:
        """
        从 SQL 中提取引用的数据库名。

        匹配模式: `database`.`table` 或 database.table（FROM / JOIN 子句中）

        Returns:
            set[str]: 数据库名集合（小写）
        """
        # 匹配 `db`.`table` 或 db.table（支持反引号和不带引号两种形式）
        pattern = r'(?:FROM|JOIN)\s+`?(\w+)`?\s*\.\s*`?\w+`?'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        # 排除 information_schema 等系统库
        system_dbs = {"information_schema", "mysql", "performance_schema"}
        return {m.lower() for m in matches if m.lower() not in system_dbs}

    def execute(self, sql: str, row_limit: int = 200, timeout: int = 30) -> dict:
        """
        执行 SQL 并返回结果集。

        Args:
            sql: 待执行的 SQL
            row_limit: 最大返回行数
            timeout: 执行超时秒数

        Returns:
            dict，包含:
                - "success" (bool)
                - "columns" (list[str]): 列名列表
                - "rows" (list[list]): 数据行
                - "row_count" (int): 实际返回行数
                - "truncated" (bool): 是否截断
                - "error" (str | None)
        """
        try:
            # 添加 LIMIT 防止返回过多数据（仅对 SELECT 生效）
            exec_sql = sql.rstrip().rstrip(";")
            if re.match(r"^\s*(SELECT|WITH)\b", exec_sql, re.IGNORECASE):
                # 检查是否已有 LIMIT
                if not re.search(r"\bLIMIT\s+\d+", exec_sql, re.IGNORECASE):
                    exec_sql = f"{exec_sql} LIMIT {row_limit + 1}"

            with self.engine.connect() as conn:
                result = conn.execute(text(f"/*+ SET_VAR(query_timeout={timeout}) */ {exec_sql}"))
                columns = list(result.keys())
                all_rows = result.fetchall()

            truncated = len(all_rows) > row_limit
            rows = all_rows[:row_limit]

            # 转换为可序列化的列表
            serialized = []
            for row in rows:
                serialized.append([self._serialize_cell(cell) for cell in row])

            logger.info(f"SQL 执行成功: {len(serialized)} 行, {len(columns)} 列")
            return {
                "success": True,
                "columns": columns,
                "rows": serialized,
                "row_count": len(serialized),
                "truncated": truncated,
                "error": None,
            }

        except Exception as e:
            error_msg = str(e)
            inner = re.search(r"\(pymysql\.err\.\w+\)\s*\((\d+),\s*[\"'](.+?)[\"']\)", error_msg)
            if inner:
                error_msg = f"Error {inner.group(1)}: {inner.group(2)}"
            logger.warning(f"SQL 执行失败: {error_msg}")
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": error_msg,
            }

    @staticmethod
    def _serialize_cell(value) -> str:
        """将单元格值转为字符串，处理特殊类型。"""
        if value is None:
            return ""
        from datetime import date, datetime
        from decimal import Decimal
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

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
