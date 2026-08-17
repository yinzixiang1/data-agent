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

    _PLACEHOLDER_RE = re.compile(r"PLACEHOLDER\s*[:：]\s*(.+)", re.IGNORECASE)

    _WRITE_KEYWORDS_RE = re.compile(
        r"\b(?:INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|CREATE|ALTER|DROP|"
        r"TRUNCATE|RENAME|GRANT|REVOKE|CALL|SET|USE|LOAD|EXPORT|BACKUP|"
        r"RESTORE|KILL|LOCK|UNLOCK)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_literals_and_comments(
        sql: str,
        preserve_quoted_identifiers: bool = False,
    ) -> str:
        """移除字符串、引用标识符和注释，保留 SQL 结构用于安全检查。"""
        output: list[str] = []
        i = 0
        state = "normal"
        while i < len(sql):
            char = sql[i]
            next_char = sql[i + 1] if i + 1 < len(sql) else ""

            if state == "normal":
                if char == "'":
                    state = "single_quote"
                    output.append(" ")
                elif char == '"':
                    state = "double_quote"
                    output.append(" ")
                elif char == "`":
                    state = "backtick"
                    output.append(char if preserve_quoted_identifiers else " ")
                elif char == "-" and next_char == "-":
                    state = "line_comment"
                    output.extend((" ", " "))
                    i += 1
                elif char == "#":
                    state = "line_comment"
                    output.append(" ")
                elif char == "/" and next_char == "*":
                    state = "block_comment"
                    output.extend((" ", " "))
                    i += 1
                else:
                    output.append(char)
            elif state == "single_quote":
                output.append("\n" if char == "\n" else " ")
                if char == "\\" and next_char:
                    output.append(" ")
                    i += 1
                elif char == "'":
                    if next_char == "'":
                        output.append(" ")
                        i += 1
                    else:
                        state = "normal"
            elif state == "double_quote":
                output.append("\n" if char == "\n" else " ")
                if char == '"':
                    if next_char == '"':
                        output.append(" ")
                        i += 1
                    else:
                        state = "normal"
            elif state == "backtick":
                output.append(
                    char
                    if preserve_quoted_identifiers
                    else ("\n" if char == "\n" else " ")
                )
                if char == "`":
                    state = "normal"
            elif state == "line_comment":
                output.append("\n" if char == "\n" else " ")
                if char == "\n":
                    state = "normal"
            elif state == "block_comment":
                output.append("\n" if char == "\n" else " ")
                if char == "*" and next_char == "/":
                    output.append(" ")
                    i += 1
                    state = "normal"
            i += 1

        return "".join(output)

    @classmethod
    def validate_read_only(cls, sql: str) -> tuple[bool, str]:
        """只允许单条 SELECT/WITH 查询，拒绝写操作和导出语句。"""
        if not sql or "\x00" in sql:
            return False, "SQL 为空或包含非法字符"

        structural_sql = cls._strip_literals_and_comments(sql)
        statements = [
            part.strip() for part in structural_sql.split(";") if part.strip()
        ]
        if len(statements) != 1:
            return False, "只允许执行一条 SQL"

        statement = statements[0]
        if not re.match(r"^(?:SELECT|WITH)\b", statement, re.IGNORECASE):
            return False, "只允许 SELECT/WITH 只读查询"
        if cls._WRITE_KEYWORDS_RE.search(statement):
            return False, "SQL 包含写入或管理语句"
        if re.search(r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", statement, re.IGNORECASE):
            return False, "不允许通过 SQL 导出文件"
        if re.search(r"\bINTO\b", statement, re.IGNORECASE):
            return False, "不允许 SELECT INTO"
        if re.search(
            r"\bFOR\s+UPDATE\b|\bLOCK\s+IN\s+SHARE\s+MODE\b", statement, re.IGNORECASE
        ):
            return False, "不允许锁定读取"
        if re.match(r"^WITH\b", statement, re.IGNORECASE) and not re.search(
            r"\bSELECT\b", statement, re.IGNORECASE
        ):
            return False, "WITH 查询必须包含 SELECT"
        return True, ""

    @classmethod
    def extract_placeholder(cls, llm_response: str) -> str:
        """从 LLM 回复中提取 PLACEHOLDER 声明。

        格式: PLACEHOLDER: field1;field2;field3

        Returns:
            str: 分号分隔的占位符字段名，未找到时返回空字符串
        """
        match = cls._PLACEHOLDER_RE.search(llm_response)
        if match:
            return match.group(1).strip()
        return ""

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
        read_only, reason = self.validate_read_only(sql)
        if not read_only:
            return {"valid": False, "error": reason, "plan": None}

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
            inner = re.search(
                r"\(pymysql\.err\.\w+\)\s*\((\d+),\s*[\"'](.+?)[\"']\)", error_msg
            )
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
        structural_sql = SQLValidator._strip_literals_and_comments(
            sql,
            preserve_quoted_identifiers=True,
        )
        pattern = r"(?:FROM|JOIN)\s+`?(\w+)`?\s*\.\s*`?\w+`?"
        return {
            match.lower()
            for match in re.findall(pattern, structural_sql, re.IGNORECASE)
        }

    @classmethod
    def validate_database_access(
        cls,
        sql: str,
        authorized_databases: set[str],
    ) -> tuple[bool, str, set[str]]:
        """要求所有物理表使用库限定名，且数据库均在 Agent 白名单内。"""
        if not authorized_databases:
            return False, "当前 Agent 没有匹配的授权数据库", set()

        structural_sql = cls._strip_literals_and_comments(
            sql,
            preserve_quoted_identifiers=True,
        )
        cte_names = {
            match.lower()
            for match in re.findall(
                r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*`?(\w+)`?"
                r"(?:\s*\([^)]*\))?\s+AS\s*\(",
                structural_sql,
                re.IGNORECASE,
            )
        }
        relation_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+(?!\s*\()"
            r"(?P<first>`?\w+`?)"
            r"(?:\s*\.\s*(?P<second>`?\w+`?))?",
            re.IGNORECASE,
        )
        referenced_databases: set[str] = set()
        unqualified_tables: set[str] = set()
        for match in relation_pattern.finditer(structural_sql):
            first = match.group("first").strip("`").lower()
            if match.group("second"):
                referenced_databases.add(first)
            elif first not in cte_names:
                unqualified_tables.add(first)

        if cls._has_implicit_comma_join(structural_sql):
            return False, "不允许使用逗号连接表，请使用显式 JOIN", referenced_databases
        if unqualified_tables:
            tables = ", ".join(sorted(unqualified_tables))
            return (
                False,
                f"SQL 中的物理表必须使用数据库限定名: {tables}",
                referenced_databases,
            )
        if not referenced_databases:
            return False, "SQL 未引用可验证的数据库限定表", set()

        authorized = {database.lower() for database in authorized_databases}
        unauthorized = referenced_databases - authorized
        if unauthorized:
            databases = ", ".join(sorted(unauthorized))
            return (
                False,
                f"数据库 {databases} 未在当前 Agent 的授权范围内",
                referenced_databases,
            )
        return True, "", referenced_databases

    @staticmethod
    def _has_implicit_comma_join(structural_sql: str) -> bool:
        """检测每个 FROM 子句顶层的逗号连接，避免遗漏后续表。"""
        terminator = re.compile(
            r"\b(?:WHERE|GROUP|HAVING|ORDER|LIMIT|UNION|EXCEPT|INTERSECT|QUALIFY|WINDOW)\b",
            re.IGNORECASE,
        )
        for from_match in re.finditer(r"\bFROM\b", structural_sql, re.IGNORECASE):
            depth = 0
            index = from_match.end()
            while index < len(structural_sql):
                char = structural_sql[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif char == "," and depth == 0:
                    return True
                elif depth == 0 and terminator.match(structural_sql, index):
                    break
                index += 1
        return False

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
        read_only, reason = self.validate_read_only(sql)
        if not read_only:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": reason,
            }

        try:
            # 添加 LIMIT 防止返回过多数据（仅对 SELECT 生效）
            exec_sql = sql.rstrip().rstrip(";")
            if re.match(r"^\s*(SELECT|WITH)\b", exec_sql, re.IGNORECASE):
                # 检查是否已有 LIMIT
                if not re.search(r"\bLIMIT\s+\d+", exec_sql, re.IGNORECASE):
                    exec_sql = f"{exec_sql} LIMIT {row_limit + 1}"

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"/*+ SET_VAR(query_timeout={timeout}) */ {exec_sql}")
                )
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
            inner = re.search(
                r"\(pymysql\.err\.\w+\)\s*\((\d+),\s*[\"'](.+?)[\"']\)", error_msg
            )
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

    @staticmethod
    def extract_where_values(sql: str) -> list[dict]:
        """提取 WHERE/HAVING 子句中的等值条件。

        Returns:
            [{"column": "status", "operator": "=", "value": "success"}, ...]
        """
        conditions = []
        # col = 'val' 或 col = "val"
        for m in re.finditer(r"`?(\w+)`?\s*=\s*['\"]([^'\"]+)['\"]", sql):
            conditions.append(
                {"column": m.group(1), "operator": "=", "value": m.group(2)}
            )
        # col = 123 (纯数字)
        for m in re.finditer(r"`?(\w+)`?\s*=\s*(\d+)(?!\w)", sql):
            col = m.group(1)
            # 排除 LIMIT/OFFSET/JOIN ON 中的数字
            if col.upper() not in ("LIMIT", "OFFSET", "INTERVAL"):
                conditions.append({"column": col, "operator": "=", "value": m.group(2)})
        return conditions

    @staticmethod
    def validate_enum_values(
        where_values: list[dict], enum_hits: list[dict]
    ) -> list[dict]:
        """将 WHERE 条件中的值与枚举定义交叉校验。

        enum_hits 是扁平列表，每条表示一个枚举值：
            {"column_name": "status", "sql_value": "1", "enum_label_cn": "成功", ...}
        需要先按 column_name 分组再校验。

        Args:
            where_values: extract_where_values 的输出
            enum_hits: RAG 检索到的枚举信息（扁平列表）

        Returns:
            不匹配的条件列表，每条含 column, sql_value, expected_values, suggestion
        """
        # 按 column_name 分组
        from collections import defaultdict

        grouped: dict[str, list[dict]] = defaultdict(list)
        for e in enum_hits:
            col = (e.get("column_name") or "").lower()
            if col:
                grouped[col].append(e)

        mismatches = []
        for cond in where_values:
            col_key = cond["column"].lower()
            if col_key not in grouped:
                continue
            col_enums = grouped[col_key]
            valid_values = {str(e.get("sql_value", "")) for e in col_enums}
            if cond["value"] in valid_values:
                continue
            # 尝试通过中文标签匹配
            label_map = {
                e.get("enum_label_cn", ""): str(e.get("sql_value", ""))
                for e in col_enums
                if e.get("enum_label_cn")
            }
            suggestion = ""
            if cond["value"] in label_map:
                suggestion = f"应使用 {cond['column']} = {label_map[cond['value']]} 表示'{cond['value']}'"
            expected_str = ", ".join(
                f"{e.get('sql_value', '')}={e.get('enum_label_cn', '')}"
                for e in col_enums[:10]
            )
            mismatches.append(
                {
                    "column": cond["column"],
                    "sql_value": cond["value"],
                    "expected_values": expected_str,
                    "suggestion": suggestion,
                }
            )
        return mismatches

    @staticmethod
    def check_result_anomalies(
        question: str, sql: str, columns: list[str], rows: list[list]
    ) -> list[str]:
        """规则型预检：快速发现查询结果中的明显异常。

        Returns:
            list[str]: 异常描述列表，无异常时为空
        """
        warnings = []

        # 1. 聚合列出现负数
        for i, col in enumerate(columns):
            if any(kw in col.lower() for kw in ("count", "sum", "total", "amount")):
                for row in rows:
                    try:
                        if row[i] not in (None, "") and float(row[i]) < 0:
                            warnings.append(f"列 {col} 存在负数值，聚合结果可能有误")
                            break
                    except (ValueError, TypeError):
                        pass

        # 2. 问时间范围但 SQL 无时间条件
        time_keywords = ["本月", "今天", "本周", "今日", "当月", "昨天", "昨日"]
        if any(kw in question for kw in time_keywords):
            if not re.search(
                r"WHERE.*(?:date|time|created|updated|create_time|complete_time)",
                sql,
                re.IGNORECASE,
            ):
                warnings.append("问题涉及时间范围，但 SQL 未包含时间条件")

        # 3. 预期多行但只有 1 行
        multi_keywords = [r"排名", r"top\s*\d", r"分组", r"按.*统计", r"各", r"每"]
        if len(rows) == 1 and any(
            re.search(kw, question, re.IGNORECASE) for kw in multi_keywords
        ):
            warnings.append("预期多行结果但仅返回 1 行，聚合粒度可能有误")

        return warnings

    @staticmethod
    def simplify_sql_for_timeout(sql: str, level: int) -> str | None:
        """按级别逐步简化 SQL 以应对超时。

        level 1: 缩小 LIMIT 到 50
        level 2: 移除 ORDER BY（保留 LIMIT）
        level 3: 返回 None（需 LLM 介入）
        """
        if level == 1:
            if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
                return re.sub(r"\bLIMIT\s+\d+", "LIMIT 50", sql, flags=re.IGNORECASE)
            return sql.rstrip().rstrip(";") + " LIMIT 50"
        if level == 2:
            simplified = re.sub(
                r"\bORDER\s+BY\s+.+?(?=\bLIMIT\b|\bUNION\b|;|\s*$)",
                "",
                sql,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return simplified.strip() if simplified.strip() != sql.strip() else None
        return None

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

        print("\n[EXPLAIN 校验] 待校验 SQL:")
        print("-" * 40)
        print(sql)
        print("-" * 40)

        result = self.explain(sql)
        result["sql"] = sql
        return result
