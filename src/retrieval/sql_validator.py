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

import json
import logging
import re
from typing import ClassVar

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

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
    _CLARIFY_RE = re.compile(
        r"^\s*NEED_CLARIFY\b(?:\s*[:：]\s*|\s+)(.+)",
        re.DOTALL | re.MULTILINE,
    )

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

    @classmethod
    def extract_clarification(cls, llm_response: str) -> dict | None:
        """将 NEED_CLARIFY 转为稳定的卡片交互结构。"""
        raw = cls.extract_clarify(llm_response)
        if not raw:
            return None

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            try:
                payload, _ = json.JSONDecoder().raw_decode(raw)
            except (json.JSONDecodeError, TypeError):
                return {"question": raw[:1000], "options": []}

        if not isinstance(payload, dict):
            return {"question": raw[:1000], "options": []}

        question = str(payload.get("question") or "").strip()
        if not question:
            return {"question": raw[:1000], "options": []}

        options = []
        for option in payload.get("options") or []:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").strip()
            value = str(option.get("value") or "").strip()
            if label and value:
                options.append({"label": label[:80], "value": value[:500]})
            if len(options) == 4:
                break

        return {"question": question[:1000], "options": options}

    _PLACEHOLDER_RE = re.compile(r"PLACEHOLDER\s*[:：]\s*(.+)", re.IGNORECASE)

    _WRITE_KEYWORDS_RE = re.compile(
        r"\b(?:INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|CREATE|ALTER|DROP|"
        r"TRUNCATE|RENAME|GRANT|REVOKE|CALL|SET|USE|LOAD|EXPORT|BACKUP|"
        r"RESTORE|KILL|LOCK|UNLOCK)\b",
        re.IGNORECASE,
    )
    _EXCHANGE_RATE_TABLE = "warehouse_sys.sys_exchange_rate"
    _TRANSIENT_DBAPI_CODES: ClassVar[frozenset[int]] = frozenset({2006, 2013, 2055})

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
        ast_valid, ast_reason = cls._validate_ast_read_only(sql)
        if not ast_valid:
            return False, ast_reason
        return True, ""

    @staticmethod
    def _parse_ast(sql: str):
        """使用 MySQL 兼容语法解析 Doris SQL；依赖缺失时保留旧校验能力。"""
        try:
            import sqlglot
        except ImportError:
            return None, ""
        try:
            return sqlglot.parse_one(sql, read="mysql"), ""
        except sqlglot.errors.ParseError as exc:
            return None, f"SQL AST 解析失败: {exc}"

    @classmethod
    def _validate_ast_read_only(cls, sql: str) -> tuple[bool, str]:
        tree, error = cls._parse_ast(sql)
        if error:
            return False, error
        if tree is None:
            return True, ""

        from sqlglot import expressions as exp

        forbidden_types = tuple(
            expression_type
            for name in (
                "Insert",
                "Update",
                "Delete",
                "Create",
                "Drop",
                "Alter",
                "Merge",
                "Command",
                "Transaction",
            )
            if (expression_type := getattr(exp, name, None)) is not None
        )
        if forbidden_types and any(tree.find_all(*forbidden_types)):
            return False, "SQL AST 包含写入或管理语句"
        for join in tree.find_all(exp.Join):
            kind = str(join.args.get("kind") or "").upper()
            if kind == "CROSS":
                return False, "不允许 CROSS JOIN"
            if not join.args.get("on") and not join.args.get("using"):
                return False, "JOIN 必须包含 ON 或 USING 条件"
        return True, ""

    @classmethod
    def validate_schema_references(
        cls,
        sql: str,
        allowed_schemas: list[dict],
    ) -> tuple[bool, str, dict]:
        """校验 SQL 的物理表和限定字段均来自本次检索上下文。"""
        tree, error = cls._parse_ast(sql)
        if error:
            return False, error, {"failure_type": "invalid_sql"}
        if tree is None:
            # 运行环境尚未安装 sqlglot 时不做伪精确的正则字段校验。
            return True, "", {"ast_validation": "unavailable"}

        from sqlglot import expressions as exp

        by_full: dict[str, dict] = {}
        by_short: dict[str, list[dict]] = {}
        for schema in allowed_schemas:
            full_name = str(schema.get("table_name") or "").replace("`", "")
            if not full_name:
                continue
            by_full[full_name.casefold()] = schema
            short_name = str(
                schema.get("table_name_short") or full_name.rsplit(".", 1)[-1]
            ).casefold()
            by_short.setdefault(short_name, []).append(schema)

        cte_names = {cte.alias_or_name.casefold() for cte in tree.find_all(exp.CTE)}
        alias_to_schema: dict[str, dict] = {}
        referenced_tables: set[str] = set()
        for table in tree.find_all(exp.Table):
            short_name = table.name.casefold()
            if short_name in cte_names:
                continue
            database = str(table.db or "").casefold()
            full_name = f"{database}.{short_name}" if database else short_name
            schema = by_full.get(full_name)
            if schema is None:
                candidates = by_short.get(short_name, [])
                if len(candidates) == 1:
                    schema = candidates[0]
            if schema is None:
                return (
                    False,
                    f"SQL 引用了本次检索上下文之外的表: {full_name}",
                    {
                        "failure_type": "table_outside_context",
                        "referenced_tables": sorted(referenced_tables | {full_name}),
                        "invalid_tables": [full_name],
                    },
                )
            canonical = str(schema.get("table_name") or full_name)
            referenced_tables.add(canonical)
            alias_to_schema[short_name] = schema
            alias_to_schema[str(table.alias_or_name).casefold()] = schema

        invalid_columns: set[str] = set()
        for column in tree.find_all(exp.Column):
            qualifier = str(column.table or "").casefold()
            if not qualifier or qualifier in cte_names:
                continue
            schema = alias_to_schema.get(qualifier)
            if schema is None:
                continue
            available_columns = {
                str(item.get("name") or "").casefold()
                for item in schema.get("columns", [])
            }
            if column.name.casefold() not in available_columns:
                invalid_columns.add(f"{qualifier}.{column.name}")
        if invalid_columns:
            return (
                False,
                "SQL 引用了本次字段上下文之外的字段: "
                + ", ".join(sorted(invalid_columns)),
                {
                    "failure_type": "column_outside_context",
                    "referenced_tables": sorted(referenced_tables),
                    "invalid_columns": sorted(invalid_columns),
                },
            )
        return True, "", {"referenced_tables": sorted(referenced_tables)}

    @classmethod
    def validate_requested_projection(
        cls,
        sql: str,
        requirements: list[dict],
    ) -> tuple[bool, str, dict]:
        """确保用户明确要求展示的字段没有在 SQL 纠错中被删除。"""
        grounded_requirements = [
            requirement for requirement in requirements if requirement.get("columns")
        ]
        if not grounded_requirements:
            return True, "", {"required": False}

        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return False, parse_error, {"required": True}

        projected_columns: set[str] = set()
        has_star = False
        if tree is not None:
            from sqlglot import expressions as exp

            select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
            if select is not None:
                for expression in select.expressions:
                    if isinstance(expression, exp.Star) or (
                        isinstance(expression, exp.Column)
                        and isinstance(expression.this, exp.Star)
                    ):
                        has_star = True
                    projected_columns.update(
                        column.name.casefold()
                        for column in expression.find_all(exp.Column)
                    )
        else:
            structural_sql = cls._strip_literals_and_comments(
                sql,
                preserve_quoted_identifiers=True,
            )
            match = re.search(
                r"\bSELECT\b(?P<select>.*?)\bFROM\b",
                structural_sql,
                re.IGNORECASE | re.DOTALL,
            )
            select_text = match.group("select") if match else ""
            has_star = bool(
                re.search(
                    r"(?:^|,)\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?"
                    r"\*\s*(?:,|$)",
                    select_text,
                )
            )
            projected_columns.update(
                value.casefold()
                for value in re.findall(
                    r"(?:\b|`)([A-Za-z_][A-Za-z0-9_]*)`?", select_text
                )
            )

        missing: list[dict] = []
        if not has_star:
            for requirement in grounded_requirements:
                candidates = {
                    str(column).rsplit(".", 1)[-1].casefold()
                    for column in requirement.get("columns", [])
                }
                if not candidates & projected_columns:
                    missing.append(requirement)

        detail = {
            "required": True,
            "projected_columns": sorted(projected_columns),
            "missing_fields": [item.get("field", "") for item in missing],
        }
        if not missing:
            return True, "", detail

        descriptions = []
        for requirement in missing:
            candidates = ", ".join(requirement.get("columns", []))
            descriptions.append(
                f"{requirement.get('field', '')}（候选列: {candidates}）"
            )
        return (
            False,
            "SQL 未在最终 SELECT 结果中保留用户明确要求的字段: "
            + "；".join(descriptions),
            detail,
        )

    @classmethod
    def aggregate_projections(cls, context: dict[str, list[str]]) -> list[str]:
        """Return SELECT projections containing an aggregate expression."""
        import sqlglot
        from sqlglot import expressions as exp

        aggregates = []
        for fragment in context.get("projections") or []:
            try:
                statement = sqlglot.parse_one(f"SELECT {fragment}", read="mysql")
            except sqlglot.errors.ParseError:
                continue
            projection = next(iter(statement.expressions), None)
            if projection is not None and next(
                projection.find_all(exp.AggFunc),
                None,
            ):
                aggregates.append(str(fragment))
        return aggregates

    @staticmethod
    def metrics_replaced(
        previous_query_state: dict | None,
        current_query_state: dict | None,
    ) -> bool:
        """Return whether all previous metrics were replaced by different metrics."""

        def normalized_metrics(state: dict | None) -> set[str]:
            payload = state if isinstance(state, dict) else {}
            return {
                str(metric).strip().casefold()
                for metric in payload.get("metrics") or []
                if str(metric).strip()
            }

        previous = normalized_metrics(previous_query_state)
        current = normalized_metrics(current_query_state)
        return bool(previous and current and previous.isdisjoint(current))

    @classmethod
    def validate_followup_inheritance(
        cls,
        sql: str,
        previous_context: dict[str, list[str]],
        relation: str,
        removed_context: dict[str, list[str]] | None = None,
        previous_query_state: dict | None = None,
        current_query_state: dict | None = None,
    ) -> tuple[bool, str, dict]:
        """Preserve every previous SQL fragment not explicitly removed."""
        if relation == "new_question" or not previous_context:
            return True, "", {"required": False, "missing": {}}

        from src.retrieval.context_compressor import ContextCompressor

        current_context = ContextCompressor.extract_sql_context(sql)

        def normalized(values: list[str], section: str) -> set[str]:
            result = set()
            for value in values:
                fragment = str(value).replace("`", "").strip()
                if not fragment:
                    continue
                if section in {"projections", "dimensions", "filters", "order_by"}:
                    # A follow-up may introduce a JOIN and therefore qualify
                    # previously unqualified columns (account_id -> o.account_id).
                    # Qualification alone is not a semantic structure change.
                    fragment = re.sub(
                        r"(?<![\w.])(?:[A-Za-z_][A-Za-z0-9_]*)\."
                        r"(?=[A-Za-z_][A-Za-z0-9_]*)",
                        "",
                        fragment,
                    )
                result.add(re.sub(r"\s+", " ", fragment).casefold())
            return result

        removed_context = removed_context or {}
        allowed_removed: dict[str, list[str]] = {}
        missing: dict[str, list[str]] = {}
        for section in (
            "tables",
            "projections",
            "dimensions",
            "filters",
            "joins",
            "order_by",
            "limit",
        ):
            previous_values = previous_context.get(section) or []
            current_values = normalized(current_context.get(section) or [], section)
            removed_values = normalized(removed_context.get(section) or [], section)
            allowed_removed[section] = [
                str(value)
                for value in previous_values
                if normalized([str(value)], section) & removed_values
            ]
            absent = [
                str(value)
                for value in previous_values
                if not normalized([str(value)], section) & current_values
                and not normalized([str(value)], section) & removed_values
            ]
            if absent:
                missing[section] = absent

        stale_metric_projections: list[str] = []
        if cls.metrics_replaced(previous_query_state, current_query_state):
            current_aggregates = normalized(
                cls.aggregate_projections(current_context),
                "projections",
            )
            stale_metric_projections = [
                fragment
                for fragment in cls.aggregate_projections(previous_context)
                if normalized([fragment], "projections") & current_aggregates
            ]

        detail = {
            "required": True,
            "missing": missing,
            "previous": previous_context,
            "current": current_context,
            "allowed_removed": allowed_removed,
            "stale_metric_projections": stale_metric_projections,
        }
        if not missing and not stale_metric_projections:
            return True, "", detail

        descriptions = [
            f"{section}: {', '.join(values)}" for section, values in missing.items()
        ]
        if stale_metric_projections:
            descriptions.append(
                "指标已替换但仍保留旧聚合投影: " + ", ".join(stale_metric_projections)
            )
        return (
            False,
            "追问 SQL 丢失上一轮结构或与本轮指标变更不一致：" + "；".join(descriptions),
            detail,
        )

    @staticmethod
    def _query_state(query_intent: dict) -> dict:
        state = query_intent.get("state")
        return state if isinstance(state, dict) else query_intent

    @classmethod
    def is_count_only(cls, query_intent: dict) -> bool:
        """Return whether the semantic frame requires one ungrouped COUNT."""
        return cls._query_state(query_intent).get("result_shape") == "count_only"

    @classmethod
    def currency_conversion_target(cls, query_intent: dict) -> str:
        """Return the normalized target currency from the semantic frame."""
        state = cls._query_state(query_intent)
        value = str(state.get("currency_conversion") or "").strip()
        currency_codes = re.findall(
            r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])",
            value,
        )
        return currency_codes[-1].upper() if currency_codes else value.upper()

    @classmethod
    def calendar_day_window(cls, query_intent: dict) -> int | None:
        """Return a rolling calendar-day window that includes the current day."""
        value = cls._query_state(query_intent).get("calendar_day_window")
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if 1 <= parsed <= 3660 else None

    @classmethod
    def requested_limit(cls, query_intent: dict) -> int | None:
        """Return only a row limit explicitly requested by the user."""
        value = cls._query_state(query_intent).get("requested_limit")
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if 1 <= parsed <= 100000 else None

    @classmethod
    def validate_calendar_day_window(
        cls,
        sql: str,
        query_intent: dict,
    ) -> tuple[bool, str, dict]:
        """Enforce the model-produced rolling natural-day contract."""
        days = cls.calendar_day_window(query_intent)
        if days is None:
            return True, "", {"required": False}

        expected_offset = days - 1
        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return False, parse_error, {"required": True, "days": days}

        lower_columns: set[str] = set()
        upper_columns: set[str] = set()
        if tree is not None:
            from sqlglot import expressions as exp

            def column_key(expression) -> str:
                if not isinstance(expression, exp.Column):
                    return ""
                return expression.sql(dialect="mysql").replace("`", "").casefold()

            def interval_value(expression, expression_type) -> int | None:
                if not isinstance(expression, expression_type):
                    return None
                if not isinstance(expression.this, exp.CurrentDate):
                    return None
                unit = str(expression.args.get("unit") or "").upper()
                if unit != "DAY":
                    return None
                interval = expression.expression
                if not isinstance(interval, exp.Literal):
                    return None
                try:
                    return int(interval.this)
                except (TypeError, ValueError):
                    return None

            def is_lower_boundary(expression) -> bool:
                if expected_offset == 0 and isinstance(expression, exp.CurrentDate):
                    return True
                return interval_value(expression, exp.DateSub) == expected_offset

            def is_upper_boundary(expression) -> bool:
                return interval_value(expression, exp.DateAdd) == 1

            for comparison in tree.find_all(exp.GTE):
                if is_lower_boundary(comparison.expression) and (
                    key := column_key(comparison.this)
                ):
                    lower_columns.add(key)
            for comparison in tree.find_all(exp.LTE):
                if is_lower_boundary(comparison.this) and (
                    key := column_key(comparison.expression)
                ):
                    lower_columns.add(key)
            for comparison in tree.find_all(exp.LT):
                if is_upper_boundary(comparison.expression) and (
                    key := column_key(comparison.this)
                ):
                    upper_columns.add(key)
            for comparison in tree.find_all(exp.GT):
                if is_upper_boundary(comparison.this) and (
                    key := column_key(comparison.expression)
                ):
                    upper_columns.add(key)
        else:
            normalized_sql = cls._strip_literals_and_comments(sql)
            lower_pattern = (
                r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*>=\s*CURDATE\s*\(\s*\)"
                if expected_offset == 0
                else (
                    r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*>=\s*DATE_SUB\s*\(\s*"
                    rf"CURDATE\s*\(\s*\)\s*,\s*INTERVAL\s+{expected_offset}\s+DAY\s*\)"
                )
            )
            lower_columns.update(
                match.casefold()
                for match in re.findall(lower_pattern, normalized_sql, re.IGNORECASE)
            )
            upper_columns.update(
                match.casefold()
                for match in re.findall(
                    r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*<\s*DATE_ADD\s*\(\s*"
                    r"CURDATE\s*\(\s*\)\s*,\s*INTERVAL\s+1\s+DAY\s*\)",
                    normalized_sql,
                    re.IGNORECASE,
                )
            )

        valid = bool(lower_columns & upper_columns)
        detail = {
            "required": True,
            "days": days,
            "includes_today": True,
            "expected_start_offset_days": expected_offset,
            "lower_bound_columns": sorted(lower_columns),
            "upper_bound_columns": sorted(upper_columns),
        }
        if valid:
            return True, "", detail
        error = (
            f"最近 {days} 天默认包含今天，时间范围必须使用同一时间字段，"
            f"下界为 DATE_SUB(CURDATE(), INTERVAL {expected_offset} DAY)，"
            "上界为 DATE_ADD(CURDATE(), INTERVAL 1 DAY) 且采用左闭右开边界"
        )
        return (
            False,
            error,
            detail,
        )

    @classmethod
    def validate_result_limit(
        cls,
        sql: str,
        query_intent: dict,
    ) -> tuple[bool, str, dict]:
        """Prevent unrequested truncation of aggregate result groups."""
        state = cls._query_state(query_intent)
        if state.get("result_shape") != "aggregate":
            return True, "", {"required": False}

        requested_limit = cls.requested_limit(query_intent)
        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return False, parse_error, {"required": True}

        actual_limit: int | None = None
        if tree is not None:
            from sqlglot import expressions as exp

            select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
            limit = select.args.get("limit") if select is not None else None
            expression = limit.expression if isinstance(limit, exp.Limit) else None
            if isinstance(expression, exp.Literal):
                try:
                    actual_limit = int(expression.this)
                except (TypeError, ValueError):
                    actual_limit = None
        else:
            matches = re.findall(r"\bLIMIT\s+(\d+)\b", sql, re.IGNORECASE)
            actual_limit = int(matches[-1]) if matches else None

        detail = {
            "required": True,
            "result_shape": "aggregate",
            "requested_limit": requested_limit,
            "actual_limit": actual_limit,
        }
        if requested_limit is None and actual_limit is None:
            return True, "", detail
        if requested_limit is not None and actual_limit == requested_limit:
            return True, "", detail
        if requested_limit is None:
            return (
                False,
                "聚合查询未明确要求限制分组数量，不得用 LIMIT 静默截断结果",
                detail,
            )
        return (
            False,
            f"用户明确要求返回 {requested_limit} 条，SQL 必须使用 LIMIT {requested_limit}",
            detail,
        )

    @classmethod
    def validate_metric_projection(
        cls,
        sql: str,
        query_intent: dict,
    ) -> tuple[bool, str, dict]:
        """Enforce exclusive metric requests after every SQL repair step."""
        count_only = cls.is_count_only(query_intent)
        if not count_only:
            return True, "", {"required": False}

        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return (
                False,
                parse_error,
                {
                    "required": True,
                    "count_only": count_only,
                },
            )

        invalid_projections: list[str] = []
        has_group_by = False
        has_exchange_rate = False
        if tree is not None:
            from sqlglot import expressions as exp

            select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
            if select is not None:
                has_group_by = select.args.get("group") is not None
                if len(select.expressions) != 1:
                    invalid_projections.extend(
                        expression.sql(dialect="mysql")
                        for expression in select.expressions
                    )
                for expression in select.expressions:
                    projected = (
                        expression.this
                        if isinstance(expression, exp.Alias)
                        else expression
                    )
                    has_count = isinstance(projected, exp.Count) or (
                        projected.find(exp.Count) is not None
                    )
                    if (
                        not has_count
                        and expression.sql(dialect="mysql") not in invalid_projections
                    ):
                        invalid_projections.append(expression.sql(dialect="mysql"))
            has_exchange_rate = any(
                table.name.casefold() == "sys_exchange_rate"
                for table in tree.find_all(exp.Table)
            )
        else:
            structural_sql = cls._strip_literals_and_comments(
                sql,
                preserve_quoted_identifiers=True,
            )
            match = re.search(
                r"\bSELECT\b(?P<select>.*?)\bFROM\b",
                structural_sql,
                re.IGNORECASE | re.DOTALL,
            )
            projection = match.group("select") if match else ""
            count_only_projection = re.fullmatch(
                r"\s*COUNT\s*\([^)]*\)\s*"
                r"(?:(?:AS\s+)?(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))?\s*",
                projection,
                flags=re.IGNORECASE,
            )
            invalid_projections = [] if count_only_projection else [projection.strip()]
            has_group_by = bool(
                re.search(r"\bGROUP\s+BY\b", structural_sql, re.IGNORECASE)
            )
            has_exchange_rate = "sys_exchange_rate" in structural_sql.casefold()

        detail = {
            "required": True,
            "count_only": count_only,
            "invalid_projections": invalid_projections,
            "has_group_by": has_group_by,
            "has_exchange_rate": has_exchange_rate,
        }
        if not invalid_projections and not has_group_by and not has_exchange_rate:
            return True, "", detail
        issues = []
        if invalid_projections:
            issues.append("SELECT 包含次数之外的结果字段")
        if has_group_by:
            issues.append("SQL 仍按维度分组")
        if has_exchange_rate:
            issues.append("SQL 仍引用汇率表")
        return (
            False,
            "用户要求仅统计次数，但" + "、".join(issues),
            detail,
        )

    @classmethod
    def validate_entity_filters(
        cls,
        sql: str,
        requirements: list[dict],
    ) -> tuple[bool, str, dict]:
        """确保配置化实体链接在最终 SQL 中仍使用指定表、字段和值。"""
        if not requirements:
            return True, "", {"required": False}

        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return False, parse_error, {"required": True}
        if tree is None:
            return cls._validate_entity_filters_without_ast(sql, requirements)

        from sqlglot import expressions as exp

        alias_to_table: dict[str, str] = {}
        referenced_tables: set[str] = set()
        for table in tree.find_all(exp.Table):
            full_name = (
                f"{table.db}.{table.name}" if table.db else str(table.name)
            ).casefold()
            referenced_tables.add(full_name)
            alias_to_table[str(table.alias_or_name).casefold()] = full_name
            alias_to_table[table.name.casefold()] = full_name

        def literal_value(expression) -> str | None:
            if not isinstance(expression, exp.Literal):
                return None
            return str(expression.this)

        missing: list[dict] = []
        for requirement in requirements:
            expected_table = str(requirement.get("table") or "").casefold()
            expected_column = str(requirement.get("column") or "").casefold()
            expected_value = str(requirement.get("value") or "")
            table_present = expected_table in referenced_tables or any(
                table.endswith(f".{expected_table}")
                for table in referenced_tables
                if "." not in expected_table
            )
            found = False
            if table_present:
                for equality in tree.find_all(exp.EQ):
                    for column_expr, value_expr in (
                        (equality.this, equality.expression),
                        (equality.expression, equality.this),
                    ):
                        if not isinstance(column_expr, exp.Column):
                            continue
                        if column_expr.name.casefold() != expected_column:
                            continue
                        actual_value = literal_value(value_expr)
                        if actual_value != expected_value:
                            continue
                        qualifier = str(column_expr.table or "").casefold()
                        if qualifier:
                            actual_table = alias_to_table.get(qualifier, qualifier)
                            if actual_table != expected_table:
                                continue
                        elif len(referenced_tables) != 1:
                            continue
                        found = True
                        break
                    if found:
                        break
            if not found:
                missing.append(requirement)

        detail = {
            "required": True,
            "referenced_tables": sorted(referenced_tables),
            "missing_filters": [
                {
                    "qualified_column": item.get("qualified_column", ""),
                    "value": item.get("value", ""),
                }
                for item in missing
            ],
        }
        if not missing:
            return True, "", detail

        descriptions = [
            f"{item.get('qualified_column', '')} = {item.get('value', '')!r}"
            for item in missing
        ]
        return (
            False,
            "SQL 未保留已确定的实体过滤条件: " + "；".join(descriptions),
            detail,
        )

    @classmethod
    def _validate_entity_filters_without_ast(
        cls,
        sql: str,
        requirements: list[dict],
    ) -> tuple[bool, str, dict]:
        """sqlglot 不可用时，用保守正则校验简单的表别名和等值条件。"""
        normalized = sql.replace("`", "")
        aliases: dict[str, str] = {}
        referenced_tables: set[str] = set()
        reserved = {
            "where",
            "join",
            "left",
            "right",
            "inner",
            "outer",
            "group",
            "order",
            "limit",
            "on",
        }
        table_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+"
            r"(?P<table>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
            r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?",
            re.IGNORECASE,
        )
        for match in table_pattern.finditer(normalized):
            table_name = match.group("table").casefold()
            alias = str(match.group("alias") or "").casefold()
            if alias in reserved:
                alias = ""
            short_name = table_name.rsplit(".", 1)[-1]
            referenced_tables.add(table_name)
            aliases[short_name] = table_name
            if alias:
                aliases[alias] = table_name

        missing: list[dict] = []
        for requirement in requirements:
            expected_table = str(requirement.get("table") or "").casefold()
            expected_column = str(requirement.get("column") or "")
            expected_value = str(requirement.get("value") or "")
            table_present = expected_table in referenced_tables
            value_pattern = rf"(?:'{re.escape(expected_value)}'|\"{re.escape(expected_value)}\"|{re.escape(expected_value)})"
            equality_pattern = re.compile(
                rf"(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?"
                rf"{re.escape(expected_column)}\s*=\s*{value_pattern}"
                r"(?![A-Za-z0-9_.@-])",
                re.IGNORECASE,
            )
            found = False
            if table_present:
                for match in equality_pattern.finditer(normalized):
                    qualifier = str(match.group("alias") or "").casefold()
                    if qualifier and aliases.get(qualifier) != expected_table:
                        continue
                    if not qualifier and len(referenced_tables) != 1:
                        continue
                    found = True
                    break
            if not found:
                missing.append(requirement)

        detail = {
            "required": True,
            "referenced_tables": sorted(referenced_tables),
            "missing_filters": [
                {
                    "qualified_column": item.get("qualified_column", ""),
                    "value": item.get("value", ""),
                }
                for item in missing
            ],
            "ast_validation": "unavailable",
        }
        if not missing:
            return True, "", detail
        descriptions = [
            f"{item.get('qualified_column', '')} = {item.get('value', '')!r}"
            for item in missing
        ]
        return (
            False,
            "SQL 未保留已确定的实体过滤条件: " + "；".join(descriptions),
            detail,
        )

    @classmethod
    def validate_currency_conversion(
        cls,
        sql: str,
        query_intent: dict,
    ) -> tuple[bool, str, dict]:
        """校验货币换算 SQL 是否完整执行了确定的业务口径。"""
        target_currency = cls.currency_conversion_target(query_intent)
        if not target_currency:
            return True, "", {"required": False}

        tree, parse_error = cls._parse_ast(sql)
        if parse_error:
            return False, parse_error, {"required": True, "issues": [parse_error]}
        if tree is None:
            return cls._validate_currency_conversion_without_ast(
                sql,
                target_currency,
            )

        from sqlglot import expressions as exp

        rate_table_found = False
        rate_join_is_left = False
        rate_aliases: set[str] = set()
        for table in tree.find_all(exp.Table):
            full_name = (
                f"{table.db}.{table.name}" if table.db else str(table.name)
            ).casefold()
            if full_name != cls._EXCHANGE_RATE_TABLE:
                continue
            rate_table_found = True
            rate_aliases.add(table.alias_or_name.casefold())
            rate_aliases.add(table.name.casefold())

        for join in tree.find_all(exp.Join):
            joined_table = join.this
            if not isinstance(joined_table, exp.Table):
                continue
            full_name = (
                f"{joined_table.db}.{joined_table.name}"
                if joined_table.db
                else str(joined_table.name)
            ).casefold()
            if full_name == cls._EXCHANGE_RATE_TABLE:
                rate_join_is_left = str(join.args.get("side") or "").upper() == "LEFT"

        all_columns = list(tree.find_all(exp.Column))
        column_names = {column.name.casefold() for column in all_columns}

        def literal_value(expression) -> str | None:
            if isinstance(expression, exp.Literal):
                return str(expression.this).upper()
            return None

        def is_rate_column(column, name: str | None = None) -> bool:
            if not isinstance(column, exp.Column):
                return False
            if name and column.name.casefold() != name:
                return False
            return str(column.table or "").casefold() in rate_aliases

        def is_fact_currency_column(column) -> bool:
            if not isinstance(column, exp.Column) or is_rate_column(column):
                return False
            name = column.name.casefold()
            return "currency" in name or name in {"ccy", "currency_code"}

        target_filter_found = False
        source_join_found = False
        date_join_found = False
        target_passthrough_found = False
        time_names = {
            "create_time",
            "complete_time",
            "settle_time",
            "transaction_time",
            "created_at",
            "completed_at",
        }

        for equality in tree.find_all(exp.EQ):
            left = equality.this
            right = equality.expression
            pairs = ((left, right), (right, left))
            for column_expr, value_expr in pairs:
                if not isinstance(column_expr, exp.Column):
                    continue
                if (
                    is_rate_column(column_expr, "target_currency")
                    and literal_value(value_expr) == target_currency
                ):
                    target_filter_found = True

            equality_columns = list(equality.find_all(exp.Column))
            rate_source_columns = [
                column
                for column in equality_columns
                if is_rate_column(column, "source_currency")
            ]
            fact_currency_columns = [
                column for column in equality_columns if is_fact_currency_column(column)
            ]
            if rate_source_columns and fact_currency_columns:
                source_join_found = True
            rate_time_columns = [
                column
                for column in equality_columns
                if is_rate_column(column, "sync_time")
            ]
            fact_time_columns = [
                column
                for column in equality_columns
                if not is_rate_column(column)
                and (
                    column.name.casefold() in time_names
                    or column.name.casefold().endswith("_time")
                    or column.name.casefold().endswith("_date")
                    or column.name.casefold().endswith("_at")
                )
            ]
            if rate_time_columns and fact_time_columns:
                date_join_found = True

        for case in tree.find_all(exp.Case):
            for equality in case.find_all(exp.EQ):
                equality_columns = list(equality.find_all(exp.Column))
                if not any(
                    is_fact_currency_column(column) for column in equality_columns
                ):
                    continue
                if any(
                    literal_value(node) == target_currency
                    for node in equality.find_all(exp.Literal)
                ):
                    target_passthrough_found = True
                    break

        mid_multiplication_found = False
        for multiplication in tree.find_all(exp.Mul):
            multiplication_columns = list(multiplication.find_all(exp.Column))
            if any(
                is_rate_column(column, "mid") for column in multiplication_columns
            ) and any(not is_rate_column(column) for column in multiplication_columns):
                mid_multiplication_found = True
                break

        rate_fallback_found = any(
            any(
                is_rate_column(column, "mid")
                for column in function.find_all(exp.Column)
            )
            for function in tree.find_all(exp.Coalesce)
        )
        missing_rate_indicator_found = any(
            is_rate_column(condition.this, "mid")
            and isinstance(condition.expression, exp.Null)
            for condition in tree.find_all(exp.Is)
        )

        issues: list[str] = []
        if not rate_table_found:
            issues.append(f"必须引用 {cls._EXCHANGE_RATE_TABLE}")
        elif not rate_join_is_left:
            issues.append("汇率表必须使用 LEFT JOIN，避免目标币种原始记录被丢弃")
        if "mid" not in column_names or not mid_multiplication_found:
            issues.append("必须使用 原金额 * mid 计算目标币种金额")
        if not source_join_found:
            issues.append("必须用 source_currency 关联交易原币种")
        if not target_filter_found:
            issues.append(f"必须限定 target_currency = '{target_currency}'")
        if not date_join_found:
            issues.append("必须按 sync_time 与交易日期关联每日汇率")
        if not target_passthrough_found:
            issues.append(
                f"原币种为 {target_currency} 时必须直接使用原金额，汇率按 1 处理"
            )
        if rate_fallback_found:
            issues.append("非目标币种缺失汇率时禁止用 COALESCE/IFNULL 按1或0兜底")
        if not missing_rate_indicator_found:
            issues.append("必须显式返回非目标币种的缺失汇率笔数（mid IS NULL）")

        detail = {
            "required": True,
            "target_currency": target_currency,
            "exchange_rate_table": cls._EXCHANGE_RATE_TABLE,
            "exchange_rate_aliases": sorted(rate_aliases),
            "rate_fallback_found": rate_fallback_found,
            "missing_rate_indicator_found": missing_rate_indicator_found,
            "issues": issues,
        }
        if issues:
            return False, "；".join(issues), detail
        return True, "", detail

    @classmethod
    def _validate_currency_conversion_without_ast(
        cls,
        sql: str,
        target_currency: str,
    ) -> tuple[bool, str, dict]:
        """在 sqlglot 不可用时保守校验固定的换汇 SQL 结构。"""
        normalized = sql.replace("`", "")
        flags = re.IGNORECASE | re.DOTALL
        rate_table = r"warehouse_sys\s*\.\s*sys_exchange_rate"
        target_literal = re.escape(target_currency)
        currency_column = r"(?:\w+\.)?(?!target_currency\b)\w*currency"
        amount_column = r"(?:\w+\.)?\w+"
        mid_column = r"(?:\w+\.)?mid"

        rate_table_found = bool(re.search(rate_table, normalized, flags))
        rate_join_is_left = bool(
            re.search(rf"\bLEFT\s+JOIN\s+{rate_table}\b", normalized, flags)
        )
        mid_multiplication_found = bool(
            re.search(
                rf"(?:{amount_column}\s*\*\s*{mid_column}|"
                rf"{mid_column}\s*\*\s*{amount_column})",
                normalized,
                flags,
            )
        )
        source_join_found = bool(
            re.search(
                rf"(?:\b(?:\w+\.)?source_currency\s*=\s*{currency_column}\b|"
                rf"\b{currency_column}\s*=\s*(?:\w+\.)?source_currency\b)",
                normalized,
                flags,
            )
        )
        target_filter_found = bool(
            re.search(
                rf"\b(?:\w+\.)?target_currency\s*=\s*['\"]{target_literal}['\"]",
                normalized,
                flags,
            )
        )
        date_join_found = bool(
            re.search(
                r"DATE\s*\(\s*(?:\w+\.)?sync_time\s*\)\s*=\s*"
                r"DATE\s*\(\s*(?:\w+\.)?\w+(?:_time|_date|_at)\s*\)",
                normalized,
                flags,
            )
            or re.search(
                r"DATE\s*\(\s*(?:\w+\.)?\w+(?:_time|_date|_at)\s*\)\s*=\s*"
                r"DATE\s*\(\s*(?:\w+\.)?sync_time\s*\)",
                normalized,
                flags,
            )
        )
        target_passthrough_found = bool(
            re.search(
                rf"\bCASE\b.*?\bWHEN\s+{currency_column}\s*=\s*"
                rf"['\"]{target_literal}['\"]\s+THEN\s+{amount_column}\b",
                normalized,
                flags,
            )
        )
        rate_fallback_found = bool(
            re.search(
                r"\b(?:COALESCE|IFNULL)\s*\([^)]*\bmid\b",
                normalized,
                flags,
            )
        )
        missing_rate_indicator_found = bool(
            re.search(r"\b(?:\w+\.)?mid\s+IS\s+NULL\b", normalized, flags)
        )

        issues: list[str] = []
        if not rate_table_found:
            issues.append(f"必须引用 {cls._EXCHANGE_RATE_TABLE}")
        elif not rate_join_is_left:
            issues.append("汇率表必须使用 LEFT JOIN，避免目标币种原始记录被丢弃")
        if not mid_multiplication_found:
            issues.append("必须使用 原金额 * mid 计算目标币种金额")
        if not source_join_found:
            issues.append("必须用 source_currency 关联交易原币种")
        if not target_filter_found:
            issues.append(f"必须限定 target_currency = '{target_currency}'")
        if not date_join_found:
            issues.append("必须按 sync_time 与交易日期关联每日汇率")
        if not target_passthrough_found:
            issues.append(
                f"原币种为 {target_currency} 时必须直接使用原金额，汇率按 1 处理"
            )
        if rate_fallback_found:
            issues.append("非目标币种缺失汇率时禁止用 COALESCE/IFNULL 按1或0兜底")
        if not missing_rate_indicator_found:
            issues.append("必须显式返回非目标币种的缺失汇率笔数（mid IS NULL）")

        detail = {
            "required": True,
            "target_currency": target_currency,
            "exchange_rate_table": cls._EXCHANGE_RATE_TABLE,
            "rate_fallback_found": rate_fallback_found,
            "missing_rate_indicator_found": missing_rate_indicator_found,
            "ast_validation": "unavailable",
            "issues": issues,
        }
        if issues:
            return False, "；".join(issues), detail
        return True, "", detail

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

    @classmethod
    def _is_retryable_connection_error(cls, exc: BaseException) -> bool:
        if isinstance(exc, DBAPIError) and exc.connection_invalidated:
            return True
        original = exc.orig if isinstance(exc, DBAPIError) else exc
        if isinstance(original, ConnectionResetError):
            return True
        args = getattr(original, "args", ())
        return bool(args and args[0] in cls._TRANSIENT_DBAPI_CODES)

    @staticmethod
    def _format_database_error(exc: BaseException) -> str:
        error_msg = str(exc)
        inner = re.search(
            r"\(pymysql\.err\.\w+\)\s*\((\d+),\s*[\"'](.+?)[\"']\)",
            error_msg,
        )
        if inner:
            return f"Error {inner.group(1)}: {inner.group(2)}"
        return error_msg

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

        connection_retries = 0
        while True:
            try:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(f"EXPLAIN {sql}")).fetchall()

                plan_lines = [
                    str(row[0]) if len(row) == 1 else str(row) for row in rows
                ]
                plan_text = "\n".join(plan_lines)

                logger.info(f"EXPLAIN 通过, 执行计划 {len(plan_lines)} 行")
                return {
                    "valid": True,
                    "error": None,
                    "plan": plan_text,
                    "connection_retries": connection_retries,
                }
            except (DBAPIError, OSError) as exc:
                connection_error = self._is_retryable_connection_error(exc)
                if connection_error and connection_retries == 0:
                    connection_retries += 1
                    logger.warning(
                        "EXPLAIN connection reset; retrying original SQL",
                        extra={
                            "connection_retries": connection_retries,
                            "error_type": type(exc).__name__,
                        },
                    )
                    self.engine.dispose()
                    continue

                error_msg = self._format_database_error(exc)
                logger.warning(
                    "EXPLAIN failed",
                    extra={
                        "error": error_msg,
                        "error_type": type(exc).__name__,
                        "connection_retries": connection_retries,
                    },
                )
                return {
                    "valid": False,
                    "error": error_msg,
                    "plan": None,
                    "connection_retries": connection_retries,
                    "infrastructure_error": connection_error,
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
            if re.match(
                r"^\s*(SELECT|WITH)\b", exec_sql, re.IGNORECASE
            ) and not re.search(r"\bLIMIT\s+\d+", exec_sql, re.IGNORECASE):
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

        except (DBAPIError, OSError) as e:
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
    def inspect_plan(plan: str, max_scan_rows: int = 100_000_000) -> dict:
        """对 Doris EXPLAIN 文本做确定性安全检查。"""
        warnings: list[str] = []
        folded = plan.upper()
        if "CARTESIAN" in folded or "CROSS JOIN" in folded:
            warnings.append("执行计划包含笛卡尔积")

        estimates: list[int] = []
        for value in re.findall(
            r"(?:CARDINALITY|ROWS|ROW_COUNT)\s*[:=]\s*([0-9][0-9,]*)",
            plan,
            re.IGNORECASE,
        ):
            try:
                estimates.append(int(value.replace(",", "")))
            except ValueError:
                continue
        max_estimated_rows = max(estimates, default=0)
        if max_scan_rows > 0 and max_estimated_rows > max_scan_rows:
            warnings.append(
                f"预计扫描/中间结果行数 {max_estimated_rows} 超过限制 {max_scan_rows}"
            )
        return {
            "safe": not warnings,
            "warnings": warnings,
            "max_estimated_rows": max_estimated_rows,
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
        if any(kw in question for kw in time_keywords) and not re.search(
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
