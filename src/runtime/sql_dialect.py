"""Execution-database dialect adapters for NL2SQL runtime behavior."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SQLDialectAdapter:
    """Database behavior that differs across supported SQL engines."""

    name: str
    display_name: str
    sqlglot_dialect: str
    identifier_quote: str
    prompt_rules: str

    def quote_identifier(self, value: str) -> str:
        quote = self.identifier_quote
        return f"{quote}{value.replace(quote, quote * 2)}{quote}"

    def qualify_table(self, namespace: str, table: str) -> str:
        return f"{self.quote_identifier(namespace)}.{self.quote_identifier(table)}"

    def prepare_execution(self, connection: Connection, timeout: int) -> None:
        """Apply a per-query timeout on the current connection when supported."""

    def execution_sql(self, sql: str, timeout: int) -> str:
        return sql

    def load_table_schema(self, engine: Engine, namespace: str, table: str) -> dict:
        raise NotImplementedError


class DorisDialectAdapter(SQLDialectAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="doris",
            display_name="Apache Doris SQL",
            sqlglot_dialect="mysql",
            identifier_quote="`",
            prompt_rules=(
                "【SQL 方言：Apache Doris】\n"
                "- 表名使用 `database`.`table` 两段式限定名。\n"
                "- DATE_TRUNC(datetime, 'unit')；DATE_ADD/DATE_SUB 使用 INTERVAL。\n"
                "- 按月文本可使用 DATE_FORMAT(datetime, '%Y-%m')。"
            ),
        )

    def execution_sql(self, sql: str, timeout: int) -> str:
        return f"/*+ SET_VAR(query_timeout={timeout}) */ {sql}"

    def load_table_schema(self, engine: Engine, namespace: str, table: str) -> dict:
        qualified = self.qualify_table(namespace, table)
        with engine.connect() as connection:
            rows = connection.execute(text(f"DESCRIBE {qualified}")).fetchall()
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": str(row[2]).upper() == "YES",
                    "key": row[3] or "",
                    "default": row[4],
                    "comment": row[5] if len(row) > 5 else "",
                }
                for row in rows
            ]
            table_comment = ""
            try:
                create_row = connection.execute(
                    text(f"SHOW CREATE TABLE {qualified}")
                ).fetchone()
                if create_row:
                    match = re.search(r"COMMENT\s*[=']?\s*'([^']*)'", create_row[1])
                    if match:
                        table_comment = match.group(1)
            except SQLAlchemyError as exc:
                logger.debug(
                    "table comment lookup skipped",
                    extra={"table": table, "error_type": type(exc).__name__},
                )
        return _schema_payload(namespace, table, table_comment, columns)


class RedshiftDialectAdapter(SQLDialectAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="redshift",
            display_name="Amazon Redshift SQL",
            sqlglot_dialect="redshift",
            identifier_quote='"',
            prompt_rules=(
                "【SQL 方言：Amazon Redshift】\n"
                '- 表名使用 "schema"."table" 两段式限定名，database 已由连接确定。\n'
                "- DATE_TRUNC('unit', timestamp)；日期加减优先使用 DATEADD(unit, n, value)。\n"
                "- 不得使用 Doris/MySQL 的反引号、DATE_FORMAT、CURDATE() 或 SET_VAR Hint。"
            ),
        )

    def prepare_execution(self, connection: Connection, timeout: int) -> None:
        connection.execute(text(f"SET statement_timeout TO {max(1, timeout) * 1000}"))

    def load_table_schema(self, engine: Engine, namespace: str, table: str) -> dict:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = :schema AND table_name = :table "
                    "ORDER BY ordinal_position"
                ),
                {"schema": namespace, "table": table},
            ).fetchall()
        if not rows:
            raise ValueError(f"Redshift 表不存在: {namespace}.{table}")
        columns = [
            {
                "name": row[0],
                "type": row[1],
                "nullable": str(row[2]).upper() == "YES",
                "key": "",
                "default": row[3],
                "comment": "",
            }
            for row in rows
        ]
        return _schema_payload(namespace, table, "", columns)


def _schema_payload(
    namespace: str, table: str, table_comment: str, columns: list[dict]
) -> dict:
    return {
        "database": namespace,
        "table_name": f"{namespace}.{table}",
        "table_name_short": table,
        "table_comment": table_comment,
        "columns": columns,
    }


_ADAPTERS = {
    "doris": DorisDialectAdapter(),
    "redshift": RedshiftDialectAdapter(),
}


def get_sql_dialect(db_type: str) -> SQLDialectAdapter:
    try:
        return _ADAPTERS[db_type.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"不支持的执行数据库类型: {db_type}") from exc
