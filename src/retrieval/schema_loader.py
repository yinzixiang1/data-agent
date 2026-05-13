"""
数据源加载 — Doris DDL + MySQL 语义层合并出完整 Schema。

连接 Doris 读取 DDL，再从 MySQL（data_agent 库）加载语义层数据，
合并为完整 Schema。MySQL 表结构对接 dataAgent-admin-api 定义的 da_* 系列表。

使用示例::

    loader = SchemaLoader()
    schemas, glossary, enums, fewshot = loader.load_all()
"""

import json
import logging
from collections import OrderedDict

from sqlalchemy import create_engine, text

from src.retrieval.config import (
    DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD, DORIS_PASSWORD_URL,
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_PASSWORD_URL, MYSQL_DATABASE,
)

logger = logging.getLogger(__name__)


class SchemaLoader:
    """
    加载 Doris DDL + MySQL 语义层，合并为完整 Schema。

    MySQL 表结构（dataAgent-admin-api 定义）:
        da_table          — 表语义
        da_table_column   — 列语义（FK → da_table.id）
        da_table_relation — 关联关系（FK → da_table.id）
        da_table_query    — 表级常见问题（FK → da_table.id）
        da_glossary       — 业务术语
        da_enum_def       — 枚举定义
        da_enum_value     — 枚举值（FK → da_enum_def.id）
        da_fewshot        — 全局 Few-shot 示例
    """

    def __init__(
        self,
        connection_string: str | None = None,
        mysql_connection_string: str | None = None,
    ):
        if connection_string is None:
            connection_string = (
                f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD_URL}"
                f"@{DORIS_HOST}:{DORIS_PORT}/information_schema?charset=utf8mb4"
            )
        self.doris_engine = create_engine(connection_string, pool_size=5, pool_recycle=3600)

        if mysql_connection_string is None:
            mysql_connection_string = (
                f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
                f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
            )
        self.mysql_engine = create_engine(mysql_connection_string, pool_size=5, pool_recycle=3600)

    # ── Doris 元数据加载 ──

    def get_all_tables(self) -> list[str]:
        """获取 Doris 数据库中所有表名。"""
        with self.doris_engine.connect() as conn:
            rows = conn.execute(text("SHOW TABLES")).fetchall()
            return [row[0] for row in rows]

    def get_table_schema(self, table_name: str, database: str) -> dict:
        """通过 DESCRIBE + SHOW CREATE TABLE 获取单张表的 Schema。"""
        db = database
        with self.doris_engine.connect() as conn:
            col_rows = conn.execute(text(f"DESCRIBE `{db}`.`{table_name}`")).fetchall()
            columns = []
            for row in col_rows:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": str(row[2]).upper() == "YES",
                    "key": row[3] or "",
                    "default": row[4],
                    "comment": row[5] if len(row) > 5 else "",
                })

            table_comment = ""
            try:
                create_rows = conn.execute(
                    text(f"SHOW CREATE TABLE `{db}`.`{table_name}`")
                ).fetchone()
                if create_rows:
                    import re
                    match = re.search(r"COMMENT\s*[=']?\s*'([^']*)'", create_rows[1])
                    if match:
                        table_comment = match.group(1)
            except Exception:
                pass

            return {
                "database": db,
                "table_name": f"{db}.{table_name}",
                "table_name_short": table_name,
                "table_comment": table_comment,
                "columns": columns,
            }

    # ── MySQL 语义层加载（对接 dataAgent-admin-api 的 da_* 表） ──

    def _load_table_semantics(self) -> dict[str, dict]:
        """从 da_table 加载表语义 → {db.table: {display_name, description, tags, query_tips}}"""
        with self.mysql_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT t.name, b.database_name, t.display_name, t.description, t.tags, t.query_tips, b.biz_line "
                "FROM da_table t "
                "JOIN da_biz_database b ON t.biz_database_id = b.id "
                "WHERE t.status = 1 AND t.available = 1"
            )).fetchall()

        result = {}
        for row in rows:
            db_name = row[1]
            full_key = f"{db_name}.{row[0]}"
            tags = []
            if row[4]:
                try:
                    tags = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    tags = [t.strip() for t in row[4].split(",") if t.strip()]
            result[full_key] = {
                "database_name": db_name,
                "table_name_short": row[0],
                "display_name": row[2] or "",
                "description": row[3] or "",
                "tags": tags,
                "query_tips": row[5] or "",
                "biz_line": row[6] or "",
            }

        logger.info(f"MySQL 表语义加载: {len(result)} 张表")
        return result

    def _load_column_semantics(self) -> dict[str, list[dict]]:
        """从 da_table_column JOIN da_table 加载列语义 → {db.table: [col_dict, ...]}"""
        with self.mysql_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT t.name, c.name, c.display_name, c.description, "
                "c.enum_values, c.business_logic, c.is_sensitive, c.is_skip_index, "
                "b.database_name "
                "FROM da_table_column c "
                "JOIN da_table t ON c.table_id = t.id "
                "JOIN da_biz_database b ON t.biz_database_id = b.id "
                "WHERE t.status = 1 AND t.available = 1 "
                "ORDER BY t.name, c.sort_order"
            )).fetchall()

        result: dict[str, list[dict]] = {}
        for row in rows:
            db_name = row[8]
            full_key = f"{db_name}.{row[0]}"
            enum_values = None
            if row[4]:
                try:
                    parsed = json.loads(row[4])
                    if parsed and parsed != []:
                        enum_values = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

            col = {
                "name": row[1],
                "display_name": row[2] or "",
                "description": row[3] or "",
                "enum_values": enum_values,
                "business_logic": row[5] or "",
                "is_sensitive": bool(row[6]),
                "is_skip_index": bool(row[7]),
            }
            result.setdefault(full_key, []).append(col)

        logger.info(f"MySQL 列语义加载: {sum(len(v) for v in result.values())} 列")
        return result

    def _load_relations(self) -> dict[str, list[dict]]:
        """从 da_table_relation JOIN da_table 加载关联关系 → {db.table: [relation_dict, ...]}"""
        with self.mysql_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT t.name, r.column_name, r.target_table, r.target_column, r.join_type, "
                "b.database_name "
                "FROM da_table_relation r "
                "JOIN da_table t ON r.table_id = t.id "
                "JOIN da_biz_database b ON t.biz_database_id = b.id "
                "WHERE t.status = 1 AND t.available = 1"
            )).fetchall()

        result: dict[str, list[dict]] = {}
        for row in rows:
            db_name = row[5]
            full_key = f"{db_name}.{row[0]}"
            rel = {
                "column": row[1],
                "target_table": row[2],
                "target_column": row[3],
                "join_type": row[4] or "LEFT JOIN",
            }
            result.setdefault(full_key, []).append(rel)

        logger.info(f"MySQL 关联关系加载: {sum(len(v) for v in result.values())} 条")
        return result

    def load_glossary(self) -> dict[str, dict]:
        """从 da_glossary 加载业务术语 → {term: info_dict}"""
        with self.mysql_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT term, definition, sql_hint, related_tables, related_columns, synonyms "
                "FROM da_glossary WHERE status = 1"
            )).fetchall()

        glossary = {}
        for row in rows:
            related_tables = []
            if row[3]:
                try:
                    related_tables = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    related_tables = [t.strip() for t in row[3].split(",") if t.strip()]

            related_columns = []
            if row[4]:
                try:
                    related_columns = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    related_columns = [c.strip() for c in row[4].split(",") if c.strip()]

            synonyms = []
            if row[5]:
                try:
                    synonyms = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    synonyms = [s.strip() for s in row[5].split(",") if s.strip()]

            glossary[row[0]] = {
                "term": row[0],
                "definition": row[1] or "",
                "sql_hint": row[2] or "",
                "related_tables": related_tables,
                "related_columns": related_columns,
                "synonyms": synonyms,
            }

        logger.info(f"MySQL 业务术语加载: {len(glossary)} 条")
        return glossary

    def load_enums(self) -> list[dict]:
        """从 da_enum_def JOIN da_enum_value 加载枚举字典 → [{table_name, field_name, field_label, values: [...]}]"""
        with self.mysql_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT d.table_name, d.field_name, d.field_label, "
                "v.code, v.label, v.label_cn, d.biz_line "
                "FROM da_enum_value v "
                "JOIN da_enum_def d ON v.definition_id = d.id "
                "ORDER BY d.table_name, d.field_name, v.sort_order"
            )).fetchall()

        groups: dict[tuple, dict] = OrderedDict()
        for row in rows:
            key = (row[0], row[1])
            if key not in groups:
                groups[key] = {
                    "table_name": row[0],
                    "field_name": row[1],
                    "field_label": row[2] or row[1],
                    "biz_line": row[6] or "",
                    "values": [],
                }
            groups[key]["values"].append({
                "code": row[3],
                "label": row[4] or "",
                "label_cn": row[5] or "",
            })

        result = list(groups.values())
        logger.info(f"MySQL 枚举字典加载: {len(result)} 个字段, {sum(len(g['values']) for g in result)} 个值")
        return result

    def load_fewshot(self) -> list[dict]:
        """从 da_fewshot + da_table_query 加载 Few-shot 示例。"""
        examples = []

        with self.mysql_engine.connect() as conn:
            # 全局 Few-shot
            rows = conn.execute(text(
                "SELECT id, question, `sql`, tables, difficulty "
                "FROM da_fewshot WHERE status = 1"
            )).fetchall()

            for row in rows:
                tables = self._parse_json_list(row[3])
                examples.append({
                    "id": row[0],
                    "question": row[1],
                    "sql": row[2],
                    "tables": tables,
                    "difficulty": row[4] or "",
                })

            # 表级常见问题（也作为 Few-shot）
            query_rows = conn.execute(text(
                "SELECT q.question, q.`sql`, q.tables, q.difficulty "
                "FROM da_table_query q "
                "JOIN da_table t ON q.table_id = t.id "
                "WHERE t.status = 1 AND t.available = 1"
            )).fetchall()

            for row in query_rows:
                tables = self._parse_json_list(row[2])
                examples.append({
                    "question": row[0],
                    "sql": row[1],
                    "tables": tables,
                    "difficulty": row[3] or "",
                })

        logger.info(f"MySQL Few-shot 加载: {len(examples)} 条 (全局 {len(rows)} + 表级 {len(query_rows)})")
        return examples

    # ── 合并 ──

    def merge_schema(
        self,
        doris_schema: dict,
        table_sem: dict | None,
        col_sems: list[dict] | None,
        relations: list[dict] | None,
    ) -> dict:
        """将 Doris DDL + MySQL 语义层合并为完整 Schema。"""
        schema = doris_schema.copy()

        # 表级信息
        if table_sem:
            schema["display_name"] = table_sem.get("display_name", "")
            schema["description"] = table_sem.get("description", schema.get("table_comment", ""))
            schema["tags"] = table_sem.get("tags", [])
            schema["query_tips"] = table_sem.get("query_tips", "")
            schema["biz_line"] = table_sem.get("biz_line", "")

        # 关联关系
        schema["relations"] = relations or []

        # 列级信息合并
        if col_sems:
            col_map = {col["name"]: col for col in col_sems}
            for col in schema["columns"]:
                if col["name"] in col_map:
                    sem_col = col_map[col["name"]]
                    if sem_col.get("display_name"):
                        col["display_name"] = sem_col["display_name"]
                    if sem_col.get("description"):
                        col["description"] = sem_col["description"]
                    if sem_col.get("enum_values"):
                        col["enum_values"] = sem_col["enum_values"]
                    if sem_col.get("business_logic"):
                        col["business_logic"] = sem_col["business_logic"]
                    if sem_col.get("is_sensitive"):
                        col["is_sensitive"] = True
                    if sem_col.get("is_skip_index"):
                        col["is_skip_index"] = True

        return schema

    # ── 统一入口 ──

    def load_all(self) -> tuple[list[dict], dict, list[dict], list[dict]]:
        """
        加载并合并所有 Schema。

        Returns:
            tuple: (schemas, glossary, enums, fewshot)
        """
        # 从 MySQL 加载语义层（只加载 status=1 AND available=1）
        table_semantics = self._load_table_semantics()
        column_semantics = self._load_column_semantics()
        relations = self._load_relations()
        glossary = self.load_glossary()
        enums = self.load_enums()
        fewshot = self.load_fewshot()

        logger.info(f"MySQL 可用表: {len(table_semantics)} 张 (status=1, available=1)")

        # 以 MySQL 为基准，逐表从 Doris 获取 DDL 补充列类型
        schemas = []
        skipped = []
        for full_key, sem in table_semantics.items():
            short_name = sem["table_name_short"]
            try:
                doris_schema = self.get_table_schema(short_name, database=sem.get("database_name"))
            except Exception as e:
                logger.warning(f"Doris DDL 获取失败，跳过: {full_key} ({e})")
                skipped.append(full_key)
                continue

            merged = self.merge_schema(
                doris_schema,
                table_sem=sem,
                col_sems=column_semantics.get(full_key),
                relations=relations.get(full_key),
            )
            schemas.append(merged)

        if skipped:
            logger.warning(f"跳过 {len(skipped)} 张表 (Doris 不可达): {skipped}")
        logger.info(f"Schema 合并完成: {len(schemas)} 张表进入索引")
        return schemas, glossary, enums, fewshot

    # ── 工具方法 ──

    @staticmethod
    def _parse_json_list(value: str | None) -> list:
        """解析 JSON 数组字符串，兼容逗号分隔格式。"""
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return [t.strip() for t in value.split(",") if t.strip()]
