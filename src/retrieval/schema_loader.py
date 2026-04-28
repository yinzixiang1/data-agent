"""
数据源加载 — 从 Doris DDL + 语义层 YAML 合并出完整 Schema。

在线模式: 连接 Doris 读取 DDL，再用语义层 YAML 补充业务信息（display_name, description 等）。
离线模式: 仅从语义层 YAML 构建 Schema（不依赖 Doris）。

使用示例::

    loader = SchemaLoader(offline=True)  # 离线模式
    schemas, glossary, enums = loader.load_all()
    # schemas: [{"table_name": "pmt_account", "columns": [...], ...}, ...]
    # glossary: {"活跃商户": {"definition": ..., "sql_hint": ...}, ...}
    # enums: [{"table_name": "pmt_account", "field_name": "account_type", "values": [...]}, ...]
"""

import logging
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text

from src.retrieval.config import (
    DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD, DORIS_DATABASE,
    SEMANTIC_LAYER_DIR,
)

logger = logging.getLogger(__name__)


class SchemaLoader:
    """
    加载 Doris Schema + 语义层 YAML，合并为完整 Schema dict 列表。

    Attributes:
        offline: 是否为离线模式（不连接 Doris）
        engine: SQLAlchemy Engine 实例（离线模式为 None）
        semantic_layer_dir: 语义层 YAML 根目录路径
    """

    def __init__(
        self,
        connection_string: str | None = None,
        semantic_layer_dir: str | Path | None = None,
        offline: bool = False,
    ):
        """
        Args:
            connection_string: SQLAlchemy 连接字符串，如
                "mysql+pymysql://root:@localhost:9030/dwd_banking?charset=utf8mb4"
                为 None 时自动从 config 中的 DORIS_* 参数拼接
            semantic_layer_dir: 语义层 YAML 根目录，默认使用 config.SEMANTIC_LAYER_DIR
            offline: 是否启用离线模式。True 时不创建数据库连接，仅从 YAML 读取
        """
        self.offline = offline
        self.engine = None
        if not offline:
            if connection_string is None:
                connection_string = (
                    f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD}"
                    f"@{DORIS_HOST}:{DORIS_PORT}/{DORIS_DATABASE}?charset=utf8mb4"
                )
            self.engine = create_engine(connection_string, pool_size=5, pool_recycle=3600)
        self.semantic_layer_dir = Path(semantic_layer_dir or SEMANTIC_LAYER_DIR)

    # ── Doris 元数据加载 ──

    def get_all_tables(self) -> list[str]:
        """
        获取 Doris 数据库中所有表名（执行 SHOW TABLES）。

        Returns:
            表名字符串列表，如 ["pmt_account", "pmt_transaction", ...]
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text("SHOW TABLES")).fetchall()
            return [row[0] for row in rows]

    def get_table_schema(self, table_name: str) -> dict:
        """
        通过 DESCRIBE + SHOW CREATE TABLE 获取单张表的完整 Schema。

        Args:
            table_name: 表名，如 "pmt_account"

        Returns:
            dict，包含:
                - "database": 数据库名
                - "table_name": 表名
                - "table_comment": 表注释（从 DDL 中提取）
                - "columns": list[dict]，每列包含 name, type, nullable, key, default, comment
        """
        with self.engine.connect() as conn:
            # 列信息
            col_rows = conn.execute(text(f"DESCRIBE `{table_name}`")).fetchall()
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

            # 尝试获取表注释
            table_comment = ""
            try:
                create_rows = conn.execute(
                    text(f"SHOW CREATE TABLE `{table_name}`")
                ).fetchone()
                if create_rows:
                    ddl_text = create_rows[1]
                    # 从 DDL 中提取 COMMENT
                    import re
                    match = re.search(r"COMMENT\s*[=']?\s*'([^']*)'", ddl_text)
                    if match:
                        table_comment = match.group(1)
            except Exception:
                pass

            return {
                "database": DORIS_DATABASE,
                "table_name": table_name,
                "table_comment": table_comment,
                "columns": columns,
            }

    def get_sample_rows(self, table_name: str, limit: int = 3) -> list[list]:
        """
        获取表的样例数据行（SELECT * LIMIT）。

        Args:
            table_name: 表名
            limit: 返回的最大行数，默认 3

        Returns:
            list[list]: 每行为一个列值列表，失败时返回空列表
        """
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(f"SELECT * FROM `{table_name}` LIMIT {limit}")).fetchall()
                return [list(row) for row in rows]
        except Exception as e:
            logger.warning(f"获取 {table_name} 样例数据失败: {e}")
            return []

    # ── 语义层 YAML 加载 ──

    def load_semantic_layer(self) -> tuple[dict[str, dict], dict]:
        """
        加载语义层 YAML 文件（表语义 + 业务术语）。

        扫描路径:
            - 表语义: semantic_layer/tables/**/*.yaml
            - 业务术语: semantic_layer/glossary/**/*.yaml

        Returns:
            tuple: (table_semantics, glossary)
                - table_semantics (dict): {table_name: yaml_data}，yaml_data 包含
                    table, columns, relations, common_queries 等顶层键
                - glossary (dict): {term: info}，info 包含
                    definition, sql_hint, related_tables, related_columns
        """
        table_semantics = {}
        glossary = {}

        # 加载表语义
        tables_dir = self.semantic_layer_dir / "tables"
        if tables_dir.exists():
            for f in tables_dir.glob("**/*.yaml"):
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if data and "table" in data:
                        table_name = data["table"]["name"]
                        table_semantics[table_name] = data
                except Exception as e:
                    logger.warning(f"加载语义层文件 {f} 失败: {e}")

        # 加载业务术语
        glossary_dir = self.semantic_layer_dir / "glossary"
        if glossary_dir.exists():
            for f in glossary_dir.glob("**/*.yaml"):
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if data and "glossary" in data:
                        for item in data["glossary"]:
                            glossary[item["term"]] = item
                except Exception as e:
                    logger.warning(f"加载术语文件 {f} 失败: {e}")

        logger.info(f"语义层加载完成: {len(table_semantics)} 张表, {len(glossary)} 条术语")
        return table_semantics, glossary

    def load_enums(self) -> list[dict]:
        """
        加载枚举字典。优先从 Doris 在线查询，失败时回退到本地 YAML。

        数据来源:
            - 在线模式: Doris warehouse_sys.sys_dim_enum_dict 表
            - 离线模式: semantic_layer/enums/*.yaml 文件

        Returns:
            list[dict]: 扁平化的枚举条目列表，每条包含:
                - "table_name" (str): 关联表名，如 "pmt_account"
                - "field_name" (str): 字段名，如 "account_type"
                - "field_label" (str): 字段中文名，如 "账户类型"
                - "values" (list[dict]): 枚举值列表，每个:
                    - "code" (str): 实际存储值，如 "1000"
                    - "label" (str): 英文标签
                    - "label_cn" (str): 中文标签，如 "普通账户"
        """
        if not self.offline and self.engine:
            try:
                enums = self._load_enums_from_doris()
                logger.info(f"枚举层从 Doris 加载完成: {len(enums)} 个字段组")
                return enums
            except Exception as e:
                logger.warning(f"从 Doris 加载枚举失败，回退到本地 YAML: {e}")

        enums = self._load_enums_from_yaml()
        logger.info(f"枚举层从本地 YAML 加载完成: {len(enums)} 个字段组")
        return enums

    def _load_enums_from_doris(self) -> list[dict]:
        """从 Doris warehouse_sys.sys_dim_enum_dict 查询枚举字典"""
        sql = text("""
            SELECT src_table_name, src_field_name, field_label,
                   src_field_value, src_value_label, status_desc
            FROM warehouse_sys.sys_dim_enum_dict
            ORDER BY src_table_name, src_field_name, sort_order
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(sql).fetchall()

        # 按 (table_name, field_name) 分组
        from collections import OrderedDict
        groups: dict[tuple, dict] = OrderedDict()
        for row in rows:
            key = (row[0], row[1])  # (src_table_name, src_field_name)
            if key not in groups:
                groups[key] = {
                    "table_name": row[0],
                    "field_name": row[1],
                    "field_label": row[2] or row[1],
                    "values": [],
                }
            groups[key]["values"].append({
                "code": str(row[3]),
                "label": row[4] or "",
                "label_cn": row[5] or "",
            })

        return list(groups.values())

    def _load_enums_from_yaml(self) -> list[dict]:
        """从本地 semantic_layer/enums/*.yaml 加载枚举"""
        enums = []
        enums_dir = self.semantic_layer_dir / "enums"
        if enums_dir.exists():
            for f in enums_dir.glob("*.yaml"):
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    if data and "enums" in data:
                        enums.extend(data["enums"])
                except Exception as e:
                    logger.warning(f"加载枚举文件 {f} 失败: {e}")
        return enums

    # ── 合并 ──

    def merge_schema(self, doris_schema: dict, semantic: dict | None) -> dict:
        """
        将 Doris DDL Schema 和语义层 YAML 合并，YAML 信息优先覆盖 DDL。

        Args:
            doris_schema: get_table_schema() 返回的 Doris 原始 Schema
            semantic: 语义层 YAML 解析结果（包含 table, columns, relations 等），
                为 None 时直接返回 doris_schema

        Returns:
            dict: 合并后的 Schema，新增 display_name, description, tags,
                query_tips, relations, common_queries 等字段；
                列级信息中 YAML 的 display_name, description, enum_values 等覆盖 DDL
        """
        if semantic is None:
            return doris_schema

        table_info = semantic.get("table", {})
        schema = doris_schema.copy()

        # 表级信息覆盖
        schema["display_name"] = table_info.get("display_name", "")
        schema["description"] = table_info.get("description", schema.get("table_comment", ""))
        schema["tags"] = table_info.get("tags", [])
        schema["query_tips"] = table_info.get("query_tips", "")

        # 关联关系
        schema["relations"] = semantic.get("relations", [])

        # 常见问题
        schema["common_queries"] = semantic.get("common_queries", [])

        # 列级信息合并：YAML 覆盖补充 DDL
        col_map = {}
        for col in semantic.get("columns", []):
            col_map[col["name"]] = col

        for col in schema["columns"]:
            if col["name"] in col_map:
                yaml_col = col_map[col["name"]]
                if "display_name" in yaml_col:
                    col["display_name"] = yaml_col["display_name"]
                if "description" in yaml_col:
                    col["description"] = yaml_col["description"]
                if "enum_values" in yaml_col:
                    col["enum_values"] = yaml_col["enum_values"]
                if "business_logic" in yaml_col:
                    col["business_logic"] = yaml_col["business_logic"]
                if yaml_col.get("sensitive"):
                    col["sensitive"] = True
                if yaml_col.get("skip_index"):
                    col["skip_index"] = True

        return schema

    # ── 统一入口 ──

    def _build_schema_from_yaml(self, semantic: dict) -> dict:
        """
        仅从语义层 YAML 构建 Schema（离线模式使用，不依赖 Doris）。

        Args:
            semantic: 单张表的语义层 YAML 解析结果，需包含 "table" 和 "columns" 键

        Returns:
            dict: 与 merge_schema 输出格式一致的 Schema dict
        """
        table_info = semantic.get("table", {})
        columns = []
        for col in semantic.get("columns", []):
            columns.append({
                "name": col["name"],
                "type": col.get("type", "VARCHAR(255)"),
                "nullable": col.get("nullable", True),
                "key": col.get("key", ""),
                "default": col.get("default"),
                "comment": col.get("description", col.get("display_name", "")),
                **{k: v for k, v in col.items() if k not in ("name", "type", "nullable", "key", "default", "description")},
            })

        return {
            "database": table_info.get("database", DORIS_DATABASE),
            "table_name": table_info["name"],
            "table_comment": table_info.get("display_name", ""),
            "display_name": table_info.get("display_name", ""),
            "description": table_info.get("description", ""),
            "tags": table_info.get("tags", []),
            "query_tips": table_info.get("query_tips", ""),
            "columns": columns,
            "relations": semantic.get("relations", []),
            "common_queries": semantic.get("common_queries", []),
        }

    def load_all(self) -> tuple[list[dict], dict, list[dict]]:
        """
        加载并合并所有 Schema。

        Returns:
            (schemas, glossary, enums)
            - schemas: 完整 Schema dict 列表
            - glossary: 业务术语表
            - enums: 独立枚举条目列表
        """
        # 语义层
        table_semantics, glossary = self.load_semantic_layer()
        enums = self.load_enums()

        if self.offline:
            # 离线模式：仅从语义层 YAML 构建
            schemas = []
            for table_name, semantic in table_semantics.items():
                schema = self._build_schema_from_yaml(semantic)
                schemas.append(schema)
            logger.info(f"离线模式: 从语义层加载 {len(schemas)} 张表")
            return schemas, glossary, enums

        # 在线模式：Doris DDL + 语义层合并
        table_names = self.get_all_tables()
        logger.info(f"从 Doris 加载到 {len(table_names)} 张表")

        schemas = []
        for table_name in table_names:
            doris_schema = self.get_table_schema(table_name)
            semantic = table_semantics.get(table_name)
            merged = self.merge_schema(doris_schema, semantic)
            schemas.append(merged)

        logger.info(f"Schema 合并完成: {len(schemas)} 张表, 其中 {len(table_semantics)} 张有语义层补充")
        return schemas, glossary, enums
