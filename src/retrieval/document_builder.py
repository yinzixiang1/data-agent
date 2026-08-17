"""
文档构建 — 将 Schema dict 转换为表级/列级/枚举级/值级检索文档。

每类文档包含一个 "text" 字段，用于向量化编码和 BM25 索引。
文本内容经过精心设计，包含表名、中文名、描述、关键列等信息，以提升检索召回率。

使用示例::

    builder = DocumentBuilder()

    # 单表文档
    table_doc = builder.build_table_document(schema)
    print(table_doc["text"])  # "表名: pmt_account\n中文名: 商户账户表\n..."

    # 批量构建
    table_docs, column_docs, enum_docs = builder.build_all(schemas, enums)
"""

import logging

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """将 Schema dict 转换为可被 BGE-M3 编码的文本文档。"""

    def build_table_document(self, schema: dict) -> dict:
        """
        构建单张表的表级检索文档。

        文本包含: 表名、中文名、描述、业务标签、关键列摘要、常见问题、关联表、查询提示。

        Args:
            schema: 合并后的表 Schema dict，需包含 table_name, display_name,
                description, tags, columns, relations 等字段

        Returns:
            dict，包含:
                - "table_name" (str): 表名
                - "doc_type" (str): 固定为 "table"
                - "text" (str): 用于向量化的拼接文本
                - "schema" (dict): 原始 Schema 引用（检索命中后取完整信息）
        """
        table_name = schema["table_name"]  # db.table 全限定名
        # SQL 中使用反引号格式
        db = schema.get("database", "")
        short = schema.get("table_name_short", table_name)
        sql_name = f"`{db}`.`{short}`" if db else f"`{short}`"

        parts = [
            f"表名: {sql_name}",
        ]

        if schema.get("display_name"):
            parts.append(f"中文名: {schema['display_name']}")

        if schema.get("description"):
            parts.append(f"描述: {schema['description']}")

        if schema.get("tags"):
            parts.append(f"业务标签: {', '.join(schema['tags'])}")

        # 关键列摘要：只挑有业务含义的列
        col_summaries = []
        for col in schema.get("columns", []):
            if col.get("is_skip_index") or col.get("is_sensitive"):
                continue
            display = col.get("display_name") or col.get("comment", "")
            if display:
                col_summaries.append(f"{col['name']}({display})")
            else:
                col_summaries.append(col["name"])
        if col_summaries:
            parts.append(f"主要字段: {', '.join(col_summaries)}")

        # 关联表
        for rel in schema.get("relations", []):
            target = rel.get("target_table", "")
            col = rel.get("column", "")
            target_col = rel.get("target_column", "")
            if target:
                parts.append(f"关联: {target} ON {short}.{col} = {target}.{target_col}")

        # 查询提示
        if schema.get("query_tips"):
            parts.append(f"查询注意: {schema['query_tips']}")

        text = "\n".join(parts)

        return {
            "table_name": table_name,
            "doc_type": "table",
            "text": text,
            "schema": schema,
        }

    def build_column_documents(self, schema: dict) -> list[dict]:
        """
        构建列级检索文档（一列一条文档）。

        跳过规则: is_skip_index=True 的列、is_sensitive=True 的列、无 comment 且无 display_name 的纯技术列。

        Args:
            schema: 合并后的表 Schema dict

        Returns:
            list[dict]: 每条文档包含:
                - "table_name" (str): 所属表名
                - "column_name" (str): 列名
                - "column_cn_name" (str): 列中文名
                - "column_type" (str): 列类型，如 "INT"、"VARCHAR(128)"
                - "column_comment" (str): 列描述
                - "is_enum" (bool): 是否为枚举列
                - "enum_values_summary" (str): 枚举值摘要
                - "doc_type" (str): 固定为 "column"
                - "text" (str): 用于向量化的拼接文本
        """
        table_name = schema["table_name"]  # db.table 全限定名
        short_name = schema.get("table_name_short", table_name)
        display_name = schema.get("display_name", "")
        docs = []

        for col in schema.get("columns", []):
            # 跳过规则
            if col.get("is_skip_index"):
                continue
            if col.get("is_sensitive"):
                continue
            # 无 comment 且无 display_name 的纯技术字段跳过
            if (
                not col.get("comment")
                and not col.get("display_name")
                and not col.get("description")
            ):
                continue
            # JSON/STRING 无注释的跳过
            if (
                col.get("type", "").upper() == "STRING"
                and not col.get("comment")
                and not col.get("display_name")
            ):
                continue

            parts = [
                f"表: {short_name}({display_name})"
                if display_name
                else f"表: {short_name}",
                f"列名: {col['name']}",
            ]

            if col.get("display_name"):
                parts.append(f"中文名: {col['display_name']}")

            if col.get("type"):
                parts.append(f"类型: {col['type']}")

            desc = col.get("description") or col.get("comment", "")
            if desc:
                parts.append(f"描述: {desc}")

            if col.get("enum_values"):
                enum_str = self._format_enum_values(col["enum_values"])
                parts.append(f"可选值: {enum_str}")

            if col.get("business_logic"):
                parts.append(f"业务逻辑: {col['business_logic']}")

            if col.get("key") and col["key"] != "":
                parts.append(f"索引: {col['key']}")

            docs.append(
                {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "column_cn_name": col.get("display_name") or col.get("comment", ""),
                    "column_type": col.get("type", ""),
                    "column_comment": desc,
                    "is_enum": bool(col.get("enum_values")),
                    "enum_values_summary": self._format_enum_values(col["enum_values"])
                    if col.get("enum_values")
                    else "",
                    "doc_type": "column",
                    "text": "\n".join(parts),
                }
            )

        return docs

    def build_enum_documents(self, enums: list[dict]) -> list[dict]:
        """
        从独立枚举层构建枚举值检索文档（每个枚举值独立成一条文档）。

        用途: 将用户自然语言（如 "持牌商户"）映射到 SQL 条件（如 account_type=2000）。

        Args:
            enums: SchemaLoader.load_enums() 返回的枚举条目列表，每条包含:
                - "table_name" (str): 关联表名
                - "field_name" (str): 字段名
                - "field_label" (str): 字段中文名
                - "values" (list[dict]): 枚举值列表，每个含 code, label, label_cn

        Returns:
            list[dict]: 每条文档包含:
                - "table_name", "column_name", "enum_code", "enum_label_cn",
                  "sql_value", "text" 等字段
        """
        docs = []

        for entry in enums:
            table_name = entry["table_name"]
            col_name = entry["field_name"]
            col_cn = entry.get("field_label", "")
            biz_line = entry.get("biz_line", "")

            for v in entry.get("values", []):
                code = str(v.get("code", ""))
                label = v.get("label", "")
                label_cn = v.get("label_cn", "")

                # 向量化文本：中文标签优先，无则用英文
                display_label = label_cn if label_cn and label_cn != label else label
                parts = [f"枚举值: {display_label}"]
                col_display = f"{table_name}.{col_name}" + (
                    f"({col_cn})" if col_cn else ""
                )
                parts.append(f"对应字段: {col_display}")
                parts.append(f"实际取值: {code}")
                if label and label != display_label:
                    parts.append(f"英文标签: {label}")

                docs.append(
                    {
                        "table_name": table_name,
                        "column_name": col_name,
                        "biz_line": biz_line,
                        "metadata": entry.get("metadata", {}),
                        "enum_code": code,
                        "enum_label_cn": label_cn or label,
                        "description": "",
                        "synonyms": "",
                        "sql_value": code,
                        "text": "\n".join(parts),
                    }
                )

        return docs

    def build_all(
        self,
        schemas: list[dict],
        enums: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        批量构建所有表的三级检索文档。

        Args:
            schemas: SchemaLoader.load_all() 返回的表 Schema 列表
            enums: SchemaLoader.load_enums() 返回的独立枚举条目列表，None 视为空

        Returns:
            tuple: (table_docs, column_docs, enum_docs)
                - table_docs (list[dict]): 表级文档，每张表一条
                - column_docs (list[dict]): 列级文档，每个有效列一条
                - enum_docs (list[dict]): 枚举值文档，每个枚举值一条
        """
        table_docs = []
        column_docs = []

        for schema in schemas:
            table_docs.append(self.build_table_document(schema))
            column_docs.extend(self.build_column_documents(schema))

        enum_docs = self.build_enum_documents(enums or [])

        logger.info(
            f"文档构建完成: {len(table_docs)} 个表级, "
            f"{len(column_docs)} 个列级, {len(enum_docs)} 个枚举值"
        )
        return table_docs, column_docs, enum_docs

    def build_value_documents(self, enums: list[dict]) -> list[dict]:
        """
        构建值级检索文档 (nl2sql_value, BM25-only Schema Linking)。

        每个枚举值独立一条文档，text 字段仅包含标签名（中英文 + 同义词），
        用于 BM25 精确匹配用户 query 中出现的实体值。

        Args:
            enums: SchemaLoader.load_enums() 返回的枚举条目列表

        Returns:
            list[dict]: 每条文档包含 table_name, column_name, enum_code,
                enum_label_cn, sql_value, text
        """
        docs = []
        for entry in enums:
            table_name = entry["table_name"]
            col_name = entry["field_name"]
            biz_line = entry.get("biz_line", "")
            for v in entry.get("values", []):
                code = str(v.get("code", ""))
                label = v.get("label", "")
                label_cn = v.get("label_cn", "")

                text_parts = []
                if label_cn:
                    text_parts.append(label_cn)
                if label and label != label_cn:
                    text_parts.append(label)
                synonyms = v.get("synonyms", [])
                if isinstance(synonyms, list):
                    text_parts.extend(synonyms)
                if not text_parts:
                    continue

                docs.append(
                    {
                        "table_name": table_name,
                        "column_name": col_name,
                        "biz_line": biz_line,
                        "metadata": entry.get("metadata", {}),
                        "enum_code": code,
                        "enum_label_cn": label_cn or label,
                        "sql_value": code,
                        "text": " ".join(text_parts),
                    }
                )

        logger.info(f"值级文档构建: {len(docs)} 条 (BM25-only)")
        return docs

    def _format_enum_values(self, enum_values) -> str:
        """
        将枚举值格式化为 "value=label, ..." 字符串。

        Args:
            enum_values: 枚举值定义，可以是:
                - list[dict]: [{"value": 1000, "label": "普通账户"}, ...]
                - list[str]: ["LOW", "MEDIUM", "HIGH"]
                - 其他: 直接 str() 转换
        """
        if isinstance(enum_values, list):
            parts = []
            for v in enum_values:
                if isinstance(v, dict):
                    parts.append(f"{v.get('value', '')}={v.get('label', '')}")
                else:
                    parts.append(str(v))
            return ", ".join(parts)
        return str(enum_values)
