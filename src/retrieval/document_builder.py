"""文档构建 — Schema dict → 表级/列级/枚举级检索文档"""

import logging

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """将 Schema dict 转换为可被 BGE-M3 编码的文本文档"""

    def build_table_document(self, schema: dict) -> dict:
        """
        构建表级检索文档。

        Returns:
            {
                "table_name": str,
                "doc_type": "table",
                "text": str,         # 用于向量化的文本
                "schema": dict,      # 原始 Schema（检索命中后取完整信息）
            }
        """
        parts = [
            f"表名: {schema['table_name']}",
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
            if col.get("skip_index") or col.get("sensitive"):
                continue
            display = col.get("display_name") or col.get("comment", "")
            if display:
                col_summaries.append(f"{col['name']}({display})")
            else:
                col_summaries.append(col["name"])
        if col_summaries:
            parts.append(f"主要字段: {', '.join(col_summaries)}")

        # 常见问题（来自语义层 common_queries）
        for q in schema.get("common_queries", [])[:5]:
            parts.append(f"常见问题: {q['question']}")

        # 关联表
        for rel in schema.get("relations", []):
            target = rel.get("target_table", "")
            col = rel.get("column", "")
            target_col = rel.get("target_column", "")
            if target:
                parts.append(f"关联: {target} ON {schema['table_name']}.{col} = {target}.{target_col}")

        # 查询提示
        if schema.get("query_tips"):
            parts.append(f"查询注意: {schema['query_tips']}")

        text = "\n".join(parts)

        return {
            "table_name": schema["table_name"],
            "doc_type": "table",
            "text": text,
            "schema": schema,
        }

    def build_column_documents(self, schema: dict) -> list[dict]:
        """
        构建列级检索文档。

        只对有业务含义的列建索引，跳过技术字段、敏感字段、JSON 配置字段。
        """
        table_name = schema["table_name"]
        display_name = schema.get("display_name", "")
        docs = []

        for col in schema.get("columns", []):
            # 跳过规则
            if col.get("skip_index"):
                continue
            if col.get("sensitive"):
                continue
            # 无 comment 且无 display_name 的纯技术字段跳过
            if not col.get("comment") and not col.get("display_name") and not col.get("description"):
                continue
            # JSON/STRING 无注释的跳过
            if col.get("type", "").upper() == "STRING" and not col.get("comment") and not col.get("display_name"):
                continue

            parts = [
                f"表: {table_name}({display_name})" if display_name else f"表: {table_name}",
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

            docs.append({
                "table_name": table_name,
                "column_name": col["name"],
                "column_cn_name": col.get("display_name") or col.get("comment", ""),
                "column_type": col.get("type", ""),
                "column_comment": desc,
                "is_enum": bool(col.get("enum_values")),
                "enum_values_summary": self._format_enum_values(col["enum_values"]) if col.get("enum_values") else "",
                "doc_type": "column",
                "text": "\n".join(parts),
            })

        return docs

    def build_enum_documents(self, enums: list[dict]) -> list[dict]:
        """
        从独立枚举层构建枚举值检索文档。

        每个枚举值独立成一条文档，用于将"持牌商户"映射到 account_type=2000。

        Args:
            enums: 枚举条目列表，每条包含 table_name, field_name, field_label,
                   values: [{code, label, label_cn}, ...]
        """
        docs = []

        for entry in enums:
            table_name = entry["table_name"]
            col_name = entry["field_name"]
            col_cn = entry.get("field_label", "")

            for v in entry.get("values", []):
                code = str(v.get("code", ""))
                label = v.get("label", "")
                label_cn = v.get("label_cn", "")

                # 向量化文本：中文标签优先，无则用英文
                display_label = label_cn if label_cn and label_cn != label else label
                parts = [f"枚举值: {display_label}"]
                col_display = f"{table_name}.{col_name}" + (f"({col_cn})" if col_cn else "")
                parts.append(f"对应字段: {col_display}")
                parts.append(f"实际取值: {code}")
                if label and label != display_label:
                    parts.append(f"英文标签: {label}")

                docs.append({
                    "table_name": table_name,
                    "column_name": col_name,
                    "enum_code": code,
                    "enum_label_cn": label_cn or label,
                    "description": "",
                    "synonyms": "",
                    "sql_value": code,
                    "text": "\n".join(parts),
                })

        return docs

    def build_all(
        self,
        schemas: list[dict],
        enums: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        批量构建所有表的文档。

        Args:
            schemas: 表 Schema 列表
            enums: 独立枚举条目列表（来自 enums/*.yaml）

        Returns:
            (table_docs, column_docs, enum_docs)
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

    def _format_enum_values(self, enum_values) -> str:
        """格式化枚举值"""
        if isinstance(enum_values, list):
            parts = []
            for v in enum_values:
                if isinstance(v, dict):
                    parts.append(f"{v.get('value', '')}={v.get('label', '')}")
                else:
                    parts.append(str(v))
            return ", ".join(parts)
        return str(enum_values)
