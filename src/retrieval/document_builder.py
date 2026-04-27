"""文档构建 — Schema dict → 表级/列级检索文档"""

import logging

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """将 Schema dict 转换为可被 BGE-M3 编码的文本文档"""

    def build_table_document(self, schema: dict) -> dict:
        """
        构建表级检索文档。

        文档内容直接决定"什么样的用户提问能匹配到这张表"。

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

        Returns:
            [{"table_name", "column_name", "doc_type": "column", "text"}, ...]
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
            if col["type"].upper() == "STRING" and not col.get("comment") and not col.get("display_name"):
                continue

            parts = [
                f"表: {table_name}({display_name})" if display_name else f"表: {table_name}",
                f"列名: {col['name']}",
            ]

            if col.get("display_name"):
                parts.append(f"中文名: {col['display_name']}")

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
                "doc_type": "column",
                "text": "\n".join(parts),
            })

        return docs

    def build_all(self, schemas: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        批量构建所有表的文档。

        Returns:
            (table_docs, column_docs)
        """
        table_docs = []
        column_docs = []

        for schema in schemas:
            table_docs.append(self.build_table_document(schema))
            column_docs.extend(self.build_column_documents(schema))

        logger.info(f"文档构建完成: {len(table_docs)} 个表级文档, {len(column_docs)} 个列级文档")
        return table_docs, column_docs

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
