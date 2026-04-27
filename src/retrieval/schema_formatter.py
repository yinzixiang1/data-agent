"""Prompt 格式化 — 将检索结果格式化为 DDL Prompt 文本"""


class SchemaFormatter:
    """
    将检索到的 Schema 格式化为 SQL 生成 Prompt。

    使用 CREATE TABLE DDL 格式（LLM 训练数据中大量出现，准确率最高）。
    """

    def format_tables(self, tables: list[dict]) -> str:
        """
        将检索命中的表格式化为 DDL 文本。

        Args:
            tables: hybrid_searcher 返回的表列表，每个需有 "schema" 字段
        """
        parts = ["## 可用数据表\n"]

        for t in tables:
            schema = t.get("schema", {})
            if not schema:
                continue
            parts.append(self._format_single_table(schema))

        return "\n".join(parts)

    def format_fewshot(self, examples: list[dict]) -> str:
        """将 Few-shot 示例格式化为 Prompt 文本"""
        if not examples:
            return ""

        parts = ["## 参考示例\n以下是类似问题的正确 SQL，请参考其模式：\n"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"### 示例 {i}")
            parts.append(f"问题：{ex['question']}")
            parts.append(f"SQL：\n```sql\n{ex['sql'].strip()}\n```\n")

        return "\n".join(parts)

    def format_context(self, business_context: str) -> str:
        """格式化业务上下文"""
        if not business_context:
            return ""
        return f"## 业务上下文\n{business_context}\n"

    def format_all(
        self,
        tables: list[dict],
        examples: list[dict],
        business_context: str,
    ) -> str:
        """组装完整的 Prompt 文本"""
        sections = []

        schema_text = self.format_tables(tables)
        if schema_text:
            sections.append(schema_text)

        context_text = self.format_context(business_context)
        if context_text:
            sections.append(context_text)

        fewshot_text = self.format_fewshot(examples)
        if fewshot_text:
            sections.append(fewshot_text)

        return "\n".join(sections)

    def _format_single_table(self, schema: dict) -> str:
        """格式化单张表为 DDL"""
        table_name = schema["table_name"]
        database = schema.get("database", "")
        full_name = f"`{database}`.`{table_name}`" if database else f"`{table_name}`"

        desc = schema.get("description") or schema.get("table_comment", "")
        header = f"-- {desc}" if desc else ""

        col_lines = []
        for col in schema.get("columns", []):
            # 跳过敏感字段
            if col.get("sensitive"):
                continue

            line = f"  `{col['name']}` {col['type']}"

            # 注释
            comment_parts = []
            display = col.get("display_name") or col.get("comment", "")
            if display:
                comment_parts.append(display)
            if col.get("enum_values"):
                enum_str = self._format_enum_inline(col["enum_values"])
                comment_parts.append(f"[{enum_str}]")
            if col.get("key") and "UNI" in col["key"]:
                comment_parts.append("(UNIQUE KEY)")

            if comment_parts:
                line += "  -- " + " ".join(comment_parts)

            col_lines.append(line)

        ddl = header + "\n" if header else ""
        ddl += f"CREATE TABLE {full_name} (\n"
        ddl += ",\n".join(col_lines)
        ddl += "\n);\n"

        # 关联关系提示
        for rel in schema.get("relations", []):
            target = rel.get("target_table", "")
            col = rel.get("column", "")
            target_col = rel.get("target_column", "")
            join_type = rel.get("join_type", "JOIN")
            if target:
                ddl += f"-- JOIN 提示: {table_name}.{col} = {target}.{target_col} ({join_type})\n"

        # 查询注意事项
        if schema.get("query_tips"):
            ddl += f"-- 注意: {schema['query_tips']}\n"

        return ddl

    def _format_enum_inline(self, enum_values) -> str:
        """内联格式化枚举值"""
        if isinstance(enum_values, list):
            parts = []
            for v in enum_values:
                if isinstance(v, dict):
                    parts.append(f"{v.get('value', '')}={v.get('label', '')}")
                else:
                    parts.append(str(v))
            return ", ".join(parts)
        return str(enum_values)
