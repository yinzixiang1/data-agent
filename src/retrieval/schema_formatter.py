"""
Prompt 格式化 — 将检索结果格式化为 DDL Prompt 文本。

使用 CREATE TABLE DDL 格式组装 Prompt，因为 LLM 训练数据中大量出现 DDL，
这种格式的 SQL 生成准确率最高。

使用示例::

    formatter = SchemaFormatter()

    # 完整 Prompt 组装
    prompt = formatter.format_all(
        tables=retrieval_results,
        examples=fewshot_examples,
        business_context="活跃商户 = account_status=1 AND is_delete=0",
        enum_hits=[{"enum_label_cn": "LPSP", "table_name": "pmt_account",
                     "column_name": "account_type", "sql_value": "2000"}],
    )
"""


class SchemaFormatter:
    """将检索到的 Schema 格式化为 SQL 生成 Prompt（DDL 格式）。"""

    def format_tables(self, tables: list[dict]) -> str:
        """
        将检索命中的表格式化为 DDL 文本。

        Args:
            tables: 检索结果列表，每个 dict 需包含 "schema" 键

        Returns:
            str: 以 "## 可用数据表" 开头的 DDL 格式文本
        """
        parts = ["## 可用数据表\n"]

        for t in tables:
            schema = t.get("schema", {})
            if not schema:
                continue
            parts.append(self._format_single_table(schema))

        return "\n".join(parts)

    def format_fewshot(self, examples: list[dict]) -> str:
        """
        将 Few-shot 示例格式化为 Prompt 文本。

        Args:
            examples: FewShotSelector.select() 返回的示例列表，
                每个 dict 包含 "question" 和 "sql" 键

        Returns:
            str: "## 参考示例" 格式文本，空列表时返回空字符串
        """
        if not examples:
            return ""

        parts = ["## 参考示例\n以下是类似问题的正确 SQL，请参考其模式：\n"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"### 示例 {i}")
            parts.append(f"问题：{ex['question']}")
            parts.append(f"SQL：\n```sql\n{ex['sql'].strip()}\n```\n")

        return "\n".join(parts)

    def format_context(self, business_context: str) -> str:
        """
        格式化业务上下文（术语解析结果）。

        Args:
            business_context: GlossaryResolver 返回的术语展开文本，
                如 "- 活跃商户 = account_status=1, SQL: ..."

        Returns:
            str: "## 业务上下文" 格式文本，空字符串输入返回空字符串
        """
        if not business_context:
            return ""
        return f"## 业务上下文\n{business_context}\n"

    def format_enums(self, enum_hits: list[dict]) -> str:
        """
        格式化枚举值映射（用户语义 → SQL 条件）。

        Args:
            enum_hits: HybridSearcher.search_enums() 返回的枚举命中列表，
                每个 dict 包含 enum_label_cn, table_name, column_name, sql_value

        Returns:
            str: "## 枚举值映射" 格式文本，如:
                '- "LPSP" → pmt_account.account_type = 2000'
                空列表时返回空字符串
        """
        if not enum_hits:
            return ""

        parts = ["## 枚举值映射\n以下是用户语义对应的实际字段取值：\n"]
        for e in enum_hits:
            parts.append(
                f"- \"{e['enum_label_cn']}\" → {e['table_name']}.{e['column_name']} = {e['sql_value']}"
            )

        return "\n".join(parts) + "\n"

    def format_values(self, value_hits: list[dict]) -> str:
        """
        格式化 Schema Linking 值匹配结果。

        Args:
            value_hits: ValueIndexer.match_values() 返回的匹配列表，
                每个 dict 包含 enum_label_cn, table_name, column_name, sql_value

        Returns:
            str: "## 已识别实体值" 格式文本，空列表时返回空字符串
        """
        if not value_hits:
            return ""

        parts = ["## 已识别实体值\n以下是从问题中识别到的字段取值：\n"]
        for v in value_hits:
            parts.append(
                f"- \"{v['enum_label_cn']}\" -> {v['table_name']}.{v['column_name']} = {v['sql_value']}"
            )

        return "\n".join(parts) + "\n"

    def format_all(
        self,
        tables: list[dict],
        examples: list[dict],
        business_context: str,
        enum_hits: list[dict] | None = None,
        value_hits: list[dict] | None = None,
    ) -> str:
        """
        组装完整的 Prompt 文本。

        组装顺序: 可用数据表 -> 已识别实体值 -> 枚举值映射 -> 业务上下文 -> 参考示例。

        Args:
            tables: 检索命中的表列表（含 schema）
            examples: Few-shot 示例列表
            business_context: 业务术语展开文本
            enum_hits: 枚举值命中列表，None 视为空
            value_hits: Schema Linking 值匹配列表，None 视为空

        Returns:
            str: 完整 Prompt 文本
        """
        sections = []

        schema_text = self.format_tables(tables)
        if schema_text:
            sections.append(schema_text)

        value_text = self.format_values(value_hits or [])
        if value_text:
            sections.append(value_text)

        enum_text = self.format_enums(enum_hits or [])
        if enum_text:
            sections.append(enum_text)

        context_text = self.format_context(business_context)
        if context_text:
            sections.append(context_text)

        fewshot_text = self.format_fewshot(examples)
        if fewshot_text:
            sections.append(fewshot_text)

        return "\n".join(sections)

    def _format_single_table(self, schema: dict) -> str:
        """
        格式化单张表为 CREATE TABLE DDL 文本。

        输出包含: 表描述注释、列定义（含类型、display_name、description、枚举值），
        JOIN 提示和查询注意事项。敏感列(is_sensitive=True)会被跳过。

        Args:
            schema: 单张表的完整 Schema dict

        Returns:
            str: DDL 文本，如 "-- 商户账户表\\nCREATE TABLE `dwd_banking`.`pmt_account` (..."
        """
        table_name = schema["table_name"]
        database = schema.get("database", "")
        full_name = f"`{database}`.`{table_name}`" if database else f"`{table_name}`"

        desc = schema.get("description") or schema.get("table_comment", "")
        header = f"-- {desc}" if desc else ""

        col_lines = []
        for col in schema.get("columns", []):
            if col.get("is_sensitive"):
                continue

            line = f"  `{col['name']}` {col.get('type', '')}"

            comment_parts = []
            display = col.get("display_name") or col.get("comment", "")
            if display:
                comment_parts.append(display)
            desc = col.get("description", "")
            if desc and desc != display:
                comment_parts.append(desc)
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

        for rel in schema.get("relations", []):
            target = rel.get("target_table", "")
            col = rel.get("column", "")
            target_col = rel.get("target_column", "")
            join_type = rel.get("join_type", "JOIN")
            if target:
                ddl += f"-- JOIN 提示: {table_name}.{col} = {target}.{target_col} ({join_type})\n"

        if schema.get("query_tips"):
            ddl += f"-- 注意: {schema['query_tips']}\n"

        return ddl

    def _format_enum_inline(self, enum_values) -> str:
        """
        内联格式化枚举值（用于 DDL 列注释中）。

        Args:
            enum_values: 枚举值定义，同 DocumentBuilder._format_enum_values

        Returns:
            str: 如 "1000=普通账户, 2000=LPSP"
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
