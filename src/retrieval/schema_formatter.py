"""
Prompt 格式化 — 析言风格，M-Schema + 【】结构化标记。

使用 M-Schema 格式描述表结构（比 DDL 更紧凑、Qwen 模型遵循度更高），
配合【】分段标记组装完整 Prompt。

使用示例::

    formatter = SchemaFormatter()

    prompt = formatter.format_all(
        tables=retrieval_results,
        examples=fewshot_examples,
        business_context="活跃商户 = account_status=1 AND is_delete=0",
        enum_hits=[{"enum_label_cn": "LPSP", "table_name": "pmt_account",
                     "column_name": "account_type", "sql_value": "2000"}],
        question="有多少活跃商户",
    )
"""

OUTPUT_RULES = """【输出要求】
1. 输出可执行的 SQL，用 ```sql ``` 包裹，不要任何多余解释
2. 使用 Schema 中的精确列名和表名
3. 先按表描述确认每张表的行粒度，选择与用户所问事实粒度一致的表
4. 如果同一张表已经直接包含所需指标和筛选维度，直接使用该表，不要通过其他表间接推导
5. amount、name、status 等通用字段必须按字段描述和业务逻辑解释，不能仅凭字段名猜测含义
6. 参考示例只用于学习 SQL 写法，不能替代当前 Schema 决定表和字段；示例与当前问题粒度或维度不一致时不得照搬
7. 优先使用已提供表中的字段，避免不必要的 JOIN
8. 状态码、类型码等枚举字段使用【枚举映射】中提供的数值
9. 时间字段使用 Doris 函数（CURDATE()、DATE_FORMAT()、DATE_TRUNC() 等）
10. 列别名使用中文双引号，如 COUNT(*) AS "数量"
11. 优先用 WHERE 条件过滤，避免全表扫描
12. 如果用户意图存在多种合理解释、无法唯一确定 SQL，输出一行 JSON：
   NEED_CLARIFY: {"question":"需要用户确认的问题","options":[{"label":"选项文案","value":"用于补充原问题的完整含义"}]}
   options 仅在有明确候选项时提供，最多 4 个；只是缺少开放式信息时输出空数组。不要因执行成本、SQL 风险或模型信心不足触发澄清。"""


class SchemaFormatter:
    """将检索到的 Schema 格式化为析言风格 Prompt（M-Schema + 【】标记）。"""

    def format_tables(self, tables: list[dict]) -> str:
        """将检索命中的表格式化为 M-Schema 文本。"""
        parts = ["【数据库 Schema】"]

        for t in tables:
            schema = t.get("schema", {})
            if not schema:
                continue
            parts.append(self._format_single_table(schema))

        return "\n".join(parts)

    def format_fewshot(self, examples: list[dict]) -> str:
        """将 Few-shot 示例格式化为【参考示例】文本。"""
        if not examples:
            return ""

        parts = [
            "【参考示例】",
            "以下示例只展示 SQL 写法，不代表当前问题应使用相同的表或字段；"
            "当前问题的事实粒度、指标和筛选维度以本次 Schema 为准。",
        ]
        for i, ex in enumerate(examples, 1):
            parts.append(f"示例 {i}:")
            parts.append(f"  问题：{ex['question']}")
            parts.append(f"  SQL：{ex['sql'].strip()}")
            parts.append("")

        return "\n".join(parts)

    def format_context(self, business_context: str) -> str:
        """格式化业务上下文为【业务术语字典】。"""
        if not business_context:
            return ""
        return f"【业务术语字典】\n{business_context}"

    def format_enums(self, enum_hits: list[dict]) -> str:
        """格式化枚举值映射为【枚举映射】。"""
        if not enum_hits:
            return ""

        parts = ["【枚举映射】"]
        for e in enum_hits:
            parts.append(
                f'- "{e["enum_label_cn"]}" -> {e["table_name"]}.{e["column_name"]} = {e["sql_value"]}'
            )

        return "\n".join(parts)

    def format_values(self, value_hits: list[dict]) -> str:
        """格式化 Schema Linking 值匹配结果，合入枚举映射区域。"""
        if not value_hits:
            return ""

        parts = []
        for v in value_hits:
            parts.append(
                f'- "{v["enum_label_cn"]}" -> {v["table_name"]}.{v["column_name"]} = {v["sql_value"]}'
            )

        return "\n".join(parts)

    def format_all(
        self,
        tables: list[dict],
        examples: list[dict],
        business_context: str,
        enum_hits: list[dict] | None = None,
        value_hits: list[dict] | None = None,
        question: str = "",
        output_rules: str = "",
        intent_context: str = "",
    ) -> str:
        """
        组装完整的析言风格 Prompt。

        组装顺序: Schema -> 术语 -> 枚举映射 -> 参考示例 -> 用户问题 -> 输出要求

        Args:
            tables: 检索命中的表列表（含 schema）
            examples: Few-shot 示例列表
            business_context: 业务术语展开文本
            enum_hits: 枚举值命中列表
            value_hits: Schema Linking 值匹配列表
            question: 用户问题（析言风格嵌入 prompt 末尾）
            output_rules: 输出要求文本，空字符串时使用内置默认值
        """
        sections = []

        # 【数据库 Schema】
        schema_text = self.format_tables(tables)
        if schema_text:
            sections.append(schema_text)

        # 【业务术语字典】
        context_text = self.format_context(business_context)
        if context_text:
            sections.append(context_text)

        if intent_context:
            sections.append(f"【查询意图】\n{intent_context}")

        # 【枚举映射】— 合并 enum_hits + value_hits
        enum_lines = self.format_enums(enum_hits or [])
        value_lines = self.format_values(value_hits or [])
        if enum_lines or value_lines:
            if enum_lines and value_lines:
                sections.append(enum_lines + "\n" + value_lines)
            else:
                sections.append(enum_lines or "【枚举映射】\n" + value_lines)

        # 【参考示例】
        fewshot_text = self.format_fewshot(examples)
        if fewshot_text:
            sections.append(fewshot_text)

        # 【用户问题】
        if question:
            sections.append(f"【用户问题】\n{question}")

        # 【输出要求】— 优先用传入的配置，否则用内置默认
        sections.append(output_rules.strip() if output_rules.strip() else OUTPUT_RULES)

        return "\n\n".join(sections)

    def _format_single_table(self, schema: dict) -> str:
        """格式化单张表为 M-Schema 文本。"""
        short_name = schema.get("table_name_short", schema["table_name"])
        database = schema.get("database", "")
        sql_name = f"`{database}`.`{short_name}`" if database else f"`{short_name}`"

        desc = schema.get("description") or schema.get("table_comment", "")
        header = f"【表】{sql_name}（{desc}）" if desc else f"【表】{sql_name}"

        col_lines = []
        for col in schema.get("columns", []):
            if col.get("is_sensitive"):
                continue
            if col.get("is_skip_index") and not col.get("_context_required"):
                continue

            line = f"  - `{col['name']}` {col.get('type', '')}"

            comment_parts = []
            display = col.get("display_name") or col.get("comment", "")
            if display:
                comment_parts.append(display)
            desc = col.get("description", "")
            if desc and desc != display:
                comment_parts.append(desc)
            business_logic = col.get("business_logic", "")
            if business_logic and business_logic not in comment_parts:
                comment_parts.append(f"业务逻辑: {business_logic}")
            if col.get("enum_values"):
                enum_str = self._format_enum_inline(col["enum_values"])
                comment_parts.append(f"[{enum_str}]")
            if col.get("key") and "UNI" in col["key"]:
                comment_parts.append("(UNIQUE KEY)")

            if comment_parts:
                line += " -- " + " | ".join(comment_parts)

            col_lines.append(line)

        result = header + "\n"
        result += "\n".join(col_lines)

        # 关联关系
        for rel in schema.get("relations", []):
            target = rel.get("target_table", "")
            col = rel.get("column", "")
            target_col = rel.get("target_column", "")
            join_type = rel.get("join_type", "JOIN")
            cardinality = rel.get("cardinality", "unknown")
            if target:
                relation_meta = join_type
                if cardinality != "unknown":
                    relation_meta += f", {cardinality}"
                result += f"\n  关联: {short_name}.{col} = {target}.{target_col} ({relation_meta})"

        # 查询注意事项
        if schema.get("query_tips"):
            result += f"\n  注意: {schema['query_tips']}"

        return result

    def _format_enum_inline(self, enum_values) -> str:
        """内联格式化枚举值。"""
        if isinstance(enum_values, list):
            parts = []
            for v in enum_values:
                if isinstance(v, dict):
                    parts.append(f"{v.get('value', '')}={v.get('label', '')}")
                else:
                    parts.append(str(v))
            return ", ".join(parts)
        return str(enum_values)
