"""基于 Agent 配置把自然语言实体值绑定到确定的表字段。"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class EntityResolver:
    """解析配置化实体规则，并校验目标表字段确实存在。"""

    _SEPARATOR = r"\s*(?:为|是|等于|=|:|：)\s*"
    _VALUE = (
        r'(?:"(?P<double>[^"\n]+)"|“(?P<curly>[^”\n]+)”|'
        r"'(?P<single>[^'\n]+)'|(?P<bare>[A-Za-z0-9_.@-]+))"
    )

    def __init__(self, rules: list[dict], table_schemas: dict[str, dict]) -> None:
        self.rules = [rule for rule in rules if isinstance(rule, dict)]
        self.table_schemas = table_schemas
        self._short_to_full: dict[str, list[str]] = {}
        for full_name in table_schemas:
            short_name = full_name.rsplit(".", 1)[-1].casefold()
            self._short_to_full.setdefault(short_name, []).append(full_name)

    def resolve(self, query: str, biz_line: str | None = None) -> dict:
        """解析问题中的实体过滤条件。"""
        filters: list[dict] = []
        unresolved: list[dict] = []
        for rule in self.rules:
            configured_business = str(rule.get("business") or "").strip()
            if biz_line and configured_business and configured_business != biz_line:
                continue

            table_name = self._resolve_table(str(rule.get("table") or ""))
            matches = self._resolve_rule(query, rule)
            for match in matches:
                column_name = match.get("column", "")
                reason = match.get("reason", "")
                value = match.get("value", "")
                issue = str(match.get("issue") or "") or self._validate_target(
                    table_name,
                    column_name,
                )
                if issue:
                    unresolved.append(
                        {
                            "rule_name": str(rule.get("name") or ""),
                            "entity": match.get("entity", ""),
                            "value": value,
                            "reason": reason,
                            "issue": issue,
                        }
                    )
                    continue
                filters.append(
                    {
                        "rule_name": str(rule.get("name") or ""),
                        "entity": match.get("entity", ""),
                        "value": value,
                        "table": table_name,
                        "column": column_name,
                        "qualified_column": f"{table_name}.{column_name}",
                        "operator": "=",
                        "reason": reason,
                    }
                )

        return {
            "filters": self._deduplicate(filters),
            "unresolved": self._deduplicate(unresolved),
        }

    @staticmethod
    def to_prompt_context(filters: list[dict]) -> str:
        """将实体绑定转换成不可被模型重新解释的查询约束。"""
        if not filters:
            return ""
        lines = ["实体过滤约束:"]
        for item in filters:
            value = str(item.get("value") or "").replace("'", "''")
            lines.append(
                f"- 用户表达“{item.get('entity', '')}”已确定为 "
                f"{item.get('qualified_column', '')} = '{value}'；"
                "必须通过该表字段过滤，不得改用其他表的同名字段"
            )
        return "\n".join(lines)

    def _resolve_rule(self, query: str, rule: dict) -> list[dict]:
        selectors = (
            ("name_terms", "name_column", "explicit_name"),
            ("long_id_terms", "long_id_column", "explicit_long_id"),
            ("short_id_terms", "short_id_column", "explicit_short_id"),
        )
        explicit_matches: list[dict] = []
        for terms_key, column_key, reason in selectors:
            column = str(rule.get(column_key) or "").strip()
            if not column:
                continue
            for entity, value in self._extract_values(query, rule.get(terms_key, [])):
                explicit_matches.append(
                    {
                        "entity": entity,
                        "value": value,
                        "column": column,
                        "reason": reason,
                    }
                )
        if explicit_matches:
            return explicit_matches

        generic_matches = self._extract_values(query, rule.get("generic_terms", []))
        if not generic_matches:
            return []

        long_id_length = rule.get("long_id_length")
        if isinstance(long_id_length, bool):
            long_id_length = None
        try:
            long_id_length = int(long_id_length)
        except (TypeError, ValueError):
            long_id_length = 0
        if long_id_length < 1:
            return [
                {
                    "entity": entity,
                    "value": value,
                    "column": "",
                    "reason": "invalid_long_id_length",
                    "issue": "实体解析规则未配置有效的长 ID 长度",
                }
                for entity, value in generic_matches
            ]

        matches: list[dict] = []
        for entity, value in generic_matches:
            issue = ""
            if len(value) == long_id_length:
                column = str(rule.get("long_id_column") or "").strip()
                reason = f"value_length_eq_{long_id_length}"
            elif len(value) < long_id_length:
                column = str(rule.get("short_id_column") or "").strip()
                reason = f"value_length_lt_{long_id_length}"
            else:
                column = ""
                reason = f"value_length_gt_{long_id_length}"
                issue = f"实体值长度超过规则允许的 {long_id_length} 位"
            matches.append(
                {
                    "entity": entity,
                    "value": value,
                    "column": column,
                    "reason": reason,
                    "issue": issue,
                }
            )
        return matches

    @classmethod
    def _extract_values(cls, query: str, terms: object) -> list[tuple[str, str]]:
        if not isinstance(terms, list):
            return []
        normalized_terms = sorted(
            {str(term).strip() for term in terms if str(term).strip()},
            key=len,
            reverse=True,
        )
        if not normalized_terms:
            return []
        term_pattern = "|".join(re.escape(term) for term in normalized_terms)
        pattern = re.compile(
            rf"(?P<entity>{term_pattern}){cls._SEPARATOR}{cls._VALUE}",
            re.IGNORECASE,
        )
        values: list[tuple[str, str]] = []
        for match in pattern.finditer(query):
            value = next(
                (
                    match.group(name)
                    for name in ("double", "curly", "single", "bare")
                    if match.group(name) is not None
                ),
                "",
            ).strip()
            if value:
                values.append((match.group("entity"), value))
        return values

    def _resolve_table(self, configured_name: str) -> str:
        normalized = configured_name.strip().strip("`")
        if normalized in self.table_schemas:
            return normalized
        candidates = self._short_to_full.get(
            normalized.rsplit(".", 1)[-1].casefold(), []
        )
        return candidates[0] if len(candidates) == 1 else normalized

    def _validate_target(self, table_name: str, column_name: str) -> str:
        if not column_name:
            return "实体解析规则未配置对应的目标字段"
        schema = self.table_schemas.get(table_name)
        if not schema:
            return f"目标语义表不存在或不可用: {table_name}"
        available_columns = {
            str(column.get("name") or "").casefold()
            for column in schema.get("columns", [])
        }
        if column_name.casefold() not in available_columns:
            return f"目标语义字段不存在: {table_name}.{column_name}"
        return ""

    @staticmethod
    def _deduplicate(items: list[dict]) -> list[dict]:
        unique: list[dict] = []
        seen: set[tuple] = set()
        for item in items:
            key = tuple(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique
