"""业务术语解析 — 从 glossary 中匹配用户问题里的术语"""

import logging

logger = logging.getLogger(__name__)


class GlossaryResolver:
    """
    业务术语解析器。

    将用户问题中出现的业务术语映射为：
    - enriched_query: 原始问题 + 术语展开词（增强检索召回）
    - business_context: 术语→口径/SQL写法（注入 Prompt）
    """

    def __init__(self):
        self.glossary: dict[str, dict] = {}

    def load(self, glossary: dict[str, dict]):
        """
        加载术语表。

        Args:
            glossary: {term: {definition, sql_hint, related_tables, related_columns}}
        """
        self.glossary = glossary
        logger.info(f"术语表加载完成: {len(glossary)} 条")

    def resolve(self, query: str) -> dict:
        """
        解析用户提问中的业务术语。

        Returns:
            {
                "enriched_query": str,      # 增强后的查询（用于检索）
                "business_context": str,    # 业务上下文（注入 Prompt）
                "matched_terms": list[str], # 命中的术语
            }
        """
        matched_terms = []
        context_parts = []
        extra_keywords = []

        query_lower = query.lower()
        for term, info in self.glossary.items():
            if term.lower() in query_lower:
                matched_terms.append(term)

                # 构建 context
                definition = info.get("definition", "")
                sql_hint = info.get("sql_hint", "")
                if definition and sql_hint:
                    context_parts.append(f"- {term} = {definition}, SQL: {sql_hint}")
                elif definition:
                    context_parts.append(f"- {term} = {definition}")

                # 提取展开关键词用于检索增强
                related_cols = info.get("related_columns", [])
                if isinstance(related_cols, list):
                    for col in related_cols:
                        # "pmt_account.verification_status" → "verification_status"
                        parts = col.split(".")
                        extra_keywords.append(parts[-1])

                related_tables = info.get("related_tables", [])
                if isinstance(related_tables, list):
                    extra_keywords.extend(related_tables)

        # 拼接 enriched_query
        enriched_query = query
        if extra_keywords:
            enriched_query = query + " " + " ".join(extra_keywords)

        business_context = "\n".join(context_parts) if context_parts else ""

        if matched_terms:
            logger.info(f"术语匹配: {matched_terms}")

        return {
            "enriched_query": enriched_query,
            "business_context": business_context,
            "matched_terms": matched_terms,
        }
