"""
业务术语解析 — 从 glossary 中匹配用户问题里的术语。

将用户自然语言中的业务术语（如 "活跃商户"）展开为:
    - 检索增强词（related_columns + related_tables 追加到 query）
    - Prompt 业务上下文（definition + sql_hint 注入 LLM Prompt）

使用示例::

    resolver = GlossaryResolver()
    resolver.load({"活跃商户": {
        "definition": "account_status=1 且 is_delete=0",
        "sql_hint": "account_status = 1 AND is_delete = 0",
        "related_tables": ["pmt_account"],
        "related_columns": ["pmt_account.account_status"],
    }})

    result = resolver.resolve("目前有多少活跃商户")
    # result["enriched_query"]   = "目前有多少活跃商户 account_status pmt_account"
    # result["business_context"] = "- 活跃商户 = account_status=1 且 is_delete=0, SQL: ..."
    # result["matched_terms"]    = ["活跃商户"]
"""

import logging

logger = logging.getLogger(__name__)


class GlossaryResolver:
    """
    业务术语解析器（大小写不敏感匹配）。

    Attributes:
        glossary: 术语字典，{term: info_dict}
    """

    def __init__(self):
        self.glossary: dict[str, dict] = {}

    def load(self, glossary: dict[str, dict]):
        """
        加载术语表。

        Args:
            glossary: SchemaLoader.load_semantic_layer() 返回的术语字典，
                格式: {term: {"definition": str, "sql_hint": str,
                              "related_tables": list[str],
                              "related_columns": list[str]}}
        """
        self.glossary = glossary
        logger.info(f"术语表加载完成: {len(glossary)} 条")

    def resolve(self, query: str) -> dict:
        """
        解析用户提问中的业务术语（大小写不敏感子串匹配）。

        Args:
            query: 用户原始查询，如 "目前有多少活跃商户"

        Returns:
            dict，包含:
                - "enriched_query" (str): 原始问题 + 展开关键词（用于检索增强），
                    如 "目前有多少活跃商户 account_status pmt_account"
                - "business_context" (str): 术语定义和 SQL 提示（注入 Prompt），
                    如 "- 活跃商户 = account_status=1, SQL: account_status = 1 AND ..."
                - "matched_terms" (list[str]): 命中的术语名列表
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
