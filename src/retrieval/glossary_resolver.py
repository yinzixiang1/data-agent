"""
业务术语解析 — 向量语义匹配用户问题中的术语。

将用户自然语言中的业务术语（如 "活跃商户"）通过向量相似度匹配，展开为:
    - 检索增强词（related_columns + related_tables 追加到 query）
    - Prompt 业务上下文（definition + sql_hint 注入 LLM Prompt）

使用示例::

    resolver = GlossaryResolver(embedding, glossary_index)

    result = resolver.resolve("目前有多少活跃商户")
    # result["enriched_query"]   = "目前有多少活跃商户 account_status pmt_account"
    # result["business_context"] = "- 活跃商户 = account_status=1 且 is_delete=0, SQL: ..."
    # result["matched_terms"]    = ["活跃商户"]
"""

import json
import logging

from src.retrieval.config import GLOSSARY_SCORE_THRESHOLD
from src.retrieval.embedding import BGEEmbedding
from src.retrieval.milvus_store import MilvusIndex

logger = logging.getLogger(__name__)


class GlossaryResolver:
    """
    业务术语解析器（基于向量语义匹配）。

    Attributes:
        embedding: BGEEmbedding 实例
        glossary_index: 术语 MilvusIndex
    """

    def __init__(self, embedding: BGEEmbedding, glossary_index: MilvusIndex):
        self.embedding = embedding
        self.glossary_index = glossary_index

    def resolve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = GLOSSARY_SCORE_THRESHOLD,
    ) -> dict:
        """
        通过向量语义匹配解析用户提问中的业务术语。

        Args:
            query: 用户原始查询，如 "目前有多少活跃商户"
            top_k: 最多返回的术语数量
            score_threshold: 余弦相似度阈值，低于此分数的匹配会被过滤

        Returns:
            dict，包含:
                - "enriched_query" (str): 原始问题 + 展开关键词（用于检索增强）
                - "business_context" (str): 术语定义和 SQL 提示（注入 Prompt）
                - "matched_terms" (list[str]): 命中的术语名列表
        """
        matched_terms = []
        context_parts = []
        extra_keywords = []

        if self.glossary_index.count == 0:
            return {
                "enriched_query": query,
                "business_context": "",
                "matched_terms": [],
            }

        q_output = self.embedding.encode_query(query)
        q_dense = q_output["dense_vecs"]
        q_sparse = q_output["lexical_weights"][0]

        results = self.glossary_index.hybrid_search(
            q_dense,
            q_sparse,
            top_k=top_k,
            recall_k=top_k,
            output_fields=["term", "definition", "sql_hint", "related_tables", "related_columns"],
        )

        # RRF 分数是相对值，用最高分 × threshold 比例作为过滤线
        max_score = results[0][1] if results else 0
        min_score = max_score * score_threshold

        for doc_id, score, entity in results:
            if score < min_score:
                continue

            term = entity["term"]
            definition = entity.get("definition", "")
            sql_hint = entity.get("sql_hint", "")

            matched_terms.append(term)

            if definition and sql_hint:
                context_parts.append(f"- {term} = {definition}, SQL: {sql_hint}")
            elif definition:
                context_parts.append(f"- {term} = {definition}")

            # 提取展开关键词用于检索增强
            try:
                related_cols = json.loads(entity.get("related_columns", "[]"))
            except (json.JSONDecodeError, TypeError):
                related_cols = []
            if isinstance(related_cols, list):
                for col in related_cols:
                    parts = col.split(".")
                    extra_keywords.append(parts[-1])

            try:
                related_tables = json.loads(entity.get("related_tables", "[]"))
            except (json.JSONDecodeError, TypeError):
                related_tables = []
            if isinstance(related_tables, list):
                extra_keywords.extend(related_tables)

        enriched_query = query
        if extra_keywords:
            enriched_query = query + " " + " ".join(extra_keywords)

        business_context = "\n".join(context_parts) if context_parts else ""

        if matched_terms:
            logger.info(f"术语混合匹配: {matched_terms} (min_score={min_score:.4f}, max_score={max_score:.4f})")
        elif results:
            top3 = [(e["term"], f"{s:.4f}") for _, s, e in results[:3]]
            logger.info(f"术语未命中 (top3={top3}, min_score={min_score:.4f})")

        return {
            "enriched_query": enriched_query,
            "business_context": business_context,
            "matched_terms": matched_terms,
        }
