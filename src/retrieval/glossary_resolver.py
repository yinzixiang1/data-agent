"""
业务术语解析 — 向量语义匹配用户问题中的术语。

将用户自然语言中的业务术语（如 "活跃商户"）通过向量相似度匹配，展开为:
    - 检索增强词（related_columns + related_tables 追加到 query）
    - Prompt 业务上下文（definition + sql_hint 注入 LLM Prompt）

使用示例::

    resolver = GlossaryResolver(embedding, glossary_index, glossary_params, ef_search=64)

    result = resolver.resolve("目前有多少活跃商户")
    # result["enriched_query"]   = "目前有多少活跃商户 account_status pmt_account"
    # result["business_context"] = "- 活跃商户 = account_status=1 且 is_delete=0, SQL: ..."
    # result["matched_terms"]    = ["活跃商户"]
"""

import json
import logging

from src.retrieval.config import GLOSSARY_SCORE_THRESHOLD
from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.ranker_strategy import CollectionSearchParams

logger = logging.getLogger(__name__)


class GlossaryResolver:
    """
    业务术语解析器（基于混合检索语义匹配）。

    Attributes:
        embedding: Qwen3Embedding 实例
        glossary_index: 术语 MilvusIndex
        search_params: glossary Collection 的检索参数
        ef_search: HNSW 搜索参数 ef
    """

    def __init__(
        self,
        embedding: Qwen3Embedding,
        glossary_index: MilvusIndex,
        search_params: CollectionSearchParams,
        ef_search: int = 64,
    ):
        self.embedding = embedding
        self.glossary_index = glossary_index
        self.search_params = search_params
        self.ef_search = ef_search

    def resolve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = GLOSSARY_SCORE_THRESHOLD,
        metadata_filter: dict | None = None,
    ) -> dict:
        """
        通过混合检索匹配用户提问中的业务术语。

        Args:
            query: 用户原始查询
            top_k: 最多返回的术语数量
            score_threshold: 分数阈值比例，低于 max_score * threshold 的匹配被过滤
            metadata_filter: 任意 KV 过滤

        Returns:
            dict: enriched_query, business_context, matched_terms
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

        # 构建 metadata 过滤表达式
        filter_expr = None
        if metadata_filter:
            parts = []
            for key, value in metadata_filter.items():
                parts.append(
                    f'(metadata["{key}"] == "{value}"'
                    f' or not exists metadata["{key}"])'
                )
            filter_expr = " and ".join(parts) if parts else None

        # 编码查询（glossary instruction）
        q_dense = self.embedding.encode_query(query, collection_type="glossary")

        # 混合检索
        results = self.glossary_index.hybrid_search(
            q_dense,
            query,
            ranker=self.search_params.ranker,
            recall_k=self.search_params.recall_limit,
            output_fields=["term", "definition", "sql_hint", "related_tables", "related_columns"],
            filter_expr=filter_expr,
            ef_search=self.ef_search,
        )

        # RRF 分数是相对值，用最高分 x threshold 比例作为过滤线
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
