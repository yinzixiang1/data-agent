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
import re

from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_filter import build_metadata_filter
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
        top_k: int = 3,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
    ) -> dict:
        """
        通过混合检索匹配用户提问中的业务术语。

        Args:
            query: 用户原始查询
            top_k: 最多返回的术语数量
            metadata_filter: 任意 KV 过滤

        Returns:
            dict: enriched_query, business_context, matched_terms
        """
        matched_terms = []
        context_parts = []
        extra_keywords = []
        required_tables: set[str] = set()
        required_columns: set[str] = set()
        table_evidence: list[dict] = []

        if self.glossary_index.count == 0:
            return {
                "enriched_query": query,
                "business_context": "",
                "matched_terms": [],
                "related_tables": [],
                "related_columns": [],
                "table_evidence": [],
                "rejected_terms": [],
            }

        effective_filter = dict(metadata_filter or {})
        if biz_line:
            effective_filter.setdefault("business", biz_line)
        filter_expr = build_metadata_filter(metadata_filter=effective_filter or None)

        # 编码查询（glossary instruction）
        q_dense = self.embedding.encode_query(query, collection_type="glossary")

        # 混合检索
        # 术语库是受控小词典，需要先召回全部可见词条再做精确落地
        # 判断。否则用户明确说出的低频术语可能被向量 top-k 挤掉。
        recall_k = min(
            max(self.search_params.recall_limit, self.glossary_index.count), 200
        )
        results = self.glossary_index.hybrid_search(
            q_dense,
            query,
            ranker=self.search_params.ranker,
            recall_k=recall_k,
            output_fields=[
                "term",
                "definition",
                "sql_hint",
                "related_tables",
                "related_columns",
                "synonyms",
            ],
            filter_expr=filter_expr,
            ef_search=self.ef_search,
        )

        grounded_results = []
        rejected_terms: list[str] = []
        for result in results:
            entity = result[2]
            term = entity["term"]
            synonyms = self._parse_synonyms(entity.get("synonyms", "[]"))
            if self._is_grounded(query, term, synonyms):
                grounded_results.append(result)
            else:
                rejected_terms.append(term)

        # 业务术语会改变表、字段和过滤条件，只有用户明确说出术语或
        # 受控同义词时才能作为权威证据。向量候选仅用于观测和后续补词，
        # 不能在低置信度下静默固定物理表。
        # top_k 只是向量候选的上限；用户在同一句中明确说出的受控
        # 术语必须全部保留，不能因为排名截断丢掉 LOCAL/SWIFT/渠道等约束。
        selected_results = grounded_results
        grounded_doc_ids = {doc_id for doc_id, _, _ in grounded_results}

        for doc_id, score, entity in selected_results:
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
                    required_columns.add(str(col))
                    parts = col.split(".")
                    extra_keywords.append(parts[-1])

            try:
                parsed_tables = json.loads(entity.get("related_tables", "[]"))
            except (json.JSONDecodeError, TypeError):
                parsed_tables = []
            if isinstance(parsed_tables, list):
                normalized_tables = [str(table) for table in parsed_tables if table]
                extra_keywords.extend(normalized_tables)
                required_tables.update(normalized_tables)
                if normalized_tables:
                    table_evidence.append(
                        {
                            "term": term,
                            "tables": normalized_tables,
                            "columns": [
                                str(column) for column in related_cols if column
                            ],
                        }
                    )

        enriched_query = query
        if extra_keywords:
            enriched_query = query + " " + " ".join(extra_keywords)

        business_context = "\n".join(context_parts) if context_parts else ""

        if matched_terms:
            logger.info(
                "术语混合匹配: matched_terms=%s grounded_count=%d candidate_count=%d",
                matched_terms,
                sum(doc_id in grounded_doc_ids for doc_id, _, _ in selected_results),
                len(results),
            )
        elif results:
            top3 = [(e["term"], f"{s:.4f}") for _, s, e in results[:3]]
            logger.info("术语未命中: top3=%s", top3)

        return {
            "enriched_query": enriched_query,
            "business_context": business_context,
            "matched_terms": matched_terms,
            "related_tables": sorted(required_tables),
            "related_columns": sorted(required_columns),
            "table_evidence": table_evidence,
            "rejected_terms": rejected_terms,
        }

    @staticmethod
    def _parse_synonyms(raw_synonyms: object) -> list[str]:
        if isinstance(raw_synonyms, list):
            return [str(value) for value in raw_synonyms if value]
        if not isinstance(raw_synonyms, str) or not raw_synonyms:
            return []
        try:
            parsed = json.loads(raw_synonyms)
        except (json.JSONDecodeError, TypeError):
            return [value.strip() for value in raw_synonyms.split(",") if value.strip()]
        if not isinstance(parsed, list):
            return []
        return [str(value) for value in parsed if value]

    @classmethod
    def _is_grounded(cls, query: str, term: str, synonyms: list[str]) -> bool:
        return any(cls._contains_phrase(query, phrase) for phrase in [term, *synonyms])

    def _passes_semantic_score(self, score: float, max_score: float) -> bool:
        if self.search_params.ranker_type == "rrf":
            max_single_lane_score = 1 / (self.search_params.rrf_k + 1)
            return score > max_single_lane_score
        return bool(max_score) and score >= max_score * 0.5

    @staticmethod
    def _contains_phrase(query: str, phrase: str) -> bool:
        normalized = phrase.strip()
        if not normalized:
            return False
        if re.fullmatch(r"[A-Za-z0-9_ -]+", normalized):
            return bool(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                    query,
                    re.IGNORECASE,
                )
            )
        # 两字中文术语直接做子串包含会产生明显误判，例如“入金”不是
        # “买入金额”的业务术语。项目已使用 jieba，因此用通用分词边界
        # 判断短词，不在代码里维护业务特判。
        if len(normalized) <= 2 and re.fullmatch(r"[\u3400-\u9fff]+", normalized):
            import jieba

            return normalized.casefold() in {
                token.strip().casefold()
                for token in jieba.lcut(query, cut_all=False)
                if token.strip()
            }
        return normalized.casefold() in query.casefold()
