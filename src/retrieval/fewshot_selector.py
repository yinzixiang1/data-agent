"""
Few-shot 示例检索 — 混合召回 + 多特征排序 + MMR 多样性选择。

选择流程:
    1. Dense + BM25 混合召回候选池
    2. 综合问题语义、表集合、查询结构、示例质量和方言一致性排序
    3. MMR 多样性选择，避免重复示例占满上下文

使用示例::

    selector = FewShotSelector(embedding, milvus_index)
    selector.build_index([
        {"question": "活跃商户数", "sql": "SELECT COUNT(*) ...", "tables": ["pmt_account"]},
    ])

    examples = selector.select(
        query="目前有多少活跃商户",
        tables=["pmt_account"],
        top_k=3,
    )
"""

import logging
import re

import numpy as np
from pymilvus import DataType

from src.retrieval.config import FEWSHOT_TOP_K, MMR_LAMBDA
from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_filter import build_metadata_filter
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.ranker_strategy import CollectionSearchParams

logger = logging.getLogger(__name__)

# Few-shot Collection Schema
FEWSHOT_FIELDS = [
    {"name": "question", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql", "dtype": DataType.VARCHAR, "max_length": 8192},
    {"name": "involved_tables", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "difficulty", "dtype": DataType.VARCHAR, "max_length": 32},
    {"name": "metadata", "dtype": DataType.JSON},
]


class FewShotSelector:
    """
    动态 Few-shot 示例选择器（Dense + 表重叠 + MMR）。

    Attributes:
        embedding: Qwen3Embedding 实例
        milvus_index: Milvus Collection
        examples: 所有 Few-shot 示例列表
        embeddings: 所有示例的 Dense 向量矩阵
        example_table_sets: 每个示例涉及的表名集合列表
    """

    def __init__(
        self, embedding: Qwen3Embedding, milvus_index: MilvusIndex | None = None
    ):
        self.embedding = embedding
        self.milvus_index = milvus_index
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.example_table_sets: list[set[str]] = []

    def build_index(self, examples: list[dict], index_config: dict | None = None):
        """
        构建 Few-shot 示例索引（编码 + 写入 Milvus）。

        Args:
            examples: 示例列表，每个 dict 包含 question, sql, tables, difficulty
            index_config: INDEX_BUILD_CONFIG 字典，传给 MilvusIndex.create()
        """
        if not examples:
            logger.info("无 Few-shot 示例，跳过索引构建")
            return

        self.examples = examples
        texts = [ex["question"] for ex in examples]

        # Dense 编码（无 instruction，文档模式）
        self.embeddings = self.embedding.encode(texts)

        # 写入 Milvus
        if self.milvus_index:
            self.milvus_index.create(FEWSHOT_FIELDS, index_config=index_config)
            rows = [
                {
                    "question": ex["question"],
                    "sql": ex["sql"],
                    "involved_tables": ",".join(ex.get("tables", [])),
                    "difficulty": ex.get("difficulty", ""),
                    "metadata": ex.get("metadata", {}),
                }
                for ex in examples
            ]
            self.milvus_index.insert(self.embeddings, texts, rows)

        self.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
        logger.info(f"Few-shot 索引构建完成: {len(examples)} 条示例")

    def select(
        self,
        query: str,
        tables: list[str] | None = None,
        top_k: int = FEWSHOT_TOP_K,
        metadata_filter: dict | None = None,
        biz_line: str | None = None,
        search_params: CollectionSearchParams | None = None,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> list[dict]:
        """
        选择最相关且多样化的 Few-shot 示例。

        Args:
            query: 用户原始查询
            tables: 当前检索命中的表名列表，用于表重叠度加权
            top_k: 最终返回的示例数量
            metadata_filter: 任意 KV 过滤

        Returns:
            list[dict]: 选中的示例列表
        """
        if not self.examples:
            return []

        effective_filter = dict(metadata_filter or {})
        if biz_line:
            effective_filter.setdefault("business", biz_line)
        filter_expr = build_metadata_filter(metadata_filter=effective_filter or None)

        # 按配置执行混合召回；无配置时兼容旧的 Dense-only 行为。
        q_dense = self.embedding.encode_query(query, collection_type="fewshot")
        recall_limit = search_params.recall_limit if search_params else top_k * 5
        candidate_k = min(len(self.examples), max(top_k * 5, recall_limit))

        if self.milvus_index and self.milvus_index.count > 0:
            if search_params and search_params.ranker is not None:
                results = self.milvus_index.hybrid_search(
                    q_dense,
                    query,
                    ranker=search_params.ranker,
                    recall_k=candidate_k,
                    output_fields=["involved_tables", "difficulty", "metadata"],
                    filter_expr=filter_expr,
                )
            else:
                results = self.milvus_index.dense_search(
                    q_dense, top_k=candidate_k, filter_expr=filter_expr
                )
            n = len(self.examples)
            candidate_indices = [doc_id for doc_id, score, _ in results if doc_id < n]
            if search_params and search_params.rerank:
                rerank_limit = max(top_k, search_params.rerank_top_n)
                candidate_indices = candidate_indices[:rerank_limit]
            similarities = np.zeros(n)
            valid_scores = [float(score) for doc_id, score, _ in results if doc_id < n]
            min_score = min(valid_scores, default=0.0)
            max_score = max(valid_scores, default=1.0)
            for doc_id, score, _ in results:
                if doc_id < n:
                    denominator = max_score - min_score
                    similarities[doc_id] = (
                        (float(score) - min_score) / denominator
                        if denominator > 1e-9
                        else 1.0
                    )
        else:
            return []

        if search_params is None or search_params.rerank:
            query_tables = self._normalized_table_set(tables or [])
            query_signature = self._question_signature(query)
            for idx in candidate_indices:
                example = self.examples[idx]
                example_tables = self._normalized_table_set(example.get("tables", []))
                table_union = query_tables | example_tables
                table_score = (
                    len(query_tables & example_tables) / len(table_union)
                    if table_union
                    else 0.0
                )
                example_signature = self._sql_signature(example.get("sql", ""))
                structure_union = query_signature | example_signature
                structure_score = (
                    len(query_signature & example_signature) / len(structure_union)
                    if structure_union
                    else 0.0
                )
                metadata = example.get("metadata") or {}
                dialect = str(metadata.get("dialect") or "doris").casefold()
                similarities[idx] = (
                    0.40 * similarities[idx]
                    + 0.25 * table_score
                    + 0.20 * structure_score
                    + 0.10 * self._quality_score(example, metadata)
                    + 0.05 * (1.0 if dialect in {"doris", "mysql"} else 0.0)
                )

        # MMR 多样性选择
        effective_top_k = top_k
        if search_params and isinstance(search_params.final_top_n, int):
            effective_top_k = min(effective_top_k, search_params.final_top_n)
        selected = self._mmr_select(
            candidate_indices, similarities, effective_top_k, lambda_param=mmr_lambda
        )
        result = []
        for i in selected:
            ex = self.examples[i].copy()
            if "id" not in ex:
                ex["id"] = None
            result.append(ex)

        logger.info(
            f"Few-shot 选择完成: {len(result)} 条, "
            f"questions={[ex['question'][:30] for ex in result]}"
        )
        return result

    @staticmethod
    def _normalized_table_set(tables: list[str]) -> set[str]:
        normalized: set[str] = set()
        for table in tables:
            value = str(table).replace("`", "").casefold().strip()
            if value:
                normalized.update((value, value.rsplit(".", 1)[-1]))
        return normalized

    @staticmethod
    def _question_signature(question: str) -> set[str]:
        signature: set[str] = set()
        rules = {
            "aggregate": r"多少|数量|总计|合计|总额|平均|均值|汇总|统计|count|sum|avg",
            "ranking": r"排名|排行|top\s*\d*|最高|最低|最多|最少",
            "trend": r"趋势|按(?:日|天|周|月|季|年)|每天|每月|同比|环比",
            "detail": r"明细|列表|哪些|逐笔|每一笔|详情",
            "distinct": r"去重|不同|唯一|distinct",
        }
        for name, pattern in rules.items():
            if re.search(pattern, question, re.IGNORECASE):
                signature.add(name)
        return signature

    @staticmethod
    def _sql_signature(sql: str) -> set[str]:
        signature: set[str] = set()
        rules = {
            "aggregate": r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
            "ranking": r"\bORDER\s+BY\b.*\bLIMIT\b|\bROW_NUMBER\s*\(",
            "trend": r"\b(?:DATE_FORMAT|DATE_TRUNC)\s*\(|\bGROUP\s+BY\b.*(?:date|time|month|day)",
            "distinct": r"\bDISTINCT\b",
            "join": r"\bJOIN\b",
        }
        for name, pattern in rules.items():
            if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
                signature.add(name)
        if "aggregate" not in signature:
            signature.add("detail")
        return signature

    @staticmethod
    def _quality_score(example: dict, metadata: dict) -> float:
        raw_score = example.get("quality_score", metadata.get("quality_score", 1.0))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 1.0
        if score > 1.0:
            score /= 100.0
        return max(0.0, min(score, 1.0))

    def _mmr_select(
        self,
        candidate_indices: list[int],
        scores: np.ndarray,
        top_k: int,
        lambda_param: float = MMR_LAMBDA,
    ) -> list[int]:
        """
        Maximal Marginal Relevance (MMR) 多样性选择。

        MMR: score = lambda * relevance - (1-lambda) * max_similarity_to_selected
        """
        if self.embeddings is None or not candidate_indices:
            return candidate_indices[:top_k]

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self.embeddings / norms

        selected: list[int] = []
        remaining = set(candidate_indices)

        for _ in range(min(top_k, len(candidate_indices))):
            if not remaining:
                break

            if not selected:
                best = min(remaining, key=lambda index: (-scores[index], index))
            else:
                best = None
                best_mmr = -float("inf")
                for c in sorted(remaining):
                    relevance = scores[c]
                    max_sim = max(float(np.dot(normed[c], normed[s])) for s in selected)
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = c

            selected.append(best)
            remaining.discard(best)

        return selected
