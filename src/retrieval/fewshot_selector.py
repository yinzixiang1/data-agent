"""
Few-shot 示例检索 — Dense 检索 + 表重叠度加权 + MMR 多样性选择。

选择流程:
    1. Dense 语义检索: 召回语义最接近的候选示例池
    2. 表重叠加权: 候选示例涉及的表和当前检索命中的表有交集时加分 (+0.1/表)
    3. MMR 多样性选择: 在相关性和多样性之间平衡，避免选出高度相似的示例

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

import json
import logging

import numpy as np
from pymilvus import DataType

from src.retrieval.config import FEWSHOT_TOP_K, MMR_LAMBDA
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import Qwen3Embedding

logger = logging.getLogger(__name__)

# Few-shot Collection Schema
FEWSHOT_FIELDS = [
    {"name": "question", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql", "dtype": DataType.VARCHAR, "max_length": 8192},
    {"name": "involved_tables", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "difficulty", "dtype": DataType.VARCHAR, "max_length": 32},
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

    def __init__(self, embedding: Qwen3Embedding, milvus_index: MilvusIndex | None = None):
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

        # Dense 检索候选池
        q_dense = self.embedding.encode_query(query, collection_type="fewshot")

        candidate_k = min(len(self.examples), top_k * 3)

        if self.milvus_index and self.milvus_index.count > 0:
            results = self.milvus_index.dense_search(q_dense, top_k=candidate_k, filter_expr=filter_expr)
            n = len(self.examples)
            candidate_indices = [doc_id for doc_id, score, _ in results if doc_id < n]
            similarities = np.zeros(n)
            for doc_id, score, _ in results:
                if doc_id < n:
                    similarities[doc_id] = score
        else:
            return []

        # 表重叠度加权
        if tables:
            query_tables = set(tables)
            for idx in candidate_indices:
                if idx < len(self.example_table_sets):
                    overlap = len(query_tables & self.example_table_sets[idx])
                    if overlap > 0:
                        similarities[idx] += 0.1 * overlap

        # MMR 多样性选择
        selected = self._mmr_select(candidate_indices, similarities, top_k)
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
                best = max(remaining, key=lambda i: scores[i])
            else:
                best = None
                best_mmr = -float("inf")
                for c in remaining:
                    relevance = scores[c]
                    max_sim = max(
                        float(np.dot(normed[c], normed[s]))
                        for s in selected
                    )
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = c

            selected.append(best)
            remaining.discard(best)

        return selected
