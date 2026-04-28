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
    # examples: [{"question": "活跃商户数", "sql": "SELECT COUNT(*) ..."}, ...]
"""

import json
import logging

import numpy as np
from pymilvus import DataType

from src.retrieval.config import FEWSHOT_TOP_K, MMR_LAMBDA
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BGEEmbedding

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
        embedding: BGEEmbedding 实例
        milvus_index: Milvus Collection（可选，用于持久化检索）
        examples: 所有 Few-shot 示例列表
        embeddings: 所有示例的 Dense 向量矩阵，shape (N, 1024)
        example_table_sets: 每个示例涉及的表名集合列表
    """

    def __init__(self, embedding: BGEEmbedding, milvus_index: MilvusIndex | None = None):
        """
        Args:
            embedding: BGEEmbedding 实例，用于编码查询和示例
            milvus_index: Milvus Collection 实例，为 None 时仅使用内存检索
        """
        self.embedding = embedding
        self.milvus_index = milvus_index
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.example_table_sets: list[set[str]] = []

    def build_index(self, examples: list[dict]):
        """
        构建 Few-shot 示例索引（编码 + 写入 Milvus）。

        Args:
            examples: 示例列表，每个 dict 包含:
                - "question" (str): 问题文本，如 "目前有多少活跃商户"
                - "sql" (str): 对应的正确 SQL
                - "tables" (list[str]): 涉及的表名列表，如 ["pmt_account"]
                - "difficulty" (str): 难度等级，如 "easy", "medium"
        """
        if not examples:
            logger.info("无 Few-shot 示例，跳过索引构建")
            return

        self.examples = examples
        texts = [ex["question"] for ex in examples]

        # Dense + Sparse 编码
        output = self.embedding.encode(texts, return_dense=True, return_sparse=True)
        self.embeddings = output["dense_vecs"]

        # 写入 Milvus
        if self.milvus_index:
            self.milvus_index.create(FEWSHOT_FIELDS)
            rows = [
                {
                    "question": ex["question"],
                    "sql": ex["sql"],
                    "involved_tables": ",".join(ex.get("tables", [])),
                    "difficulty": ex.get("difficulty", ""),
                }
                for ex in examples
            ]
            self.milvus_index.insert(output["dense_vecs"], output["lexical_weights"], rows)

        self.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
        logger.info(f"Few-shot 索引构建完成: {len(examples)} 条示例")

    def select(
        self,
        query: str,
        tables: list[str] | None = None,
        top_k: int = FEWSHOT_TOP_K,
    ) -> list[dict]:
        """
        选择最相关且多样化的 Few-shot 示例。

        Args:
            query: 用户原始查询，如 "目前有多少活跃商户"
            tables: 当前检索命中的表名列表，用于表重叠度加权。
                示例的涉及表和 tables 有交集时，每个重叠表 +0.1 分
            top_k: 最终返回的示例数量

        Returns:
            list[dict]: 选中的示例列表，每个包含 question, sql, tables, difficulty
        """
        if not self.examples:
            return []

        # Dense 检索候选池
        q_output = self.embedding.encode([query], return_dense=True, return_sparse=False)
        q_dense = q_output["dense_vecs"]

        candidate_k = min(len(self.examples), top_k * 3)

        if self.milvus_index and self.milvus_index.count > 0:
            results = self.milvus_index.dense_search(q_dense, top_k=candidate_k)
            candidate_indices = [doc_id for doc_id, score, _ in results]
            similarities = np.zeros(len(self.examples))
            for doc_id, score, _ in results:
                if doc_id < len(self.examples):
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
        result = [self.examples[i] for i in selected]

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

        MMR 公式: score = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            candidate_indices: 候选示例的索引列表（指向 self.examples）
            scores: 全量分数数组，scores[i] 为第 i 个示例的综合得分
            top_k: 选择数量
            lambda_param: 相关性权重（0→纯多样性，1→纯相关性），默认 0.7

        Returns:
            list[int]: 选中的示例索引列表
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
