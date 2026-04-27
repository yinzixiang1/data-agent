"""Few-shot 示例检索 — Milvus Dense 检索 + 表重叠度加权 + MMR 多样性选择"""

import json
import logging

import numpy as np

from src.retrieval.config import FEWSHOT_TOP_K, MMR_LAMBDA
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BGEEmbedding

logger = logging.getLogger(__name__)


class FewShotSelector:
    """
    动态 Few-shot 示例选择器。

    示例来源：语义层 YAML 中的 common_queries。
    """

    def __init__(self, embedding: BGEEmbedding, milvus_index: MilvusIndex | None = None):
        self.embedding = embedding
        self.milvus_index = milvus_index
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.example_table_sets: list[set[str]] = []

    def build_index(self, examples: list[dict]):
        """
        构建 Few-shot 示例索引（写入 Milvus）。

        Args:
            examples: [{"question", "sql", "tables": [], "difficulty"}, ...]
        """
        if not examples:
            logger.info("无 Few-shot 示例，跳过索引构建")
            return

        self.examples = examples
        texts = [ex["question"] for ex in examples]

        # Dense 编码
        output = self.embedding.encode(texts, return_dense=True, return_sparse=True)
        self.embeddings = output["dense_vecs"]

        # 写入 Milvus
        if self.milvus_index:
            self.milvus_index.create()
            doc_jsons = [json.dumps(ex, ensure_ascii=False) for ex in examples]
            self.milvus_index.insert(
                output["dense_vecs"].copy(),
                output["lexical_weights"],
                doc_jsons,
            )

        self.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
        logger.info(f"Few-shot 索引构建完成: {len(examples)} 条示例")

    def select(
        self,
        query: str,
        tables: list[str] | None = None,
        top_k: int = FEWSHOT_TOP_K,
    ) -> list[dict]:
        """
        选择最相关的 Few-shot 示例。

        排序策略：
        1. Milvus Dense 语义检索召回候选
        2. 表重叠度加权
        3. MMR 多样性选择
        """
        if not self.examples:
            return []

        # Milvus Dense 检索候选池
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
        """MMR 多样性选择"""
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
