"""FAISS Dense 向量索引 — 构建、检索、持久化"""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DenseIndex:
    """基于 FAISS IndexFlatIP 的稠密向量索引（余弦相似度，向量需归一化）"""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index: faiss.IndexFlatIP | None = None
        self.count = 0

    def build(self, embeddings: np.ndarray):
        """从向量矩阵构建索引"""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim
        # 归一化，使内积等价于余弦相似度
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.count = embeddings.shape[0]
        logger.info(f"Dense 索引构建完成, 文档数={self.count}, 维度={self.dim}")

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """
        检索最相似的 top_k 个文档。

        Args:
            query_vec: (1, dim) 的查询向量
            top_k: 返回数量

        Returns:
            [(doc_idx, score), ...] 按 score 降序
        """
        if self.index is None or self.count == 0:
            return []
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        top_k = min(top_k, self.count)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for i in range(top_k):
            idx = int(indices[0][i])
            if idx >= 0:
                results.append((idx, float(scores[0][i])))
        return results

    def save(self, path: str | Path):
        """持久化到文件"""
        if self.index is None:
            return
        path = str(path)
        faiss.write_index(self.index, path)
        logger.info(f"Dense 索引已保存: {path}")

    def load(self, path: str | Path):
        """从文件加载"""
        path = str(path)
        self.index = faiss.read_index(path)
        self.count = self.index.ntotal
        logger.info(f"Dense 索引已加载: {path}, 文档数={self.count}")
