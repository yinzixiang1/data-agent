"""Sparse 倒排索引 — 基于 BGE-M3 lexical_weights 构建"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SparseIndex:
    """
    基于 BGE-M3 Learned Sparse 的倒排索引。

    结构: {token_id: [(doc_idx, weight), ...]}
    查询: 遍历 query 的 sparse weights，从倒排索引中累加文档得分。
    """

    def __init__(self):
        self.inverted_index: dict[int, list[tuple[int, float]]] = {}
        self.count = 0

    def build(self, lexical_weights_list: list[dict]):
        """
        从 BGE-M3 的 lexical_weights 列表构建倒排索引。

        Args:
            lexical_weights_list: BGE-M3 encode 输出的 list[dict{token_id: weight}]
        """
        self.inverted_index = {}
        self.count = len(lexical_weights_list)

        for doc_idx, weights in enumerate(lexical_weights_list):
            for token_id_str, weight in weights.items():
                token_id = int(token_id_str)
                if token_id not in self.inverted_index:
                    self.inverted_index[token_id] = []
                self.inverted_index[token_id].append((doc_idx, float(weight)))

        logger.info(f"Sparse 索引构建完成, 文档数={self.count}, 词项数={len(self.inverted_index)}")

    def search(self, query_weights: dict, top_k: int = 20) -> list[tuple[int, float]]:
        """
        稀疏检索：遍历 query 的 token weights，累加命中文档的得分。

        Args:
            query_weights: BGE-M3 encode 输出的 dict{token_id: weight}
            top_k: 返回数量

        Returns:
            [(doc_idx, score), ...] 按 score 降序
        """
        if not self.inverted_index:
            return []

        scores: dict[int, float] = {}
        for token_id_str, q_weight in query_weights.items():
            token_id = int(token_id_str)
            if token_id in self.inverted_index:
                for doc_idx, d_weight in self.inverted_index[token_id]:
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + q_weight * d_weight

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def save(self, path: str | Path):
        """持久化到 JSON"""
        # 将 int key 转为 str（JSON 要求）
        data = {
            "count": self.count,
            "inverted_index": {
                str(k): v for k, v in self.inverted_index.items()
            },
        }
        path = Path(path)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Sparse 索引已保存: {path}")

    def load(self, path: str | Path):
        """从 JSON 加载"""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.count = data["count"]
        self.inverted_index = {
            int(k): [tuple(item) for item in v]
            for k, v in data["inverted_index"].items()
        }
        logger.info(f"Sparse 索引已加载: {path}, 文档数={self.count}, 词项数={len(self.inverted_index)}")
