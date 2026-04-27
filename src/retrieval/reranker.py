"""Reranker 精排 — BGE-Reranker-v2-M3 交叉编码器"""

import logging
from typing import Optional

from src.retrieval.config import RERANKER_MODEL, ENABLE_RERANKER

logger = logging.getLogger(__name__)

_instance: Optional["SchemaReranker"] = None


class SchemaReranker:
    """基于 Cross-Encoder 的 Schema 精排"""

    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        logger.info(f"加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选表做精排。

        Args:
            query: 用户原始提问
            candidates: hybrid_searcher 返回的候选列表，每个 dict 需有 "doc" 字段
            top_k: 精排后返回数量

        Returns:
            精排后的候选列表（附加 rerank_score 字段）
        """
        if not candidates:
            return []

        # 构建 (query, doc_text) 对
        pairs = []
        for c in candidates:
            doc_text = c.get("doc", {}).get("text", "")
            pairs.append((query, doc_text))

        # Cross-Encoder 打分
        scores = self.model.predict(pairs)

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranker 精排完成: {len(candidates)} → {min(top_k, len(ranked))}, "
            f"top={[r['table_name'] for r in ranked[:top_k]]}"
        )
        return ranked[:top_k]


def get_reranker() -> SchemaReranker | None:
    """获取 Reranker 单例，未启用则返回 None"""
    if not ENABLE_RERANKER:
        return None
    global _instance
    if _instance is None:
        _instance = SchemaReranker()
    return _instance
