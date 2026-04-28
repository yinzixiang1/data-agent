"""
Reranker 精排 — BGE-Reranker-v2-M3 交叉编码器。

对混合检索的粗排候选做二次精排，提升 top-k 表的准确率。

使用示例::

    from src.retrieval.reranker import get_reranker

    reranker = get_reranker()  # ENABLE_RERANKER=false 时返回 None
    if reranker:
        ranked = reranker.rerank(
            query="活跃商户数量",
            candidates=[{"table_name": "pmt_account", "doc": {"text": "..."}, ...}],
            top_k=5,
        )
        # ranked: 按 rerank_score 降序排列的前 5 个候选
"""

import logging
from typing import Optional

from src.retrieval.config import RERANKER_MODEL, ENABLE_RERANKER

logger = logging.getLogger(__name__)

_instance: Optional["SchemaReranker"] = None


class SchemaReranker:
    """
    基于 Cross-Encoder 的 Schema 精排器。

    Cross-Encoder 将 (query, document) 对作为整体输入 Transformer，
    比 Bi-Encoder 更准确但更慢，因此只在粗排之后对少量候选做精排。

    Attributes:
        model: sentence-transformers CrossEncoder 实例
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径，如 "BAAI/bge-reranker-v2-m3"
        """
        from sentence_transformers import CrossEncoder
        logger.info(f"加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选表做精排，返回 top_k 个最相关的候选。

        Args:
            query: 用户原始提问，如 "目前有多少活跃商户"
            candidates: hybrid_searcher 返回的候选列表，每个 dict 需包含:
                - "doc": {"text": str} — 用于和 query 配对做 Cross-Encoder 打分
                - "table_name": str — 表名（用于日志）
            top_k: 精排后返回的最大数量

        Returns:
            list[dict]: 按 rerank_score 降序排列的候选列表，每个 dict 附加:
                - "rerank_score": float — Cross-Encoder 打分（值越大越相关）
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
    """
    获取 Reranker 单例（懒加载）。

    当 config.ENABLE_RERANKER 为 False 时返回 None，调用方需自行判断。

    Returns:
        SchemaReranker 实例，或 None（未启用时）
    """
    if not ENABLE_RERANKER:
        return None
    global _instance
    if _instance is None:
        _instance = SchemaReranker()
    return _instance
