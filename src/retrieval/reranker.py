"""
Reranker 精排 — Qwen3-Reranker / BGE-Reranker 交叉编码器。

对混合检索的粗排候选做二次精排，提升 top-k 准确率。
支持 Qwen3-Reranker (trust_remote_code) 和 BGE-Reranker-v2-M3。

使用示例::

    from src.retrieval.reranker import get_reranker

    reranker = get_reranker()  # ENABLE_RERANKER=false 时返回 None
    if reranker:
        ranked = reranker.rerank(
            query="活跃商户数量",
            candidates=[{"table_name": "pmt_account", "doc": {"text": "..."}, ...}],
            top_k=5,
        )
"""

import logging
from typing import Optional

from src.retrieval.config import RERANKER_MODEL, ENABLE_RERANKER

logger = logging.getLogger(__name__)

_instance: Optional["SchemaReranker"] = None


class SchemaReranker:
    """
    基于 Cross-Encoder 的 Schema 精排器。

    Attributes:
        model: sentence-transformers CrossEncoder 实例
        score_threshold: 精排分数阈值 (供调用方读取，rerank() 内不自动过滤)
    """

    def __init__(self, model_name: str = RERANKER_MODEL, score_threshold: float = 0.3):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径
            score_threshold: 精排分数阈值，存储为属性供调用方使用
        """
        from sentence_transformers import CrossEncoder

        logger.info(f"加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        self.score_threshold = score_threshold
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选做精排，返回 top_k 个最相关的候选。

        Args:
            query: 用户原始提问
            candidates: 候选列表，每个 dict 需包含:
                - "doc": {"text": str} — 用于配对打分
                - "table_name": str — 表名 (用于日志)
            top_k: 精排后返回的最大数量

        Returns:
            list[dict]: 按 rerank_score 降序排列，每个 dict 附加 "rerank_score"
        """
        if not candidates:
            return []

        pairs = [(query, c.get("doc", {}).get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranker 精排完成: {len(candidates)} -> {min(top_k, len(ranked))}, "
            f"top={[r.get('table_name', '?') for r in ranked[:top_k]]}"
        )
        return ranked[:top_k]


def get_reranker(index_build_config: dict | None = None) -> SchemaReranker | None:
    """
    获取 Reranker 单例（懒加载）。

    Args:
        index_build_config: INDEX_BUILD_CONFIG 字典，仅首次调用时生效。
            读取 reranker.model 和 reranker.score_threshold。

    Returns:
        SchemaReranker 实例，ENABLE_RERANKER=False 时返回 None
    """
    if not ENABLE_RERANKER:
        return None
    global _instance
    if _instance is None:
        cfg = (index_build_config or {}).get("reranker", {})
        model = cfg.get("model", RERANKER_MODEL)
        threshold = cfg.get("score_threshold", 0.3)
        _instance = SchemaReranker(model_name=model, score_threshold=threshold)
    return _instance
