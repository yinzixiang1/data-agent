"""
差异化 Ranker 策略 — 根据 collection_search_config 为每个 Collection 生成检索参数。

6 个 Collection 使用不同的融合策略:
    - table / column / enum / fewshot: WeightedRanker (Dense/BM25 加权)
    - glossary: RRFRanker
    - value: 纯 BM25 (不走 hybrid_search)

使用示例::

    from src.retrieval.ranker_strategy import get_search_params

    params = get_search_params(config.collection_search_config, "table")
    # params.ranker → WeightedRanker(0.4, 0.6)
    # params.recall_limit → 30
    # params.rerank → True
"""

import logging
from dataclasses import dataclass
from typing import Any

from pymilvus import RRFRanker, WeightedRanker

logger = logging.getLogger(__name__)


@dataclass
class CollectionSearchParams:
    """单个 Collection 的完整检索参数。

    Attributes:
        ranker_type: 融合策略类型 (weighted / rrf / bm25_only)
        ranker: Milvus Ranker 实例, bm25_only 时为 None
        recall_limit: Dense/BM25 各路召回数量
        rerank: 是否启用 Reranker 精排
        rerank_top_n: 送入 Reranker 的候选数量
        final_top_n: 最终保留数量, "all" 表示全部保留
    """

    ranker_type: str
    ranker: Any
    recall_limit: int
    rerank: bool
    rerank_top_n: int
    final_top_n: int | str


def get_search_params(
    collection_search_config: dict,
    collection_type: str,
) -> CollectionSearchParams:
    """
    从配置中解析指定 Collection 的检索参数。

    Args:
        collection_search_config: COLLECTION_SEARCH_CONFIG 字典
        collection_type: Collection 类型名 (table/column/enum/value/fewshot/glossary)

    Returns:
        CollectionSearchParams 实例
    """
    col_cfg = collection_search_config.get(collection_type, {})
    ranker_type = col_cfg.get("ranker_type", "weighted")

    if ranker_type == "weighted":
        dw = col_cfg.get("dense_weight", 0.5)
        bw = col_cfg.get("bm25_weight", 0.5)
        ranker = WeightedRanker(dw, bw)
    elif ranker_type == "rrf":
        ranker = RRFRanker(k=col_cfg.get("rrf_k", 60))
    elif ranker_type == "bm25_only":
        ranker = None
    else:
        logger.warning(f"未知 ranker_type '{ranker_type}'，fallback 到 weighted(0.5, 0.5)")
        ranker = WeightedRanker(0.5, 0.5)
        ranker_type = "weighted"

    return CollectionSearchParams(
        ranker_type=ranker_type,
        ranker=ranker,
        recall_limit=col_cfg.get("recall_limit", 20),
        rerank=col_cfg.get("rerank", False),
        rerank_top_n=col_cfg.get("rerank_top_n", 10),
        final_top_n=col_cfg.get("final_top_n", 10),
    )
