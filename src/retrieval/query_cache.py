"""
查询缓存 — 基于语义相似度的结果缓存。

对于相同或高度相似的问题，直接返回缓存结果，避免重复 RAG + LLM 调用。

使用示例::

    cache = QueryCache(embedding, ttl=3600)
    cache.put("本月交易总额", {"sql": "SELECT ...", ...})
    hit = cache.get("本月的交易总额多少")  # 命中
"""

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from src.retrieval.embedding import BaseEmbedding

logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600  # 默认缓存 1 小时
DEFAULT_MAX_SIZE = 500  # 最大缓存条目数
SIMILARITY_THRESHOLD = 0.95  # 余弦相似度阈值


@dataclass
class CacheEntry:
    """缓存条目。"""

    query: str
    vector: np.ndarray
    result: dict
    created_at: float
    context_key: str = ""
    hit_count: int = 0


class QueryCache:
    """
    基于语义向量相似度的查询缓存。

    策略:
        1. 新查询编码为向量
        2. 与已有缓存条目计算余弦相似度
        3. 超过阈值则命中，返回缓存结果
        4. 未命中则返回 None
        5. 条目超过 TTL 或容量上限时淘汰

    Attributes:
        embedding: 向量编码器
        ttl: 缓存过期时间 (秒)
        max_size: 最大缓存条目数
        threshold: 语义相似度阈值
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        ttl: int = DEFAULT_TTL,
        max_size: int = DEFAULT_MAX_SIZE,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.embedding = embedding
        self.ttl = ttl
        self.max_size = max_size
        self.threshold = threshold
        self._entries: list[CacheEntry] = []

    def get(self, query: str, context_key: str = "") -> dict | None:
        """
        查询缓存。

        Args:
            query: 用户问题
            context_key: 上下文标识（biz_line + metadata_filter 序列化），不同 context 互不命中

        Returns:
            缓存的查询结果 dict，未命中返回 None
        """
        if not self._entries:
            return None

        self._evict_expired()

        q_vec = self.embedding.encode([query])[0]
        best_sim = -1.0
        best_entry = None

        for entry in self._entries:
            if entry.context_key != context_key:
                continue
            sim = self._cosine_sim(q_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.threshold and best_entry is not None:
            best_entry.hit_count += 1
            context_digest = hashlib.sha256(context_key.encode()).hexdigest()[:12]
            logger.info(
                f"缓存命中: sim={best_sim:.4f}, ctx_hash={context_digest}, "
                f"cached='{best_entry.query[:50]}', query='{query[:50]}'"
            )
            return best_entry.result

        return None

    def put(self, query: str, result: dict, context_key: str = ""):
        """
        写入缓存。

        Args:
            query: 用户问题
            result: 查询结果 dict
            context_key: 上下文标识
        """
        self._evict_expired()

        # 容量淘汰
        if len(self._entries) >= self.max_size:
            # 淘汰最老且命中最少的条目
            self._entries.sort(key=lambda e: (e.hit_count, e.created_at))
            self._entries.pop(0)

        q_vec = self.embedding.encode([query])[0]
        self._entries.append(
            CacheEntry(
                query=query,
                vector=q_vec,
                result=result,
                created_at=time.time(),
                context_key=context_key,
            )
        )

    def invalidate(self):
        """清空所有缓存。"""
        count = len(self._entries)
        self._entries.clear()
        if count:
            logger.info(f"缓存已清空: {count} 条")

    @property
    def size(self) -> int:
        return len(self._entries)

    def _evict_expired(self):
        """淘汰过期条目。"""
        now = time.time()
        before = len(self._entries)
        self._entries = [e for e in self._entries if now - e.created_at < self.ttl]
        evicted = before - len(self._entries)
        if evicted:
            logger.debug(f"缓存淘汰: {evicted} 条过期")

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
