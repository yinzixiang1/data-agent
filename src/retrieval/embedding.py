"""
BGE-M3 Embedding 封装 — 单例模式，一次 encode 同时输出 Dense + Sparse 向量。

使用示例::

    from src.retrieval.embedding import get_embedding

    emb = get_embedding()  # 首次调用会加载模型，后续返回同一实例

    # 批量编码
    result = emb.encode(["查询活跃商户", "交易总额"])
    dense = result["dense_vecs"]    # shape (2, 1024)
    sparse = result["lexical_weights"]  # [dict, dict]

    # 单条查询编码
    q = emb.encode_query("有多少活跃商户")
    q_dense = q["dense_vecs"]          # shape (1, 1024)
    q_sparse = q["lexical_weights"][0]  # dict {token_id: weight}
"""

import logging
import numpy as np
from typing import Optional

from src.retrieval.config import EMBEDDING_MODEL, EMBEDDING_USE_FP16

logger = logging.getLogger(__name__)

_instance: Optional["BGEEmbedding"] = None


class BGEEmbedding:
    """
    BGE-M3 编码器，支持 Dense + Sparse 混合输出。

    Attributes:
        model: BGEM3FlagModel 实例，负责实际的向量编码
        dense_dim: Dense 向量维度，BGE-M3 固定为 1024
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, use_fp16: bool = EMBEDDING_USE_FP16):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径，如 "BAAI/bge-m3"
            use_fp16: 是否使用半精度推理，True 可减少显存并加速（推荐 GPU 环境开启）
        """
        from FlagEmbedding import BGEM3FlagModel
        logger.info(f"加载 BGE-M3 模型: {model_name}, fp16={use_fp16}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.dense_dim = 1024
        logger.info("BGE-M3 模型加载完成")

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> dict:
        """
        编码文本列表，一次调用同时返回 Dense 和 Sparse 向量。

        Args:
            texts: 待编码的文本列表，如 ["商户账户表", "交易流水表"]
            return_dense: 是否返回 Dense 向量（用于语义检索）
            return_sparse: 是否返回 Sparse 向量（用于关键词检索）
            batch_size: 编码批大小，显存不足时可调小
            max_length: 输入文本最大 token 长度，超出会被截断

        Returns:
            dict，包含以下可选键:
                - "dense_vecs": np.ndarray, shape (N, 1024), float32 密集向量
                - "lexical_weights": list[dict], 每个 dict 为 {token_id: weight} 稀疏权重
        """
        output = self.model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
            batch_size=batch_size,
            max_length=max_length,
        )
        result = {}
        if return_dense:
            dense = output["dense_vecs"]
            if not isinstance(dense, np.ndarray):
                dense = np.array(dense)
            result["dense_vecs"] = dense.astype(np.float32)
        if return_sparse:
            result["lexical_weights"] = output["lexical_weights"]
        return result

    def encode_query(self, query: str) -> dict:
        """
        编码单条查询文本（encode 的便捷封装）。

        Args:
            query: 用户的自然语言查询，如 "目前有多少活跃商户"

        Returns:
            dict，同 encode() 返回格式，但 dense_vecs shape 为 (1, 1024)
        """
        return self.encode([query], return_dense=True, return_sparse=True)


def get_embedding() -> BGEEmbedding:
    """
    获取全局 BGEEmbedding 单例（懒加载，首次调用时初始化模型）。

    Returns:
        BGEEmbedding 实例
    """
    global _instance
    if _instance is None:
        _instance = BGEEmbedding()
    return _instance
