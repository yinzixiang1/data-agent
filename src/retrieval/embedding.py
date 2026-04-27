"""BGE-M3 Embedding 封装 — 单例模式，一次 encode 同时输出 Dense + Sparse"""

import logging
import numpy as np
from typing import Optional

from src.retrieval.config import EMBEDDING_MODEL, EMBEDDING_USE_FP16

logger = logging.getLogger(__name__)

_instance: Optional["BGEEmbedding"] = None


class BGEEmbedding:
    """BGE-M3 编码器，支持 Dense + Sparse 混合输出"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, use_fp16: bool = EMBEDDING_USE_FP16):
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
        编码文本，一次调用同时返回 Dense 和 Sparse 向量。

        Returns:
            {
                "dense_vecs": np.ndarray (N, 1024) 或 None,
                "lexical_weights": list[dict{token_id: weight}] 或 None,
            }
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
        """编码单条查询"""
        return self.encode([query], return_dense=True, return_sparse=True)


def get_embedding() -> BGEEmbedding:
    """获取全局单例"""
    global _instance
    if _instance is None:
        _instance = BGEEmbedding()
    return _instance
