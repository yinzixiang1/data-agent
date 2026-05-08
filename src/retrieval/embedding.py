"""
Qwen3-Embedding 编码器 — 仅输出 Dense 向量，Sparse 由 Milvus BM25 Function 处理。

Dense / Sparse 完全解耦:
    - Dense: 本模块通过 sentence-transformers 编码
    - Sparse: Milvus BM25 Function 在 insert 时自动生成，search 时自动匹配

支持 MRL (Matryoshka Representation Learning):
    dev 环境使用 1024 维截断 + L2 重归一化，prod 使用满血 2560 维。

使用示例::

    from src.retrieval.embedding import get_embedding

    emb = get_embedding()

    # 文档编码 (无 instruction)
    vecs = emb.encode(["商户账户表", "交易流水表"])  # shape (2, 1024)

    # 查询编码 (自动注入 per-collection instruction)
    q_vec = emb.encode_query("有多少活跃商户", collection_type="table")  # shape (1, 1024)
"""

import logging
from typing import Optional

import numpy as np

from src.retrieval.config import DENSE_MODEL, DENSE_DIM, DENSE_DEVICE, NL2SQL_ENV

logger = logging.getLogger(__name__)

_instance: Optional["Qwen3Embedding"] = None


class Qwen3Embedding:
    """
    Qwen3-Embedding 编码器，仅输出 Dense 向量。

    Attributes:
        model: SentenceTransformer 实例
        dim: 输出向量维度 (MRL 截断后)
        mrl_renormalize: 是否 MRL 截断后 L2 重归一化
        instructions: per-collection 查询 instruction 映射
        batch_size: 编码批大小
    """

    def __init__(self, embedding_config: dict | None = None):
        """
        Args:
            embedding_config: EMBEDDING_CONFIG 字典，为 None 时使用 config.py 默认值
        """
        from sentence_transformers import SentenceTransformer
        import torch

        cfg = embedding_config or {}
        model_name = cfg.get("model", DENSE_MODEL)
        self.dim = cfg.get("dim", DENSE_DIM)
        self.mrl_renormalize = cfg.get("mrl_renormalize", NL2SQL_ENV == "dev")
        self.instructions = cfg.get("instructions", {})
        self.batch_size = cfg.get("batch_size", 8)
        self.max_length = cfg.get("max_length", 512)

        dtype_str = cfg.get("dtype", "float16")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype_str, torch.float16)

        logger.info(
            f"加载 Embedding 模型: {model_name}, dim={self.dim}, "
            f"device={DENSE_DEVICE}, dtype={dtype_str}, mrl={self.mrl_renormalize}"
        )
        self.model = SentenceTransformer(
            model_name,
            device=DENSE_DEVICE,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        self.model.max_seq_length = self.max_length
        logger.info("Embedding 模型加载完成")

    def encode(self, texts: list[str], instruction: str = "") -> np.ndarray:
        """
        编码文本列表，返回 Dense 向量。

        Args:
            texts: 待编码文本列表
            instruction: 查询 instruction (文档编码时留空)。
                非空时自动格式化为 ``Instruct: {instruction}\\nQuery: `` 前缀。

        Returns:
            np.ndarray, shape (N, dim), float32
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": False,
        }
        if instruction:
            kwargs["prompt"] = f"Instruct: {instruction}\nQuery: "

        vecs = self.model.encode(texts, **kwargs)

        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs)

        # MRL: 截断到目标维度
        if vecs.shape[1] > self.dim:
            vecs = vecs[:, :self.dim]

        # MRL 重归一化: L2 normalize 截断后的向量
        if self.mrl_renormalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms

        return vecs.astype(np.float32)

    def encode_query(self, query: str, collection_type: str = "table") -> np.ndarray:
        """
        编码单条查询，自动注入 per-collection instruction。

        Args:
            query: 用户查询文本
            collection_type: Collection 类型 (table/column/enum/fewshot/glossary)

        Returns:
            np.ndarray, shape (1, dim), float32
        """
        instruction = self.instructions.get(collection_type, "")
        return self.encode([query], instruction=instruction)


# 向后兼容别名 (Phase 3 完成后可移除)
BGEEmbedding = Qwen3Embedding


def get_embedding(embedding_config: dict | None = None) -> Qwen3Embedding:
    """
    获取全局 Embedding 单例（懒加载）。

    Args:
        embedding_config: EMBEDDING_CONFIG 字典，仅首次调用时生效

    Returns:
        Qwen3Embedding 实例
    """
    global _instance
    if _instance is None:
        _instance = Qwen3Embedding(embedding_config)
    return _instance
