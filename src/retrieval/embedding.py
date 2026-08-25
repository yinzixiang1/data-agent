"""
Embedding 编码器 — 仅输出 Dense 向量，Sparse 由 Milvus BM25 Function 处理。

支持两种来源:
    - local: 通过 sentence-transformers 本地推理
    - api: 通过 OpenAI-compatible /v1/embeddings API 在线调用

Dense / Sparse 完全解耦:
    - Dense: 本模块编码
    - Sparse: Milvus BM25 Function 在 insert 时自动生成，search 时自动匹配

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

from src.retrieval.config import DENSE_DEVICE, DENSE_DIM, DENSE_MODEL, NL2SQL_ENV

logger = logging.getLogger(__name__)

_instance: Optional["BaseEmbedding"] = None


class BaseEmbedding:
    """Embedding 基类，定义统一接口。"""

    def __init__(self, dim: int, instructions: dict | None = None):
        self.dim = dim
        self.instructions = instructions or {}

    def encode(self, texts: list[str], instruction: str = "") -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, query: str, collection_type: str = "table") -> np.ndarray:
        instruction = self.instructions.get(collection_type, "")
        return self.encode([query], instruction=instruction)


class Qwen3Embedding(BaseEmbedding):
    """
    本地 Embedding 编码器 (sentence-transformers)。

    Attributes:
        model: SentenceTransformer 实例
        dim: 输出向量维度 (MRL 截断后)
        mrl_renormalize: 是否 MRL 截断后 L2 重归一化
        instructions: per-collection 查询 instruction 映射
        batch_size: 编码批大小
    """

    def __init__(self, embedding_config: dict | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = embedding_config or {}
        model_name = cfg.get("model", DENSE_MODEL)
        dim = cfg.get("dim", DENSE_DIM)
        super().__init__(dim=dim, instructions=cfg.get("instructions", {}))

        self.mrl_renormalize = cfg.get("mrl_renormalize", NL2SQL_ENV == "dev")
        self.batch_size = cfg.get("batch_size", 8)
        self.max_length = cfg.get("max_length", 512)

        dtype_str = cfg.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
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
            vecs = vecs[:, : self.dim]

        # MRL 重归一化: L2 normalize 截断后的向量
        if self.mrl_renormalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms

        return vecs.astype(np.float32)


class ApiEmbedding(BaseEmbedding):
    """
    在线 API Embedding 编码器 (OpenAI-compatible /v1/embeddings)。

    Attributes:
        base_url: API base URL
        api_key: 认证密钥
        api_model_name: API 请求中的 model 参数
        dim: 输出向量维度
        batch_size: 每次 API 请求的最大文本数
    """

    def __init__(self, embedding_config: dict | None = None):
        cfg = embedding_config or {}
        dim = cfg.get("dim", DENSE_DIM)
        super().__init__(dim=dim, instructions=cfg.get("instructions", {}))

        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.api_key = cfg.get("api_key", "")
        self.api_model_name = cfg.get("api_model_name", cfg.get("model", ""))
        self.batch_size = cfg.get("batch_size", 64)
        self.max_length = cfg.get("max_seq_length", 8192)
        self.send_dimensions = cfg.get("send_dimensions", False)

        if not self.base_url:
            raise ValueError("API Embedding 需要 base_url 配置")

        logger.info(
            f"初始化 API Embedding: base_url={self.base_url}, "
            f"model={self.api_model_name}, dim={self.dim}"
        )

    def encode(self, texts: list[str], instruction: str = "") -> np.ndarray:
        import httpx

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # 如有 instruction，加入前缀
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        all_vecs = []
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body = {"model": self.api_model_name, "input": batch}
            if self.dim and self.send_dimensions:
                body["dimensions"] = self.dim

            resp = httpx.post(url, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            embeddings = sorted(data["data"], key=lambda x: x["index"])
            for item in embeddings:
                vec = item["embedding"]
                all_vecs.append(vec)

        result = np.array(all_vecs, dtype=np.float32)
        # 截断到目标维度（兼容 API 不支持 dimensions 参数的情况）
        if result.shape[1] > self.dim:
            result = result[:, : self.dim]

        return result


# 向后兼容别名
BGEEmbedding = Qwen3Embedding


def get_embedding(embedding_config: dict | None = None) -> BaseEmbedding:
    """
    获取全局 Embedding 单例（懒加载）。

    根据 embedding_config.source 选择本地或 API 实现:
        - "local" (默认): Qwen3Embedding (sentence-transformers)
        - "api": ApiEmbedding (OpenAI-compatible HTTP API)

    Args:
        embedding_config: EMBEDDING_CONFIG 字典，仅首次调用时生效

    Returns:
        BaseEmbedding 实例
    """
    global _instance
    if _instance is None:
        cfg = embedding_config or {}
        source = cfg.get("source", "local")
        if source == "api":
            _instance = ApiEmbedding(cfg)
        else:
            _instance = Qwen3Embedding(cfg)
    return _instance
