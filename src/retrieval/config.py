"""
全局配置 — 所有模块共享的常量和环境变量。

所有配置项均可通过 .env 文件或环境变量覆盖。

使用示例::

    from src.retrieval.config import DORIS_HOST, MILVUS_URI, TABLE_SEARCH_TOP_K

    print(DORIS_HOST)          # "localhost"（默认）或 .env 中配置的值
    print(TABLE_SEARCH_TOP_K)  # 5（默认的表检索 top-k）
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 防止 faiss-cpu 和 torch 在 ARM Mac 上的 OpenMP 冲突导致 segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

load_dotenv(override=True)

# 项目根目录（推导自本文件位置: src/retrieval/config.py → 上溯两级）
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Doris 连接 ──
# DORIS_HOST: Doris FE 节点地址
# DORIS_PORT: Doris MySQL 协议端口（默认 9030）
# DORIS_USER: 登录用户名
# DORIS_PASSWORD: 登录密码
# DORIS_DATABASE: 默认数据库名
DORIS_HOST = os.getenv("DORIS_HOST", "localhost")
DORIS_PORT = int(os.getenv("DORIS_PORT", "9030"))
DORIS_USER = os.getenv("DORIS_USER", "root")
DORIS_PASSWORD = os.getenv("DORIS_PASSWORD", "")
DORIS_DATABASE = os.getenv("DORIS_DATABASE", "dwd_banking")

# ── BGE-M3 Embedding 模型 ──
# EMBEDDING_MODEL: HuggingFace 模型名称或本地路径，用于生成 Dense + Sparse 向量
# EMBEDDING_USE_FP16: 是否使用半精度推理（加速 + 省显存）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"

# ── BGE-Reranker 精排模型 ──
# RERANKER_MODEL: Cross-Encoder 模型，对检索候选做精排
# ENABLE_RERANKER: 是否启用 Reranker（关闭后直接用混合检索分数排序）
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# ── 语义层目录 ──
# SEMANTIC_LAYER_DIR: 存放 tables/*.yaml、glossary/*.yaml、enums/*.yaml 的根目录
SEMANTIC_LAYER_DIR = PROJECT_ROOT / os.getenv("SEMANTIC_LAYER_DIR", "semantic_layer")

# ── Milvus 向量数据库 ──
# MILVUS_URI: Milvus 服务地址（含端口）
# MILVUS_DB: 使用的数据库名（自动创建）
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB = os.getenv("MILVUS_DB", "nl2sql")

# ── 检索参数 ──
# TABLE_SEARCH_TOP_K: 最终返回的表数量（Reranker 之后）
# COLUMN_SEARCH_TOP_K: 列级检索返回数量
# RECALL_TOP_K: Dense/Sparse 混合检索的召回数量（Reranker 输入池大小的上限）
# RERANK_INPUT_TOP_K: 送入 Reranker 精排的候选数量（> TABLE_SEARCH_TOP_K 才有意义）
# FEWSHOT_TOP_K: Few-shot 示例返回数量
# RRF_K: Reciprocal Rank Fusion 参数，值越大排名差距越平滑（推荐 40~80）
# MMR_LAMBDA: Maximal Marginal Relevance 相关性权重，0→纯多样性，1→纯相关性
TABLE_SEARCH_TOP_K = int(os.getenv("TABLE_SEARCH_TOP_K", "5"))
COLUMN_SEARCH_TOP_K = int(os.getenv("COLUMN_SEARCH_TOP_K", "20"))
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "20"))
RERANK_INPUT_TOP_K = int(os.getenv("RERANK_INPUT_TOP_K", "10"))
FEWSHOT_TOP_K = int(os.getenv("FEWSHOT_TOP_K", "3"))
RRF_K = int(os.getenv("RRF_K", "60"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
