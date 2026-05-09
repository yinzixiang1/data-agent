"""
全局配置 — 所有模块共享的常量和环境变量。

分两层：
  - 本文件: .env 级启动默认值（连接地址、设备、模型 fallback）
  - agent_config.py: 运行时从 MySQL sys_config 加载结构化 JSON 配置

NL2SQL_ENV 环境变量控制 dev / prod 默认值差异。

使用示例::

    from src.retrieval.config import DORIS_HOST, MILVUS_URI, NL2SQL_ENV

    print(NL2SQL_ENV)    # "dev"（默认）或 .env 中配置的值
    print(DORIS_HOST)    # "localhost"（默认）或 .env 中配置的值
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

# ── 环境标识 ──
# NL2SQL_ENV: 运行环境，影响模型、维度、索引参数的默认值
NL2SQL_ENV = os.getenv("NL2SQL_ENV", "dev")

# ── 硬件设备 ──
# DENSE_DEVICE: Embedding / Reranker 推理设备
DENSE_DEVICE = os.getenv("DENSE_DEVICE", "mps" if NL2SQL_ENV == "dev" else "cuda")

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

# ── Dense Embedding 模型 ──
# DENSE_MODEL: 默认 Embedding 模型（会被 sys_config EMBEDDING_CONFIG.model 覆盖）
# DENSE_DIM: 默认向量维度（dev 用 MRL 截断 1024，prod 用满血 2560）
DENSE_MODEL = os.getenv("DENSE_MODEL", "Qwen/Qwen3-Embedding-4B")
DENSE_DIM = int(os.getenv("DENSE_DIM", "1024" if NL2SQL_ENV == "dev" else "2560"))

# 向后兼容：旧变量名映射到新变量（Phase 2 完成后可移除）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DENSE_MODEL)
EMBEDDING_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"

# ── Reranker 精排模型 ──
# RERANKER_MODEL: 默认 Reranker 模型（会被 sys_config INDEX_BUILD_CONFIG.reranker.model 覆盖）
# ENABLE_RERANKER: 是否启用 Reranker（关闭后直接用混合检索分数排序）
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "Qwen/Qwen3-Reranker-0.6B" if NL2SQL_ENV == "dev" else "Qwen/Qwen3-Reranker-4B",
)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# ── MySQL 语义层数据库 ──
# MYSQL_HOST: 语义层 MySQL 地址
# MYSQL_PORT: MySQL 端口
# MYSQL_USER: MySQL 用户名
# MYSQL_PASSWORD: MySQL 密码
# MYSQL_DATABASE: 语义层数据库名
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "yzx12345.")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "data_agent")

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

# ── 术语检索参数 ──
# GLOSSARY_SCORE_THRESHOLD: 术语向量检索的余弦相似度阈值，低于此分数的匹配会被过滤
GLOSSARY_SCORE_THRESHOLD = float(os.getenv("GLOSSARY_SCORE_THRESHOLD", "0.5"))

# ── 启动行为 ──
# REBUILD_INDEX_ON_STARTUP: 启动时是否全量重建索引（true=重建, false=复用已有 Collection）
REBUILD_INDEX_ON_STARTUP = os.getenv("REBUILD_INDEX_ON_STARTUP", "false").lower() == "true"

# ── API 安全 ──
# DEFAULT_AGENT_TOKEN: Agent 未单独配置 token 时使用的默认值
DEFAULT_AGENT_TOKEN = os.getenv("DEFAULT_AGENT_TOKEN", "")
