import os
from pathlib import Path
from dotenv import load_dotenv

# 防止 faiss-cpu 和 torch 在 ARM Mac 上的 OpenMP 冲突导致 segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

load_dotenv(override=True)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Doris 连接
DORIS_HOST = os.getenv("DORIS_HOST", "localhost")
DORIS_PORT = int(os.getenv("DORIS_PORT", "9030"))
DORIS_USER = os.getenv("DORIS_USER", "root")
DORIS_PASSWORD = os.getenv("DORIS_PASSWORD", "")
DORIS_DATABASE = os.getenv("DORIS_DATABASE", "dwd_banking")

# BGE-M3 Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"

# BGE-Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# 语义层目录
SEMANTIC_LAYER_DIR = PROJECT_ROOT / os.getenv("SEMANTIC_LAYER_DIR", "semantic_layer")

# 索引持久化目录
INDEX_STORE_DIR = PROJECT_ROOT / os.getenv("INDEX_STORE_DIR", "index_store")

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB = os.getenv("MILVUS_DB", "nl2sql")

# 检索参数
TABLE_SEARCH_TOP_K = int(os.getenv("TABLE_SEARCH_TOP_K", "5"))
COLUMN_SEARCH_TOP_K = int(os.getenv("COLUMN_SEARCH_TOP_K", "20"))
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "20"))
RERANK_INPUT_TOP_K = int(os.getenv("RERANK_INPUT_TOP_K", "10"))
FEWSHOT_TOP_K = int(os.getenv("FEWSHOT_TOP_K", "3"))
RRF_K = int(os.getenv("RRF_K", "60"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
