"""
全局配置 — 所有模块共享的常量和环境变量。

分两层：
  - 本文件: YAML / 环境变量级启动默认值（连接地址、设备、模型 fallback）
  - agent_config.py: 运行时从 MySQL sys_config 加载结构化 JSON 配置

默认读取 configs/config.yaml，也可通过 NL2SQL_CONFIG_FILE 指定其他路径。
进程环境变量优先于 YAML。

使用示例::

    from src.retrieval.config import DORIS_HOST, MILVUS_URI, NL2SQL_ENV

    print(NL2SQL_ENV)    # "dev"（默认）或启动配置中的值
    print(DORIS_HOST)    # "localhost"（默认）或启动配置中的值
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import yaml

# 防止 faiss-cpu 和 torch 在 ARM Mac 上的 OpenMP 冲突导致 segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 项目根目录（推导自本文件位置: src/retrieval/config.py → 上溯两级）
PROJECT_ROOT = Path(__file__).parent.parent.parent


class StartupConfigurationError(RuntimeError):
    """Raised when startup configuration is incomplete or invalid."""


def _config_value_to_string(key: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    raise StartupConfigurationError(f"Configuration value for {key} must be a scalar")


def _load_yaml_config() -> None:
    configured_path = os.getenv("NL2SQL_CONFIG_FILE", "").strip()
    config_path = (
        Path(configured_path)
        if configured_path
        else PROJECT_ROOT / "configs" / "config.yaml"
    )
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        if configured_path:
            raise StartupConfigurationError(
                f"Configuration file does not exist: {config_path}"
            )
        return

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise StartupConfigurationError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc

    if values is None:
        return
    if not isinstance(values, Mapping):
        raise StartupConfigurationError(
            f"Configuration root must be a mapping: {config_path}"
        )

    for key, value in values.items():
        if not isinstance(key, str) or not key.isupper():
            raise StartupConfigurationError(
                "Configuration keys must be uppercase environment variable names"
            )
        os.environ.setdefault(key, _config_value_to_string(key, value))


_load_yaml_config()

# ── 环境标识 ──
# NL2SQL_ENV: 运行环境，影响模型、维度、索引参数的默认值
NL2SQL_ENV = os.getenv("NL2SQL_ENV", "dev").strip().lower()

# ── 硬件设备 ──
# DENSE_DEVICE: Embedding / Reranker 推理设备
DENSE_DEVICE = os.getenv("DENSE_DEVICE", "mps" if NL2SQL_ENV == "dev" else "cuda")

# ── Doris 连接 ──
# DORIS_HOST: Doris FE 节点地址
# DORIS_PORT: Doris MySQL 协议端口（默认 9030）
# DORIS_USER: 登录用户名
# DORIS_PASSWORD: 登录密码
DORIS_HOST = os.getenv("DORIS_HOST", "localhost")
DORIS_PORT = int(os.getenv("DORIS_PORT", "9030"))
DORIS_USER = os.getenv("DORIS_USER", "root")
DORIS_PASSWORD = os.getenv("DORIS_PASSWORD", "")
DORIS_PASSWORD_URL = quote_plus(DORIS_PASSWORD)

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
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_PASSWORD_URL = quote_plus(MYSQL_PASSWORD)
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "data_agent")

# ── Milvus 向量数据库 ──
# MILVUS_URI: Milvus 服务地址（含端口）
# MILVUS_DB: 使用的数据库名（自动创建）
# MILVUS_USER / MILVUS_PASSWORD: 认证信息（可选，开启 auth 时必填）
# MILVUS_TOKEN: Token 认证（可选，与 user/password 二选一）
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB = os.getenv("MILVUS_DB", "nl2sql")
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")

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

# ── 配置来源 ──
# CONFIG_SOURCE: 配置加载方式（mysql=从数据库加载, local=从本地文件加载）
# CONFIG_PROFILE: mysql 模式下为 Agent ID（如 "1"），local 模式下为配置文件路径
CONFIG_SOURCE = os.getenv("CONFIG_SOURCE", "mysql").strip().lower()
CONFIG_PROFILE = os.getenv("CONFIG_PROFILE", "").strip()

# ── 启动行为 ──
# REBUILD_INDEX_ON_STARTUP: 启动时是否全量重建索引（true=重建, false=复用已有 Collection）
REBUILD_INDEX_ON_STARTUP = (
    os.getenv("REBUILD_INDEX_ON_STARTUP", "false").lower() == "true"
)

# ── 日志 ──
# LOG_DIR: 日志输出目录（相对于项目根目录或绝对路径）
# LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR）
# LOG_RETENTION_DAYS: 日志保留天数（超过后自动删除压缩文件）
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))

# ── API 安全 ──
# DEFAULT_AGENT_TOKEN: Agent 未单独配置 token 时使用的默认值
DEFAULT_AGENT_TOKEN = os.getenv("DEFAULT_AGENT_TOKEN", "")
# AGENT_ADMIN_TOKEN: 控制面调用管理接口使用，与查询 Token 分离
AGENT_ADMIN_TOKEN = os.getenv("AGENT_ADMIN_TOKEN", "")


def validate_startup_config(environ: Mapping[str, str] | None = None) -> None:
    """Reject incomplete production configuration before external clients start."""
    values = os.environ if environ is None else environ
    runtime_env = values.get("NL2SQL_ENV", "dev").strip().lower()
    if runtime_env != "prod":
        return

    admin_token = values.get("AGENT_ADMIN_TOKEN", "").strip()
    if len(admin_token) < 32:
        raise StartupConfigurationError(
            "AGENT_ADMIN_TOKEN must be at least 32 characters in production"
        )
    default_token = values.get("DEFAULT_AGENT_TOKEN", "").strip()
    if default_token and admin_token == default_token:
        raise StartupConfigurationError(
            "AGENT_ADMIN_TOKEN must not reuse DEFAULT_AGENT_TOKEN"
        )

    config_source = values.get("CONFIG_SOURCE", "mysql").strip().lower()
    if config_source not in {"mysql", "local"}:
        raise StartupConfigurationError(
            "CONFIG_SOURCE must be either 'mysql' or 'local' in production"
        )

    if config_source == "local":
        if not values.get("CONFIG_PROFILE", "").strip():
            raise StartupConfigurationError(
                "CONFIG_PROFILE is required for local production configuration"
            )
        return

    required_mysql_fields = (
        "MYSQL_HOST",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
    )
    missing = [
        field for field in required_mysql_fields if not values.get(field, "").strip()
    ]
    if missing:
        raise StartupConfigurationError(
            "Missing required production configuration: " + ", ".join(missing)
        )

    profile = (
        values.get("CONFIG_PROFILE", "").strip()
        or values.get("DEFAULT_AGENT_ID", "").strip()
    )
    try:
        profile_id = int(profile)
    except (TypeError, ValueError):
        profile_id = 0
    if profile_id <= 0:
        raise StartupConfigurationError(
            "CONFIG_PROFILE or DEFAULT_AGENT_ID must be a positive Agent ID"
        )
