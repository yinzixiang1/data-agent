"""
NL2SQL Data Agent — FastAPI HTTP 服务入口。

启动方式::

    uvicorn app:app --host 0.0.0.0 --port 9090 --reload

接口:
    GET  /health                — 健康检查
    POST /query                 — NL2SQL 查询（RAG + LLM + EXPLAIN）
    POST /admin/index-rebuild   — 触发索引全量重建
    POST /admin/config-reload   — 重新加载 Agent 配置
    POST /evaluation/run        — 执行评估
"""

import json
import re
import time
import logging
import uuid
import os
import secrets
import signal
from datetime import date as _date

from contextlib import asynccontextmanager

# 确保 Ctrl+C / kill 能强制终止进程（gRPC 线程会吞默认信号处理）
def _force_exit(*_):
    os._exit(1)

try:
    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)
except ValueError:
    pass  # 非主线程忽略
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy import create_engine, text

from src.retrieval.retriever import SchemaRetriever
from src.retrieval.sql_validator import SQLValidator
from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.query_logger import QueryLogger
from src.retrieval.config import (
    DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD, DORIS_PASSWORD_URL,
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_PASSWORD_URL, MYSQL_DATABASE,
    MILVUS_URI, MILVUS_DB, MILVUS_USER, MILVUS_PASSWORD, MILVUS_TOKEN,
    EMBEDDING_MODEL, RERANKER_MODEL,
    DEFAULT_AGENT_TOKEN,
    CONFIG_SOURCE, CONFIG_PROFILE,
    LOG_DIR, LOG_LEVEL, LOG_RETENTION_DAYS, PROJECT_ROOT,
)

# ── 日志配置：控制台 + 文件双写，按天轮转压缩 ──
import gzip
import shutil
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_log_dir = Path(LOG_DIR) if Path(LOG_DIR).is_absolute() else PROJECT_ROOT / LOG_DIR
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "app.log"
_log_fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def _namer(default_name: str) -> str:
    return default_name + ".gz"


def _rotator(source: str, dest: str) -> None:
    with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(source)


_file_handler = TimedRotatingFileHandler(
    filename=str(_log_file),
    when="midnight",
    interval=1,
    backupCount=LOG_RETENTION_DAYS,
    encoding="utf-8",
)
_file_handler.namer = _namer
_file_handler.rotator = _rotator
_file_handler.setFormatter(logging.Formatter(_log_fmt))

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(_log_fmt))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    handlers=[_console_handler, _file_handler],
)
# uvicorn 日志统一走 root handler（清除自带 handler，靠 propagate 传递）
for _uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_logger = logging.getLogger(_uv_name)
    _uv_logger.handlers.clear()
    _uv_logger.propagate = True

logger = logging.getLogger(__name__)

# ── 全局状态 ──

retriever: SchemaRetriever | None = None
validator: SQLValidator | None = None
llm_client: BaseChatModel | None = None
agent_config: AgentRuntimeConfig | None = None
query_logger: QueryLogger | None = None
config_loader: AgentConfigLoader | None = None

# Phase 3: 知识库 + 意图分类
from src.retrieval.intent_classifier import IntentClassifier
from src.retrieval.knowledge_retriever import KnowledgeRetriever
from src.retrieval.query_cache import QueryCache

intent_classifier: IntentClassifier = IntentClassifier()
knowledge_retriever: KnowledgeRetriever | None = None
query_cache: QueryCache | None = None


# ── 启动配置打印 ──

def print_infra_config(config: AgentRuntimeConfig | None = None):
    """打印基础设施配置。传入 agent_config 后会显示实际使用的 Milvus 连接。"""
    # Milvus: 优先 agent_config 资源绑定，fallback .env
    m_uri = (config.milvus_uri if config and config.milvus_uri else MILVUS_URI)
    m_db = (config.milvus_db if config and config.milvus_db else MILVUS_DB)
    m_user = (config.milvus_user if config and config.milvus_user else MILVUS_USER)
    m_pass = (config.milvus_password if config and config.milvus_password else MILVUS_PASSWORD)
    m_token = (config.milvus_token if config and config.milvus_token else MILVUS_TOKEN)
    m_source = "资源绑定" if config and config.milvus_uri else ".env"

    # Embedding / Reranker: 优先 agent_config 覆盖值，fallback .env
    emb_model = EMBEDDING_MODEL
    emb_source = ".env"
    if config and config.embedding_config.get("model"):
        emb_model = config.embedding_config["model"]
        emb_source = "Agent 配置"
    rnk_model = RERANKER_MODEL
    rnk_source = ".env"
    if config and config.index_build_config.get("reranker", {}).get("model"):
        rnk_model = config.index_build_config["reranker"]["model"]
        rnk_source = "Agent 配置"

    lines = [
        "",
        "=" * 60,
        "  NL2SQL Data Agent 基础设施配置",
        "=" * 60,
        "",
        "  [Doris]",
        f"    Host:     {DORIS_HOST}:{DORIS_PORT}",
        f"    User:     {DORIS_USER}",
        "",
        "  [MySQL 语义层]",
        f"    Host:     {MYSQL_HOST}:{MYSQL_PORT}",
        f"    User:     {MYSQL_USER}",
        f"    Database: {MYSQL_DATABASE}",
        "",
        f"  [Milvus] (来源: {m_source})",
        f"    URI:      {m_uri}",
        f"    Database: {m_db}",
        f"    User:     {m_user or '(anonymous)'}",
        f"    Auth:     {'token' if m_token else ('password' if m_pass else 'none')}",
        "",
        f"  [Embedding] (来源: {emb_source})",
        f"    Model:    {emb_model}",
        "",
        f"  [Reranker] (来源: {rnk_source})",
        f"    Model:    {rnk_model}",
        "",
        "=" * 60,
    ]
    print("\n".join(lines))


def create_doris_engine():
    url = (
        f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD_URL}"
        f"@{DORIS_HOST}:{DORIS_PORT}/information_schema?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def _get_mysql_engine():
    """获取 MySQL 语义层连接引擎（复用）。"""
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def load_agent_databases(agent_id: int | None, metadata_filter: dict | None = None) -> set[str]:
    """从 da_agent_exec_db 加载 Agent 绑定的可执行数据库名集合。

    Args:
        agent_id: Agent ID
        metadata_filter: 通用 KV 过滤，匹配 da_agent_exec_db.meta_json。
            对于每个 key，meta_json 中不含该 key 的行视为公共数据（始终通过）。
    """
    if not agent_id:
        raise RuntimeError("执行 SQL 必须绑定 Agent")
    try:
        eng = _get_mysql_engine()
        with eng.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT database_name, meta_json FROM da_agent_exec_db "
                "WHERE agent_id = :agent_id AND status = 1"
            ), {"agent_id": agent_id}).fetchall()
        eng.dispose()

        if not metadata_filter:
            result = {row[0].lower() for row in rows if row[0]}
        else:
            result = set()
            for row in rows:
                if not row[0]:
                    continue
                try:
                    meta = json.loads(row[1]) if row[1] else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                # 公共数据规则：meta 中不含该 key → 视为公共，始终通过
                match = all(
                    key not in meta or meta[key] == value
                    for key, value in metadata_filter.items()
                )
                if match:
                    result.add(row[0].lower())

        logger.info(f"Agent {agent_id} 授权数据库: {result}" + (f" (filter={metadata_filter})" if metadata_filter else ""))
        return result
    except Exception as e:
        logger.error(f"加载 Agent 授权数据库失败: agent_id={agent_id}, error={e}")
        raise RuntimeError("无法确认 Agent 的数据库授权范围") from e


def load_doris_config(agent_id: int | None) -> dict:
    """
    加载 Agent 绑定的 Doris 连接配置。

    优先级: da_agent_exec_db.resource_id → sys_resource → .env 默认值
    """
    from urllib.parse import quote_plus
    result = {
        "host": DORIS_HOST,
        "port": int(DORIS_PORT),
        "user": DORIS_USER,
        "password": DORIS_PASSWORD,
        "password_url": DORIS_PASSWORD_URL,
        "source": "env",
    }
    if not agent_id:
        return result

    try:
        eng = _get_mysql_engine()
        with eng.connect() as conn:
            # 从 da_agent_exec_db 取第一个启用的 resource_id
            ds_row = conn.execute(text(
                "SELECT resource_id FROM da_agent_exec_db "
                "WHERE agent_id = :agent_id AND status = 1 "
                "ORDER BY sort_order LIMIT 1"
            ), {"agent_id": agent_id}).fetchone()
            if not ds_row:
                eng.dispose()
                return result

            resource_id = ds_row[0]
            # 从 sys_resource 读取连接配置
            res_row = conn.execute(text(
                "SELECT name, config_json FROM sys_resource "
                "WHERE id = :resource_id AND status = 1"
            ), {"resource_id": resource_id}).fetchone()
        eng.dispose()

        if res_row and res_row[1]:
            import json
            resource_name = res_row[0]
            cfg = json.loads(res_row[1])
            pwd = cfg.get("password", "")
            result = {
                "host": cfg.get("host", DORIS_HOST),
                "port": int(cfg.get("port", DORIS_PORT)),
                "user": cfg.get("user", DORIS_USER),
                "password": pwd,
                "password_url": quote_plus(pwd) if pwd else DORIS_PASSWORD_URL,
                "source": f"resource:{resource_name}(id={resource_id})",
            }
            logger.info(f"Doris 配置来自资源: {resource_name}(id={resource_id}) ({result['host']}:{result['port']})")
    except Exception as e:
        logger.warning(f"加载 Doris 资源配置失败 (fallback .env): {e}")

    return result


def create_doris_engine_for_agent(agent_id: int | None = None):
    """根据 Agent 绑定的 Doris 资源创建引擎，fallback 到 .env。"""
    cfg = load_doris_config(agent_id)
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password_url']}"
        f"@{cfg['host']}:{cfg['port']}/information_schema?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def _register_engine_url(agent_id: int):
    """启动时将本机 engine_url 写入 da_agent 表。"""
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        ip = "127.0.0.1"
    port = int(os.getenv("PORT", "9090"))
    engine_url = f"http://{ip}:{port}"
    try:
        from sqlalchemy import create_engine, text
        url = (
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
            f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
        )
        eng = create_engine(url)
        with eng.connect() as conn:
            conn.execute(
                text("UPDATE da_agent SET engine_url = :url WHERE id = :id"),
                {"url": engine_url, "id": agent_id},
            )
            conn.commit()
        eng.dispose()
        logger.info(f"engine_url 已注册: agent_id={agent_id}, url={engine_url}")
    except Exception as e:
        logger.warning(f"engine_url 注册失败 (非致命): {e}")


def create_llm_client(config: AgentRuntimeConfig) -> BaseChatModel:
    """根据 Agent 配置创建 LangChain ChatModel。"""
    from src.retrieval.llm_factory import create_chat_model
    return create_chat_model(config)


def load_agent_config(agent_id: int | None = None) -> AgentRuntimeConfig:
    """加载 Agent 配置，打印配置信息。"""
    global config_loader
    if not config_loader:
        config_loader = AgentConfigLoader()

    config = config_loader.load(agent_id=agent_id)
    config_loader.print_config(config)
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, validator, llm_client, agent_config, query_logger, knowledge_retriever, query_cache

    # 加载 Agent 配置（CONFIG_SOURCE + CONFIG_PROFILE 控制来源）
    if CONFIG_SOURCE == "local":
        agent_config = load_agent_config()
    else:
        profile = CONFIG_PROFILE or os.getenv("DEFAULT_AGENT_ID", "")
        agent_config = load_agent_config(
            agent_id=int(profile) if profile else None
        )

    if CONFIG_SOURCE == "mysql" and not agent_config.agent_id:
        raise SystemExit(
            "启动中止: mysql 配置模式必须通过 CONFIG_PROFILE 或 DEFAULT_AGENT_ID 绑定 Agent"
        )

    # 打印基础设施配置（含 Agent 绑定的 Milvus 信息）
    print_infra_config(agent_config)

    # 注入 Milvus 连接配置（Agent 资源绑定 > .env 默认值）
    from src.retrieval.milvus_store import configure as configure_milvus
    if agent_config.milvus_uri:
        configure_milvus(
            uri=agent_config.milvus_uri,
            db=agent_config.milvus_db,
            user=agent_config.milvus_user,
            password=agent_config.milvus_password,
            token=agent_config.milvus_token,
        )
        logger.info(f"Milvus 配置来自资源绑定: {agent_config.milvus_uri}")
    else:
        logger.info(f"Milvus 配置来自 .env: {MILVUS_URI}")

    # 验证 Milvus 连接（5s 超时，连不上直接退出）
    import socket
    milvus_uri = agent_config.milvus_uri or MILVUS_URI
    try:
        from urllib.parse import urlparse
        parsed = urlparse(milvus_uri)
        m_host = parsed.hostname or "localhost"
        m_port = parsed.port or 19530
        logger.info(f"验证 Milvus 连接 ({m_host}:{m_port})...")
        sock = socket.create_connection((m_host, m_port), timeout=5)
        sock.close()
        logger.info("Milvus 连接成功")
    except (socket.timeout, OSError) as e:
        logger.error(f"Milvus 连接失败: {milvus_uri} — {e}")
        raise SystemExit(f"启动中止: Milvus 不可达 ({milvus_uri})")

    # 初始化 RAG（传入 agent_config 以使用 Agent 级 Embedding/Reranker 配置）
    engine_type = agent_config.engine_type

    # 验证 Doris 连接（仅 NL2SQL / Hybrid 模式需要）
    if engine_type in ("nl2sql", "hybrid"):
        logger.info(f"连接 Doris ({DORIS_HOST}:{DORIS_PORT})...")
        try:
            engine = create_doris_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            logger.info("Doris 连接成功")
        except Exception as e:
            logger.error(f"Doris 连接失败: {e}")
            raise

    if engine_type in ("nl2sql", "hybrid"):
        retriever = SchemaRetriever()
        retriever.initialize(config=agent_config)

    # 初始化 LLM
    llm_client = create_llm_client(agent_config)
    logger.info(f"LLM 已就绪 (provider={agent_config.llm_provider}, model={agent_config.llm_model})")

    # 初始化知识库检索器（knowledge / hybrid 模式）
    if engine_type in ("knowledge", "hybrid"):
        from src.retrieval.embedding import get_embedding
        knowledge_retriever = KnowledgeRetriever(
            get_embedding(),
            agent_id=agent_config.agent_id,
        )
        knowledge_retriever.connect()
        logger.info(f"知识库检索器已就绪 (engine_type={engine_type})")

    # 初始化查询缓存（默认关闭，需在 Agent 配置中开启）
    if agent_config and agent_config.enable_query_cache:
        from src.retrieval.embedding import get_embedding
        query_cache = QueryCache(get_embedding(), ttl=3600, max_size=500)
        logger.info("查询缓存已启用")
    else:
        logger.info("查询缓存已关闭 (enable_query_cache=false)")

    # 初始化 EXPLAIN 校验器
    if engine_type in ("nl2sql", "hybrid"):
        validator = SQLValidator(create_doris_engine())
        logger.info("EXPLAIN 校验器已就绪")

    # 初始化查询日志
    query_logger = QueryLogger()
    logger.info("查询日志已就绪")

    # 自动注册 engine_url 到 da_agent 表
    if CONFIG_SOURCE == "mysql" and CONFIG_PROFILE:
        _register_engine_url(int(CONFIG_PROFILE))

    # 初始化请求参数（注册到 admin DB / 从 admin DB 拉取）
    if CONFIG_SOURCE == "mysql":
        from src.request_params import init_request_params
        param_source = os.getenv("PARAM_SOURCE", "local")
        init_request_params(param_source, agent_id=int(CONFIG_PROFILE) if CONFIG_PROFILE else None)

    logger.info("NL2SQL Data Agent 服务启动完成")

    yield

    logger.info("NL2SQL Data Agent 服务关闭")


# ── FastAPI App ──

app = FastAPI(
    title="NL2SQL Data Agent",
    description="NL2SQL RAG 检索 + LLM 生成 + EXPLAIN 校验",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ──

class QueryMetadata(BaseModel):
    caller: str = Field(default="", description="调用方标识")
    user_id: str = Field(default="", description="外部用户唯一标识")
    user_name: str = Field(default="", description="外部用户显示名")
    trace_id: str = Field(default="", description="链路追踪 ID")
    filter: dict | None = Field(default=None, description="通用 KV 过滤 (如 {\"business\":\"banking\",\"scenario\":\"bi\"}), 用于语义层 + 执行层隔离")
    context: dict | None = Field(default=None, description="场景专属元数据 (如 {\"param_mode\":true, \"placeholder_fields\":[...]})")


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户自然语言问题")
    session_id: str = Field(default="", description="会话 ID，为空则自动生成")
    agent_id: int | None = Field(default=None, description="Agent ID，为空则使用默认配置")
    enable_explain: bool | None = Field(default=None, description="是否启用 EXPLAIN 校验，为空则使用 Agent 配置")
    metadata: QueryMetadata | None = Field(default=None, description="业务元数据（场景/业务线/调用方/追踪ID）")
    history_summary: str = Field(default="", description="上一轮对话摘要（用于多轮上下文压缩），格式: question|||sql")
    expand_info: dict | None = Field(default=None, description="扩展参数，按需覆盖 Agent 配置（如 enable_execute, row_limit 等）")


class QueryResponse(BaseModel):
    session_id: str
    question: str
    sql: str = ""
    raw_answer: str = ""
    matched_tables: list[str] = []
    matched_terms: list[str] = []
    enum_hits: list[dict] = []
    is_success: bool = True
    retry_count: int = 0
    execution_time_ms: int = 0
    error: str = ""
    log_id: int | None = None
    context_summary: str = Field(default="", description="本轮对话摘要，前端缓存后下一轮带回")
    trace: dict | None = Field(default=None, description="详细链路追踪数据，仅调试时返回")
    summary: str = Field(default="", description="LLM 对执行结果的自然语言总结")
    query_result: dict | None = Field(default=None, description="SQL 执行结果: {columns, rows, row_count, truncated}")
    execution_error: str = Field(default="", description="SQL 执行错误信息")
    script: str = Field(default="", description="参数化 SQL 模板（? 占位符），仅 param_mode 时返回")
    placeholder: str = Field(default="", description="占位符字段声明（分号分隔），按 ? 出现顺序对应")


class IndexRebuildRequest(BaseModel):
    force: bool = Field(default=True, description="是否强制重建")
    collections: list[str] = Field(default=[], description="要重建的 collection 类型列表: table/glossary/enum/fewshot，为空则全量重建")


class IndexRebuildResponse(BaseModel):
    status: str
    message: str
    table_count: int = 0


class ConfigReloadRequest(BaseModel):
    agent_id: int | None = Field(default=None, description="Agent ID")


class ConfigReloadResponse(BaseModel):
    status: str
    message: str
    agent_name: str = ""
    config_source: str = ""


class EvalRunRequest(BaseModel):
    run_id: int = Field(..., description="评估运行 ID")
    cases: list[dict] = Field(..., description="评估用例列表，每条含 id, question, expected_sql")


class EvalRunResponse(BaseModel):
    run_id: int
    status: str
    case_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    accuracy: float = 0.0
    duration_ms: int = 0
    results: list[dict] = []


# ── 核心查询逻辑 ──

def run_query(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
) -> dict:
    """
    执行一次完整的 NL2SQL 查询（RAG + LLM + EXPLAIN）。

    Args:
        question: 用户原始问题
        config: Agent 运行时配置
        client: LangChain ChatModel
        history_summary: 上一轮摘要，非空时触发上下文压缩

    Returns:
        dict 包含 sql, raw_answer, matched_tables, matched_terms, enum_hits,
        is_success, retry_count, error, context_summary, trace
    """
    import time as _time

    trace_steps = []
    t_start = _time.monotonic()

    def _elapsed_ms(t0):
        return int((_time.monotonic() - t0) * 1000)

    # 多轮上下文压缩
    effective_question = question
    if history_summary:
        t0 = _time.monotonic()
        from src.retrieval.context_compressor import ContextCompressor
        compressor = ContextCompressor(client, custom_prompt=config.compress_prompt)
        effective_question = compressor.compress(history_summary, question)
        trace_steps.append({
            "step": "context_compress",
            "duration_ms": _elapsed_ms(t0),
            "input": question,
            "output": effective_question,
        })

    # RAG 检索（用压缩后的完整问题）
    t0 = _time.monotonic()
    result = retriever.retrieve(
        effective_question,
        top_k=config.table_search_top_k,
        fewshot_k=config.fewshot_top_k,
        glossary_score_threshold=config.glossary_score_threshold,
        biz_line=biz_line,
        metadata_filter=metadata_filter,
    )
    retrieval_ms = _elapsed_ms(t0)

    matched_tables = [t["table_name"] for t in result.relevant_tables]
    matched_terms = result.matched_terms

    # trace: 术语解析
    trace_steps.append({
        "step": "glossary",
        "duration_ms": retrieval_ms,
        "matched_terms": matched_terms,
        "business_context": result.business_context or "",
    })

    # trace: Schema 检索 + Reranker
    table_details = []
    for t in result.relevant_tables:
        td = {"table_name": t["table_name"]}
        if "rerank_score" in t:
            td["rerank_score"] = round(t["rerank_score"], 4)
        if "doc" in t and "score" in t["doc"]:
            td["search_score"] = round(t["doc"]["score"], 4)
        table_details.append(td)
    trace_steps.append({
        "step": "schema_retrieval",
        "tables": table_details,
        "count": len(matched_tables),
    })

    # trace: Value 匹配
    if result.value_hits:
        trace_steps.append({
            "step": "value_matching",
            "hits": [
                {"table": v.get("table_name", ""), "column": v.get("column_name", ""), "value": f'{v.get("enum_label_cn", "")} → {v.get("sql_value", "")}'}
                for v in result.value_hits[:10]
            ],
            "count": len(result.value_hits),
        })

    # trace: 枚举
    if result.enum_hits:
        trace_steps.append({
            "step": "enum_lookup",
            "hits": [
                {
                    "table_name": e.get("table_name", ""),
                    "column_name": e.get("column_name", ""),
                    "label": e.get("enum_label_cn", ""),
                    "sql_value": e.get("sql_value", ""),
                }
                for e in result.enum_hits[:10]
            ],
            "count": len(result.enum_hits),
        })

    # trace: Few-shot
    fewshot_details = [
        {"id": ex.get("id"), "question": ex.get("question", ""), "sql": ex.get("sql", "")}
        for ex in result.relevant_examples
        if ex.get("id") is not None
    ]
    if fewshot_details:
        trace_steps.append({
            "step": "fewshot",
            "examples": fewshot_details,
            "count": len(fewshot_details),
        })

    # param_mode: 从 metadata.context 读取，替换输出规则生成 ? 占位符 SQL
    _ctx = metadata_context or {}
    _param_mode = _ctx.get("param_mode", False)
    _placeholder_fields = _ctx.get("placeholder_fields", [])

    prompt_text = result.prompt_text
    if _param_mode and _placeholder_fields:
        fields_str = ", ".join(_placeholder_fields)
        param_mode_rules = (
            "【输出要求】\n"
            f"1. 生成参数化 SQL 模板，对以下字段使用 ? 位置占位符（JDBC 风格）：{fields_str}\n"
            "2. 除上述字段外，其他条件使用具体值（枚举码、时间函数等）\n"
            "3. SQL 用 ```sql ``` 包裹\n"
            "4. SQL 之后另起一行输出占位符声明，格式：PLACEHOLDER: field1;field2（按 ? 在 SQL 中出现的顺序，分号分隔）\n"
            "5. 使用 Schema 中的精确列名和表名\n"
            "6. 状态码、类型码等枚举字段使用【枚举映射】中提供的数值\n"
            "7. 时间字段使用 Doris 函数（CURDATE()、DATE_FORMAT()、DATE_TRUNC() 等）\n"
            "8. 如果问题过于模糊导致无法生成精确 SQL，输出：NEED_CLARIFY: <你的澄清问题>"
        )
        prompt_text = re.sub(r"【输出要求】.*", param_mode_rules, prompt_text, flags=re.DOTALL)

    # 构建对话（注入当前日期，避免 LLM 因知识截止而误判年份）
    _system_content = f"{config.system_prompt}\n\n当前日期: {_date.today().isoformat()}"
    messages = [
        {"role": "system", "content": _system_content},
        {"role": "user", "content": prompt_text},
    ]

    def _to_lc_messages(msgs):
        """dict 列表 → LangChain Message 列表。"""
        _map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        return [_map.get(m["role"], HumanMessage)(content=m["content"]) for m in msgs]

    def _invoke(msgs):
        """调用 LLM 并返回 (answer, usage_metadata)。"""
        resp = client.invoke(_to_lc_messages(msgs))
        return resp.content, getattr(resp, "usage_metadata", None)

    # 调用 LLM
    llm_calls = []
    t0 = _time.monotonic()
    answer, usage = _invoke(messages)
    llm_calls.append({
        "role": "initial",
        "duration_ms": _elapsed_ms(t0),
        "model": config.llm_model,
        "output": answer,
        "input_tokens": usage.get("input_tokens") if usage else None,
        "output_tokens": usage.get("output_tokens") if usage else None,
    })
    messages.append({"role": "assistant", "content": answer})

    # 检测 NEED_CLARIFY
    clarify_msg = SQLValidator.extract_clarify(answer)

    # 提取 SQL
    extracted_sql = SQLValidator.extract_sql(answer) if not clarify_msg else None
    retry_count = 0
    is_success = True
    error_msg = ""

    if not extracted_sql and not clarify_msg:
        messages.append({
            "role": "user",
            "content": "你没有生成 SQL，请根据上面的 Schema 生成可执行的 Doris SQL，用 ```sql ``` 包裹。",
        })
        t0 = _time.monotonic()
        answer, usage = _invoke(messages)
        llm_calls.append({
            "role": "retry_no_sql",
            "duration_ms": _elapsed_ms(t0),
            "model": config.llm_model,
            "output": answer,
            "input_tokens": usage.get("input_tokens") if usage else None,
            "output_tokens": usage.get("output_tokens") if usage else None,
        })
        messages.append({"role": "assistant", "content": answer})
        extracted_sql = SQLValidator.extract_sql(answer)

    # EXPLAIN 校验（param_mode 下跳过，? 占位符无法通过 EXPLAIN）
    explain_details = []
    if extracted_sql and config.enable_explain and validator and not _param_mode:
        syntax_ok = False
        check = None
        for attempt in range(config.max_fix_retries):
            check = validator.validate(answer)
            explain_details.append({
                "attempt": attempt + 1,
                "valid": check["valid"],
                "error": check.get("error", ""),
            })
            if check["valid"]:
                syntax_ok = True
                break

            retry_count = attempt + 1
            logger.info(f"EXPLAIN 失败 (第 {retry_count}/{config.max_fix_retries} 次): {check['error']}")

            if attempt < config.max_fix_retries - 1:
                messages.append({"role": "user", "content":
                    f"你生成的 SQL 执行 EXPLAIN 校验失败。\n\n"
                    f"## EXPLAIN 报错\n{check['error']}\n\n"
                    f"请分析错误原因（1-2句），然后输出修复后的 SQL，用 ```sql ``` 包裹。"
                })
                t0 = _time.monotonic()
                answer, usage = _invoke(messages)
                llm_calls.append({
                    "role": f"explain_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                })
                messages.append({"role": "assistant", "content": answer})

        if not syntax_ok:
            is_success = False
            error_msg = check["error"] if check else "EXPLAIN 校验失败"

        # 执行计划分析
        if syntax_ok and check and check.get("plan"):
            messages.append({"role": "user", "content":
                f"请分析这条 SQL 的 EXPLAIN 执行计划，判断是否有明显性能问题。\n\n"
                f"## EXPLAIN 执行计划\n```\n{check['plan']}\n```\n\n"
                f"关注：笛卡尔积、扫描行数过大、缺少分区裁剪、JOIN 顺序。\n"
                f"如果有优化空间，输出优化后的 SQL，用 ```sql ``` 包裹。如果没问题，只回复：LGTM"
            })
            t0 = _time.monotonic()
            review_result, usage = _invoke(messages)
            llm_calls.append({
                "role": "plan_review",
                "duration_ms": _elapsed_ms(t0),
                "model": config.llm_model,
                "output": review_result,
                "input_tokens": usage.get("input_tokens") if usage else None,
                "output_tokens": usage.get("output_tokens") if usage else None,
            })
            if "LGTM" not in review_result.upper():
                recheck = validator.validate(review_result)
                if recheck["valid"]:
                    answer = review_result

    # trace: LLM 调用
    trace_steps.append({
        "step": "llm_generation",
        "calls": llm_calls,
        "total_calls": len(llm_calls),
    })

    # trace: EXPLAIN 校验
    if explain_details:
        trace_steps.append({
            "step": "explain_validate",
            "attempts": explain_details,
            "final_valid": is_success,
        })

    final_sql = SQLValidator.extract_sql(answer) or ""

    if final_sql:
        read_only, read_only_error = SQLValidator.validate_read_only(final_sql)
        if not read_only:
            is_success = False
            error_msg = read_only_error
            trace_steps.append({
                "step": "sql_read_only_guard",
                "success": False,
                "error": read_only_error,
            })

    # NEED_CLARIFY: 模型认为问题模糊，需要澄清
    if not final_sql and clarify_msg:
        is_success = False
        error_msg = f"NEED_CLARIFY: {clarify_msg}"

    # ── P2: 枚举值预校验（param_mode 跳过）──
    if final_sql and is_success and config.enable_enum_validate and result.enum_hits and not _param_mode:
        where_values = SQLValidator.extract_where_values(final_sql)
        if where_values:
            enum_mismatches = SQLValidator.validate_enum_values(where_values, result.enum_hits)
            if enum_mismatches:
                t0 = _time.monotonic()
                mismatch_lines = []
                for mm in enum_mismatches:
                    line = f"- `{mm['column']} = '{mm['sql_value']}'`：该字段的枚举值为 {mm['expected_values']}"
                    if mm.get("suggestion"):
                        line += f"（{mm['suggestion']}）"
                    mismatch_lines.append(line)
                enum_fix_prompt = (
                    f"你生成的 SQL 中以下条件值可能有误：\n\n"
                    + "\n".join(mismatch_lines) + "\n\n"
                    f"请根据枚举定义修正 SQL，用 ```sql ``` 包裹。"
                )
                messages.append({"role": "user", "content": enum_fix_prompt})
                enum_fix_answer, enum_fix_usage = _invoke(messages)
                llm_calls.append({
                    "role": "enum_fix",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": enum_fix_answer,
                    "input_tokens": enum_fix_usage.get("input_tokens") if enum_fix_usage else None,
                    "output_tokens": enum_fix_usage.get("output_tokens") if enum_fix_usage else None,
                })
                messages.append({"role": "assistant", "content": enum_fix_answer})
                fixed_sql = SQLValidator.extract_sql(enum_fix_answer)
                enum_fixed = False
                if fixed_sql and validator:
                    recheck = validator.explain(fixed_sql)
                    if recheck["valid"]:
                        final_sql = fixed_sql
                        answer = enum_fix_answer
                        enum_fixed = True
                trace_steps.append({
                    "step": "enum_validate",
                    "mismatches": enum_mismatches,
                    "fixed": enum_fixed,
                    "duration_ms": _elapsed_ms(t0),
                })

    # ── param_mode: 提取 script + placeholder，跳过执行 ──
    script_text = ""
    placeholder_text = ""
    if _param_mode and _placeholder_fields and final_sql:
        script_text = final_sql
        placeholder_text = SQLValidator.extract_placeholder(answer) or ";".join(_placeholder_fields)
        trace_steps.append({
            "step": "param_mode",
            "script": script_text,
            "placeholder": placeholder_text,
        })

    # SQL 执行 & 结果总结（param_mode 下跳过执行，? 占位符无法直接执行）
    query_result_data = None
    execution_error = ""
    summary = ""

    if final_sql and is_success and config.enable_execute and not _param_mode and validator:
        # 授权信息无法确认时必须拒绝执行，不能回退到环境变量权限。
        referenced_dbs: set[str] = set()
        authorized_dbs: set[str] = set()
        try:
            authorized_dbs = load_agent_databases(
                config.agent_id,
                metadata_filter=metadata_filter,
            )
            access_allowed, access_error, referenced_dbs = (
                SQLValidator.validate_database_access(final_sql, authorized_dbs)
            )
            if not access_allowed:
                execution_error = f"安全拦截: {access_error}"
        except RuntimeError as e:
            execution_error = f"安全拦截: {e}"

        if execution_error:
            logger.warning(
                f"数据库授权校验拒绝执行: agent_id={config.agent_id}, "
                f"referenced={referenced_dbs}, authorized={authorized_dbs}, "
                f"reason={execution_error}"
            )
            trace_steps.append({
                "step": "sql_execution",
                "success": False,
                "error": execution_error,
                "database": ", ".join(sorted(referenced_dbs)),
                "duration_ms": 0,
            })

        if not execution_error:
            t0 = _time.monotonic()
            # 使用 Agent 绑定的 Doris 连接执行 SQL（可能与 EXPLAIN 引擎不同）
            exec_engine = create_doris_engine_for_agent(config.agent_id)
            exec_validator = SQLValidator(exec_engine)
            current_exec_sql = final_sql
            try:
                exec_result = exec_validator.execute(
                    current_exec_sql,
                    row_limit=config.execute_row_limit,
                    timeout=config.execute_timeout,
                )
            finally:
                exec_engine.dispose()
            exec_ms = _elapsed_ms(t0)

            trace_steps.append({
                "step": "sql_execution",
                "success": exec_result["success"],
                "row_count": exec_result["row_count"],
                "truncated": exec_result.get("truncated", False),
                "error": exec_result.get("error", ""),
                "database": ", ".join(referenced_dbs) if referenced_dbs else "",
                "duration_ms": exec_ms,
            })

            # ── P4: 超时降级 ──
            _is_timeout = (not exec_result["success"]
                           and exec_result.get("error")
                           and re.search(r"(?:timeout|timed?\s*out|query_timeout)", exec_result["error"], re.IGNORECASE))
            if _is_timeout and config.enable_timeout_fallback:
                for level in (1, 2, 3):
                    t0 = _time.monotonic()
                    simplified = SQLValidator.simplify_sql_for_timeout(current_exec_sql, level)
                    if simplified is None:
                        # level 3: LLM 简化
                        timeout_prompt = (
                            f"以下 SQL 执行超时（{config.execute_timeout}s）：\n\n"
                            f"```sql\n{current_exec_sql}\n```\n\n"
                            f"请简化查询以提高性能（减少 JOIN、去掉子查询、缩小时间范围等）。\n"
                            f"输出简化后的 SQL，用 ```sql ``` 包裹。保持查询意图不变。"
                        )
                        messages.append({"role": "user", "content": timeout_prompt})
                        timeout_answer, t_usage = _invoke(messages)
                        llm_calls.append({
                            "role": "timeout_simplify",
                            "duration_ms": _elapsed_ms(t0),
                            "model": config.llm_model,
                            "output": timeout_answer,
                        })
                        messages.append({"role": "assistant", "content": timeout_answer})
                        simplified = SQLValidator.extract_sql(timeout_answer)
                        if not simplified:
                            break

                    # EXPLAIN 校验简化后的 SQL
                    access_allowed, access_error, simplified_dbs = (
                        SQLValidator.validate_database_access(simplified, authorized_dbs)
                    )
                    if not access_allowed:
                        trace_steps.append({
                            "step": "timeout_fallback",
                            "level": level,
                            "simplified_sql": simplified,
                            "success": False,
                            "error": f"安全拦截: {access_error}",
                            "database": ", ".join(sorted(simplified_dbs)),
                            "duration_ms": _elapsed_ms(t0),
                        })
                        break

                    simp_check = validator.explain(simplified)
                    if not simp_check["valid"]:
                        continue

                    exec_engine = create_doris_engine_for_agent(config.agent_id)
                    exec_validator_t = SQLValidator(exec_engine)
                    try:
                        exec_result = exec_validator_t.execute(
                            simplified,
                            row_limit=config.execute_row_limit,
                            timeout=config.execute_timeout,
                        )
                    finally:
                        exec_engine.dispose()

                    trace_steps.append({
                        "step": "timeout_fallback",
                        "level": level,
                        "simplified_sql": simplified,
                        "success": exec_result["success"],
                        "duration_ms": _elapsed_ms(t0),
                    })

                    if exec_result["success"]:
                        final_sql = simplified
                        current_exec_sql = simplified
                        logger.info(f"超时降级成功 (level {level})")
                        break

            # ── P0: 执行失败纠错循环 ──
            if not exec_result["success"] and config.max_execute_fix_retries > 0:
                for exec_fix_i in range(config.max_execute_fix_retries):
                    t0 = _time.monotonic()
                    fix_prompt = (
                        f"你生成的 SQL 执行失败。\n\n"
                        f"## 原始 SQL\n```sql\n{current_exec_sql}\n```\n\n"
                        f"## 执行错误\n{exec_result['error']}\n\n"
                        f"请分析错误原因（1-2句），然后输出修复后的 SQL，用 ```sql ``` 包裹。"
                    )
                    messages.append({"role": "user", "content": fix_prompt})
                    fix_answer, fix_usage = _invoke(messages)
                    llm_calls.append({
                        "role": f"execution_fix_{exec_fix_i + 1}",
                        "duration_ms": _elapsed_ms(t0),
                        "model": config.llm_model,
                        "output": fix_answer,
                        "input_tokens": fix_usage.get("input_tokens") if fix_usage else None,
                        "output_tokens": fix_usage.get("output_tokens") if fix_usage else None,
                    })
                    messages.append({"role": "assistant", "content": fix_answer})

                    new_sql = SQLValidator.extract_sql(fix_answer)
                    if not new_sql:
                        trace_steps.append({"step": "execution_fix", "attempt": exec_fix_i + 1, "success": False, "reason": "无法提取 SQL"})
                        break

                    # 白名单校验
                    access_allowed, access_error, new_dbs = (
                        SQLValidator.validate_database_access(new_sql, authorized_dbs)
                    )
                    if not access_allowed:
                        trace_steps.append({
                            "step": "execution_fix",
                            "attempt": exec_fix_i + 1,
                            "success": False,
                            "reason": f"安全拦截: {access_error}",
                            "database": ", ".join(sorted(new_dbs)),
                        })
                        break

                    # EXPLAIN 校验
                    fix_check = validator.explain(new_sql)
                    if not fix_check["valid"]:
                        trace_steps.append({"step": "execution_fix", "attempt": exec_fix_i + 1, "success": False, "reason": f"EXPLAIN 失败: {fix_check['error']}"})
                        break

                    # 重新执行
                    current_exec_sql = new_sql
                    exec_engine = create_doris_engine_for_agent(config.agent_id)
                    exec_validator_retry = SQLValidator(exec_engine)
                    try:
                        exec_result = exec_validator_retry.execute(
                            current_exec_sql,
                            row_limit=config.execute_row_limit,
                            timeout=config.execute_timeout,
                        )
                    finally:
                        exec_engine.dispose()

                    trace_steps.append({
                        "step": "execution_fix",
                        "attempt": exec_fix_i + 1,
                        "fixed_sql": current_exec_sql,
                        "success": exec_result["success"],
                        "error": exec_result.get("error", ""),
                        "duration_ms": _elapsed_ms(t0),
                    })

                    if exec_result["success"]:
                        final_sql = current_exec_sql
                        answer = fix_answer
                        logger.info(f"执行纠错成功 (第 {exec_fix_i + 1} 次)")
                        break
                else:
                    logger.warning(f"执行纠错 {config.max_execute_fix_retries} 次后仍失败")

            if exec_result["success"]:
                query_result_data = {
                    "columns": exec_result["columns"],
                    "rows": exec_result["rows"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                }

                # ── P1: 空结果智能分析 ──
                if exec_result["row_count"] == 0 and config.enable_empty_analysis:
                    t0 = _time.monotonic()
                    empty_prompt = (
                        f"用户问题：{effective_question}\n\n"
                        f"执行的 SQL：\n```sql\n{final_sql}\n```\n\n"
                        f"查询结果为空（0 行）。请分析可能的原因：\n"
                        f"1. WHERE 条件是否过于严格？（时间范围、状态值、币种等）\n"
                        f"2. 表或字段是否选择正确？\n"
                        f"3. JOIN 条件是否导致数据被过滤？\n\n"
                        f"请给出可能的原因（1-3 条），用简洁的自然语言回答。"
                    )
                    empty_msgs = [
                        {"role": "system", "content": "你是一个数据分析助手，负责分析 SQL 查询结果为空的原因。"},
                        {"role": "user", "content": empty_prompt},
                    ]
                    try:
                        summary, e_usage = _invoke(empty_msgs)
                        trace_steps.append({
                            "step": "empty_analysis",
                            "duration_ms": _elapsed_ms(t0),
                            "output": summary,
                        })
                    except Exception as e:
                        logger.warning(f"空结果分析失败: {e}")

                # ── P3: 结果合理性检验 + LLM 总结 ──
                elif config.enable_summarize and exec_result["row_count"] > 0:
                    t0 = _time.monotonic()
                    summary_rows = exec_result["rows"][:50]
                    result_text = " | ".join(exec_result["columns"]) + "\n"
                    for row in summary_rows:
                        result_text += " | ".join(str(c) for c in row) + "\n"
                    if len(exec_result["rows"]) > 50:
                        result_text += f"... (共 {exec_result['row_count']} 行，仅展示前 50 行)\n"

                    # 规则型预检
                    result_warnings = []
                    if config.enable_result_check:
                        result_warnings = SQLValidator.check_result_anomalies(
                            effective_question, final_sql,
                            exec_result["columns"], exec_result["rows"],
                        )

                    # 构建总结 Prompt（含合理性审查指令）
                    if config.enable_result_check:
                        summarize_prompt = (
                            f"用户问题：{effective_question}\n\n"
                            f"执行的 SQL：\n```sql\n{final_sql}\n```\n\n"
                            f"查询结果：\n{result_text}\n\n"
                        )
                        if result_warnings:
                            summarize_prompt += f"规则预检发现以下异常：\n" + "\n".join(f"- {w}" for w in result_warnings) + "\n\n"
                        summarize_prompt += (
                            f"请完成两项任务：\n\n"
                            f"**1. 合理性审查**\n"
                            f"检查：数值是否合理、时间范围是否与问题一致、结果行数是否匹配查询意图、是否有异常。\n"
                            f"如发现异常，在回答开头用 [数据提示] 标注。\n\n"
                            f"**2. 结果总结**\n"
                            f"用简洁的自然语言回答用户的问题。如果数据量较多，概括关键数据点。不要重复 SQL。"
                        )
                    else:
                        summarize_prompt = (
                            f"用户问题：{effective_question}\n\n"
                            f"执行的 SQL：\n```sql\n{final_sql}\n```\n\n"
                            f"查询结果：\n{result_text}\n\n"
                            f"请用简洁的自然语言总结以上查询结果，直接回答用户的问题。"
                            f"如果数据量较多，概括关键数据点。不要重复 SQL。"
                        )

                    summarize_msgs = [
                        {"role": "system", "content": "你是一个数据分析助手，负责将 SQL 查询结果转化为用户易懂的自然语言回答。"},
                        {"role": "user", "content": summarize_prompt},
                    ]
                    try:
                        summary, s_usage = _invoke(summarize_msgs)
                        summarize_ms = _elapsed_ms(t0)
                        trace_data = {
                            "step": "result_summarize",
                            "duration_ms": summarize_ms,
                            "output": summary,
                            "tokens": (s_usage.get("input_tokens", 0) + s_usage.get("output_tokens", 0)) if s_usage else 0,
                        }
                        if result_warnings:
                            trace_data["result_warnings"] = result_warnings
                        trace_steps.append(trace_data)
                    except Exception as e:
                        logger.warning(f"结果总结失败: {e}")
                        summary = ""
            else:
                execution_error = exec_result["error"] or "SQL 执行失败"

    # 提取命中的 fewshot 信息（id + question）
    matched_fewshot = [
        {"id": ex.get("id"), "question": ex.get("question", "")}
        for ex in result.relevant_examples
        if ex.get("id") is not None
    ]

    # 构建本轮摘要供下一轮使用（仅成功时更新，失败轮不污染上下文）
    from src.retrieval.context_compressor import ContextCompressor
    context_summary = ""
    if is_success and final_sql:
        context_summary = ContextCompressor.build_summary(effective_question, matched_tables)

    # 汇总 trace
    trace = {
        "question": question,
        "effective_question": effective_question,
        "steps": trace_steps,
        "total_duration_ms": _elapsed_ms(t_start),
    }

    return {
        "sql": final_sql,
        "raw_answer": answer,
        "matched_tables": matched_tables,
        "matched_terms": matched_terms,
        "enum_hits": result.enum_hits,
        "is_success": is_success,
        "retry_count": retry_count,
        "error": error_msg,
        "matched_fewshot": matched_fewshot,
        "context_summary": context_summary,
        "trace": trace,
        "summary": summary,
        "query_result": query_result_data,
        "execution_error": execution_error,
        "script": script_text,
        "placeholder": placeholder_text,
    }


# ── 接口 ──

@app.get("/health")
async def health():
    et = agent_config.engine_type if agent_config else "nl2sql"
    return {
        "status": "ok",
        "initialized": (retriever is not None and retriever._initialized) or (knowledge_retriever is not None and knowledge_retriever._initialized),
        "agent": agent_config.agent_name if agent_config and agent_config.agent_id else None,
        "config_source": agent_config.config_source if agent_config else "none",
        "engine_type": et,
        "capabilities": {
            "nl2sql": et in ("nl2sql", "hybrid"),
            "knowledge_qa": knowledge_retriever is not None and knowledge_retriever._initialized,
            "intent_classification": et == "hybrid",
            "explain_validate": validator is not None,
            "sql_execution": agent_config.enable_execute if agent_config else False,
            "query_cache": query_cache is not None,
        },
    }


def _extract_bearer_token(request: Request) -> str:
    """从 Authorization header 提取 Bearer token。"""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return auth


def _verify_token(request: Request, config: AgentRuntimeConfig):
    """校验请求 token。token 始终强制校验（Agent 未单独配置时已 fallback 到默认值）。"""
    expected = config.token
    if not expected:
        raise HTTPException(status_code=401, detail="Agent token not configured")
    token = _extract_bearer_token(request)
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _verify_admin_token(request: Request):
    """校验管理接口 token，使用 DEFAULT_AGENT_TOKEN。"""
    if not DEFAULT_AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token not configured")
    token = _extract_bearer_token(request)
    if not secrets.compare_digest(token, DEFAULT_AGENT_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


# ── 知识问答管道 ──

async def _handle_knowledge_query(
    req: QueryRequest,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    session_id: str,
    start_time: float,
) -> QueryResponse:
    """处理知识问答意图的查询。"""
    from langchain_core.messages import SystemMessage, HumanMessage

    if not knowledge_retriever or not knowledge_retriever._initialized:
        raise HTTPException(status_code=503, detail="知识库未初始化，请先执行 /admin/sync")

    try:
        t0 = time.time()
        trace_steps = []

        # 1. 检索相关知识 chunk
        chunks = knowledge_retriever.retrieve(req.question, top_k=5)
        trace_steps.append({
            "step": "knowledge_retrieval",
            "duration_ms": int((time.time() - t0) * 1000),
            "chunk_count": len(chunks),
            "sources": [{"title": c["title"], "kb_name": c["kb_name"], "score": c["score"]} for c in chunks],
        })

        # 2. 组装 Prompt 并调用 LLM
        prompt_text = knowledge_retriever.format_prompt(req.question, chunks)
        system_prompt = "你是一个知识问答助手，请根据提供的参考文档准确回答用户问题。回答时引用来源。"
        if config.system_prompt and "知识" in config.system_prompt:
            system_prompt = config.system_prompt

        t0 = time.time()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)]
        resp = client.invoke(messages)
        answer = resp.content
        usage = getattr(resp, "usage_metadata", None)

        trace_steps.append({
            "step": "llm_generation",
            "duration_ms": int((time.time() - t0) * 1000),
            "model": config.llm_model,
            "input_tokens": usage.get("input_tokens") if usage else None,
            "output_tokens": usage.get("output_tokens") if usage else None,
        })

        elapsed_ms = int((time.time() - start_time) * 1000)
        source_docs = list({c["title"] for c in chunks if c.get("title")})

        # 记录查询日志
        log_id = None
        if query_logger:
            meta = req.metadata or QueryMetadata()
            _kf = meta.filter or {}
            log_id = query_logger.log(
                session_id=session_id,
                user_query=req.question,
                is_success=True,
                execution_time_ms=elapsed_ms,
                agent_id=config.agent_id,
                scenario=_kf.get("scenario", ""),
                business=_kf.get("business", ""),
                caller=meta.caller,
                user_id=meta.user_id,
                user_name=meta.user_name,
                trace_id=meta.trace_id,
            )

        return QueryResponse(
            session_id=session_id,
            question=req.question,
            sql="",
            raw_answer=answer,
            matched_tables=source_docs,
            is_success=True,
            execution_time_ms=elapsed_ms,
            log_id=log_id,
            summary=answer,
            trace={
                "question": req.question,
                "intent": "knowledge",
                "steps": trace_steps,
                "total_duration_ms": elapsed_ms,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"知识问答异常: {e}", exc_info=True)
        return QueryResponse(
            session_id=session_id,
            question=req.question,
            is_success=False,
            execution_time_ms=elapsed_ms,
            error=str(e),
        )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    start_time = time.time()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    if not agent_config or not llm_client:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")

    if req.agent_id is not None and req.agent_id != agent_config.agent_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"当前引擎绑定 Agent {agent_config.agent_id}，"
                f"不能处理 Agent {req.agent_id} 的请求"
            ),
        )

    config = agent_config
    client = llm_client

    # Token 鉴权
    _verify_token(request, config)

    # 意图分类
    intent = intent_classifier.classify(req.question, engine_type=config.engine_type)
    logger.info(f"意图分类: intent={intent}, engine_type={config.engine_type}")

    # 知识问答管道
    if intent == "knowledge":
        return await _handle_knowledge_query(req, config, client, session_id, start_time)

    # SQL 管道（原有逻辑）
    if not retriever or not retriever._initialized:
        raise HTTPException(status_code=503, detail="服务未就绪，NL2SQL RAG 尚未初始化")

    meta = req.metadata or QueryMetadata()

    # 所有会影响回答的请求上下文都必须参与缓存隔离。
    _filter = meta.filter or {}
    _cache_ctx = json.dumps(
        {
            "agent_id": config.agent_id,
            "filter": _filter,
            "history_summary": req.history_summary,
            "metadata_context": meta.context or {},
            "expand_info": req.expand_info or {},
            "enable_explain": req.enable_explain,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if query_cache:
        cached = query_cache.get(req.question, context_key=_cache_ctx)
        if cached:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return QueryResponse(
                session_id=session_id,
                question=req.question,
                execution_time_ms=elapsed_ms,
                **{k: v for k, v in cached.items() if k in QueryResponse.model_fields},
            )

    # 请求级配置覆盖
    overrides = {}
    if req.enable_explain is not None:
        overrides["enable_explain"] = req.enable_explain
    if req.expand_info:
        from src.request_params import ALLOWED_KEYS
        from src.retrieval.agent_config import get_expand
        for key in req.expand_info:
            if key in ALLOWED_KEYS and hasattr(config, key):
                field_type = type(getattr(config, key))
                cast = field_type if field_type in (int, float, bool) else None
                val = get_expand(req.expand_info, key, default=getattr(config, key), cast=cast)
                overrides[key] = val
    if overrides:
        config = AgentRuntimeConfig(**{**config.__dict__, **overrides})

    try:
        result = run_query(
            req.question, config, client,
            history_summary=req.history_summary,
            biz_line=_filter.get("business", ""),
            metadata_filter=_filter or None,
            metadata_context=meta.context,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 写入查询缓存（仅成功的 SQL 查询）
        if query_cache and result["is_success"] and result["sql"]:
            query_cache.put(req.question, {
                "sql": result["sql"],
                "raw_answer": result["raw_answer"],
                "matched_tables": result["matched_tables"],
                "matched_terms": result["matched_terms"],
                "enum_hits": result["enum_hits"],
                "is_success": True,
                "retry_count": result["retry_count"],
                "error": "",
                "context_summary": result.get("context_summary", ""),
                "summary": result.get("summary", ""),
                "query_result": result.get("query_result"),
                "execution_error": result.get("execution_error", ""),
                "script": result.get("script", ""),
                "placeholder": result.get("placeholder", ""),
            }, context_key=_cache_ctx)

        # 记录查询日志
        log_id = None
        if query_logger:
            log_id = query_logger.log(
                session_id=session_id,
                user_query=req.question,
                matched_tables=result["matched_tables"],
                matched_terms=result["matched_terms"],
                generated_sql=result["sql"],
                is_success=result["is_success"],
                execution_time_ms=elapsed_ms,
                retry_count=result["retry_count"],
                agent_id=config.agent_id,
                scenario=_filter.get("scenario", ""),
                business=_filter.get("business", ""),
                caller=meta.caller,
                user_id=meta.user_id,
                user_name=meta.user_name,
                trace_id=meta.trace_id,
                matched_fewshot=result.get("matched_fewshot"),
                enum_hits=result.get("enum_hits"),
                trace_detail=result.get("trace"),
            )

        return QueryResponse(
            session_id=session_id,
            question=req.question,
            sql=result["sql"],
            raw_answer=result["raw_answer"],
            matched_tables=result["matched_tables"],
            matched_terms=result["matched_terms"],
            enum_hits=result["enum_hits"],
            is_success=result["is_success"],
            retry_count=result["retry_count"],
            execution_time_ms=elapsed_ms,
            error=result["error"],
            log_id=log_id,
            context_summary=result.get("context_summary", ""),
            trace=result.get("trace"),
            summary=result.get("summary", ""),
            query_result=result.get("query_result"),
            execution_error=result.get("execution_error", ""),
            script=result.get("script", ""),
            placeholder=result.get("placeholder", ""),
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"查询处理异常: {e}", exc_info=True)

        # 异常也记录日志
        log_id = None
        meta = req.metadata or QueryMetadata()
        _ef = meta.filter or {}
        if query_logger:
            log_id = query_logger.log(
                session_id=session_id,
                user_query=req.question,
                is_success=False,
                execution_time_ms=elapsed_ms,
                agent_id=config.agent_id,
                scenario=_ef.get("scenario", ""),
                business=_ef.get("business", ""),
                caller=meta.caller,
                user_id=meta.user_id,
                user_name=meta.user_name,
                trace_id=meta.trace_id,
            )

        return QueryResponse(
            session_id=session_id,
            question=req.question,
            is_success=False,
            execution_time_ms=elapsed_ms,
            error=str(e),
            log_id=log_id,
        )


@app.post("/admin/index-rebuild", response_model=IndexRebuildResponse)
async def index_rebuild(req: IndexRebuildRequest, request: Request):
    _verify_admin_token(request)
    if not retriever:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        if req.collections:
            logger.info(f"收到局部索引重建请求: {req.collections}")
            table_count = retriever.rebuild_partial(req.collections)
            rebuilt = ", ".join(req.collections)
            return IndexRebuildResponse(
                status="success",
                message=f"局部索引重建完成 [{rebuilt}]，共 {table_count} 张表",
                table_count=table_count,
            )
        else:
            logger.info("收到全量索引重建请求，开始重建...")
            retriever.initialize()
            table_count = len(retriever.table_schemas)
            logger.info(f"索引重建完成: {table_count} 张表")
            return IndexRebuildResponse(
                status="success",
                message=f"索引重建完成，共 {table_count} 张表",
                table_count=table_count,
            )
    except Exception as e:
        logger.error(f"索引重建失败: {e}", exc_info=True)
        return IndexRebuildResponse(status="error", message=str(e))


@app.post("/admin/config-reload", response_model=ConfigReloadResponse)
async def config_reload(req: ConfigReloadRequest, request: Request):
    """重新加载 Agent 配置（不重建索引）。"""
    _verify_admin_token(request)
    global agent_config, llm_client

    try:
        if req.agent_id is not None and agent_config and req.agent_id != agent_config.agent_id:
            raise ValueError(
                f"当前引擎绑定 Agent {agent_config.agent_id}，不能加载 Agent {req.agent_id}"
            )
        target_agent_id = req.agent_id
        if target_agent_id is None and agent_config:
            target_agent_id = agent_config.agent_id
        agent_config = load_agent_config(agent_id=target_agent_id)
        llm_client = create_llm_client(agent_config)

        # 同步更新 retriever 和 searcher 的运行时配置
        if retriever and retriever._initialized:
            retriever.config = agent_config
            if retriever.searcher:
                retriever.searcher.config = agent_config
        if query_cache:
            query_cache.invalidate()

        return ConfigReloadResponse(
            status="success",
            message=f"配置已重新加载 (source={agent_config.config_source})",
            agent_name=agent_config.agent_name,
            config_source=agent_config.config_source,
        )
    except Exception as e:
        logger.error(f"配置重载失败: {e}", exc_info=True)
        return ConfigReloadResponse(status="error", message=str(e))


class KnowledgeSyncRequest(BaseModel):
    agent_id: int | None = Field(default=None, description="Agent ID，为空则同步当前绑定 Agent")


class KnowledgeSyncResponse(BaseModel):
    status: str
    message: str


@app.post("/admin/sync", response_model=KnowledgeSyncResponse)
async def knowledge_sync(req: KnowledgeSyncRequest, request: Request):
    """触发知识库文档同步（分块 + 向量化 → Milvus）。"""
    _verify_admin_token(request)
    global knowledge_retriever

    if req.agent_id is not None and agent_config and req.agent_id != agent_config.agent_id:
        return KnowledgeSyncResponse(
            status="error",
            message=(
                f"当前引擎绑定 Agent {agent_config.agent_id}，"
                f"不能同步 Agent {req.agent_id}"
            ),
        )

    if knowledge_retriever is None:
        from src.retrieval.embedding import get_embedding
        knowledge_retriever = KnowledgeRetriever(
            get_embedding(),
            agent_id=agent_config.agent_id if agent_config else None,
        )

    try:
        agent_id = req.agent_id or (agent_config.agent_id if agent_config else None)
        logger.info(f"收到知识库同步请求: agent_id={agent_id}")
        knowledge_retriever.sync_from_db(agent_id=agent_id)

        # 同步后清空查询缓存
        if query_cache:
            query_cache.invalidate()

        return KnowledgeSyncResponse(
            status="success",
            message=f"知识库同步完成 (agent_id={agent_id})",
        )
    except Exception as e:
        logger.error(f"知识库同步失败: {e}", exc_info=True)
        return KnowledgeSyncResponse(status="error", message=str(e))


@app.post("/evaluation/run", response_model=EvalRunResponse)
async def evaluation_run(req: EvalRunRequest, request: Request):
    """执行评估：逐条运行 case，返回结果。"""
    _verify_admin_token(request)
    global agent_config, llm_client

    if not retriever or not retriever._initialized:
        raise HTTPException(status_code=503, detail="服务未就绪")

    config = agent_config
    client = llm_client
    start_time = time.time()
    results = []
    pass_count = 0
    fail_count = 0

    for case in req.cases:
        case_id = case.get("id", 0)
        question = case.get("question", "")
        expected_sql = case.get("expected_sql", "")

        case_start = time.time()
        try:
            query_result = run_query(question, config, client)

            generated_sql = query_result["sql"]
            # 简单的 SQL 匹配：去除空白后比较
            sql_match = _normalize_sql(generated_sql) == _normalize_sql(expected_sql) if expected_sql else None

            # 执行结果匹配：两条 SQL 都能通过 EXPLAIN 就算执行匹配
            execution_match = query_result["is_success"]

            # 综合评分
            score = 0.0
            if execution_match:
                score += 0.5
            if sql_match:
                score += 0.5

            is_pass = execution_match and (sql_match is None or sql_match)
            if is_pass:
                pass_count += 1
            else:
                fail_count += 1

            results.append({
                "case_id": case_id,
                "generated_sql": generated_sql,
                "execution_match": execution_match,
                "sql_match": sql_match,
                "score": score,
                "error_message": query_result["error"],
            })

        except Exception as e:
            fail_count += 1
            results.append({
                "case_id": case_id,
                "generated_sql": "",
                "execution_match": False,
                "sql_match": False,
                "score": 0.0,
                "error_message": str(e),
            })

    total_ms = int((time.time() - start_time) * 1000)
    case_count = len(req.cases)
    accuracy = pass_count / case_count if case_count > 0 else 0.0

    return EvalRunResponse(
        run_id=req.run_id,
        status="completed",
        case_count=case_count,
        pass_count=pass_count,
        fail_count=fail_count,
        accuracy=round(accuracy, 4),
        duration_ms=total_ms,
        results=results,
    )


def _normalize_sql(sql: str) -> str:
    """标准化 SQL 用于比较。"""
    import re
    if not sql:
        return ""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.upper()
