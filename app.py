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
import time
import logging
import uuid
import os
import signal

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


def load_allowed_databases(agent_id: int | None) -> set[str]:
    """从 da_agent_ref + da_biz_database 加载 Agent 绑定的数据库名白名单。"""
    if not agent_id:
        return set()
    try:
        eng = _get_mysql_engine()
        with eng.connect() as conn:
            rows = conn.execute(text(
                "SELECT b.database_name FROM da_agent_ref r "
                "JOIN da_biz_database b ON r.resource_key = b.id "
                "WHERE r.agent_id = :agent_id AND r.resource_type = 'biz_database'"
            ), {"agent_id": agent_id}).fetchall()
        eng.dispose()
        result = {row[0].lower() for row in rows if row[0]}
        logger.info(f"Agent {agent_id} 数据库白名单: {result}")
        return result
    except Exception as e:
        logger.warning(f"加载数据库白名单失败: {e}")
        return set()


def load_doris_config(agent_id: int | None) -> dict:
    """
    加载 Agent 绑定的 Doris 连接配置。

    优先级: da_agent_ref(type='doris') → sys_resource → .env 默认值
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
            # 查 Agent 绑定的数据库资源（优先 database，兼容旧 doris）
            ref_row = conn.execute(text(
                "SELECT resource_key FROM da_agent_ref "
                "WHERE agent_id = :agent_id AND resource_type IN ('database', 'doris') "
                "ORDER BY sort_order LIMIT 1"
            ), {"agent_id": agent_id}).fetchone()
            if not ref_row:
                eng.dispose()
                return result

            resource_name = ref_row[0]
            # 从 sys_resource 读取连接配置（兼容 database 和旧 doris 类型）
            res_row = conn.execute(text(
                "SELECT config_json FROM sys_resource "
                "WHERE resource_type IN ('database', 'doris') AND name = :name AND status = 1"
            ), {"name": resource_name}).fetchone()
        eng.dispose()

        if res_row and res_row[0]:
            import json
            cfg = json.loads(res_row[0])
            pwd = cfg.get("password", "")
            result = {
                "host": cfg.get("host", DORIS_HOST),
                "port": int(cfg.get("port", DORIS_PORT)),
                "user": cfg.get("user", DORIS_USER),
                "password": pwd,
                "password_url": quote_plus(pwd) if pwd else DORIS_PASSWORD_URL,
                "source": f"resource:{resource_name}",
            }
            logger.info(f"Doris 配置来自资源绑定: {resource_name} ({result['host']}:{result['port']})")
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
    global retriever, validator, llm_client, agent_config, query_logger

    # 加载 Agent 配置（CONFIG_SOURCE + CONFIG_PROFILE 控制来源）
    if CONFIG_SOURCE == "local":
        agent_config = load_agent_config()
    else:
        profile = CONFIG_PROFILE or os.getenv("DEFAULT_AGENT_ID", "")
        agent_config = load_agent_config(
            agent_id=int(profile) if profile else None
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

    # 验证 Doris 连接
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

    # 初始化 RAG（传入 agent_config 以使用 Agent 级 Embedding/Reranker 配置）
    retriever = SchemaRetriever()
    retriever.initialize(config=agent_config)

    # 初始化 LLM
    llm_client = create_llm_client(agent_config)
    logger.info(f"LLM 已就绪 (provider={agent_config.llm_provider}, model={agent_config.llm_model})")

    # 初始化 EXPLAIN 校验器
    validator = SQLValidator(create_doris_engine())
    logger.info("EXPLAIN 校验器已就绪")

    # 初始化查询日志
    query_logger = QueryLogger()
    logger.info("查询日志已就绪")

    # 自动注册 engine_url 到 da_agent 表
    if CONFIG_SOURCE == "mysql" and CONFIG_PROFILE:
        _register_engine_url(int(CONFIG_PROFILE))

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
    scenario: str = Field(default="", description="使用场景: bi/risk/ops/ad-hoc")
    business: str = Field(default="", description="业务线: banking/issuing/acquiring/payment")
    caller: str = Field(default="", description="调用方标识")
    user_id: str = Field(default="", description="外部用户唯一标识")
    user_name: str = Field(default="", description="外部用户显示名")
    trace_id: str = Field(default="", description="链路追踪 ID")


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户自然语言问题")
    session_id: str = Field(default="", description="会话 ID，为空则自动生成")
    agent_id: int | None = Field(default=None, description="Agent ID，为空则使用默认配置")
    enable_explain: bool | None = Field(default=None, description="是否启用 EXPLAIN 校验，为空则使用 Agent 配置")
    metadata: QueryMetadata | None = Field(default=None, description="业务元数据（场景/业务线/调用方/追踪ID）")
    history_summary: str = Field(default="", description="上一轮对话摘要（用于多轮上下文压缩），格式: question|||sql")


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
                {"column": e.get("column", ""), "value": e.get("value", "")}
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

    # 构建对话
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": result.prompt_text},
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

    # EXPLAIN 校验
    explain_details = []
    if extracted_sql and config.enable_explain and validator:
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

    # NEED_CLARIFY: 模型认为问题模糊，需要澄清
    if not final_sql and clarify_msg:
        is_success = False
        error_msg = f"NEED_CLARIFY: {clarify_msg}"

    # SQL 执行 & 结果总结
    query_result_data = None
    execution_error = ""
    summary = ""

    if final_sql and is_success and config.enable_execute and validator:
        # 数据库白名单校验
        referenced_dbs = SQLValidator.extract_databases(final_sql)
        allowed_dbs = load_allowed_databases(config.agent_id)

        if allowed_dbs and referenced_dbs:
            unauthorized = referenced_dbs - allowed_dbs
            if unauthorized:
                execution_error = f"安全拦截: 数据库 {', '.join(unauthorized)} 未在当前 Agent 的授权范围内"
                logger.warning(f"数据库白名单校验失败: {unauthorized}, 允许: {allowed_dbs}")
                trace_steps.append({
                    "step": "sql_execution",
                    "success": False,
                    "error": execution_error,
                    "database": ", ".join(referenced_dbs),
                    "duration_ms": 0,
                })

        if not execution_error:
            t0 = _time.monotonic()
            # 使用 Agent 绑定的 Doris 连接执行 SQL（可能与 EXPLAIN 引擎不同）
            exec_engine = create_doris_engine_for_agent(config.agent_id)
            exec_validator = SQLValidator(exec_engine)
            try:
                exec_result = exec_validator.execute(
                    final_sql,
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

            if exec_result["success"]:
                query_result_data = {
                    "columns": exec_result["columns"],
                    "rows": exec_result["rows"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                }

                # LLM 结果总结
                if config.enable_summarize and exec_result["row_count"] > 0:
                    t0 = _time.monotonic()
                    # 构建结果摘要（限制行数避免 token 过多）
                    summary_rows = exec_result["rows"][:50]
                    result_text = " | ".join(exec_result["columns"]) + "\n"
                    for row in summary_rows:
                        result_text += " | ".join(str(c) for c in row) + "\n"
                    if len(exec_result["rows"]) > 50:
                        result_text += f"... (共 {exec_result['row_count']} 行，仅展示前 50 行)\n"

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
                        trace_steps.append({
                            "step": "result_summarize",
                            "duration_ms": summarize_ms,
                            "output": summary,
                            "tokens": (s_usage.get("input_tokens", 0) + s_usage.get("output_tokens", 0)) if s_usage else 0,
                        })
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
    }


# ── 接口 ──

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "initialized": retriever is not None and retriever._initialized,
        "agent": agent_config.agent_name if agent_config and agent_config.agent_id else None,
        "config_source": agent_config.config_source if agent_config else "none",
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
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _verify_admin_token(request: Request):
    """校验管理接口 token，使用 DEFAULT_AGENT_TOKEN。"""
    if not DEFAULT_AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token not configured")
    token = _extract_bearer_token(request)
    if token != DEFAULT_AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    global agent_config, llm_client

    if not retriever or not retriever._initialized:
        raise HTTPException(status_code=503, detail="服务未就绪，RAG 尚未初始化")

    start_time = time.time()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # 如果请求指定了 agent_id，动态加载该 Agent 的配置
    config = agent_config
    client = llm_client
    if req.agent_id and (not agent_config or req.agent_id != agent_config.agent_id):
        config = load_agent_config(agent_id=req.agent_id)
        client = create_llm_client(config)

    # Token 鉴权
    _verify_token(request, config)

    # enable_explain 可以在请求级别覆盖
    if req.enable_explain is not None:
        config = AgentRuntimeConfig(**{
            **config.__dict__,
            "enable_explain": req.enable_explain,
        })

    try:
        meta = req.metadata or QueryMetadata()
        result = run_query(req.question, config, client, history_summary=req.history_summary, biz_line=meta.business)
        elapsed_ms = int((time.time() - start_time) * 1000)

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
                agent_id=req.agent_id,
                scenario=meta.scenario,
                business=meta.business,
                caller=meta.caller,
                user_id=meta.user_id,
                user_name=meta.user_name,
                trace_id=meta.trace_id,
                matched_fewshot=result.get("matched_fewshot"),
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
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"查询处理异常: {e}", exc_info=True)

        # 异常也记录日志
        log_id = None
        meta = req.metadata or QueryMetadata()
        if query_logger:
            log_id = query_logger.log(
                session_id=session_id,
                user_query=req.question,
                is_success=False,
                execution_time_ms=elapsed_ms,
                agent_id=req.agent_id,
                scenario=meta.scenario,
                business=meta.business,
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
        agent_config = load_agent_config(agent_id=req.agent_id)
        llm_client = create_llm_client(agent_config)

        return ConfigReloadResponse(
            status="success",
            message=f"配置已重新加载 (source={agent_config.config_source})",
            agent_name=agent_config.agent_name,
            config_source=agent_config.config_source,
        )
    except Exception as e:
        logger.error(f"配置重载失败: {e}", exc_info=True)
        return ConfigReloadResponse(status="error", message=str(e))


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
