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

import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial

import anyio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import (
    CodexStatusResponse,
    CodexTestRequest,
    CodexTestResponse,
    ConfigReloadRequest,
    ConfigReloadResponse,
    EvalRunRequest,
    EvalRunResponse,
    IndexRebuildRequest,
    IndexRebuildResponse,
    PreparedQueryContext,
    QueryMetadata,
    QueryPrepareRequest,
    QueryRequest,
    QueryResponse,
    SavedQueryRequest,
    SavedQueryResponse,
)
from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.config import (
    AGENT_ADMIN_TOKEN,
    CONFIG_PROFILE,
    CONFIG_SOURCE,
    EMBEDDING_MODEL,
    MILVUS_DB,
    MILVUS_PASSWORD,
    MILVUS_TOKEN,
    MILVUS_URI,
    MILVUS_USER,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD_URL,
    MYSQL_PORT,
    MYSQL_USER,
    RERANKER_MODEL,
    validate_startup_config,
)
from src.retrieval.query_cache import QueryCache
from src.retrieval.query_logger import QueryLogger
from src.retrieval.query_pipeline import run_query_pipeline
from src.retrieval.query_preparation import (
    prepare_query_context,
    query_context_fingerprint,
)
from src.retrieval.retriever import SchemaRetriever
from src.retrieval.sql_validator import SQLValidator
from src.runtime.database import (
    AgentDatasourceNotConfiguredError,
    create_database_runtime,
    load_agent_databases,
    load_execution_config,
)
from src.runtime.logging_setup import configure_process_runtime
from src.tools.executor import execute_agent_result_tools
from src.tools.result_snapshot import ResultSnapshotStore

configure_process_runtime()

logger = logging.getLogger(__name__)

# ── 全局状态 ──

retriever: SchemaRetriever | None = None
validator: SQLValidator | None = None
llm_client: BaseChatModel | None = None
agent_config: AgentRuntimeConfig | None = None
query_logger: QueryLogger | None = None
config_loader: AgentConfigLoader | None = None

query_cache: QueryCache | None = None
result_snapshot_store = ResultSnapshotStore(ttl_seconds=1800, max_size=500)
codex_query_semaphore: asyncio.Semaphore | None = None
codex_query_capacity = 0
codex_runtime_swap_lock: asyncio.Lock | None = None


# ── 启动配置打印 ──


def print_infra_config(config: AgentRuntimeConfig | None = None) -> None:
    """打印基础设施配置。传入 agent_config 后会显示实际使用的 Milvus 连接。"""
    # Milvus: 优先 agent_config 资源绑定，fallback 启动配置
    m_uri = config.milvus_uri if config and config.milvus_uri else MILVUS_URI
    m_db = config.milvus_db if config and config.milvus_db else MILVUS_DB
    m_user = config.milvus_user if config and config.milvus_user else MILVUS_USER
    m_pass = (
        config.milvus_password if config and config.milvus_password else MILVUS_PASSWORD
    )
    m_token = config.milvus_token if config and config.milvus_token else MILVUS_TOKEN
    m_source = "资源绑定" if config and config.milvus_uri else "config.yaml"

    # Embedding / Reranker: 优先 agent_config 覆盖值，fallback 启动配置
    emb_model = EMBEDDING_MODEL
    emb_source = "config.yaml"
    if config and config.embedding_config.get("model"):
        emb_model = config.embedding_config["model"]
        emb_source = "Agent 配置"
    rnk_model = RERANKER_MODEL
    rnk_source = "config.yaml"
    if config and config.index_build_config.get("reranker", {}).get("model"):
        rnk_model = config.index_build_config["reranker"]["model"]
        rnk_source = "Agent 配置"

    execution_config = load_execution_config(config.agent_id if config else None)
    if execution_config:
        execution_endpoint = f"{execution_config['host']}:{execution_config['port']}"
        execution_source = str(execution_config["source"])
        execution_type = str(execution_config.get("db_type") or "doris").upper()
    else:
        execution_endpoint = "(尚未绑定)"
        execution_source = "全局数据库资源"
        execution_type = "执行数据库"

    lines = [
        "",
        "=" * 60,
        "  NL2SQL Data Agent 基础设施配置",
        "=" * 60,
        "",
        f"  [{execution_type}] (来源: {execution_source})",
        f"    Host:     {execution_endpoint}",
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


def _initialize_nl2sql_runtime(config: AgentRuntimeConfig) -> None:
    """使用 Agent 引用的全局数据库资源初始化 NL2SQL 运行时。"""
    global retriever, validator

    runtime = create_database_runtime(config.agent_id)
    engine = runtime.engine
    candidate = SchemaRetriever(
        execution_engine=engine,
        dialect=runtime.dialect,
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        candidate.initialize(config=config)
    except Exception:
        engine.dispose()
        raise

    old_validator = validator
    retriever = candidate
    validator = SQLValidator(engine, runtime.dialect)
    if old_validator:
        old_validator.engine.dispose()
    logger.info(
        "%s、Schema 索引与 EXPLAIN 校验器已就绪",
        runtime.dialect.display_name,
    )


def _register_engine_url(agent_id: int):
    """显式配置公共地址时，将 engine_url 写入 da_agent 表。"""
    engine_url = os.getenv("ENGINE_PUBLIC_URL", "").strip().rstrip("/")
    if not engine_url:
        logger.info("未配置 ENGINE_PUBLIC_URL，保留后台绑定的 engine_url")
        return
    if not engine_url.startswith(("http://", "https://")):
        logger.warning("ENGINE_PUBLIC_URL 必须以 http:// 或 https:// 开头，跳过注册")
        return
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
    except (OSError, SQLAlchemyError, ValueError) as e:
        logger.warning(f"engine_url 注册失败 (非致命): {e}")


def create_llm_client(config: AgentRuntimeConfig) -> BaseChatModel:
    """根据 Agent 配置创建 LangChain ChatModel。"""
    from src.retrieval.llm_factory import create_chat_model

    return create_chat_model(config)


def _is_codex_config(config: AgentRuntimeConfig | None) -> bool:
    return bool(config and (config.llm_provider or "").strip().lower() == "codex")


def _new_codex_query_semaphore(
    config: AgentRuntimeConfig,
) -> tuple[asyncio.Semaphore | None, int]:
    if not _is_codex_config(config):
        return None, 0
    return asyncio.Semaphore(config.codex_max_concurrency), config.codex_max_concurrency


def _close_llm_client(client: BaseChatModel | None) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "LLM client cleanup failed", extra={"error_type": type(exc).__name__}
        )


async def _close_llm_after_queries(
    client: BaseChatModel | None,
    semaphore: asyncio.Semaphore | None,
    capacity: int,
) -> None:
    """Wait for every old Codex pipeline slot before closing its SDK client."""
    acquired = 0
    try:
        if semaphore is not None:
            for _ in range(capacity):
                await semaphore.acquire()
                acquired += 1
        await anyio.to_thread.run_sync(_close_llm_client, client)
    finally:
        if semaphore is not None:
            for _ in range(acquired):
                semaphore.release()


async def _run_query_in_worker(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    *,
    semaphore: asyncio.Semaphore | None = None,
    runtime_swap_lock: asyncio.Lock | None = None,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
    prepared_context: dict | None = None,
    previous_query_result: dict | None = None,
) -> dict:
    """Run the complete blocking pipeline outside the FastAPI event loop."""
    call = partial(
        run_query,
        question,
        config,
        client,
        history_summary=history_summary,
        biz_line=biz_line,
        metadata_filter=metadata_filter,
        metadata_context=metadata_context,
        prepared_context=prepared_context,
        previous_query_result=previous_query_result,
    )
    if not _is_codex_config(config):
        return await anyio.to_thread.run_sync(call)

    from src.retrieval.codex_chat_model import (
        CodexModelError,
        CodexTimeoutError,
        codex_query_budget,
        codex_remaining_timeout,
    )

    with codex_query_budget(config.codex_timeout_seconds):
        if semaphore is None:
            return await anyio.to_thread.run_sync(call)
        try:
            if runtime_swap_lock is None:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=codex_remaining_timeout(config.codex_timeout_seconds),
                )
            else:
                async with runtime_swap_lock:
                    if (
                        client is not llm_client
                        or semaphore is not codex_query_semaphore
                    ):
                        raise CodexModelError("Codex 配置刚刚更新，请重新提交查询")
                    await asyncio.wait_for(
                        semaphore.acquire(),
                        timeout=codex_remaining_timeout(config.codex_timeout_seconds),
                    )
        except TimeoutError:
            raise CodexTimeoutError("Codex 查询已超过时间限制，请稍后重试") from None
        try:
            return await anyio.to_thread.run_sync(call)
        finally:
            semaphore.release()


async def _prepare_context_in_worker(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    *,
    semaphore: asyncio.Semaphore | None = None,
    runtime_swap_lock: asyncio.Lock | None = None,
    history_summary: str = "",
    enabled_tools: list[str] | None = None,
) -> dict:
    """Run the one-time turn classifier outside the FastAPI event loop."""
    call = partial(
        prepare_query_context,
        question,
        config,
        client,
        history_summary=history_summary,
        enabled_tools=enabled_tools,
    )
    if not _is_codex_config(config):
        return await anyio.to_thread.run_sync(call)

    from src.retrieval.codex_chat_model import (
        CodexModelError,
        CodexTimeoutError,
        codex_query_budget,
        codex_remaining_timeout,
    )

    with codex_query_budget(config.codex_timeout_seconds):
        if semaphore is None:
            return await anyio.to_thread.run_sync(call)
        try:
            if runtime_swap_lock is None:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=codex_remaining_timeout(config.codex_timeout_seconds),
                )
            else:
                async with runtime_swap_lock:
                    if (
                        client is not llm_client
                        or semaphore is not codex_query_semaphore
                    ):
                        raise CodexModelError("Codex 配置刚刚更新，请重新提交查询")
                    await asyncio.wait_for(
                        semaphore.acquire(),
                        timeout=codex_remaining_timeout(config.codex_timeout_seconds),
                    )
        except TimeoutError:
            raise CodexTimeoutError(
                "Codex 意图识别已超过时间限制，请稍后重试"
            ) from None
        try:
            return await anyio.to_thread.run_sync(call)
        finally:
            semaphore.release()


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
    global llm_client, agent_config, query_logger, query_cache
    global codex_query_semaphore, codex_query_capacity
    global codex_runtime_swap_lock

    validate_startup_config()

    # 加载 Agent 配置（CONFIG_SOURCE + CONFIG_PROFILE 控制来源）
    if CONFIG_SOURCE == "local":
        agent_config = load_agent_config()
    else:
        profile = CONFIG_PROFILE or os.getenv("DEFAULT_AGENT_ID", "")
        agent_config = load_agent_config(agent_id=int(profile) if profile else None)

    if CONFIG_SOURCE == "mysql" and not agent_config.agent_id:
        raise SystemExit(
            "启动中止: mysql 配置模式必须通过 CONFIG_PROFILE 或 DEFAULT_AGENT_ID 绑定 Agent"
        )

    # 打印基础设施配置（含 Agent 绑定的 Milvus 信息）
    print_infra_config(agent_config)

    # 注入 Milvus 连接配置（Agent 资源绑定 > 启动配置默认值）
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
        logger.info(f"Milvus 配置来自 config.yaml: {MILVUS_URI}")

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
    except (TimeoutError, OSError) as e:
        logger.error(f"Milvus 连接失败: {milvus_uri} — {e}")
        raise SystemExit(f"启动中止: Milvus 不可达 ({milvus_uri})")

    # 初始化 NL2SQL RAG（传入 agent_config 以使用 Agent 级模型配置）
    try:
        _initialize_nl2sql_runtime(agent_config)
    except AgentDatasourceNotConfiguredError:
        logger.info(
            f"Agent {agent_config.agent_id} 尚未绑定执行数据库，NL2SQL 以待配置状态启动"
        )

    # 初始化 LLM
    llm_client = create_llm_client(agent_config)
    codex_query_semaphore, codex_query_capacity = _new_codex_query_semaphore(
        agent_config
    )
    codex_runtime_swap_lock = asyncio.Lock()
    logger.info(
        f"LLM 已就绪 (provider={agent_config.llm_provider}, model={agent_config.llm_model})"
    )

    # 初始化查询缓存（默认关闭，需在 Agent 配置中开启）
    if agent_config and agent_config.enable_query_cache:
        from src.retrieval.embedding import get_embedding

        query_cache = QueryCache(get_embedding(), ttl=3600, max_size=500)
        logger.info("查询缓存已启用")
    else:
        logger.info("查询缓存已关闭 (enable_query_cache=false)")

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
        init_request_params(
            param_source, agent_id=int(CONFIG_PROFILE) if CONFIG_PROFILE else None
        )

    logger.info("NL2SQL Data Agent 服务启动完成")

    try:
        yield
    finally:
        await _close_llm_after_queries(
            llm_client, codex_query_semaphore, codex_query_capacity
        )
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


# ── 核心查询逻辑 ──


def _run_query_impl(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
    prepared_context: dict | None = None,
    previous_query_result: dict | None = None,
) -> dict:
    """Adapt process-level runtime dependencies to the query pipeline."""
    return run_query_pipeline(
        question,
        config,
        client,
        retriever=retriever,
        validator=validator,
        history_summary=history_summary,
        biz_line=biz_line,
        metadata_filter=metadata_filter,
        metadata_context=metadata_context,
        prepared_context=prepared_context,
        previous_query_result=previous_query_result,
        result_tool_executor=execute_agent_result_tools,
    )


def run_query(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
    prepared_context: dict | None = None,
    previous_query_result: dict | None = None,
) -> dict:
    """Run one pipeline with a shared cumulative deadline for Codex turns."""
    query_args = (
        question,
        config,
        client,
    )
    query_kwargs = {
        "history_summary": history_summary,
        "biz_line": biz_line,
        "metadata_filter": metadata_filter,
        "metadata_context": metadata_context,
        "prepared_context": prepared_context,
        "previous_query_result": previous_query_result,
    }
    if _is_codex_config(config):
        from src.retrieval.codex_chat_model import codex_query_budget

        with codex_query_budget(config.codex_timeout_seconds):
            return _run_query_impl(*query_args, **query_kwargs)
    return _run_query_impl(*query_args, **query_kwargs)


# ── 接口 ──


@app.get("/health")
async def health(response: Response):
    ready = retriever is not None and retriever._initialized and validator is not None
    if not ready:
        response.status_code = 503
    return {
        "status": "ok",
        "ready": ready,
        "state": "ready" if ready else "not_configured",
        "initialized": ready,
        "agent": agent_config.agent_name
        if agent_config and agent_config.agent_id
        else None,
        "config_source": agent_config.config_source if agent_config else "none",
        "capabilities": {
            "nl2sql": True,
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
    """校验仅供控制面使用的独立管理令牌。"""
    if not AGENT_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token not configured")
    token = _extract_bearer_token(request)
    if not secrets.compare_digest(token, AGENT_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


@app.get(
    "/admin/codex/status",
    response_model=CodexStatusResponse,
    response_model_exclude_none=True,
)
async def codex_status(request: Request) -> CodexStatusResponse:
    """Return live, identity-free status for the configured Codex SDK client."""
    _verify_admin_token(request)
    config = agent_config
    client = llm_client
    if not _is_codex_config(config) or client is None:
        return CodexStatusResponse(
            status="not_configured", message="当前 Agent 未启用 Codex"
        )
    status_reader = getattr(client, "safe_status", None)
    if not callable(status_reader):
        return CodexStatusResponse(status="unavailable", message="Codex 状态检查不可用")
    raw_status = await anyio.to_thread.run_sync(status_reader)
    cli_version = raw_status.get("cli_version")
    models = raw_status.get("models")
    return CodexStatusResponse(
        status=str(raw_status.get("status", "unavailable")),
        message=str(raw_status.get("message", "Codex 暂时不可用")),
        cli_version=cli_version if isinstance(cli_version, str) else None,
        models=[str(model) for model in models] if isinstance(models, list) else None,
    )


def _run_codex_smoke_test(model_name: str) -> None:
    from src.retrieval.codex_chat_model import (
        CodexChatModel,
        CodexModelError,
        codex_query_budget,
    )

    client = CodexChatModel(
        model_name=model_name,
        reasoning_effort="low",
        timeout_seconds=60,
        max_concurrency=1,
    )
    try:
        with codex_query_budget(60):
            response = client.invoke([HumanMessage(content="只回复 OK。")])
        if not str(response.content).strip():
            raise CodexModelError("Codex 未返回有效结果，请稍后重试")
    finally:
        client.close()


@app.post(
    "/admin/codex/test",
    response_model=CodexTestResponse,
    response_model_exclude_none=True,
)
async def codex_test(req: CodexTestRequest, request: Request) -> CodexTestResponse:
    """Run an isolated minimal generation with the selected Codex model."""
    _verify_admin_token(request)
    model_name = req.model.strip()
    start = time.monotonic()
    try:
        await anyio.to_thread.run_sync(partial(_run_codex_smoke_test, model_name))
    except (OSError, RuntimeError, ValueError) as exc:
        from src.retrieval.codex_chat_model import CodexModelError

        latency_ms = int((time.monotonic() - start) * 1000)
        if isinstance(exc, CodexModelError):
            return CodexTestResponse(
                status="error",
                message=str(exc),
                latency_ms=latency_ms,
            )
        logger.error(
            "Codex model test failed",
            extra={"model": model_name, "error_type": type(exc).__name__},
        )
        return CodexTestResponse(
            status="error",
            message="Codex 测试请求无法完成",
            latency_ms=latency_ms,
        )
    return CodexTestResponse(
        status="success",
        message=f"{model_name} 调用成功",
        latency_ms=int((time.monotonic() - start) * 1000),
    )


@app.post("/query/execute-saved", response_model=SavedQueryResponse)
async def execute_saved_query(req: SavedQueryRequest, request: Request):
    """Execute an already validated read-only query without invoking a model."""
    config = agent_config
    if config is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    if req.agent_id is not None and req.agent_id != config.agent_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"当前引擎绑定 Agent {config.agent_id}，"
                f"不能处理 Agent {req.agent_id} 的请求"
            ),
        )
    _verify_token(request, config)

    read_only, read_only_error = SQLValidator.validate_read_only(req.sql)
    if not read_only:
        return SavedQueryResponse(
            is_success=False,
            execution_error=f"安全拦截: {read_only_error}",
        )

    metadata_filter = (req.metadata.filter if req.metadata else None) or {}
    try:
        authorized_dbs = load_agent_databases(
            config.agent_id,
            metadata_filter=metadata_filter,
        )
        access_allowed, access_error, _ = SQLValidator.validate_database_access(
            req.sql,
            authorized_dbs,
        )
    except RuntimeError as exc:
        return SavedQueryResponse(
            is_success=False,
            execution_error=f"安全拦截: {exc}",
        )
    if not access_allowed:
        return SavedQueryResponse(
            is_success=False,
            execution_error=f"安全拦截: {access_error}",
        )

    runtime = create_database_runtime(config.agent_id)
    engine = runtime.engine
    try:
        result = await anyio.to_thread.run_sync(
            lambda: SQLValidator(engine, runtime.dialect).execute(
                req.sql,
                row_limit=req.row_limit,
                timeout=config.execute_timeout,
            )
        )
    finally:
        engine.dispose()
    if not result.get("success"):
        return SavedQueryResponse(
            is_success=False,
            execution_error=str(result.get("error") or "查询执行失败")[:2000],
        )
    return SavedQueryResponse(
        is_success=True,
        query_result={
            "columns": result.get("columns") or [],
            "rows": result.get("rows") or [],
            "row_count": int(result.get("row_count") or 0),
            "truncated": bool(result.get("truncated")),
        },
    )


@app.post("/query/prepare", response_model=PreparedQueryContext)
async def prepare_query(req: QueryPrepareRequest, request: Request):
    """Recognize the turn once so channels can show progress before SQL work."""
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
    _verify_token(request, agent_config)
    try:
        prepared = await _prepare_context_in_worker(
            req.question,
            agent_config,
            llm_client,
            semaphore=(
                codex_query_semaphore if _is_codex_config(agent_config) else None
            ),
            runtime_swap_lock=codex_runtime_swap_lock,
            history_summary=req.history_summary,
            enabled_tools=(req.metadata.context or {}).get("enabled_tools")
            if req.metadata and isinstance(req.metadata.context, dict)
            else None,
        )
        return PreparedQueryContext.model_validate(prepared)
    except Exception as exc:
        logger.exception("查询预处理异常")
        meta = req.metadata or QueryMetadata()
        metadata_filter = meta.filter or {}
        if query_logger:
            prepare_trace_id = f"{meta.trace_id[:120]}:prepare" if meta.trace_id else ""
            failure = {
                "stage": "query_prepare",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            query_logger.log(
                tenant_key=str((meta.context or {}).get("tenant_key") or ""),
                session_id=req.session_id,
                user_query=req.question,
                intent="prepare_failed",
                execution_result=failure,
                is_success=False,
                agent_id=agent_config.agent_id,
                scenario=metadata_filter.get("scenario", ""),
                business=metadata_filter.get("business", ""),
                caller=meta.caller,
                user_id=meta.user_id,
                user_name=meta.user_name,
                trace_id=prepare_trace_id,
                trace_detail=failure,
            )
        raise


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    start_time = time.time()
    session_id = req.session_id or uuid.uuid4().hex

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
    query_limit = codex_query_semaphore if _is_codex_config(config) else None

    # Token 鉴权
    _verify_token(request, config)

    if req.prepared_context and req.prepared_context.input_fingerprint != (
        query_context_fingerprint(req.question, req.history_summary)
    ):
        raise HTTPException(
            status_code=422,
            detail="prepared_context 与当前问题或历史上下文不匹配",
        )

    # NL2SQL 管道
    if not retriever or not retriever._initialized:
        raise HTTPException(status_code=503, detail="服务未就绪，NL2SQL RAG 尚未初始化")

    meta = req.metadata or QueryMetadata()
    is_lark_request = bool(
        query_logger and meta.caller == "lark" and meta.trace_id and meta.user_id
    )

    def attach_result_snapshot(response: QueryResponse) -> QueryResponse:
        if not response.is_success or not isinstance(response.query_result, dict):
            return response
        snapshot_id = result_snapshot_store.put(
            response.query_result,
            session_id=session_id,
            agent_id=config.agent_id,
            user_id=meta.user_id,
            context_summary=response.context_summary,
        )
        return response.model_copy(update={"result_snapshot_id": snapshot_id})

    previous_query_result = None
    if req.previous_result_snapshot_id:
        previous_query_result = result_snapshot_store.get(
            req.previous_result_snapshot_id,
            session_id=session_id,
            agent_id=config.agent_id,
            user_id=meta.user_id,
            context_summary=req.history_summary,
        )

    def replay_lark_response() -> QueryResponse | None:
        if not is_lark_request:
            return None
        replay = query_logger.get_lark_response(
            tenant_key=str((meta.context or {}).get("tenant_key") or ""),
            trace_id=meta.trace_id,
            user_id=meta.user_id,
            agent_id=config.agent_id,
            user_query=req.question,
        )
        if replay is not None:
            replay_log_id, replay_response = replay
            try:
                return attach_result_snapshot(
                    QueryResponse.model_validate(
                        {**replay_response, "log_id": replay_log_id}
                    )
                )
            except ValidationError:
                logger.warning(
                    "忽略无效的 Lark 幂等响应快照",
                    extra={"trace_id": meta.trace_id, "log_id": replay_log_id},
                )
        return None

    replayed_response = replay_lark_response()
    if replayed_response is not None:
        return replayed_response

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
            "previous_result_snapshot_id": req.previous_result_snapshot_id,
            "prepared_context": (
                req.prepared_context.model_dump() if req.prepared_context else None
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if query_cache:
        cached = query_cache.get(req.question, context_key=_cache_ctx)
        if cached:
            elapsed_ms = int((time.time() - start_time) * 1000)
            cached_response = attach_result_snapshot(
                QueryResponse(
                    session_id=session_id,
                    question=req.question,
                    execution_time_ms=elapsed_ms,
                    **{
                        k: v
                        for k, v in cached.items()
                        if k in QueryResponse.model_fields
                    },
                )
            )
            if is_lark_request:
                snapshot = QueryLogger.lark_response_snapshot(
                    cached_response.model_dump(mode="json", exclude={"log_id"})
                )
                log_id = query_logger.log(
                    tenant_key=str((meta.context or {}).get("tenant_key") or ""),
                    session_id=session_id,
                    user_query=req.question,
                    matched_tables=cached_response.matched_tables,
                    matched_terms=cached_response.matched_terms,
                    generated_sql=cached_response.sql,
                    execution_result=snapshot,
                    is_success=cached_response.is_success,
                    execution_time_ms=elapsed_ms,
                    retry_count=cached_response.retry_count,
                    agent_id=config.agent_id,
                    scenario=_filter.get("scenario", ""),
                    business=_filter.get("business", ""),
                    caller=meta.caller,
                    user_id=meta.user_id,
                    user_name=meta.user_name,
                    trace_id=meta.trace_id,
                    enum_hits=cached_response.enum_hits,
                )
                if log_id is None:
                    replayed_response = replay_lark_response()
                    if replayed_response is not None:
                        return replayed_response
                return cached_response.model_copy(update={"log_id": log_id})
            return cached_response

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
                val = get_expand(
                    req.expand_info, key, default=getattr(config, key), cast=cast
                )
                overrides[key] = val
    if overrides:
        config = AgentRuntimeConfig(**{**config.__dict__, **overrides})

    try:
        result = await _run_query_in_worker(
            req.question,
            config,
            client,
            semaphore=query_limit,
            runtime_swap_lock=codex_runtime_swap_lock,
            history_summary=req.history_summary,
            biz_line=_filter.get("business", ""),
            metadata_filter=_filter or None,
            metadata_context=meta.context,
            prepared_context=(
                req.prepared_context.model_dump() if req.prepared_context else None
            ),
            previous_query_result=previous_query_result,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 写入查询缓存（仅成功的 SQL 查询）
        if query_cache and result["is_success"] and result["sql"]:
            query_cache.put(
                req.question,
                {
                    "sql": result["sql"],
                    "raw_answer": result["raw_answer"],
                    "matched_tables": result["matched_tables"],
                    "matched_terms": result["matched_terms"],
                    "enum_hits": result["enum_hits"],
                    "retrieval_context": result.get("retrieval_context", {}),
                    "is_success": True,
                    "retry_count": result["retry_count"],
                    "error": "",
                    "context_summary": result.get("context_summary", ""),
                    "summary": result.get("summary", ""),
                    "query_result": result.get("query_result"),
                    "execution_error": result.get("execution_error", ""),
                    "script": result.get("script", ""),
                    "placeholder": result.get("placeholder", ""),
                    "interpretation": result.get("interpretation", ""),
                    "query_state": result.get("query_state", {}),
                    "turn_intent": result.get("turn_intent", "sql_query"),
                    "context_relation": result.get("context_relation", "new_question"),
                    "interaction_calls": result.get("interaction_calls", []),
                    "tool_calls": result.get("tool_calls", []),
                    "tool_results": result.get("tool_results", []),
                },
                context_key=_cache_ctx,
            )

        response = attach_result_snapshot(
            QueryResponse(
                session_id=session_id,
                question=req.question,
                sql=result["sql"],
                raw_answer=result["raw_answer"],
                matched_tables=result["matched_tables"],
                matched_terms=result["matched_terms"],
                enum_hits=result["enum_hits"],
                retrieval_context=result.get("retrieval_context", {}),
                is_success=result["is_success"],
                retry_count=result["retry_count"],
                execution_time_ms=elapsed_ms,
                error=result["error"],
                context_summary=result.get("context_summary", ""),
                trace=result.get("trace"),
                summary=result.get("summary", ""),
                query_result=result.get("query_result"),
                execution_error=result.get("execution_error", ""),
                script=result.get("script", ""),
                placeholder=result.get("placeholder", ""),
                needs_clarification=result.get("needs_clarification", False),
                clarification=result.get("clarification"),
                interpretation=result.get("interpretation", ""),
                query_state=result.get("query_state", {}),
                turn_intent=result.get("turn_intent", "sql_query"),
                context_relation=result.get("context_relation", "new_question"),
                interaction_calls=result.get("interaction_calls", []),
                tool_calls=result.get("tool_calls", []),
                tool_results=result.get("tool_results", []),
            )
        )

        # Lark 请求把完整、JSON-safe 的响应快照与查询日志一起提交；Admin
        # 在收到响应前退出时，同一 trace_id 可直接重放而不再次执行 SQL。
        log_id = None
        if query_logger:
            execution_result = ""
            if is_lark_request:
                execution_result = QueryLogger.lark_response_snapshot(
                    response.model_dump(mode="json", exclude={"log_id"})
                )
            log_id = query_logger.log(
                tenant_key=str((meta.context or {}).get("tenant_key") or ""),
                session_id=session_id,
                user_query=req.question,
                matched_tables=result["matched_tables"],
                matched_terms=result["matched_terms"],
                generated_sql=result["sql"],
                execution_result=execution_result,
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
            if is_lark_request and log_id is None:
                replayed_response = replay_lark_response()
                if replayed_response is not None:
                    return replayed_response

        return response.model_copy(update={"log_id": log_id})

    except (OSError, RuntimeError, SQLAlchemyError, TypeError, ValueError) as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.exception("查询处理异常")

        error_response = QueryResponse(
            session_id=session_id,
            question=req.question,
            is_success=False,
            execution_time_ms=elapsed_ms,
            error=str(e),
        )

        # 异常也记录日志
        log_id = None
        meta = req.metadata or QueryMetadata()
        _ef = meta.filter or {}
        if query_logger:
            execution_result = ""
            if is_lark_request:
                execution_result = QueryLogger.lark_response_snapshot(
                    error_response.model_dump(mode="json", exclude={"log_id"})
                )
            log_id = query_logger.log(
                tenant_key=str((meta.context or {}).get("tenant_key") or ""),
                session_id=session_id,
                user_query=req.question,
                execution_result=execution_result,
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
            if is_lark_request and log_id is None:
                replayed_response = replay_lark_response()
                if replayed_response is not None:
                    return replayed_response

        return error_response.model_copy(update={"log_id": log_id})


@app.post("/admin/index-rebuild", response_model=IndexRebuildResponse)
async def index_rebuild(req: IndexRebuildRequest, request: Request):
    _verify_admin_token(request)
    if not agent_config:
        raise HTTPException(status_code=503, detail="Agent 配置未就绪")

    try:
        if req.collections:
            if not retriever or not retriever._initialized:
                raise AgentDatasourceNotConfiguredError(
                    "索引尚未初始化，请先执行一次全量重建"
                )
            logger.info(f"收到局部索引重建请求: {req.collections}")
            table_count = retriever.rebuild_partial(req.collections)
            if query_cache:
                query_cache.invalidate()
            rebuilt = ", ".join(req.collections)
            return IndexRebuildResponse(
                status="success",
                message=f"局部索引重建完成 [{rebuilt}]，共 {table_count} 张表",
                table_count=table_count,
            )
        else:
            logger.info("收到全量索引重建请求，开始重建...")
            # 先重新加载 Agent 配置，再真正删除并重建所有 Milvus
            # collection。仅调用 initialize 会在
            # REBUILD_INDEX_ON_STARTUP=false 时复用旧索引，与接口语义不符。
            _initialize_nl2sql_runtime(agent_config)
            table_count = retriever.rebuild_all()
            if query_cache:
                query_cache.invalidate()
            logger.info(f"索引重建完成: {table_count} 张表")
            return IndexRebuildResponse(
                status="success",
                message=f"索引重建完成，共 {table_count} 张表",
                table_count=table_count,
            )
    except AgentDatasourceNotConfiguredError as e:
        logger.info(f"索引重建等待数据源配置: {e}")
        return IndexRebuildResponse(status="error", message=str(e))
    except (OSError, RuntimeError, SQLAlchemyError, TypeError, ValueError) as e:
        logger.exception("索引重建失败")
        return IndexRebuildResponse(status="error", message=str(e))


@app.post("/admin/config-reload", response_model=ConfigReloadResponse)
async def config_reload(req: ConfigReloadRequest, request: Request):
    """重新加载 Agent 配置（不重建索引）。"""
    _verify_admin_token(request)
    global agent_config, llm_client, codex_query_semaphore, codex_query_capacity
    global codex_runtime_swap_lock

    try:
        if (
            req.agent_id is not None
            and agent_config
            and req.agent_id != agent_config.agent_id
        ):
            raise ValueError(
                f"当前引擎绑定 Agent {agent_config.agent_id}，不能加载 Agent {req.agent_id}"
            )
        target_agent_id = req.agent_id
        if target_agent_id is None and agent_config:
            target_agent_id = agent_config.agent_id
        new_config = await anyio.to_thread.run_sync(
            partial(load_agent_config, agent_id=target_agent_id)
        )
        new_client = await anyio.to_thread.run_sync(create_llm_client, new_config)
        new_semaphore, new_capacity = _new_codex_query_semaphore(new_config)

        if codex_runtime_swap_lock is None:
            codex_runtime_swap_lock = asyncio.Lock()
        async with codex_runtime_swap_lock:
            old_client = llm_client
            old_semaphore = codex_query_semaphore
            old_capacity = codex_query_capacity
            agent_config = new_config
            llm_client = new_client
            codex_query_semaphore = new_semaphore
            codex_query_capacity = new_capacity

        try:
            # 同步更新 retriever 和 searcher 的运行时配置
            if retriever and retriever._initialized:
                retriever.config = agent_config
                if retriever.searcher:
                    retriever.searcher.config = agent_config
            if query_cache:
                query_cache.invalidate()
        finally:
            await _close_llm_after_queries(old_client, old_semaphore, old_capacity)

        return ConfigReloadResponse(
            status="success",
            message=f"配置已重新加载 (source={agent_config.config_source})",
            agent_name=agent_config.agent_name,
            config_source=agent_config.config_source,
        )
    except (OSError, RuntimeError, SQLAlchemyError, TypeError, ValueError) as e:
        logger.exception("配置重载失败")
        return ConfigReloadResponse(status="error", message=str(e))


@app.post("/evaluation/run", response_model=EvalRunResponse)
async def evaluation_run(req: EvalRunRequest, request: Request):
    """执行评估：逐条运行 case，返回结果。"""
    _verify_admin_token(request)
    if (
        not retriever
        or not retriever._initialized
        or agent_config is None
        or llm_client is None
    ):
        raise HTTPException(status_code=503, detail="服务未就绪")
    start_time = time.time()
    results = []
    pass_count = 0
    fail_count = 0

    for case in req.cases:
        case_id = case.get("id", 0)
        question = case.get("question", "")
        expected_sql = case.get("expected_sql", "")

        try:
            config = agent_config
            client = llm_client
            if config is None or client is None:
                raise RuntimeError("服务配置正在重载，请稍后重试")
            query_result = await _run_query_in_worker(
                question,
                config,
                client,
                semaphore=codex_query_semaphore if _is_codex_config(config) else None,
                runtime_swap_lock=codex_runtime_swap_lock,
            )

            generated_sql = query_result["sql"]
            # 简单的 SQL 匹配：去除空白后比较
            sql_match = (
                _normalize_sql(generated_sql) == _normalize_sql(expected_sql)
                if expected_sql
                else None
            )

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

            results.append(
                {
                    "case_id": case_id,
                    "generated_sql": generated_sql,
                    "execution_match": execution_match,
                    "sql_match": sql_match,
                    "score": score,
                    "error_message": query_result["error"],
                }
            )

        except (OSError, RuntimeError, SQLAlchemyError, TypeError, ValueError) as e:
            fail_count += 1
            results.append(
                {
                    "case_id": case_id,
                    "generated_sql": "",
                    "execution_match": False,
                    "sql_match": False,
                    "score": 0.0,
                    "error_message": str(e),
                }
            )

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

    if not sql:
        return ""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.upper()
