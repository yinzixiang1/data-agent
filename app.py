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
import gzip
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime as _datetime
from functools import partial
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Literal

import anyio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.config import (
    AGENT_ADMIN_TOKEN,
    CONFIG_PROFILE,
    CONFIG_SOURCE,
    DORIS_HOST,
    DORIS_PASSWORD,
    DORIS_PASSWORD_URL,
    DORIS_PORT,
    DORIS_USER,
    EMBEDDING_MODEL,
    LOG_DIR,
    LOG_LEVEL,
    LOG_RETENTION_DAYS,
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
    PROJECT_ROOT,
    RERANKER_MODEL,
    validate_startup_config,
)
from src.retrieval.query_cache import QueryCache
from src.retrieval.query_logger import QueryLogger
from src.retrieval.retriever import RetrievalResult, SchemaRetriever
from src.retrieval.schema_loader import AgentDatasourceNotConfiguredError
from src.retrieval.sql_validator import SQLValidator
from src.retrieval.tool_planner import (
    declared_action_count,
    explicitly_requested_tools,
    extract_planned_tool_calls,
    tool_planning_messages,
)


# gRPC 后台线程可能阻止默认信号处理完成，服务进程必须直接退出。
def _force_exit(*_: object) -> None:
    os._exit(1)


try:
    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)
except ValueError:
    pass  # 非主线程忽略

# ── 日志配置：控制台 + 文件双写，按天轮转压缩 ──
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

query_cache: QueryCache | None = None
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

    doris_config = load_doris_config(config.agent_id if config else None)
    if doris_config:
        doris_endpoint = f"{doris_config['host']}:{doris_config['port']}"
        doris_source = str(doris_config["source"])
    else:
        doris_endpoint = "(尚未绑定)"
        doris_source = "全局数据库资源"

    lines = [
        "",
        "=" * 60,
        "  NL2SQL Data Agent 基础设施配置",
        "=" * 60,
        "",
        f"  [Doris] (来源: {doris_source})",
        f"    Host:     {doris_endpoint}",
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


def _get_mysql_engine() -> Engine:
    """获取 MySQL 语义层连接引擎（复用）。"""
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def load_agent_databases(
    agent_id: int | None, metadata_filter: dict | None = None
) -> set[str]:
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
            rows = conn.execute(
                text(
                    "SELECT DISTINCT database_name, meta_json FROM da_agent_exec_db "
                    "WHERE agent_id = :agent_id AND status = 1"
                ),
                {"agent_id": agent_id},
            ).fetchall()
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

        logger.info(
            f"Agent {agent_id} 授权数据库: {result}"
            + (f" (filter={metadata_filter})" if metadata_filter else "")
        )
        return result
    except (OSError, SQLAlchemyError, ValueError) as e:
        logger.error(f"加载 Agent 授权数据库失败: agent_id={agent_id}, error={e}")
        raise RuntimeError("无法确认 Agent 的数据库授权范围") from e


def load_doris_config(agent_id: int | None) -> dict[str, str | int] | None:
    """
    加载 Agent 绑定的 Doris 连接配置。

    mysql 配置模式只允许从全局资源读取；local 模式使用启动配置。
    """
    from urllib.parse import quote_plus

    if CONFIG_SOURCE == "local":
        return {
            "host": DORIS_HOST,
            "port": int(DORIS_PORT),
            "user": DORIS_USER,
            "password": DORIS_PASSWORD,
            "password_url": DORIS_PASSWORD_URL,
            "source": "config.yaml (local)",
        }
    if not agent_id:
        return None

    eng = _get_mysql_engine()
    try:
        with eng.connect() as conn:
            # 从 da_agent_exec_db 取第一个启用的 resource_id
            ds_row = conn.execute(
                text(
                    "SELECT resource_id FROM da_agent_exec_db "
                    "WHERE agent_id = :agent_id AND status = 1 "
                    "ORDER BY sort_order LIMIT 1"
                ),
                {"agent_id": agent_id},
            ).fetchone()
            if not ds_row:
                return None

            resource_id = ds_row[0]
            # 从 sys_resource 读取连接配置
            res_row = conn.execute(
                text(
                    "SELECT name, resource_type, config_json FROM sys_resource "
                    "WHERE id = :resource_id AND status = 1"
                ),
                {"resource_id": resource_id},
            ).fetchone()
    finally:
        eng.dispose()

    if not res_row:
        raise RuntimeError(
            f"Agent {agent_id} 绑定的数据库资源 {resource_id} 不存在或未启用"
        )
    resource_name, resource_type, raw_config = res_row
    if resource_type != "database":
        raise RuntimeError(f"资源 {resource_name}(id={resource_id}) 不是数据库资源")

    try:
        cfg = json.loads(raw_config) if raw_config else {}
        host = str(cfg["host"]).strip()
        port = int(cfg["port"])
        user = str(cfg["user"]).strip()
        password = str(cfg.get("password", ""))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"数据库资源 {resource_name}(id={resource_id}) 配置不完整"
        ) from exc
    if not host or not user:
        raise RuntimeError(f"数据库资源 {resource_name}(id={resource_id}) 配置不完整")

    logger.info(
        f"Doris 配置来自全局资源: {resource_name}(id={resource_id}) ({host}:{port})"
    )
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "password_url": quote_plus(password),
        "source": f"全局资源:{resource_name}(id={resource_id})",
    }


def create_doris_engine_for_agent(agent_id: int | None = None) -> Engine:
    """根据 Agent 引用的全局 Doris 资源创建引擎。"""
    cfg = load_doris_config(agent_id)
    if cfg is None:
        raise AgentDatasourceNotConfiguredError(f"Agent {agent_id} 尚未绑定执行数据库")
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password_url']}"
        f"@{cfg['host']}:{cfg['port']}/information_schema?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def _initialize_nl2sql_runtime(config: AgentRuntimeConfig) -> None:
    """使用 Agent 引用的全局数据库资源初始化 NL2SQL 运行时。"""
    global retriever, validator

    engine = create_doris_engine_for_agent(config.agent_id)
    candidate = SchemaRetriever(
        connection_string=engine.url.render_as_string(hide_password=False)
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
    validator = SQLValidator(engine)
    if old_validator:
        old_validator.engine.dispose()
    logger.info("Doris、Schema 索引与 EXPLAIN 校验器已就绪")


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
) -> dict:
    """Run the one-time turn classifier outside the FastAPI event loop."""
    call = partial(
        prepare_query_context,
        question,
        config,
        client,
        history_summary=history_summary,
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


# ── 请求/响应模型 ──


class QueryMetadata(BaseModel):
    caller: str = Field(default="", description="调用方标识")
    user_id: str = Field(default="", description="外部用户唯一标识")
    user_name: str = Field(default="", description="外部用户显示名")
    trace_id: str = Field(default="", description="链路追踪 ID")
    filter: dict | None = Field(
        default=None,
        description='通用 KV 过滤 (如 {"business":"banking","scenario":"bi"}), 用于语义层 + 执行层隔离',
    )
    context: dict | None = Field(
        default=None,
        description='场景专属元数据 (如 {"param_mode":true, "placeholder_fields":[...]})',
    )


class PreparedQueryContext(BaseModel):
    """One-time model output reused by the remaining NL2SQL pipeline."""

    input_fingerprint: str = Field(..., min_length=64, max_length=64)
    effective_question: str = Field(..., min_length=1, max_length=2000)
    query_state: dict = Field(default_factory=dict)
    relation: Literal[
        "new_question",
        "follow_up_add",
        "follow_up_modify",
        "correction_override",
    ] = "new_question"
    turn_intent: Literal[
        "sql_query",
        "result_operation",
        "result_explanation",
        "non_query",
    ] = "sql_query"
    presentation_relation: Literal["inherit", "add", "replace", "clear"] = "clear"
    interpretation: str = Field(default="", max_length=500)
    direct_response: str = Field(default="", max_length=1000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification: dict = Field(default_factory=dict)
    changes: dict = Field(default_factory=dict)
    removed_sql_context: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户自然语言问题")
    session_id: str = Field(
        default="", max_length=32, description="32 位会话 ID，为空则自动生成"
    )
    agent_id: int | None = Field(
        default=None, description="Agent ID，为空则使用默认配置"
    )
    enable_explain: bool | None = Field(
        default=None, description="是否启用 EXPLAIN 校验，为空则使用 Agent 配置"
    )
    metadata: QueryMetadata | None = Field(
        default=None, description="业务元数据（场景/业务线/调用方/追踪ID）"
    )
    history_summary: str = Field(
        default="",
        description="上一轮对话摘要（用于多轮上下文压缩），格式: question|||sql",
    )
    expand_info: dict | None = Field(
        default=None,
        description="扩展参数，按需覆盖 Agent 配置（如 enable_execute, row_limit 等）",
    )
    prepared_context: PreparedQueryContext | None = Field(
        default=None,
        description="由 /query/prepare 返回的一次性轮次识别结果；传入后不重复调用模型识别",
    )


class QueryPrepareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    agent_id: int | None = Field(default=None)
    session_id: str = Field(default="", max_length=32)
    metadata: QueryMetadata | None = Field(default=None)
    history_summary: str = Field(default="")


class SavedQueryRequest(BaseModel):
    """A previously validated query that only needs deterministic execution."""

    sql: str = Field(..., min_length=1, max_length=50000)
    agent_id: int | None = Field(default=None)
    metadata: QueryMetadata | None = Field(default=None)
    row_limit: int = Field(default=500, ge=1, le=2000)


class SavedQueryResponse(BaseModel):
    is_success: bool
    query_result: dict | None = None
    execution_error: str = ""


class ClarificationOption(BaseModel):
    label: str
    value: str


class ClarificationTableReference(BaseModel):
    name: str
    description: str = ""


class Clarification(BaseModel):
    question: str
    options: list[ClarificationOption] = Field(default_factory=list)
    table_references: list[ClarificationTableReference] = Field(
        default_factory=list,
        description="澄清候选涉及的已检索物理表",
    )


class QueryResponse(BaseModel):
    session_id: str
    question: str
    sql: str = ""
    raw_answer: str = ""
    matched_tables: list[str] = []
    matched_terms: list[str] = []
    enum_hits: list[dict] = []
    retrieval_context: dict = Field(
        default_factory=dict,
        description="字段裁剪、Join 路径、业务口径和 Prompt 规模信息",
    )
    is_success: bool = True
    retry_count: int = 0
    execution_time_ms: int = 0
    error: str = ""
    log_id: int | None = None
    context_summary: str = Field(
        default="", description="本轮对话摘要，前端缓存后下一轮带回"
    )
    trace: dict | None = Field(
        default=None, description="详细链路追踪数据，仅调试时返回"
    )
    summary: str = Field(default="", description="LLM 对执行结果的自然语言总结")
    query_result: dict | None = Field(
        default=None, description="SQL 执行结果: {columns, rows, row_count, truncated}"
    )
    execution_error: str = Field(default="", description="SQL 执行错误信息")
    script: str = Field(
        default="", description="参数化 SQL 模板（? 占位符），仅 param_mode 时返回"
    )
    placeholder: str = Field(
        default="", description="占位符字段声明（分号分隔），按 ? 出现顺序对应"
    )
    needs_clarification: bool = Field(
        default=False, description="是否需要用户澄清业务意图"
    )
    clarification: Clarification | None = Field(
        default=None, description="结构化澄清问题和候选项"
    )
    interpretation: str = Field(
        default="", description="本轮对历史查询状态的保留、修改和移除说明"
    )
    query_state: dict = Field(
        default_factory=dict, description="合并后的结构化查询状态"
    )
    turn_intent: str = Field(
        default="sql_query", description="本轮通用意图，不包含具体业务语义"
    )
    context_relation: str = Field(
        default="new_question", description="本轮对上一轮查询状态的更新关系"
    )
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="经注册表与输入 Schema 校验后的结果工具调用",
    )


class IndexRebuildRequest(BaseModel):
    force: bool = Field(default=True, description="是否强制重建")
    collections: list[str] = Field(
        default=[],
        description="要重建的 collection 类型列表: table/glossary/enum/fewshot，为空则全量重建",
    )


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


class CodexStatusResponse(BaseModel):
    status: str
    message: str
    cli_version: str | None = None
    models: list[str] | None = None


class CodexTestRequest(BaseModel):
    model: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r".*\S.*",
        description="要实际调用的 Codex 模型名称",
    )


class CodexTestResponse(BaseModel):
    status: str
    message: str
    latency_ms: int | None = None


class EvalRunRequest(BaseModel):
    run_id: int = Field(..., description="评估运行 ID")
    cases: list[dict] = Field(
        ..., description="评估用例列表，每条含 id, question, expected_sql"
    )


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
    prepared_context: dict | None = None,
) -> dict:
    """Run one pipeline with a shared cumulative deadline for Codex turns."""
    if _is_codex_config(config):
        from src.retrieval.codex_chat_model import codex_query_budget

        with codex_query_budget(config.codex_timeout_seconds):
            return _run_query_impl(
                question,
                config,
                client,
                history_summary=history_summary,
                biz_line=biz_line,
                metadata_filter=metadata_filter,
                metadata_context=metadata_context,
                prepared_context=prepared_context,
            )
    return _run_query_impl(
        question,
        config,
        client,
        history_summary=history_summary,
        biz_line=biz_line,
        metadata_filter=metadata_filter,
        metadata_context=metadata_context,
        prepared_context=prepared_context,
    )


def prepare_query_context(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
) -> dict:
    """Classify and merge one turn once, before retrieval and SQL generation."""
    from src.retrieval.context_compressor import ContextCompressor

    compressor = ContextCompressor(client, custom_prompt=config.compress_prompt)
    result = compressor.merge(history_summary, question)
    return PreparedQueryContext(
        input_fingerprint=_query_context_fingerprint(question, history_summary),
        effective_question=result.effective_question,
        query_state=result.query_state.to_dict(),
        relation=result.relation,
        turn_intent=result.turn_intent,
        presentation_relation=result.presentation_relation,
        interpretation=result.interpretation,
        direct_response=result.direct_response,
        confidence=result.confidence,
        needs_clarification=result.needs_clarification,
        clarification=result.clarification,
        changes=result.changes,
        removed_sql_context=result.removed_sql_context,
    ).model_dump()


def _query_context_fingerprint(question: str, history_summary: str) -> str:
    payload = json.dumps(
        {"question": question, "history_summary": history_summary},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_clarification_table_references(
    relevant_tables: list[dict],
) -> list[dict[str, str]]:
    """Build verified table labels from retrieval results for clarification cards."""
    references = []
    seen_table_names = set()
    for table in relevant_tables:
        if not isinstance(table, dict):
            continue
        schema = table.get("schema")
        schema = schema if isinstance(schema, dict) else {}
        table_name = str(
            table.get("table_name") or schema.get("table_name") or ""
        ).strip()
        if not table_name or table_name in seen_table_names:
            continue
        seen_table_names.add(table_name)
        description = str(
            schema.get("display_name")
            or schema.get("description")
            or schema.get("table_comment")
            or ""
        ).strip()
        references.append(
            {
                "name": table_name[:200],
                "description": description[:200],
            }
        )
        if len(references) == 6:
            break
    return references


def _run_query_impl(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
    prepared_context: dict | None = None,
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
    from src.retrieval.context_compressor import (
        ContextCompressor,
        ContextMergeResult,
        QueryState,
    )

    effective_question = question
    previous_sql = ""
    previous_sql_context: dict[str, list[str]] = {}
    previous_presentation_state: dict = {"tool_calls": []}
    inherited_tables: set[str] = set()
    inherited_columns: set[str] = set()
    query_state = ContextCompressor.infer_state(question)
    turn_intent = "sql_query"
    interpretation = ""
    context_relation = "new_question"
    presentation_relation = "clear"
    context_changes = {"kept": [], "set": [question], "removed": []}
    removed_sql_context: dict[str, list[str]] = {}
    pending_clarification: dict | None = None
    t0 = _time.monotonic()
    previous_state = ContextCompressor.parse_summary(history_summary)
    if history_summary:
        previous_sql = previous_state["sql"]
        previous_sql_context = previous_state["sql_context"]
        previous_presentation_state = previous_state["presentation_state"]
    if prepared_context is None:
        compressor = ContextCompressor(client, custom_prompt=config.compress_prompt)
        merge_result = compressor.merge(history_summary, question)
    else:
        prepared = PreparedQueryContext.model_validate(prepared_context)
        if prepared.input_fingerprint != _query_context_fingerprint(
            question, history_summary
        ):
            raise ValueError("prepared_context 与当前问题或历史上下文不匹配")
        merge_result = ContextMergeResult(
            effective_question=prepared.effective_question,
            query_state=QueryState.from_value(prepared.query_state),
            relation=prepared.relation,
            turn_intent=prepared.turn_intent,
            presentation_relation=prepared.presentation_relation,
            interpretation=prepared.interpretation,
            direct_response=prepared.direct_response,
            confidence=prepared.confidence,
            needs_clarification=prepared.needs_clarification,
            clarification=prepared.clarification,
            changes=prepared.changes,
            removed_sql_context=prepared.removed_sql_context,
        )
    effective_question = merge_result.effective_question
    query_state = merge_result.query_state
    turn_intent = merge_result.turn_intent
    interpretation = merge_result.interpretation
    context_relation = merge_result.relation
    presentation_relation = merge_result.presentation_relation
    context_changes = merge_result.changes
    removed_sql_context = merge_result.removed_sql_context
    if turn_intent in {"result_explanation", "result_operation"} and previous_sql:
        # Result-only turns do not mutate the semantic query.  The classifier
        # chooses the route; state preservation is a deterministic contract.
        query_state = QueryState.from_value(previous_state["query_state"])
        effective_question = previous_state["question"] or effective_question
        context_relation = "follow_up_add"
        context_changes = {
            "kept": ["上一轮完整查询"],
            "set": [],
            "removed": [],
        }
        removed_sql_context = {section: [] for section in previous_sql_context}
    context_needs_clarification = (
        merge_result.needs_clarification and turn_intent == "sql_query"
    )
    trace_steps.append(
        {
            "step": "context_compress",
            "duration_ms": _elapsed_ms(t0),
            "input": question,
            "output": effective_question,
            "relation": context_relation,
            "turn_intent": merge_result.turn_intent,
            "resolved_turn_intent": turn_intent,
            "presentation_relation": presentation_relation,
            "previous_presentation_tools": [
                call.get("name")
                for call in previous_presentation_state.get("tool_calls", [])
                if isinstance(call, dict) and call.get("name")
            ],
            "changes": context_changes,
            "removed_sql_context": removed_sql_context,
            "query_state": query_state.to_dict(),
            "interpretation": merge_result.interpretation,
            "confidence": merge_result.confidence,
            "needs_clarification": context_needs_clarification,
        }
    )
    if history_summary and context_relation != "new_question":
        inherited_tables.update(previous_state["tables"])
        inherited_tables.update(previous_sql_context.get("tables", []))
        inherited_columns.update(previous_sql_context.get("columns", []))
    if context_needs_clarification:
        pending_clarification = merge_result.clarification

    if turn_intent == "non_query":
        direct_response = merge_result.direct_response or (
            "当前 Agent 用于数据库查询，请描述要查询的数据、范围和展示方式。"
        )
        return {
            "sql": "",
            "raw_answer": direct_response,
            "matched_tables": [],
            "matched_terms": [],
            "enum_hits": [],
            "retrieval_context": {},
            "is_success": True,
            "retry_count": 0,
            "error": "",
            "matched_fewshot": [],
            "context_summary": history_summary,
            "trace": {
                "question": question,
                "effective_question": effective_question,
                "steps": trace_steps,
                "total_duration_ms": _elapsed_ms(t_start),
                "tool_calls": [],
            },
            "summary": direct_response,
            "query_result": None,
            "execution_error": "",
            "script": "",
            "placeholder": "",
            "needs_clarification": False,
            "clarification": None,
            "interpretation": interpretation,
            "query_state": previous_state["query_state"],
            "turn_intent": turn_intent,
            "context_relation": context_relation,
            "tool_calls": [],
        }

    if turn_intent in {"result_explanation", "result_operation"} and not previous_sql:
        clarification = {
            "question": "当前会话没有可复用的上一轮查询，请先完成一次数据查询。",
            "options": [],
            "table_references": [],
        }
        return {
            "sql": "",
            "raw_answer": "NEED_CLARIFY: "
            + json.dumps(clarification, ensure_ascii=False),
            "matched_tables": [],
            "matched_terms": [],
            "enum_hits": [],
            "retrieval_context": {},
            "is_success": False,
            "retry_count": 0,
            "error": f"NEED_CLARIFY: {clarification['question']}",
            "matched_fewshot": [],
            "context_summary": history_summary,
            "trace": {
                "question": question,
                "effective_question": effective_question,
                "steps": trace_steps,
                "total_duration_ms": _elapsed_ms(t_start),
                "tool_calls": [],
            },
            "summary": "",
            "query_result": None,
            "execution_error": "",
            "script": "",
            "placeholder": "",
            "needs_clarification": True,
            "clarification": clarification,
            "interpretation": interpretation,
            "query_state": query_state.to_dict(),
            "turn_intent": turn_intent,
            "context_relation": context_relation,
            "tool_calls": [],
        }

    # SQL 查询轮执行 RAG。结果解释/操作轮只恢复上一轮表的当前
    # 活动 Schema，不让操作性文本干扰业务召回，也不重新生成 SQL。
    t0 = _time.monotonic()
    if turn_intent in {"result_explanation", "result_operation"}:
        table_schemas = getattr(retriever, "table_schemas", {})
        schemas_by_name = {
            str(name).replace("`", "").casefold(): schema
            for name, schema in table_schemas.items()
        }
        reused_tables = []
        previous_table_names = list(
            dict.fromkeys(
                [
                    *previous_state["tables"],
                    *previous_sql_context.get("tables", []),
                ]
            )
        )
        for table_name in previous_table_names:
            normalized_name = str(table_name).replace("`", "").casefold()
            schema = schemas_by_name.get(normalized_name)
            if schema is not None:
                reused_tables.append(
                    {
                        "table_name": str(schema.get("table_name") or table_name),
                        "schema": schema,
                        "score": 1.0,
                        "selected_columns": [
                            str(column.get("name") or "")
                            for column in schema.get("columns", [])
                            if column.get("name")
                        ],
                        "pinned": True,
                    }
                )
        result = RetrievalResult(
            relevant_tables=reused_tables,
            context_stats={"reused_previous_sql": True},
            query_intent={"state": query_state.to_dict()},
        )
    else:
        result = retriever.retrieve(
            effective_question,
            top_k=config.table_search_top_k,
            fewshot_k=config.fewshot_top_k,
            biz_line=biz_line,
            metadata_filter=metadata_filter,
            query_state=query_state.to_dict(),
            original_query=question,
            inherited_tables=inherited_tables,
            inherited_columns=inherited_columns,
        )
    retrieval_ms = _elapsed_ms(t0)

    matched_tables = [t["table_name"] for t in result.relevant_tables]
    matched_terms = result.matched_terms
    semantic_table_evidence = getattr(result, "semantic_table_evidence", [])
    requested_field_contract = result.requested_fields
    entity_filter_contract = result.entity_filters

    def _validate_requested_projection(sql: str) -> tuple[bool, str, dict]:
        projection_ok, projection_error, projection_detail = (
            SQLValidator.validate_requested_projection(
                sql,
                requested_field_contract,
            )
        )
        metric_ok, metric_error, metric_detail = (
            SQLValidator.validate_metric_projection(sql, result.query_intent)
        )
        errors = [
            error
            for valid, error in (
                (projection_ok, projection_error),
                (metric_ok, metric_error),
            )
            if not valid and error
        ]
        return (
            projection_ok and metric_ok,
            "；".join(errors),
            {**projection_detail, "metric_contract": metric_detail},
        )

    def _validate_entity_filters(sql: str) -> tuple[bool, str, dict]:
        entity_ok, entity_error, entity_detail = SQLValidator.validate_entity_filters(
            sql,
            entity_filter_contract,
        )
        return entity_ok, entity_error, entity_detail

    def _validate_followup_inheritance(sql: str) -> tuple[bool, str, dict]:
        return SQLValidator.validate_followup_inheritance(
            sql,
            previous_sql_context,
            context_relation,
            removed_sql_context,
        )

    # trace: 术语解析
    trace_steps.append(
        {
            "step": "glossary",
            "duration_ms": retrieval_ms,
            "matched_terms": matched_terms,
            "rejected_terms": result.rejected_terms,
            "table_evidence": semantic_table_evidence,
            "business_context": result.business_context or "",
        }
    )

    # trace: Schema 检索 + Reranker
    table_details = []
    for t in result.relevant_tables:
        td = {
            "table_name": t["table_name"],
            "search_score": round(float(t.get("score", 0)), 4),
            "hit_by_column": bool(t.get("hit_by_column")),
            "semantic_coverage": int(t.get("semantic_coverage", 0)),
            "matched_columns": t.get("matched_columns", [])[:5],
            "hit_by_enum": bool(t.get("hit_by_enum")),
            "pinned": bool(t.get("pinned")),
            "relation_bridge": bool(t.get("relation_bridge")),
            "selected_columns": t.get("selected_columns", []),
        }
        if "rerank_score" in t:
            td["rerank_score"] = round(t["rerank_score"], 4)
        table_details.append(td)
    trace_steps.append(
        {
            "step": "schema_retrieval",
            "tables": table_details,
            "count": len(matched_tables),
            "join_paths": result.join_paths,
            "inferred_biz_line": result.inferred_biz_line,
            "context_stats": result.context_stats,
            "query_intent": result.query_intent,
            "entity_filters": entity_filter_contract,
            "unresolved_entities": result.unresolved_entities,
        }
    )

    # trace: Value 匹配
    if result.value_hits:
        trace_steps.append(
            {
                "step": "value_matching",
                "hits": [
                    {
                        "table": v.get("table_name", ""),
                        "column": v.get("column_name", ""),
                        "value": f"{v.get('enum_label_cn', '')} → {v.get('sql_value', '')}",
                    }
                    for v in result.value_hits[:10]
                ],
                "count": len(result.value_hits),
            }
        )

    # trace: 枚举
    if result.enum_hits:
        trace_steps.append(
            {
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
            }
        )

    # trace: Few-shot
    fewshot_details = [
        {
            "id": ex.get("id"),
            "question": ex.get("question", ""),
            "sql": ex.get("sql", ""),
        }
        for ex in result.relevant_examples
        if ex.get("id") is not None
    ]
    if fewshot_details:
        trace_steps.append(
            {
                "step": "fewshot",
                "examples": fewshot_details,
                "count": len(fewshot_details),
            }
        )

    # 上下文压缩发生在 Schema/术语检索之前，它提出的疑问只是预检信号。
    # 检索已经给出候选表时，应让最终 SQL 模型基于完整证据作决定；只有
    # 完全没有 Schema 证据时才提前澄清，避免已配置口径仍被预检拦截。
    if pending_clarification is not None and not result.relevant_tables:
        table_references = _build_clarification_table_references(result.relevant_tables)
        clarification = {
            **pending_clarification,
            "table_references": table_references,
        }
        return {
            "sql": "",
            "raw_answer": "NEED_CLARIFY: "
            + json.dumps(clarification, ensure_ascii=False),
            "matched_tables": matched_tables,
            "matched_terms": matched_terms,
            "enum_hits": result.enum_hits,
            "retrieval_context": {
                "selected_columns": {
                    table["table_name"]: table.get("selected_columns", [])
                    for table in result.relevant_tables
                },
                "join_paths": result.join_paths,
                "matched_terms": result.matched_terms,
                "required_columns": result.required_columns,
                "inferred_biz_line": result.inferred_biz_line,
                "context_stats": result.context_stats,
                "query_intent": result.query_intent,
                "requested_fields": result.requested_fields,
                "semantic_table_evidence": semantic_table_evidence,
            },
            "is_success": False,
            "retry_count": 0,
            "error": "NEED_CLARIFY:" + str(pending_clarification.get("question") or ""),
            "matched_fewshot": fewshot_details,
            "context_summary": history_summary,
            "trace": {
                "question": question,
                "effective_question": effective_question,
                "steps": trace_steps,
                "total_duration_ms": _elapsed_ms(t_start),
            },
            "summary": "",
            "query_result": None,
            "execution_error": "",
            "script": "",
            "placeholder": "",
            "needs_clarification": True,
            "clarification": clarification,
            "interpretation": interpretation,
            "query_state": query_state.to_dict(),
            "turn_intent": turn_intent,
            "context_relation": context_relation,
            "tool_calls": [],
        }

    # param_mode: 从 metadata.context 读取，替换输出规则生成 ? 占位符 SQL
    _ctx = metadata_context or {}
    _param_mode = _ctx.get("param_mode", False)
    _placeholder_fields = _ctx.get("placeholder_fields", [])
    allowed_tool_names = _ctx.get("enabled_tools")
    available_tools = list(config.tools)
    agent_bound_tool_names = [
        str(tool.get("name") or "")
        for tool in available_tools
        if str(tool.get("name") or "")
    ]
    if isinstance(allowed_tool_names, list):
        normalized_tool_names = [
            str(name).strip()
            for name in allowed_tool_names
            if isinstance(name, str) and name.strip()
        ]
        allowed = set(normalized_tool_names)
        available_tools = [
            tool for tool in available_tools if str(tool.get("name") or "") in allowed
        ]
    pending_result_tool_names = {
        str(name).strip()
        for name in _ctx.get("pending_result_tools", [])
        if isinstance(name, str) and name.strip()
    }
    pending_result_tools = [
        tool
        for tool in available_tools
        if str(tool.get("name") or "") in pending_result_tool_names
    ]
    available_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in available_tools
        if str(tool.get("name") or "").strip()
    }
    filtered_previous_presentation_state = {
        "tool_calls": [
            call
            for call in previous_presentation_state.get("tool_calls", [])
            if isinstance(call, dict)
            and str(call.get("name") or "").strip() in available_tool_names
        ]
    }
    should_execute_inherited_presentation = (
        context_relation != "new_question"
        and presentation_relation in {"inherit", "add"}
        and turn_intent in {"sql_query", "result_operation"}
    )
    inherited_presentation_calls = (
        filtered_previous_presentation_state["tool_calls"]
        if should_execute_inherited_presentation
        else []
    )
    inherited_result_tool_names = {
        str(call.get("name") or "").strip()
        for call in inherited_presentation_calls
        if str(call.get("name") or "").strip()
    }
    inherited_result_tools = [
        tool
        for tool in available_tools
        if str(tool.get("name") or "").strip() in inherited_result_tool_names
    ]

    prompt_text = result.prompt_text
    if previous_sql and context_relation != "new_question":
        prompt_text += (
            "\n\n【上一轮成功结果（本轮结构基线）】\n"
            "根据本轮完整查询状态修改此 SQL。用户未明确替换或删除的表、字段、"
            "展示维度、聚合方式和过滤条件必须保留；用户明确修改的内容以本轮为准。\n"
            f"本轮语义变更：{json.dumps(context_changes, ensure_ascii=False)}\n"
            f"上一轮 SQL 结构：{json.dumps(previous_sql_context, ensure_ascii=False)}\n"
            f"```sql\n{previous_sql}\n```"
        )
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
            "8. 只有用户业务意图有歧义且不同解释会改变查询结果时，才输出："
            'NEED_CLARIFY: {"question":"需要确认的问题",'
            '"options":[{"label":"选项文案","value":"用于补充原问题的含义"}]}。'
            "候选项最多 4 个；没有可靠候选项时 options 输出空数组"
        )
        prompt_text = re.sub(
            r"【输出要求】.*", param_mode_rules, prompt_text, flags=re.DOTALL
        )
    # 构建对话（注入当前日期，避免 LLM 因知识截止而误判年份）
    current_date = _datetime.now().astimezone().date().isoformat()
    _system_content = f"{config.system_prompt}\n\n当前日期: {current_date}"
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
    unresolved_entity_issues = [
        str(item.get("issue") or "")
        for item in result.unresolved_entities
        if item.get("issue")
    ]
    if unresolved_entity_issues:
        answer = "NEED_CLARIFY: " + json.dumps(
            {
                "question": (
                    "；".join(unresolved_entity_issues)
                    + "请说明它对应哪个业务实体或标识。"
                ),
                "options": [],
            },
            ensure_ascii=False,
        )
        llm_calls.append(
            {
                "role": "unresolved_entity",
                "duration_ms": 0,
                "model": "deterministic_guard",
                "output": answer,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
    elif turn_intent in {"result_explanation", "result_operation"} and previous_sql:
        answer = f"```sql\n{previous_sql}\n```"
        llm_calls.append(
            {
                "role": "reuse_previous_sql",
                "duration_ms": 0,
                "model": "deterministic_state",
                "output": answer,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
    else:
        t0 = _time.monotonic()
        answer, usage = _invoke(messages)
        llm_calls.append(
            {
                "role": "initial",
                "duration_ms": _elapsed_ms(t0),
                "model": config.llm_model,
                "output": answer,
                "input_tokens": usage.get("input_tokens") if usage else None,
                "output_tokens": usage.get("output_tokens") if usage else None,
            }
        )
    messages.append({"role": "assistant", "content": answer})

    # 检测 NEED_CLARIFY
    clarification = SQLValidator.extract_clarification(answer)
    clarify_msg = clarification["question"] if clarification else None

    # 提取 SQL
    extracted_sql = SQLValidator.extract_sql(answer) if not clarify_msg else None
    retry_count = 0
    is_success = True
    error_msg = ""

    if not extracted_sql and not clarify_msg:
        messages.append(
            {
                "role": "user",
                "content": "你没有生成 SQL，请根据上面的 Schema 生成可执行的 Doris SQL，用 ```sql ``` 包裹。",
            }
        )
        t0 = _time.monotonic()
        answer, usage = _invoke(messages)
        llm_calls.append(
            {
                "role": "retry_no_sql",
                "duration_ms": _elapsed_ms(t0),
                "model": config.llm_model,
                "output": answer,
                "input_tokens": usage.get("input_tokens") if usage else None,
                "output_tokens": usage.get("output_tokens") if usage else None,
            }
        )
        messages.append({"role": "assistant", "content": answer})
        clarification = SQLValidator.extract_clarification(answer)
        clarify_msg = clarification["question"] if clarification else None
        extracted_sql = SQLValidator.extract_sql(answer) if not clarify_msg else None

    tool_calls: list[dict] = []

    if not extracted_sql and not clarify_msg:
        is_success = False
        error_msg = "模型未生成可执行 SQL"

    inheritance_validation_attempts = []
    must_preserve_previous_structure = bool(
        previous_sql and context_relation != "new_question"
    )
    if extracted_sql and must_preserve_previous_structure:
        max_inheritance_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_inheritance_fix_retries + 1):
            inheritance_ok, inheritance_error, inheritance_detail = (
                _validate_followup_inheritance(extracted_sql)
            )
            inheritance_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": inheritance_ok,
                    "error": inheritance_error,
                    "missing": inheritance_detail.get("missing", {}),
                }
            )
            if inheritance_ok:
                break
            if attempt >= max_inheritance_fix_retries:
                is_success = False
                error_msg = inheritance_error
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "这是增量追问，但新 SQL 丢失了上一轮已验证结构。\n\n"
                        f"## 缺失结构\n{inheritance_error}\n\n"
                        f"## 本轮语义变更\n{json.dumps(context_changes, ensure_ascii=False)}\n\n"
                        "## 允许删除的上一轮 SQL 结构\n"
                        f"{json.dumps(removed_sql_context, ensure_ascii=False)}\n\n"
                        "请以上一轮 SQL 为基线，只删除明确授权删除的结构，并恢复其他缺失的投影、"
                        "过滤、分组、关联、排序和限制。输出完整 Doris SQL，用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"followup_inheritance_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "追问继承修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "followup_inheritance_validate",
                "attempts": inheritance_validation_attempts,
                "final_valid": bool(
                    inheritance_validation_attempts
                    and inheritance_validation_attempts[-1]["valid"]
                ),
            }
        )

    # QueryState 中的自然日窗口是通用时间契约，不依赖具体业务表或时间字段名。
    time_validation_attempts = []
    calendar_day_window = SQLValidator.calendar_day_window(result.query_intent)
    if extracted_sql and calendar_day_window and is_success:
        max_time_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_time_fix_retries + 1):
            time_ok, time_error, time_detail = (
                SQLValidator.validate_calendar_day_window(
                    extracted_sql,
                    result.query_intent,
                )
            )
            time_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": time_ok,
                    "error": time_error,
                    **time_detail,
                }
            )
            if time_ok:
                break
            if attempt >= max_time_fix_retries:
                is_success = False
                error_msg = f"时间范围校验失败: {time_error}"
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你生成的 SQL 不符合已确定的滚动自然日时间范围。\n\n"
                        f"## 时间范围问题\n{time_error}\n\n"
                        "请保留当前查询选择的时间字段和其他查询结构，只修正时间边界。"
                        "输出完整 Doris SQL，用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"calendar_day_window_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "时间范围修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "calendar_day_window_validate",
                "attempts": time_validation_attempts,
                "final_valid": bool(
                    time_validation_attempts and time_validation_attempts[-1]["valid"]
                ),
            }
        )

    # 业务语义校验必须先于 EXPLAIN。语法正确不代表换汇口径正确。
    currency_validation_attempts = []
    if (
        extracted_sql
        and SQLValidator.currency_conversion_target(result.query_intent)
        and is_success
    ):
        max_currency_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_currency_fix_retries + 1):
            currency_ok, currency_error, currency_detail = (
                SQLValidator.validate_currency_conversion(
                    extracted_sql,
                    result.query_intent,
                )
            )
            currency_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": currency_ok,
                    "error": currency_error,
                    **currency_detail,
                }
            )
            if currency_ok:
                break
            if attempt >= max_currency_fix_retries:
                is_success = False
                error_msg = f"货币换算口径校验失败: {currency_error}"
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你生成的 SQL 没有完整执行货币换算口径。\n\n"
                        f"## 口径问题\n{currency_error}\n\n"
                        "请使用 `warehouse_sys`.`sys_exchange_rate`，按原币种关联 "
                        "source_currency，按目标币种限定 target_currency，按交易日期关联 "
                        "sync_time，使用 原金额 * mid 换算；原币种已经等于目标币种时直接使用原金额。"
                        "汇率表必须 LEFT JOIN。请输出完整修复后的 Doris SQL，用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"currency_semantic_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "货币换算口径修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "currency_conversion_validate",
                "attempts": currency_validation_attempts,
                "final_valid": bool(
                    currency_validation_attempts
                    and currency_validation_attempts[-1]["valid"]
                ),
            }
        )

    # 明细查询保留默认资源保护；聚合查询只有用户明确要求 Top N 时才允许 LIMIT。
    result_limit_validation_attempts = []
    if (
        extracted_sql
        and turn_intent == "sql_query"
        and result.query_intent.get("state", {}).get("result_shape") == "aggregate"
        and is_success
    ):
        max_limit_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_limit_fix_retries + 1):
            limit_ok, limit_error, limit_detail = SQLValidator.validate_result_limit(
                extracted_sql,
                result.query_intent,
            )
            result_limit_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": limit_ok,
                    "error": limit_error,
                    **limit_detail,
                }
            )
            if limit_ok:
                break
            if attempt >= max_limit_fix_retries:
                is_success = False
                error_msg = f"结果数量校验失败: {limit_error}"
                break

            requested_limit = SQLValidator.requested_limit(result.query_intent)
            repair_instruction = (
                f"将最终查询限制为 LIMIT {requested_limit}。"
                if requested_limit is not None
                else "移除最终聚合查询的 LIMIT，返回完整分组结果。"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你生成的 SQL 会错误截断聚合结果。\n\n"
                        f"## 结果数量问题\n{limit_error}\n\n"
                        f"{repair_instruction}保留其他查询结构不变。"
                        "输出完整 Doris SQL，用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"result_limit_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "结果数量修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "result_limit_validate",
                "attempts": result_limit_validation_attempts,
                "final_valid": bool(
                    result_limit_validation_attempts
                    and result_limit_validation_attempts[-1]["valid"]
                ),
            }
        )

    # 用户明确要求展示的字段属于结果契约，任何生成或纠错都不能静默删除。
    projection_validation_attempts = []
    if (
        extracted_sql
        and (result.requested_fields or SQLValidator.is_count_only(result.query_intent))
        and is_success
    ):
        max_projection_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_projection_fix_retries + 1):
            projection_ok, projection_error, projection_detail = (
                _validate_requested_projection(extracted_sql)
            )
            projection_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": projection_ok,
                    "error": projection_error,
                    **projection_detail,
                }
            )
            if projection_ok:
                break
            if attempt >= max_projection_fix_retries:
                is_success = False
                error_msg = projection_error
                break

            if SQLValidator.is_count_only(result.query_intent):
                repair_instruction = (
                    "请把最终 SELECT 改为唯一一个 COUNT 结果列，移除金额、币种、汇率等其他结果列，"
                    "移除 GROUP BY 和汇率表关联；保留当前查询状态中仍适用的筛选条件。"
                )
            else:
                repair_instruction = (
                    "请从候选列中选择与查询实体语义最匹配的真实字段，必要时补充正确的 "
                    "JOIN，并确保该字段出现在最终 SELECT 结果中。禁止通过删除用户要求的字段来修复 SQL。"
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你生成的 SQL 不符合用户本轮明确要求的结果结构。\n\n"
                        f"## 结果字段问题\n{projection_error}\n\n"
                        f"{repair_instruction}"
                        "请输出完整修复后的 Doris SQL，用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"requested_projection_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "结果字段修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "requested_projection_validate",
                "requirements": {
                    "requested_fields": result.requested_fields,
                    "count_only": SQLValidator.is_count_only(result.query_intent),
                },
                "attempts": projection_validation_attempts,
                "final_valid": bool(
                    projection_validation_attempts
                    and projection_validation_attempts[-1]["valid"]
                ),
            }
        )

    # 配置化实体绑定是强约束；同时复核换汇和展示字段，避免一次修复破坏另一项口径。
    entity_validation_attempts = []
    if extracted_sql and entity_filter_contract and is_success:
        max_entity_fix_retries = min(max(config.max_fix_retries, 0), 2)
        for attempt in range(max_entity_fix_retries + 1):
            entity_ok, entity_error, entity_detail = _validate_entity_filters(
                extracted_sql
            )
            currency_ok, currency_error, _ = SQLValidator.validate_currency_conversion(
                extracted_sql,
                result.query_intent,
            )
            projection_ok, projection_error, _ = _validate_requested_projection(
                extracted_sql
            )
            contract_errors = [
                error
                for valid, error in (
                    (entity_ok, entity_error),
                    (currency_ok, currency_error),
                    (projection_ok, projection_error),
                )
                if not valid and error
            ]
            contract_ok = entity_ok and currency_ok and projection_ok
            entity_validation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": contract_ok,
                    "error": "；".join(contract_errors),
                    **entity_detail,
                }
            )
            if contract_ok:
                break
            if attempt >= max_entity_fix_retries:
                is_success = False
                error_msg = "；".join(contract_errors)
                break

            expected_filters = "\n".join(
                f"- `{item['qualified_column']}` = {item['value']!r}"
                for item in entity_filter_contract
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "SQL 违反了已确定的查询契约。\n\n"
                        f"## 校验问题\n{'；'.join(contract_errors)}\n\n"
                        f"## 必须保留的实体过滤\n{expected_filters}\n\n"
                        "请使用指定表、指定字段和原值修复 WHERE 条件；需要时补充正确 JOIN。"
                        "同时保留完整换汇口径和用户要求展示的字段。请输出完整 Doris SQL，"
                        "用 ```sql ``` 包裹。"
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"entity_contract_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                is_success = False
                error_msg = "实体过滤修复后未生成可执行 SQL"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "entity_filter_validate",
                "requirements": entity_filter_contract,
                "attempts": entity_validation_attempts,
                "final_valid": bool(
                    entity_validation_attempts
                    and entity_validation_attempts[-1]["valid"]
                ),
            }
        )

    # EXPLAIN 校验（param_mode 下跳过，? 占位符无法通过 EXPLAIN）
    explain_details = []
    if (
        extracted_sql
        and is_success
        and config.enable_explain
        and validator
        and not _param_mode
    ):
        syntax_ok = False
        check = None
        for attempt in range(config.max_fix_retries):
            check = validator.validate(answer)
            if check["valid"] and check.get("sql"):
                currency_ok, currency_error, _ = (
                    SQLValidator.validate_currency_conversion(
                        check["sql"],
                        result.query_intent,
                    )
                )
                if not currency_ok:
                    check = {
                        **check,
                        "valid": False,
                        "error": f"货币换算口径校验失败: {currency_error}",
                    }
                projection_ok, projection_error, _ = _validate_requested_projection(
                    check["sql"]
                )
                if check["valid"] and not projection_ok:
                    check = {
                        **check,
                        "valid": False,
                        "error": projection_error,
                    }
                entity_ok, entity_error, _ = _validate_entity_filters(check["sql"])
                if check["valid"] and not entity_ok:
                    check = {
                        **check,
                        "valid": False,
                        "error": entity_error,
                    }
            explain_details.append(
                {
                    "attempt": attempt + 1,
                    "valid": check["valid"],
                    "error": check.get("error", ""),
                    "connection_retries": check.get("connection_retries", 0),
                    "infrastructure_error": bool(check.get("infrastructure_error")),
                }
            )
            if check["valid"]:
                syntax_ok = True
                break

            retry_count = attempt + 1
            logger.info(
                f"EXPLAIN 失败 (第 {retry_count}/{config.max_fix_retries} 次): {check['error']}"
            )

            if check.get("infrastructure_error"):
                break

            if attempt < config.max_fix_retries - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": f"你生成的 SQL 执行 EXPLAIN 校验失败。\n\n"
                        f"## EXPLAIN 报错\n{check['error']}\n\n"
                        "请分析错误原因（1-2句），然后输出修复后的 SQL，用 ```sql ``` 包裹。"
                        "修复时必须保留用户明确要求展示的所有结果字段。",
                    }
                )
                t0 = _time.monotonic()
                answer, usage = _invoke(messages)
                llm_calls.append(
                    {
                        "role": f"explain_fix_{attempt + 1}",
                        "duration_ms": _elapsed_ms(t0),
                        "model": config.llm_model,
                        "output": answer,
                        "input_tokens": usage.get("input_tokens") if usage else None,
                        "output_tokens": usage.get("output_tokens") if usage else None,
                    }
                )
                messages.append({"role": "assistant", "content": answer})

        if not syntax_ok:
            is_success = False
            error_msg = check["error"] if check else "EXPLAIN 校验失败"

        # 执行计划规则检查；仅在规则判定不安全时调用模型修复。
        if syntax_ok and check and check.get("plan"):
            plan_safety = SQLValidator.inspect_plan(
                check["plan"], max_scan_rows=config.max_explain_scan_rows
            )
            trace_steps.append({"step": "plan_safety", **plan_safety})
            if not plan_safety["safe"]:
                is_success = False
                error_msg = "；".join(plan_safety["warnings"])
                messages.append(
                    {
                        "role": "user",
                        "content": f"这条 SQL 的 EXPLAIN 执行计划存在明确风险。\n\n"
                        f"## EXPLAIN 执行计划\n```\n{check['plan']}\n```\n\n"
                        f"## 规则检查\n{'; '.join(plan_safety['warnings'])}\n\n"
                        "请修复这些风险并输出优化后的 SQL，用 ```sql ``` 包裹。",
                    }
                )
                t0 = _time.monotonic()
                review_result, usage = _invoke(messages)
                llm_calls.append(
                    {
                        "role": "plan_fix",
                        "duration_ms": _elapsed_ms(t0),
                        "model": config.llm_model,
                        "output": review_result,
                        "input_tokens": usage.get("input_tokens") if usage else None,
                        "output_tokens": usage.get("output_tokens") if usage else None,
                    }
                )
                recheck = validator.validate(review_result)
                if recheck["valid"]:
                    optimized_safety = SQLValidator.inspect_plan(
                        recheck.get("plan") or "",
                        max_scan_rows=config.max_explain_scan_rows,
                    )
                    optimized_sql = recheck.get("sql") or ""
                    optimized_currency_ok, optimized_currency_error, _ = (
                        SQLValidator.validate_currency_conversion(
                            optimized_sql,
                            result.query_intent,
                        )
                    )
                    optimized_projection_ok, optimized_projection_error, _ = (
                        _validate_requested_projection(optimized_sql)
                    )
                    optimized_entity_ok, optimized_entity_error, _ = (
                        _validate_entity_filters(optimized_sql)
                    )
                    if (
                        optimized_safety["safe"]
                        and optimized_currency_ok
                        and optimized_projection_ok
                        and optimized_entity_ok
                    ):
                        answer = review_result
                        is_success = True
                        error_msg = ""
                    trace_steps.append(
                        {
                            "step": "optimized_plan_safety",
                            **optimized_safety,
                            "currency_conversion_valid": optimized_currency_ok,
                            "currency_conversion_error": optimized_currency_error,
                            "requested_projection_valid": optimized_projection_ok,
                            "requested_projection_error": optimized_projection_error,
                            "entity_filter_valid": optimized_entity_ok,
                            "entity_filter_error": optimized_entity_error,
                        }
                    )

    # trace: LLM 调用
    trace_steps.append(
        {
            "step": "llm_generation",
            "calls": llm_calls,
            "total_calls": len(llm_calls),
        }
    )

    # trace: EXPLAIN 校验
    if explain_details:
        trace_steps.append(
            {
                "step": "explain_validate",
                "attempts": explain_details,
                "final_valid": is_success,
            }
        )

    # 澄清信号优先级高于同一回复中可能夹带的 SQL，避免在意图未确认时执行。
    final_clarification = SQLValidator.extract_clarification(answer)
    if final_clarification is not None:
        clarification = final_clarification
        clarify_msg = clarification["question"]
    final_sql = (
        "" if clarification is not None else SQLValidator.extract_sql(answer) or ""
    )

    if clarification is not None:
        table_references = _build_clarification_table_references(result.relevant_tables)
        if table_references:
            clarification = {
                **clarification,
                "table_references": table_references,
            }

    if final_sql:
        read_only, read_only_error = SQLValidator.validate_read_only(final_sql)
        if not read_only:
            is_success = False
            error_msg = read_only_error
            trace_steps.append(
                {
                    "step": "sql_read_only_guard",
                    "success": False,
                    "error": read_only_error,
                }
            )

    if final_sql and is_success:
        schema_allowed, schema_error, schema_detail = (
            SQLValidator.validate_schema_references(
                final_sql,
                [table.get("schema", {}) for table in result.relevant_tables],
            )
        )
        trace_steps.append(
            {
                "step": "sql_schema_guard",
                "success": schema_allowed,
                "error": schema_error,
                **schema_detail,
            }
        )
        if not schema_allowed:
            is_success = False
            error_msg = schema_error

    # NEED_CLARIFY: 模型认为问题模糊，需要澄清
    if clarification is not None and clarify_msg:
        is_success = False
        error_msg = f"NEED_CLARIFY: {clarify_msg}"

    # ── P2: 枚举值预校验（param_mode 跳过）──
    if (
        final_sql
        and is_success
        and config.enable_enum_validate
        and result.enum_hits
        and not _param_mode
    ):
        where_values = SQLValidator.extract_where_values(final_sql)
        if where_values:
            enum_mismatches = SQLValidator.validate_enum_values(
                where_values, result.enum_hits
            )
            if enum_mismatches:
                t0 = _time.monotonic()
                mismatch_lines = []
                for mm in enum_mismatches:
                    line = f"- `{mm['column']} = '{mm['sql_value']}'`：该字段的枚举值为 {mm['expected_values']}"
                    if mm.get("suggestion"):
                        line += f"（{mm['suggestion']}）"
                    mismatch_lines.append(line)
                enum_fix_prompt = (
                    "你生成的 SQL 中以下条件值可能有误：\n\n"
                    + "\n".join(mismatch_lines)
                    + "\n\n"
                    "请根据枚举定义修正 SQL，用 ```sql ``` 包裹。"
                )
                messages.append({"role": "user", "content": enum_fix_prompt})
                enum_fix_answer, enum_fix_usage = _invoke(messages)
                llm_calls.append(
                    {
                        "role": "enum_fix",
                        "duration_ms": _elapsed_ms(t0),
                        "model": config.llm_model,
                        "output": enum_fix_answer,
                        "input_tokens": enum_fix_usage.get("input_tokens")
                        if enum_fix_usage
                        else None,
                        "output_tokens": enum_fix_usage.get("output_tokens")
                        if enum_fix_usage
                        else None,
                    }
                )
                messages.append({"role": "assistant", "content": enum_fix_answer})
                fixed_sql = SQLValidator.extract_sql(enum_fix_answer)
                enum_fixed = False
                if fixed_sql and validator:
                    recheck = validator.explain(fixed_sql)
                    currency_ok, _, _ = SQLValidator.validate_currency_conversion(
                        fixed_sql,
                        result.query_intent,
                    )
                    schema_ok, _, _ = SQLValidator.validate_schema_references(
                        fixed_sql,
                        [table.get("schema", {}) for table in result.relevant_tables],
                    )
                    projection_ok, _, _ = _validate_requested_projection(fixed_sql)
                    entity_ok, _, _ = _validate_entity_filters(fixed_sql)
                    if (
                        recheck["valid"]
                        and currency_ok
                        and schema_ok
                        and projection_ok
                        and entity_ok
                    ):
                        final_sql = fixed_sql
                        answer = enum_fix_answer
                        enum_fixed = True
                trace_steps.append(
                    {
                        "step": "enum_validate",
                        "mismatches": enum_mismatches,
                        "fixed": enum_fixed,
                        "duration_ms": _elapsed_ms(t0),
                    }
                )

    # 任意后处理都不能绕过时间、换汇、结果数量和检索 Schema 约束。
    if final_sql and is_success:
        time_ok, time_error, time_detail = SQLValidator.validate_calendar_day_window(
            final_sql,
            result.query_intent,
        )
        currency_ok, currency_error, currency_detail = (
            SQLValidator.validate_currency_conversion(
                final_sql,
                result.query_intent,
            )
        )
        if turn_intent == "sql_query":
            limit_ok, limit_error, limit_detail = SQLValidator.validate_result_limit(
                final_sql,
                result.query_intent,
            )
        else:
            limit_ok, limit_error, limit_detail = True, "", {"required": False}
        schema_ok, schema_error, schema_detail = (
            SQLValidator.validate_schema_references(
                final_sql,
                [table.get("schema", {}) for table in result.relevant_tables],
            )
        )
        projection_ok, projection_error, projection_detail = (
            _validate_requested_projection(final_sql)
        )
        entity_ok, entity_error, entity_detail = _validate_entity_filters(final_sql)
        if must_preserve_previous_structure:
            inheritance_ok, inheritance_error, inheritance_detail = (
                _validate_followup_inheritance(final_sql)
            )
        else:
            inheritance_ok, inheritance_error, inheritance_detail = (
                True,
                "",
                {"required": False, "missing": {}},
            )
        trace_steps.append(
            {
                "step": "sql_post_rewrite_guard",
                "calendar_day_window_valid": time_ok,
                "calendar_day_window_error": time_error,
                "calendar_day_window_detail": time_detail,
                "currency_conversion_valid": currency_ok,
                "currency_conversion_error": currency_error,
                "currency_conversion_detail": currency_detail,
                "result_limit_valid": limit_ok,
                "result_limit_error": limit_error,
                "result_limit_detail": limit_detail,
                "schema_valid": schema_ok,
                "schema_error": schema_error,
                "schema_detail": schema_detail,
                "requested_projection_valid": projection_ok,
                "requested_projection_error": projection_error,
                "requested_projection_detail": projection_detail,
                "entity_filter_valid": entity_ok,
                "entity_filter_error": entity_error,
                "entity_filter_detail": entity_detail,
                "followup_inheritance_valid": inheritance_ok,
                "followup_inheritance_error": inheritance_error,
                "followup_inheritance_detail": inheritance_detail,
            }
        )
        if (
            not time_ok
            or not currency_ok
            or not limit_ok
            or not schema_ok
            or not projection_ok
            or not entity_ok
            or not inheritance_ok
        ):
            is_success = False
            error_msg = (
                time_error
                or currency_error
                or limit_error
                or schema_error
                or projection_error
                or entity_error
                or inheritance_error
            )

    presentation_clears_existing = (
        presentation_relation == "clear" and context_relation != "new_question"
    )
    explicit_tools = (
        []
        if presentation_clears_existing
        else explicitly_requested_tools(question, available_tools)
    )
    forced_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in [
            *pending_result_tools,
            *explicit_tools,
            *inherited_result_tools,
        ]
        if str(tool.get("name") or "").strip()
    }
    forced_tools = [
        tool
        for tool in available_tools
        if str(tool.get("name") or "").strip() in forced_tool_names
    ]
    if presentation_clears_existing:
        planner_tools = []
        planner_choice = "none"
    else:
        # Forced and inherited calls remain active actions, but they must not hide
        # another Agent-bound tool explicitly requested in the same turn.
        planner_tools = available_tools
        planner_choice = "required" if forced_tools else config.tool_choice
    deferred_tools = forced_tools
    if clarification is not None and deferred_tools:
        tool_calls = [
            {
                "name": str(tool.get("name") or ""),
                "arguments": {},
                "requires_query_result": bool(tool.get("requires_query_result")),
            }
            for tool in deferred_tools
            if str(tool.get("name") or "")
        ]
        trace_steps.append(
            {
                "step": "tool_intent_deferred",
                "selected_tools": [call["name"] for call in tool_calls],
                "reason": "awaiting_query_clarification",
            }
        )
    elif final_sql and is_success and planner_tools and planner_choice != "none":
        t0 = _time.monotonic()
        planner_answer = ""
        planner_error = ""
        planner_usage = None
        planner_attempts: list[str] = []
        try:
            planner_messages = tool_planning_messages(
                question,
                planner_tools,
                choice=planner_choice,
                query_context=(
                    effective_question if effective_question != question else ""
                ),
                active_tool_calls=inherited_presentation_calls,
                query_projection=(
                    ContextCompressor.extract_sql_context(final_sql).get(
                        "projections", []
                    )
                    if final_sql
                    else []
                ),
            )
            planner_answer, planner_usage = _invoke(planner_messages)
            planner_attempts.append(planner_answer)
            tool_calls = extract_planned_tool_calls(
                planner_answer,
                planner_tools,
                max_calls=config.tool_max_calls,
            )
            if not tool_calls and (
                declared_action_count(planner_answer) > 0
                or planner_choice == "required"
            ):
                planner_messages.extend(
                    [
                        {"role": "assistant", "content": planner_answer},
                        {
                            "role": "user",
                            "content": (
                                "上一条 JSON 未通过输入 Schema 的结构校验。"
                                "保留动作语义，只修复结构：arguments 必须包含全部 required "
                                "字段；枚举值只能来自 enum；additionalProperties 为 false 时"
                                "不得增加未声明字段。返回完整 JSON。"
                            ),
                        },
                    ]
                )
                planner_answer, planner_usage = _invoke(planner_messages)
                planner_attempts.append(planner_answer)
                tool_calls = extract_planned_tool_calls(
                    planner_answer,
                    planner_tools,
                    max_calls=config.tool_max_calls,
                )
        # Result-tool planning must not discard an otherwise valid query result.
        except Exception as exc:  # noqa: BLE001
            planner_error = type(exc).__name__
            logger.warning(
                "result tool planning failed",
                extra={
                    "agent_id": config.agent_id,
                    "model": config.llm_model,
                    "tool_count": len(available_tools),
                    "error_type": planner_error,
                },
            )
        trace_steps.append(
            {
                "step": "tool_planning",
                "duration_ms": _elapsed_ms(t0),
                "model": config.llm_model,
                "available_tools": [
                    str(tool.get("name") or "") for tool in planner_tools
                ],
                "agent_bound_tools": agent_bound_tool_names,
                "channel_allowed_tools": [
                    str(tool.get("name") or "") for tool in available_tools
                ],
                "explicit_tools": [
                    str(tool.get("name") or "") for tool in explicit_tools
                ],
                "inherited_tools": [
                    str(tool.get("name") or "") for tool in inherited_result_tools
                ],
                "presentation_relation": presentation_relation,
                "selected_tools": [call["name"] for call in tool_calls],
                "output": planner_answer,
                "attempts": planner_attempts,
                "error": planner_error,
                "input_tokens": planner_usage.get("input_tokens")
                if planner_usage
                else None,
                "output_tokens": planner_usage.get("output_tokens")
                if planner_usage
                else None,
            }
        )

    # ── param_mode: 提取 script + placeholder，跳过执行 ──
    script_text = ""
    placeholder_text = ""
    if _param_mode and _placeholder_fields and final_sql:
        script_text = final_sql
        placeholder_text = SQLValidator.extract_placeholder(answer) or ";".join(
            _placeholder_fields
        )
        trace_steps.append(
            {
                "step": "param_mode",
                "script": script_text,
                "placeholder": placeholder_text,
            }
        )

    # SQL 执行（param_mode 下跳过执行，? 占位符无法直接执行）
    query_result_data = None
    execution_error = ""
    summary = ""

    if (
        final_sql
        and is_success
        and (
            config.enable_execute
            or any(call.get("requires_query_result") for call in tool_calls)
        )
        and not _param_mode
        and validator
    ):
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
            trace_steps.append(
                {
                    "step": "sql_execution",
                    "success": False,
                    "error": execution_error,
                    "database": ", ".join(sorted(referenced_dbs)),
                    "duration_ms": 0,
                }
            )

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

            trace_steps.append(
                {
                    "step": "sql_execution",
                    "success": exec_result["success"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result.get("truncated", False),
                    "error": exec_result.get("error", ""),
                    "database": ", ".join(referenced_dbs) if referenced_dbs else "",
                    "duration_ms": exec_ms,
                }
            )

            # ── P4: 超时降级 ──
            _is_timeout = (
                not exec_result["success"]
                and exec_result.get("error")
                and re.search(
                    r"(?:timeout|timed?\s*out|query_timeout)",
                    exec_result["error"],
                    re.IGNORECASE,
                )
            )
            if _is_timeout and config.enable_timeout_fallback:
                for level in (1, 2, 3):
                    t0 = _time.monotonic()
                    simplified = SQLValidator.simplify_sql_for_timeout(
                        current_exec_sql, level
                    )
                    if simplified is None:
                        # level 3: LLM 简化
                        timeout_prompt = (
                            f"以下 SQL 执行超时（{config.execute_timeout}s）：\n\n"
                            f"```sql\n{current_exec_sql}\n```\n\n"
                            f"请简化查询以提高性能（减少 JOIN、去掉子查询、缩小时间范围等）。\n"
                            f"输出简化后的 SQL，用 ```sql ``` 包裹。保持查询意图不变。"
                        )
                        messages.append({"role": "user", "content": timeout_prompt})
                        timeout_answer, _t_usage = _invoke(messages)
                        llm_calls.append(
                            {
                                "role": "timeout_simplify",
                                "duration_ms": _elapsed_ms(t0),
                                "model": config.llm_model,
                                "output": timeout_answer,
                            }
                        )
                        messages.append(
                            {"role": "assistant", "content": timeout_answer}
                        )
                        simplified = SQLValidator.extract_sql(timeout_answer)
                        if not simplified:
                            break

                    currency_ok, currency_error, _ = (
                        SQLValidator.validate_currency_conversion(
                            simplified,
                            result.query_intent,
                        )
                    )
                    if not currency_ok:
                        trace_steps.append(
                            {
                                "step": "timeout_fallback",
                                "level": level,
                                "simplified_sql": simplified,
                                "success": False,
                                "error": f"货币换算口径校验失败: {currency_error}",
                                "duration_ms": _elapsed_ms(t0),
                            }
                        )
                        continue

                    projection_ok, projection_error, _ = _validate_requested_projection(
                        simplified
                    )
                    if not projection_ok:
                        trace_steps.append(
                            {
                                "step": "timeout_fallback",
                                "level": level,
                                "simplified_sql": simplified,
                                "success": False,
                                "error": projection_error,
                                "duration_ms": _elapsed_ms(t0),
                            }
                        )
                        continue

                    entity_ok, entity_error, _ = _validate_entity_filters(simplified)
                    if not entity_ok:
                        trace_steps.append(
                            {
                                "step": "timeout_fallback",
                                "level": level,
                                "simplified_sql": simplified,
                                "success": False,
                                "error": entity_error,
                                "duration_ms": _elapsed_ms(t0),
                            }
                        )
                        continue

                    # EXPLAIN 校验简化后的 SQL
                    access_allowed, access_error, simplified_dbs = (
                        SQLValidator.validate_database_access(
                            simplified, authorized_dbs
                        )
                    )
                    if not access_allowed:
                        trace_steps.append(
                            {
                                "step": "timeout_fallback",
                                "level": level,
                                "simplified_sql": simplified,
                                "success": False,
                                "error": f"安全拦截: {access_error}",
                                "database": ", ".join(sorted(simplified_dbs)),
                                "duration_ms": _elapsed_ms(t0),
                            }
                        )
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

                    trace_steps.append(
                        {
                            "step": "timeout_fallback",
                            "level": level,
                            "simplified_sql": simplified,
                            "success": exec_result["success"],
                            "duration_ms": _elapsed_ms(t0),
                        }
                    )

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
                    llm_calls.append(
                        {
                            "role": f"execution_fix_{exec_fix_i + 1}",
                            "duration_ms": _elapsed_ms(t0),
                            "model": config.llm_model,
                            "output": fix_answer,
                            "input_tokens": fix_usage.get("input_tokens")
                            if fix_usage
                            else None,
                            "output_tokens": fix_usage.get("output_tokens")
                            if fix_usage
                            else None,
                        }
                    )
                    messages.append({"role": "assistant", "content": fix_answer})

                    new_sql = SQLValidator.extract_sql(fix_answer)
                    if not new_sql:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": "无法提取 SQL",
                            }
                        )
                        break

                    currency_ok, currency_error, _ = (
                        SQLValidator.validate_currency_conversion(
                            new_sql,
                            result.query_intent,
                        )
                    )
                    if not currency_ok:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": (f"货币换算口径校验失败: {currency_error}"),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一版修复 SQL 破坏了货币换算口径："
                                    f"{currency_error}。后续修复必须保留完整换汇规则。"
                                ),
                            }
                        )
                        continue

                    projection_ok, projection_error, _ = _validate_requested_projection(
                        new_sql
                    )
                    if not projection_ok:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": projection_error,
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一版修复 SQL 删除了用户明确要求的结果字段："
                                    f"{projection_error}。后续修复必须保留这些字段。"
                                ),
                            }
                        )
                        continue

                    entity_ok, entity_error, _ = _validate_entity_filters(new_sql)
                    if not entity_ok:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": entity_error,
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一版修复 SQL 破坏了已确定的实体过滤条件："
                                    f"{entity_error}。后续修复必须使用配置指定的表、字段和值。"
                                ),
                            }
                        )
                        continue

                    # 白名单校验
                    access_allowed, access_error, new_dbs = (
                        SQLValidator.validate_database_access(new_sql, authorized_dbs)
                    )
                    if not access_allowed:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": f"安全拦截: {access_error}",
                                "database": ", ".join(sorted(new_dbs)),
                            }
                        )
                        break

                    # EXPLAIN 校验
                    fix_check = validator.explain(new_sql)
                    if not fix_check["valid"]:
                        trace_steps.append(
                            {
                                "step": "execution_fix",
                                "attempt": exec_fix_i + 1,
                                "success": False,
                                "reason": f"EXPLAIN 失败: {fix_check['error']}",
                            }
                        )
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

                    trace_steps.append(
                        {
                            "step": "execution_fix",
                            "attempt": exec_fix_i + 1,
                            "fixed_sql": current_exec_sql,
                            "success": exec_result["success"],
                            "error": exec_result.get("error", ""),
                            "duration_ms": _elapsed_ms(t0),
                        }
                    )

                    if exec_result["success"]:
                        final_sql = current_exec_sql
                        answer = fix_answer
                        logger.info(f"执行纠错成功 (第 {exec_fix_i + 1} 次)")
                        break
                else:
                    logger.warning(
                        f"执行纠错 {config.max_execute_fix_retries} 次后仍失败"
                    )

            if exec_result["success"]:
                query_result_data = {
                    "columns": exec_result["columns"],
                    "rows": exec_result["rows"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                }
            else:
                execution_error = exec_result["error"] or "SQL 执行失败"

    if turn_intent == "result_explanation" and final_sql and is_success:
        result_excerpt = None
        if query_result_data is not None:
            result_excerpt = {
                "columns": query_result_data.get("columns", []),
                "rows": query_result_data.get("rows", [])[:20],
                "row_count": query_result_data.get("row_count", 0),
                "truncated": query_result_data.get("truncated", False),
            }
        explanation_prompt = (
            "用户要求解释上一轮查询。请仅依据给出的 SQL 和执行结果，用简洁中文说明"
            "查询口径、关键过滤、分组维度和结果含义；不得补造未提供的数据。"
            "如果没有执行结果，只解释 SQL 口径并明确说明没有可解释的结果数据。\n\n"
            f"用户问题：{question}\n"
            f"SQL：\n{final_sql}\n"
            "执行结果：" + json.dumps(result_excerpt, ensure_ascii=False, default=str)
        )
        t0 = _time.monotonic()
        explanation_answer, explanation_usage = _invoke(
            [
                {
                    "role": "system",
                    "content": "你是数据库查询结果解释助手，只能使用提供的证据。",
                },
                {"role": "user", "content": explanation_prompt},
            ]
        )
        summary = str(explanation_answer or "").strip()
        trace_steps.append(
            {
                "step": "result_explanation",
                "duration_ms": _elapsed_ms(t0),
                "has_query_result": result_excerpt is not None,
                "input_tokens": explanation_usage.get("input_tokens")
                if explanation_usage
                else None,
                "output_tokens": explanation_usage.get("output_tokens")
                if explanation_usage
                else None,
            }
        )

    # 提取命中的 fewshot 信息（id + question）
    matched_fewshot = [
        {"id": ex.get("id"), "question": ex.get("question", "")}
        for ex in result.relevant_examples
        if ex.get("id") is not None
    ]

    # 构建本轮摘要供下一轮使用（仅成功时更新，失败轮不污染上下文）
    context_summary = ""
    presentation_state = ContextCompressor.update_presentation_state(
        filtered_previous_presentation_state,
        tool_calls,
        relation=presentation_relation,
        context_relation=context_relation,
    )
    if is_success and final_sql:
        summary_question = effective_question
        summary_query_state = query_state
        if turn_intent in {"result_explanation", "result_operation"}:
            summary_question = previous_state["question"] or effective_question
            summary_query_state = previous_state["query_state"]
        context_summary = ContextCompressor.build_summary(
            summary_question,
            matched_tables,
            final_sql,
            query_state=summary_query_state,
            presentation_state=presentation_state,
        )

    # 汇总 trace
    trace = {
        "question": question,
        "effective_question": effective_question,
        "steps": trace_steps,
        "total_duration_ms": _elapsed_ms(t_start),
        "tool_calls": tool_calls,
        "presentation_state": presentation_state,
    }

    return {
        "sql": final_sql,
        "raw_answer": answer,
        "matched_tables": matched_tables,
        "matched_terms": matched_terms,
        "enum_hits": result.enum_hits,
        "retrieval_context": {
            "selected_columns": {
                table["table_name"]: table.get("selected_columns", [])
                for table in result.relevant_tables
            },
            "join_paths": result.join_paths,
            "matched_terms": result.matched_terms,
            "required_columns": result.required_columns,
            "semantic_table_evidence": semantic_table_evidence,
            "inferred_biz_line": result.inferred_biz_line,
            "context_stats": result.context_stats,
            "query_intent": result.query_intent,
            "requested_fields": result.requested_fields,
        },
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
        "needs_clarification": clarification is not None,
        "clarification": clarification,
        "interpretation": interpretation,
        "query_state": query_state.to_dict(),
        "turn_intent": turn_intent,
        "context_relation": context_relation,
        "tool_calls": tool_calls,
    }


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

    engine = create_doris_engine_for_agent(config.agent_id)
    try:
        result = await anyio.to_thread.run_sync(
            lambda: SQLValidator(engine).execute(
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
        _query_context_fingerprint(req.question, req.history_summary)
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

    def replay_lark_response() -> QueryResponse | None:
        if not is_lark_request:
            return None
        replay = query_logger.get_lark_response(
            trace_id=meta.trace_id,
            user_id=meta.user_id,
            agent_id=config.agent_id,
            user_query=req.question,
        )
        if replay is not None:
            replay_log_id, replay_response = replay
            try:
                return QueryResponse.model_validate(
                    {**replay_response, "log_id": replay_log_id}
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
            cached_response = QueryResponse(
                session_id=session_id,
                question=req.question,
                execution_time_ms=elapsed_ms,
                **{k: v for k, v in cached.items() if k in QueryResponse.model_fields},
            )
            if is_lark_request:
                snapshot = QueryLogger.lark_response_snapshot(
                    cached_response.model_dump(mode="json", exclude={"log_id"})
                )
                log_id = query_logger.log(
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
                    "tool_calls": result.get("tool_calls", []),
                },
                context_key=_cache_ctx,
            )

        response = QueryResponse(
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
            tool_calls=result.get("tool_calls", []),
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
    import re

    if not sql:
        return ""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.upper()
