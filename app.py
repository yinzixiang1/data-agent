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

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from sqlalchemy import create_engine, text

from src.retrieval.retriever import SchemaRetriever
from src.retrieval.sql_validator import SQLValidator
from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.query_logger import QueryLogger
from src.retrieval.config import (
    DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASSWORD, DORIS_DATABASE,
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    MILVUS_URI, MILVUS_DB,
    EMBEDDING_MODEL, RERANKER_MODEL,
    DEFAULT_AGENT_TOKEN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 全局状态 ──

retriever: SchemaRetriever | None = None
validator: SQLValidator | None = None
llm_client: OpenAI | None = None
agent_config: AgentRuntimeConfig | None = None
query_logger: QueryLogger | None = None
config_loader: AgentConfigLoader | None = None


# ── 启动配置打印 ──

def print_infra_config():
    """打印基础设施配置（不随 Agent 变化的部分）。"""
    lines = [
        "",
        "=" * 60,
        "  NL2SQL Data Agent 基础设施配置",
        "=" * 60,
        "",
        "  [Doris]",
        f"    Host:     {DORIS_HOST}:{DORIS_PORT}",
        f"    User:     {DORIS_USER}",
        f"    Database: {DORIS_DATABASE}",
        "",
        "  [MySQL 语义层]",
        f"    Host:     {MYSQL_HOST}:{MYSQL_PORT}",
        f"    User:     {MYSQL_USER}",
        f"    Database: {MYSQL_DATABASE}",
        "",
        "  [Milvus]",
        f"    URI:      {MILVUS_URI}",
        f"    Database: {MILVUS_DB}",
        "",
        "  [Embedding]",
        f"    Model:    {EMBEDDING_MODEL}",
        "",
        "  [Reranker]",
        f"    Model:    {RERANKER_MODEL}",
        "",
        "=" * 60,
    ]
    print("\n".join(lines))


def create_doris_engine():
    url = (
        f"mysql+pymysql://{DORIS_USER}:{DORIS_PASSWORD}"
        f"@{DORIS_HOST}:{DORIS_PORT}/{DORIS_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def create_llm_client(config: AgentRuntimeConfig) -> OpenAI:
    """根据 Agent 配置创建 LLM 客户端。"""
    return OpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )


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

    # 打印基础设施配置
    print_infra_config()

    # 加载 Agent 配置
    default_agent_id = os.getenv("DEFAULT_AGENT_ID")
    agent_config = load_agent_config(
        agent_id=int(default_agent_id) if default_agent_id else None
    )

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

    # 初始化 RAG
    retriever = SchemaRetriever()
    retriever.initialize()

    # 初始化 LLM
    llm_client = create_llm_client(agent_config)
    logger.info(f"LLM 已就绪 (provider={agent_config.llm_provider}, model={agent_config.llm_model})")

    # 初始化 EXPLAIN 校验器
    validator = SQLValidator(create_doris_engine())
    logger.info("EXPLAIN 校验器已就绪")

    # 初始化查询日志
    query_logger = QueryLogger()
    logger.info("查询日志已就绪")

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
    client: OpenAI,
    history_summary: str = "",
) -> dict:
    """
    执行一次完整的 NL2SQL 查询（RAG + LLM + EXPLAIN）。

    Args:
        question: 用户原始问题
        config: Agent 运行时配置
        client: OpenAI 客户端
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

    # 构建 LLM 调用的公共参数（某些供应商不支持 temperature）
    llm_kwargs = {"model": config.llm_model}
    is_anthropic = "anthropic" in (config.llm_base_url or "")
    if not is_anthropic:
        llm_kwargs["temperature"] = config.llm_temperature

    # 多轮上下文压缩
    effective_question = question
    if history_summary:
        t0 = _time.monotonic()
        from src.retrieval.context_compressor import ContextCompressor
        compressor = ContextCompressor(client, model=config.llm_model, custom_prompt=config.compress_prompt)
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
                {"column": v.get("column", ""), "value": v.get("value", ""), "table": v.get("table_name", "")}
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
        {"role": "user", "content": f"## 用户问题\n{effective_question}\n\n{result.prompt_text}"},
    ]

    # 调用 LLM
    llm_calls = []
    t0 = _time.monotonic()
    resp = client.chat.completions.create(**llm_kwargs, messages=messages)
    answer = resp.choices[0].message.content
    usage = resp.usage
    llm_calls.append({
        "role": "initial",
        "duration_ms": _elapsed_ms(t0),
        "model": config.llm_model,
        "output": answer,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    })
    messages.append({"role": "assistant", "content": answer})

    # 提取 SQL
    extracted_sql = SQLValidator.extract_sql(answer)
    retry_count = 0
    is_success = True
    error_msg = ""

    if not extracted_sql:
        messages.append({
            "role": "user",
            "content": "你没有生成 SQL，请根据上面的表结构生成可执行的 SQL，用 ```sql ``` 包裹。",
        })
        t0 = _time.monotonic()
        resp = client.chat.completions.create(**llm_kwargs, messages=messages)
        answer = resp.choices[0].message.content
        usage = resp.usage
        llm_calls.append({
            "role": "retry_no_sql",
            "duration_ms": _elapsed_ms(t0),
            "model": config.llm_model,
            "output": answer,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
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
                resp = client.chat.completions.create(**llm_kwargs, messages=messages)
                answer = resp.choices[0].message.content
                usage = resp.usage
                llm_calls.append({
                    "role": f"explain_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.prompt_tokens if usage else None,
                    "output_tokens": usage.completion_tokens if usage else None,
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
            resp = client.chat.completions.create(**llm_kwargs, messages=messages)
            review_result = resp.choices[0].message.content
            usage = resp.usage
            llm_calls.append({
                "role": "plan_review",
                "duration_ms": _elapsed_ms(t0),
                "model": config.llm_model,
                "output": review_result,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
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

    # 提取命中的 fewshot 信息（id + question）
    matched_fewshot = [
        {"id": ex.get("id"), "question": ex.get("question", "")}
        for ex in result.relevant_examples
        if ex.get("id") is not None
    ]

    # 构建本轮摘要供下一轮使用
    from src.retrieval.context_compressor import ContextCompressor
    context_summary = ContextCompressor.build_summary(effective_question, final_sql)

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
        result = run_query(req.question, config, client, history_summary=req.history_summary)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 记录查询日志
        log_id = None
        meta = req.metadata or QueryMetadata()
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
