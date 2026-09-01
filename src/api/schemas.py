"""Validated request and response contracts for the Agent HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


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
        description='场景专属元数据 (如 {"param_mode":true,"placeholder_fields":[...]})',
    )


class PreparedQueryContext(BaseModel):
    """One-time model output reused by the remaining NL2SQL pipeline."""

    input_fingerprint: str = Field(..., min_length=64, max_length=64)
    effective_question: str = Field(..., min_length=1, max_length=2000)
    query_question: str = Field(..., min_length=1, max_length=2000)
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
    interaction_calls: list[dict] = Field(
        default_factory=list,
        description="经当前 Agent、渠道和输入 Schema 校验后的会话交互工具调用",
    )


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
        description="上一轮由 Agent 返回的结构化 context_summary",
    )
    expand_info: dict | None = Field(
        default=None,
        description="扩展参数，按需覆盖 Agent 配置（如 enable_execute, row_limit 等）",
    )
    prepared_context: PreparedQueryContext | None = Field(
        default=None,
        description="由 /query/prepare 返回的一次性轮次识别结果；传入后不重复调用模型识别",
    )
    previous_result_snapshot_id: str = Field(
        default="",
        max_length=128,
        description="上一轮成功查询返回的短期结果快照 ID，供结果操作复用",
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
    matched_tables: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    enum_hits: list[dict] = Field(default_factory=list)
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
    result_snapshot_id: str = Field(
        default="",
        description="本次查询结果的短期快照 ID，后续结果操作应原样带回",
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
    interaction_calls: list[dict] = Field(
        default_factory=list,
        description="需要由接入渠道呈现或确认的会话交互工具调用",
    )
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="经注册表与输入 Schema 校验后的结果工具调用",
    )
    tool_results: list[dict] = Field(
        default_factory=list,
        description="Agent 侧受控结果工具的结构化执行结果",
    )


class IndexRebuildRequest(BaseModel):
    force: bool = Field(default=True, description="是否强制重建")
    collections: list[str] = Field(
        default_factory=list,
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
    results: list[dict] = Field(default_factory=list)
