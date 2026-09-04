"""End-to-end NL2SQL retrieval, generation, validation, and execution pipeline."""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime as _datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.api.schemas import PreparedQueryContext
from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.clarification import (
    is_schema_grounding_failure,
    schema_grounding_clarification,
    table_references,
)
from src.retrieval.query_preparation import query_context_fingerprint
from src.retrieval.retriever import RetrievalResult, SchemaRetriever
from src.retrieval.sql_validator import SQLValidator
from src.retrieval.tool_planner import (
    declared_action_count,
    explicitly_requested_tools,
    extract_planned_tool_calls,
    tool_planning_messages,
)
from src.runtime.database import create_database_runtime, load_agent_databases
from src.tools.executor import execute_agent_result_tools

logger = logging.getLogger(__name__)


def run_query_pipeline(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    *,
    retriever: SchemaRetriever | None,
    validator: SQLValidator | None,
    history_summary: str = "",
    biz_line: str = "",
    metadata_filter: dict | None = None,
    metadata_context: dict | None = None,
    prepared_context: dict | None = None,
    previous_query_result: dict | None = None,
    result_tool_executor: Callable[..., list[dict]] = execute_agent_result_tools,
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
    dialect = getattr(validator, "dialect", None)
    dialect_sql_name = dialect.display_name if dialect is not None else "SQL"

    _ctx = metadata_context or {}
    allowed_tool_names = _ctx.get("enabled_tools")
    allowed = (
        {
            str(name).strip()
            for name in allowed_tool_names
            if isinstance(name, str) and name.strip()
        }
        if isinstance(allowed_tool_names, list)
        else None
    )
    callable_tools = [
        tool
        for tool in config.tools
        if allowed is None or str(tool.get("name") or "") in allowed
    ]
    interaction_tools = [
        tool
        for tool in callable_tools
        if str(tool.get("capability_kind") or "result") == "interaction"
        and str(tool.get("execution_stage") or "") == "conversation_pre_query"
    ]
    result_tools = [
        tool
        for tool in callable_tools
        if str(tool.get("capability_kind") or "result") == "result"
    ]

    def _elapsed_ms(t0):
        return int((_time.monotonic() - t0) * 1000)

    # 多轮上下文压缩
    from src.retrieval.context_compressor import (
        ContextCompressor,
        ContextMergeResult,
        QueryState,
    )

    effective_question = question
    query_question = question
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
    interaction_calls: list[dict] = []
    t0 = _time.monotonic()
    previous_state = ContextCompressor.parse_summary(history_summary)
    previous_query_state = QueryState.from_value(previous_state["query_state"])
    if history_summary:
        previous_sql = previous_state["sql"]
        previous_sql_context = previous_state["sql_context"]
        previous_presentation_state = previous_state["presentation_state"]
    if prepared_context is None:
        compressor = ContextCompressor(
            client,
            custom_prompt=config.compress_prompt,
            interaction_tools=interaction_tools,
        )
        merge_result = compressor.merge(history_summary, question)
    else:
        prepared = PreparedQueryContext.model_validate(prepared_context)
        if prepared.input_fingerprint != query_context_fingerprint(
            question, history_summary
        ):
            raise ValueError("prepared_context 与当前问题或历史上下文不匹配")
        merge_result = ContextMergeResult(
            effective_question=prepared.effective_question,
            query_question=prepared.query_question,
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
            interaction_calls=prepared.interaction_calls,
        )
    effective_question = merge_result.effective_question
    query_question = merge_result.query_question
    query_state = merge_result.query_state
    turn_intent = merge_result.turn_intent
    interpretation = merge_result.interpretation
    context_relation = merge_result.relation
    presentation_relation = merge_result.presentation_relation
    context_changes = merge_result.changes
    removed_sql_context = merge_result.removed_sql_context
    interaction_calls = merge_result.interaction_calls
    if turn_intent in {"result_explanation", "result_operation"} and previous_sql:
        # Result-only turns do not mutate the semantic query.  The classifier
        # chooses the route; state preservation is a deterministic contract.
        query_state = QueryState.from_value(previous_state["query_state"])
        effective_question = previous_state["question"] or effective_question
        query_question = previous_state["question"] or query_question
        context_relation = "follow_up_add"
        context_changes = {
            "kept": ["上一轮完整查询"],
            "set": [],
            "removed": [],
        }
        removed_sql_context = {section: [] for section in previous_sql_context}
    result_only_turn = turn_intent in {"result_explanation", "result_operation"}
    reuse_previous_result = result_only_turn and isinstance(previous_query_result, dict)
    metrics_were_replaced = (
        turn_intent == "sql_query"
        and context_relation != "new_question"
        and SQLValidator.metrics_replaced(
            previous_query_state.to_dict(),
            query_state.to_dict(),
        )
    )
    if metrics_were_replaced:
        normalized_removed_context = {
            section: list(removed_sql_context.get(section) or [])
            for section in previous_sql_context
        }
        removed_projections = normalized_removed_context.setdefault("projections", [])
        for projection in SQLValidator.aggregate_projections(previous_sql_context):
            if projection not in removed_projections:
                removed_projections.append(projection)
        removed_sql_context = normalized_removed_context
    context_needs_clarification = (
        merge_result.needs_clarification and turn_intent == "sql_query"
    )
    trace_steps.append(
        {
            "step": "context_compress",
            "duration_ms": _elapsed_ms(t0),
            "input": question,
            "output": effective_question,
            "query_question": query_question,
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
            "reuses_result_snapshot": reuse_previous_result,
        }
    )
    if history_summary and context_relation != "new_question":
        inherited_tables.update(previous_state["tables"])
        inherited_tables.update(previous_sql_context.get("tables", []))
        inherited_columns.update(previous_sql_context.get("columns", []))
    if context_needs_clarification:
        pending_clarification = merge_result.clarification

    if interaction_calls and turn_intent != "non_query":
        interaction_calls = []
        pending_clarification = {
            "question": "你同时提出了数据查询和配置操作。请确认这次先执行哪一个？",
            "options": [
                {"label": "先查询数据", "value": query_question},
                {"label": "先处理配置", "value": question},
            ],
            "table_references": [],
        }
        context_needs_clarification = True

    if turn_intent == "non_query" or interaction_calls:
        direct_response = merge_result.direct_response or (
            "已识别交互操作，请在接入渠道中继续完成。"
            if interaction_calls
            else "当前 Agent 用于数据库查询，请描述要查询的数据、范围和展示方式。"
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
                "query_question": query_question,
                "steps": trace_steps,
                "total_duration_ms": _elapsed_ms(t_start),
                "tool_calls": [],
                "interaction_calls": interaction_calls,
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
            "interaction_calls": interaction_calls,
            "tool_calls": [],
            "tool_results": [],
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
                "query_question": query_question,
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
            "tool_results": [],
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
            query_question,
            top_k=config.table_search_top_k,
            fewshot_k=config.fewshot_top_k,
            biz_line=biz_line,
            metadata_filter=metadata_filter,
            query_state=query_state.to_dict(),
            original_query=query_question,
            inherited_tables=inherited_tables,
            inherited_columns=inherited_columns,
        )
    retrieval_ms = _elapsed_ms(t0)

    matched_tables = [t["table_name"] for t in result.relevant_tables]
    matched_terms = result.matched_terms
    semantic_table_evidence = getattr(result, "semantic_table_evidence", [])
    requested_field_contract = result.requested_fields
    entity_filter_contract = result.entity_filters

    retrieved_schemas = [table.get("schema", {}) for table in result.relevant_tables]

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
            previous_query_state.to_dict(),
            query_state.to_dict(),
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

    # 召回候选只表示存在可参考的证据，不能推翻已经识别出的用户意图歧义。
    # 否则 Top1 表或最高分会在未确认口径时替用户作决定。
    if pending_clarification is not None:
        clarification_tables = table_references(result.relevant_tables)
        clarification = {
            **pending_clarification,
            "table_references": clarification_tables,
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
                "query_question": query_question,
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
            "tool_results": [],
        }

    # param_mode: 从 metadata.context 读取，替换输出规则生成 ? 占位符 SQL
    _param_mode = _ctx.get("param_mode", False)
    _placeholder_fields = _ctx.get("placeholder_fields", [])
    available_tools = list(result_tools)
    agent_bound_tool_names = [
        str(tool.get("name") or "")
        for tool in available_tools
        if str(tool.get("name") or "")
    ]
    raw_runtime_configs = _ctx.get("tool_runtime_configs")
    runtime_configs = (
        raw_runtime_configs if isinstance(raw_runtime_configs, dict) else {}
    )
    available_tools = [
        {
            **tool,
            "runtime_config": (
                runtime_configs.get(str(tool.get("name") or ""), {})
                if isinstance(
                    runtime_configs.get(str(tool.get("name") or ""), {}), dict
                )
                else {}
            ),
        }
        for tool in available_tools
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
    sticky_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in available_tools
        if str(tool.get("name") or "").strip()
        and str(tool.get("state_policy") or "sticky") == "sticky"
    }
    filtered_previous_presentation_state = {
        "tool_calls": [
            call
            for call in previous_presentation_state.get("tool_calls", [])
            if isinstance(call, dict)
            and str(call.get("name") or "").strip() in sticky_tool_names
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
        metric_change_rule = (
            "\n本轮已替换统计指标：必须重新生成聚合投影及其别名，"
            "不得保留上一轮指标的结果列别名。"
            if metrics_were_replaced
            else ""
        )
        prompt_text += (
            "\n\n【上一轮成功结果（本轮结构基线）】\n"
            "根据本轮完整查询状态修改此 SQL。用户未明确替换或删除的表、字段、"
            "展示维度、聚合方式和过滤条件必须保留；用户明确修改的内容以本轮为准。\n"
            f"本轮语义变更：{json.dumps(context_changes, ensure_ascii=False)}\n"
            "本轮允许替换或删除的上一轮 SQL 结构："
            f"{json.dumps(removed_sql_context, ensure_ascii=False)}"
            f"{metric_change_rule}\n"
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
            "7. 时间函数必须遵循本轮【SQL 方言】规则\n"
            "8. 只有当前证据能够唯一确定事实表、指标、字段、关联、筛选值和时间口径时才生成 SQL。"
            "只要存在多种合理解释、证据不足或证据冲突，必须输出："
            'NEED_CLARIFY: {"question":"需要确认的问题",'
            '"options":[{"label":"选项文案","value":"用于补充原问题的含义"}]}。'
            "候选项最多 4 个；没有可靠候选项时 options 输出空数组。"
            "不得用检索排名、最高分、Few-shot、常见做法或默认习惯替用户作决定"
        )
        prompt_text = re.sub(
            r"【输出要求】.*", param_mode_rules, prompt_text, flags=re.DOTALL
        )
    # 构建对话（注入当前日期，避免 LLM 因知识截止而误判年份）
    current_date = _datetime.now().astimezone().date().isoformat()
    dialect_rules = dialect.prompt_rules if dialect is not None else ""
    _system_content = (
        f"{config.system_prompt}\n\n{dialect_rules}\n\n当前日期: {current_date}"
    )
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
                "content": (
                    "你没有返回有效结果。请先判断现有证据能否唯一确定 SQL："
                    f"能够唯一确定时生成可执行的 {dialect_sql_name}，"
                    "并用 ```sql ``` 包裹；"
                    "不能唯一确定时必须返回 NEED_CLARIFY，不得猜测。"
                ),
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
    tool_results: list[dict] = []

    if not extracted_sql and not clarify_msg:
        is_success = False
        error_msg = "模型未生成可执行 SQL"

    inheritance_validation_attempts = []
    must_preserve_previous_structure = bool(
        previous_sql and context_relation != "new_question"
    )
    if extracted_sql and must_preserve_previous_structure and not result_only_turn:
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
                        "这是追问，但新 SQL 与已继承结构或本轮指标变更不一致。\n\n"
                        f"## 缺失结构或过期指标\n{inheritance_error}\n\n"
                        f"## 本轮语义变更\n{json.dumps(context_changes, ensure_ascii=False)}\n\n"
                        "## 允许删除的上一轮 SQL 结构\n"
                        f"{json.dumps(removed_sql_context, ensure_ascii=False)}\n\n"
                        "请以上一轮 SQL 为基线，只删除明确授权删除的结构，并恢复其他缺失的投影、"
                        "过滤、分组、关联、排序和限制。指标被替换时同步更新聚合投影别名。"
                        f"输出完整 {dialect_sql_name}，用 ```sql ``` 包裹。"
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
    if extracted_sql and calendar_day_window and is_success and not result_only_turn:
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
                        f"输出完整 {dialect_sql_name}，用 ```sql ``` 包裹。"
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
        and not result_only_turn
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
                        "汇率表必须 LEFT JOIN。"
                        f"请输出完整修复后的 {dialect_sql_name}，用 ```sql ``` 包裹。"
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
                        f"输出完整 {dialect_sql_name}，用 ```sql ``` 包裹。"
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
        and not result_only_turn
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
                        f"请输出完整修复后的 {dialect_sql_name}，用 ```sql ``` 包裹。"
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
    if extracted_sql and entity_filter_contract and is_success and not result_only_turn:
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
                        "同时保留完整换汇口径和用户要求展示的字段。"
                        f"请输出完整 {dialect_sql_name}，"
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

    schema_grounding_attempts = []
    if extracted_sql and is_success and not result_only_turn:
        max_schema_fix_retries = min(max(config.max_fix_retries, 0), 1)
        for attempt in range(max_schema_fix_retries + 1):
            schema_ok, schema_error, schema_detail = (
                SQLValidator.validate_schema_references(
                    extracted_sql,
                    retrieved_schemas,
                )
            )
            grounding_failure = is_schema_grounding_failure(schema_detail)
            schema_grounding_attempts.append(
                {
                    "attempt": attempt + 1,
                    "valid": schema_ok,
                    "error": schema_error,
                    "grounding_failure": grounding_failure,
                    **schema_detail,
                }
            )
            if schema_ok:
                break
            if not grounding_failure:
                break
            if attempt >= max_schema_fix_retries:
                clarification = schema_grounding_clarification(
                    effective_question,
                    schema_detail,
                )
                clarify_msg = str(clarification["question"])
                answer = "NEED_CLARIFY: " + json.dumps(
                    clarification,
                    ensure_ascii=False,
                )
                extracted_sql = None
                is_success = False
                error_msg = f"NEED_CLARIFY: {clarify_msg}"
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你生成的 SQL 使用了本轮检索证据未提供的表或字段。\n\n"
                        f"## Schema 证据问题\n{schema_error}\n\n"
                        "只能使用当前对话中已经提供的 Schema、术语、关系和枚举证据。"
                        "如果这些证据足以表达用户要求，"
                        f"请输出完整修复后的 {dialect_sql_name}；"
                        "如果证据不足，请不要猜测，输出 "
                        'NEED_CLARIFY: {"question":"需要用户补充的业务口径或字段映射",'
                        '"options":[]}。'
                    ),
                }
            )
            t0 = _time.monotonic()
            answer, usage = _invoke(messages)
            llm_calls.append(
                {
                    "role": f"schema_grounding_fix_{attempt + 1}",
                    "duration_ms": _elapsed_ms(t0),
                    "model": config.llm_model,
                    "output": answer,
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                }
            )
            messages.append({"role": "assistant", "content": answer})
            repaired_clarification = SQLValidator.extract_clarification(answer)
            if repaired_clarification is not None:
                clarification = repaired_clarification
                clarify_msg = clarification["question"]
                extracted_sql = None
                is_success = False
                error_msg = f"NEED_CLARIFY: {clarify_msg}"
                break
            repaired_sql = SQLValidator.extract_sql(answer)
            if not repaired_sql:
                clarification = schema_grounding_clarification(
                    effective_question,
                    schema_detail,
                )
                clarify_msg = str(clarification["question"])
                answer = "NEED_CLARIFY: " + json.dumps(
                    clarification,
                    ensure_ascii=False,
                )
                extracted_sql = None
                is_success = False
                error_msg = f"NEED_CLARIFY: {clarify_msg}"
                break
            extracted_sql = repaired_sql

        trace_steps.append(
            {
                "step": "schema_grounding_validate",
                "attempts": schema_grounding_attempts,
                "final_valid": bool(
                    schema_grounding_attempts and schema_grounding_attempts[-1]["valid"]
                ),
                "needs_clarification": clarification is not None,
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
        and not result_only_turn
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
        clarification_tables = table_references(result.relevant_tables)
        if clarification_tables:
            clarification = {
                **clarification,
                "table_references": clarification_tables,
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

    if final_sql and is_success and not result_only_turn:
        schema_allowed, schema_error, schema_detail = (
            SQLValidator.validate_schema_references(
                final_sql,
                retrieved_schemas,
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
            if is_schema_grounding_failure(schema_detail):
                clarification = schema_grounding_clarification(
                    effective_question,
                    schema_detail,
                )
                clarify_msg = str(clarification["question"])
                answer = "NEED_CLARIFY: " + json.dumps(
                    clarification,
                    ensure_ascii=False,
                )
                final_sql = ""
                error_msg = f"NEED_CLARIFY: {clarify_msg}"
            else:
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
        and not result_only_turn
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
                        retrieved_schemas,
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
    if final_sql and is_success and not result_only_turn:
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
                retrieved_schemas,
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
            *[
                tool
                for tool in available_tools
                if str(tool.get("trigger_mode") or "") == "always"
            ],
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
    query_result_data = previous_query_result if reuse_previous_result else None
    execution_error = ""
    summary = ""

    if result_only_turn:
        trace_steps.append(
            {
                "step": "result_snapshot_reuse",
                "available": reuse_previous_result,
                "sql_reexecuted": False,
            }
        )

    if (
        final_sql
        and is_success
        and (
            config.enable_execute
            or any(call.get("requires_query_result") for call in tool_calls)
        )
        and not _param_mode
        and validator
        and not result_only_turn
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
            execution_runtime = create_database_runtime(config.agent_id)
            exec_engine = execution_runtime.engine
            exec_validator = SQLValidator(exec_engine, execution_runtime.dialect)
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

                    execution_runtime = create_database_runtime(config.agent_id)
                    exec_engine = execution_runtime.engine
                    exec_validator_t = SQLValidator(
                        exec_engine, execution_runtime.dialect
                    )
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
                    execution_runtime = create_database_runtime(config.agent_id)
                    exec_engine = execution_runtime.engine
                    exec_validator_retry = SQLValidator(
                        exec_engine, execution_runtime.dialect
                    )
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

    should_execute_agent_tools = query_result_data is not None or (
        turn_intent == "result_operation" and not execution_error
    )
    if final_sql and is_success and tool_calls and should_execute_agent_tools:
        t0 = _time.monotonic()
        tool_results = result_tool_executor(
            tool_calls,
            available_tools,
            query_result=query_result_data,
            analysis_context={
                "query_state": query_state.to_dict(),
                "sql": final_sql,
                "sql_dialect": (
                    dialect.sqlglot_dialect if dialect is not None else "mysql"
                ),
            },
            missing_result_error=(
                "上一轮查询结果快照不存在或已过期，请重新执行数据查询后再分析"
                if turn_intent == "result_operation"
                else "查询没有产生可供工具处理的结果"
            ),
            invoke=_invoke,
        )
        if tool_results:
            trace_steps.append(
                {
                    "step": "agent_tool_execution",
                    "duration_ms": _elapsed_ms(t0),
                    "results": [
                        {
                            "name": item.get("name"),
                            "status": item.get("status"),
                            "duration_ms": item.get("duration_ms"),
                            "error": item.get("error"),
                        }
                        for item in tool_results
                    ],
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
    sticky_tool_calls = [
        call
        for call in tool_calls
        if str(call.get("name") or "").strip() in sticky_tool_names
    ]
    presentation_state = ContextCompressor.update_presentation_state(
        filtered_previous_presentation_state,
        sticky_tool_calls,
        relation=presentation_relation,
        context_relation=context_relation,
    )
    if is_success and final_sql:
        summary_question = query_question
        summary_query_state = query_state
        if turn_intent in {"result_explanation", "result_operation"}:
            summary_question = previous_state["question"] or query_question
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
        "query_question": query_question,
        "steps": trace_steps,
        "total_duration_ms": _elapsed_ms(t_start),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
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
        "interaction_calls": interaction_calls,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    }
