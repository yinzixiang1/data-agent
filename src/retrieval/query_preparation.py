"""One-time turn recognition shared by prepare and query endpoints."""

import hashlib
import json

from langchain_core.language_models import BaseChatModel

from src.api.schemas import PreparedQueryContext
from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.context_compressor import ContextCompressor


def query_context_fingerprint(question: str, history_summary: str) -> str:
    payload = json.dumps(
        {"question": question, "history_summary": history_summary},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_query_context(
    question: str,
    config: AgentRuntimeConfig,
    client: BaseChatModel,
    history_summary: str = "",
    enabled_tools: list[str] | None = None,
) -> dict:
    """Classify and merge one turn once, before retrieval and SQL generation."""
    allowed = (
        None
        if enabled_tools is None
        else {str(name).strip() for name in enabled_tools if str(name).strip()}
    )
    interaction_tools = [
        tool
        for tool in config.tools
        if str(tool.get("capability_kind") or "result") == "interaction"
        and str(tool.get("execution_stage") or "") == "conversation_pre_query"
        and (allowed is None or str(tool.get("name") or "") in allowed)
    ]
    compressor = ContextCompressor(
        client,
        custom_prompt=config.compress_prompt,
        interaction_tools=interaction_tools,
    )
    result = compressor.merge(history_summary, question)
    return PreparedQueryContext(
        input_fingerprint=query_context_fingerprint(question, history_summary),
        effective_question=result.effective_question,
        query_question=result.query_question,
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
        interaction_calls=result.interaction_calls,
    ).model_dump()
