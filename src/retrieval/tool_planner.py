"""Generic structured tool-call planning for query result actions."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALLS_PATTERN = re.compile(r"(?:^|\n)TOOL_CALLS\s*:\s*", re.IGNORECASE)
_SUPPORTED_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

_TOOL_PLANNER_SYSTEM_PROMPT = """你是查询结果交付方式分类器。
只做声明式分类，不执行任何操作，不生成 SQL，也不补充业务语义。
latest_user_request 是用户本轮的原始请求，必须优先用它判断动作意图；
query_context 只说明动作所作用的数据查询，不得覆盖或改写本轮动作意图。
active_actions 是用户在上一轮已明确要求并仍然生效的展示动作；必须继续输出，不能因为本轮只修改查询条件、维度或指标就删除。
active_actions 的参数需要与 query_projection 重新对齐；仍适用的参数应保留，引用失效字段的参数必须按当前投影修正。
只能依据这些请求信息和提供的动作名称、显示名称、描述及意图短语做决定。
如果请求中的结果呈现、交付、保存或传递意图与某项动作的能力语义匹配，必须选择该动作；
请求同时包含数据查询不影响动作选择，也不要求用户逐字说出动作名称。
每个 arguments 必须满足对应 input_schema：不得遗漏 required 字段，枚举值必须来自 enum，
additionalProperties 为 false 时不得增加未声明字段，数组数量必须满足 minItems 和 maxItems。
input_schema 中的可选参数在用户请求存在明确依据时应一并填写；没有依据时不得猜测。
query_projection 非空时，它是本轮查询的输出字段证据；动作参数引用字段时必须逐字使用其中的输出别名。
返回且只返回一个 JSON 对象，格式为 {"actions":[{"name":"动作名称","arguments":{}}]}。
没有符合条件的动作时返回 {"actions":[]}。"""


def tool_instructions(
    tools: list[dict[str, Any]],
    *,
    choice: str = "auto",
) -> str:
    """Return a compact prompt section containing only callable tool evidence."""
    public_tools = _public_tool_contracts(tools)
    if not public_tools:
        return ""
    rendered = json.dumps(public_tools, ensure_ascii=False, separators=(",", ":"))
    decision_rule = (
        "本轮必须从可用工具中选择至少一项。"
        if choice == "required"
        else "根据用户的结果呈现意图决定是否调用；证据不足时不调用。"
    )
    ordinary_rule = "" if choice == "required" else "普通数据查询不要输出 TOOL_CALLS。"
    return (
        "\n\n【可用结果工具】\n"
        f"{rendered}\n"
        f"{decision_rule}需要调用时，在 SQL 代码块之后追加一行：\n"
        'TOOL_CALLS: [{"name":"工具名","arguments":{...}}]\n'
        "arguments 必须遵守对应 input_schema；不得调用未列出的工具。"
        f"{ordinary_rule}"
    )


def tool_planning_messages(
    question: str,
    tools: list[dict[str, Any]],
    *,
    choice: str = "auto",
    query_projection: list[str] | None = None,
    query_context: str = "",
    active_tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a focused, business-agnostic result-action planning request."""
    public_tools = _public_tool_contracts(tools)
    decision_rule = (
        "必须选择至少一个最符合请求的工具。"
        if choice == "required"
        else (
            "用户的结果动作意图与任一工具能力匹配时必须选择；"
            "只有完全没有结果动作意图时才返回空数组。"
        )
    )
    request = {
        "latest_user_request": question,
        "query_context": query_context,
        "active_actions": _active_action_contracts(active_tool_calls, tools),
        "available_actions": public_tools,
        "query_projection": [
            str(item) for item in (query_projection or []) if str(item).strip()
        ],
        "selection_rule": decision_rule,
    }
    return [
        {"role": "system", "content": _TOOL_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _active_action_contracts(
    calls: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available_names = {
        str(tool.get("name") or "").strip()
        for tool in tools
        if str(tool.get("name") or "").strip()
    }
    active = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments")
        if name not in available_names or not isinstance(arguments, dict):
            continue
        active.append({"name": name, "arguments": arguments})
    return active[:5]


def explicitly_requested_tools(
    question: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return tools whose registered action phrases occur in the latest request."""
    normalized_question = _normalize_intent_text(question)
    if not normalized_question:
        return []

    matched = []
    for tool in tools:
        phrases = tool.get("intent_phrases")
        if not isinstance(phrases, list):
            continue
        if any(
            normalized_phrase and normalized_phrase in normalized_question
            for phrase in phrases
            if (normalized_phrase := _normalize_intent_text(phrase))
        ):
            matched.append(tool)
    return matched


def extract_planned_tool_calls(
    answer: str,
    tools: list[dict[str, Any]],
    *,
    max_calls: int = 5,
) -> list[dict[str, Any]]:
    """Parse and validate the dedicated planner's JSON response."""
    payload = _decode_planner_payload(answer)
    if not isinstance(payload, dict):
        return []
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return []
    return _validate_tool_calls(actions, tools, max_calls=max_calls)


def declared_action_count(answer: str) -> int:
    """Return the number of syntactically declared actions before validation."""
    payload = _decode_planner_payload(answer)
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        return 0
    return len(payload["actions"])


def extract_tool_calls(
    answer: str,
    tools: list[dict[str, Any]],
    *,
    max_calls: int = 5,
) -> list[dict[str, Any]]:
    """Parse and validate model-proposed calls against registered tool schemas."""
    match = _TOOL_CALLS_PATTERN.search(answer or "")
    if match is None:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(
            (answer or "")[match.end() :].lstrip()
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []

    return _validate_tool_calls(payload, tools, max_calls=max_calls)


def _public_tool_contracts(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts = []
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        contracts.append(
            {
                "name": name,
                "display_name": str(tool.get("display_name") or name).strip(),
                "description": str(tool.get("description") or "").strip(),
                "intent_phrases": [
                    str(item).strip()
                    for item in (tool.get("intent_phrases") or [])
                    if str(item).strip()
                ],
                "input_schema": tool.get("input_schema")
                or {"type": "object", "properties": {}},
            }
        )
    return contracts


def _normalize_intent_text(value: Any) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").casefold())


def _decode_planner_payload(answer: str) -> Any:
    rendered = (answer or "").strip()
    if rendered.startswith("```"):
        first_line, separator, remainder = rendered.partition("\n")
        if separator and first_line.lower() in {"```", "```json"}:
            rendered = remainder
        if rendered.rstrip().endswith("```"):
            rendered = rendered.rstrip()[:-3].rstrip()
    object_start = rendered.find("{")
    if object_start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(rendered[object_start:])
        return payload
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _validate_tool_calls(
    payload: list[Any],
    tools: list[dict[str, Any]],
    *,
    max_calls: int,
) -> list[dict[str, Any]]:
    registry = {
        str(tool.get("name") or "").strip(): tool
        for tool in tools
        if str(tool.get("name") or "").strip()
    }
    validated = []
    for item in payload[: max(0, max_calls)]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        tool = registry.get(name)
        arguments = item.get("arguments")
        if tool is None or not isinstance(arguments, dict):
            continue
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        if not _valid_object(arguments, schema):
            continue
        validated.append(
            {
                "name": name,
                "arguments": arguments,
                "requires_query_result": bool(tool.get("requires_query_result")),
            }
        )
    return validated


def _valid_object(value: dict[str, Any], schema: dict[str, Any]) -> bool:
    if schema.get("type", "object") != "object":
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    required_names = (
        {str(item) for item in required} if isinstance(required, list) else set()
    )
    if not required_names.issubset(value):
        return False
    if schema.get("additionalProperties") is False and any(
        key not in properties for key in value
    ):
        return False
    return all(
        key in properties and _valid_value(item, properties[key])
        for key, item in value.items()
    )


def _valid_value(value: Any, schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    expected = schema.get("type")
    if expected not in _SUPPORTED_TYPES:
        return False
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }[expected]
    if not valid:
        return False
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    if expected == "array":
        min_items = schema.get("minItems")
        if (
            isinstance(min_items, int)
            and not isinstance(min_items, bool)
            and len(value) < min_items
        ):
            return False
        max_items = schema.get("maxItems")
        if (
            isinstance(max_items, int)
            and not isinstance(max_items, bool)
            and len(value) > max_items
        ):
            return False
        item_schema = schema.get("items")
        return isinstance(item_schema, dict) and all(
            _valid_value(item, item_schema) for item in value
        )
    if expected == "object":
        return _valid_object(value, schema)
    return True
