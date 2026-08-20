"""Generic structured tool-call planning for query result actions."""

from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALLS_PATTERN = re.compile(r"(?:^|\n)TOOL_CALLS\s*:\s*", re.IGNORECASE)
_SUPPORTED_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def tool_instructions(
    tools: list[dict[str, Any]],
    *,
    choice: str = "auto",
) -> str:
    """Return a compact prompt section containing only callable tool evidence."""
    public_tools = []
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        public_tools.append(
            {
                "name": name,
                "description": str(tool.get("description") or "").strip(),
                "input_schema": tool.get("input_schema")
                or {
                    "type": "object",
                    "properties": {},
                },
            }
        )
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
        item_schema = schema.get("items")
        return isinstance(item_schema, dict) and all(
            _valid_value(item, item_schema) for item in value
        )
    if expected == "object":
        return _valid_object(value, schema)
    return True
