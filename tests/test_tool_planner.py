"""Structured result tool planning tests."""

from src.retrieval.tool_planner import (
    declared_action_count,
    explicitly_requested_tools,
    extract_planned_tool_calls,
    extract_tool_calls,
    tool_instructions,
    tool_planning_messages,
)


TOOLS = [
    {
        "name": "export_result",
        "display_name": "Download data",
        "description": "Export the current query result.",
        "intent_phrases": ["export data", "download data"],
        "requires_query_result": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["csv", "xlsx"]},
                "file_name": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["format"],
            "additionalProperties": False,
        },
    }
]


def test_instructions_only_expose_public_tool_contract() -> None:
    prompt = tool_instructions(TOOLS)

    assert "export_result" in prompt
    assert "TOOL_CALLS" in prompt
    assert "requires_query_result" not in prompt


def test_extract_tool_calls_validates_registered_name_and_schema() -> None:
    answer = (
        "```sql\nSELECT 1\n```\n"
        'TOOL_CALLS: [{"name":"export_result","arguments":'
        '{"format":"csv","columns":["渠道","交易笔数"]}}]\n'
    )

    assert extract_tool_calls(answer, TOOLS) == [
        {
            "name": "export_result",
            "arguments": {"format": "csv", "columns": ["渠道", "交易笔数"]},
            "requires_query_result": True,
        }
    ]


def test_extract_tool_calls_rejects_unknown_or_invalid_calls() -> None:
    answer = (
        'TOOL_CALLS: [{"name":"unknown","arguments":{}},'
        '{"name":"export_result","arguments":{"format":"pdf"}},'
        '{"name":"export_result","arguments":{"format":"csv","sql":"DROP"}}]\n'
    )

    assert extract_tool_calls(answer, TOOLS) == []


def test_extract_tool_calls_enforces_array_size_constraints() -> None:
    report_tool = {
        "name": "publish_report",
        "requires_query_result": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "charts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": {"type": "string"},
                }
            },
            "required": ["charts"],
            "additionalProperties": False,
        },
    }

    assert (
        extract_planned_tool_calls(
            '{"actions":[{"name":"publish_report","arguments":{"charts":[]}}]}',
            [report_tool],
        )
        == []
    )
    assert extract_planned_tool_calls(
        '{"actions":[{"name":"publish_report","arguments":{"charts":["趋势"]}}]}',
        [report_tool],
    )[0]["arguments"] == {"charts": ["趋势"]}
    assert (
        extract_planned_tool_calls(
            '{"actions":[{"name":"publish_report","arguments":'
            '{"charts":["趋势","比较","占比"]}}]}',
            [report_tool],
        )
        == []
    )


def test_planning_messages_only_use_registered_tool_evidence() -> None:
    messages = tool_planning_messages(
        "Please provide the result as a file",
        TOOLS,
        query_context="Show monthly transaction counts",
        query_projection=['channel AS "渠道"', 'COUNT(*) AS "交易笔数"'],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "export_result" in messages[1]["content"]
    assert "Download data" in messages[1]["content"]
    assert "Export the current query result." in messages[1]["content"]
    assert (
        '"latest_user_request":"Please provide the result as a file"'
        in messages[1]["content"]
    )
    assert '"query_context":"Show monthly transaction counts"' in messages[1]["content"]
    assert "download data" in messages[1]["content"]
    assert 'COUNT(*) AS \\"交易笔数\\"' in messages[1]["content"]
    assert "requires_query_result" not in messages[1]["content"]


def test_explicitly_requested_tools_match_registered_intent_phrases() -> None:
    assert explicitly_requested_tools("Please export data as Excel", TOOLS) == TOOLS
    assert explicitly_requested_tools("Show monthly transaction counts", TOOLS) == []

    report_tool = {
        "name": "publish_lark_report",
        "intent_phrases": ["生成报表", "生成看板"],
    }
    assert explicitly_requested_tools("生成报表", [report_tool]) == [report_tool]


def test_extract_planned_tool_calls_accepts_validated_json_object() -> None:
    answer = (
        '```json\n{"actions":[{"name":"export_result","arguments":'
        '{"format":"xlsx","file_name":"result"}}]}\n```'
    )

    assert extract_planned_tool_calls(answer, TOOLS) == [
        {
            "name": "export_result",
            "arguments": {"format": "xlsx", "file_name": "result"},
            "requires_query_result": True,
        }
    ]


def test_extract_planned_tool_calls_rejects_unregistered_or_invalid_calls() -> None:
    answer = (
        '{"actions":[{"name":"unknown","arguments":{}},'
        '{"name":"export_result","arguments":{"format":"pdf"}}]}'
    )

    assert extract_planned_tool_calls(answer, TOOLS) == []
    assert declared_action_count(answer) == 2
