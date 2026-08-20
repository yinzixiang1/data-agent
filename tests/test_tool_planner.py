"""Structured result tool planning tests."""

from src.retrieval.tool_planner import extract_tool_calls, tool_instructions


TOOLS = [
    {
        "name": "export_result",
        "description": "Export the current query result.",
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
