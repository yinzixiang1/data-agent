from src.retrieval.tool_planner import (
    explicitly_requested_tools,
    extract_planned_tool_calls,
    tool_planning_messages,
)


def test_chart_type_phrase_matches_the_registered_result_tool() -> None:
    tools = [
        {
            "name": "render_chart",
            "display_name": "生成图表",
            "intent_phrases": ["生成图表", "折线图", "柱状图", "饼图"],
        },
        {
            "name": "export_result",
            "display_name": "下载数据",
            "intent_phrases": ["下载数据", "导出文件"],
        },
    ]

    matched = explicitly_requested_tools(
        "查询渠道今年所有的出金交易，按照月份生成折线图",
        tools,
    )

    assert [tool["name"] for tool in matched] == ["render_chart"]


def test_analysis_phrase_matches_the_registered_result_tool() -> None:
    tools = [
        {
            "name": "analyze_result",
            "display_name": "智能分析",
            "intent_phrases": ["分析结果", "趋势分析", "异常分析"],
        },
        {
            "name": "export_result",
            "display_name": "下载数据",
            "intent_phrases": ["下载数据", "导出文件"],
        },
    ]

    matched = explicitly_requested_tools("查询月度金额并分析结果", tools)

    assert [tool["name"] for tool in matched] == ["analyze_result"]


def test_analysis_tool_accepts_compound_modes() -> None:
    tools = [
        {
            "name": "analyze_result",
            "requires_query_result": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "modes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["trend", "anomaly"],
                        },
                        "minItems": 1,
                        "maxItems": 2,
                    }
                },
                "additionalProperties": False,
            },
        }
    ]

    calls = extract_planned_tool_calls(
        '{"actions":[{"name":"analyze_result",'
        '"arguments":{"modes":["trend","anomaly"]}}]}',
        tools,
    )

    assert calls == [
        {
            "name": "analyze_result",
            "arguments": {"modes": ["trend", "anomaly"]},
            "requires_query_result": True,
        }
    ]


def test_personal_runtime_config_never_enters_tool_planning_prompt() -> None:
    messages = tool_planning_messages(
        "分析当前结果",
        [
            {
                "name": "analyze_result",
                "display_name": "智能分析",
                "description": "分析已查询的数据",
                "input_schema": {"type": "object", "properties": {}},
                "runtime_config": {
                    "user_config": {"detail_level": "concise"},
                    "user_skill": {
                        "name": "private-skill-name",
                        "instructions": "private-instructions",
                    },
                },
            }
        ],
    )

    prompt = messages[-1]["content"]
    assert "analyze_result" in prompt
    assert "runtime_config" not in prompt
    assert "private-skill-name" not in prompt
    assert "private-instructions" not in prompt
