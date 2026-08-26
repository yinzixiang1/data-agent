from src.retrieval.tool_planner import (
    explicitly_requested_tools,
    extract_planned_tool_calls,
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
