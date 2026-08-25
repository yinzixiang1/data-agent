from src.retrieval.tool_planner import explicitly_requested_tools


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
