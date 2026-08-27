import json

import pytest

from src.tools.analysis import AnalysisSkipped, build_analysis_facts, execute_analysis
from src.tools.executor import execute_agent_result_tools


def _monthly_result(*, truncated: bool = False) -> dict:
    return {
        "columns": ["month", "amount"],
        "rows": [
            ["2026-01", 10],
            ["2026-02", 12],
            ["2026-03", 11],
            ["2026-04", 13],
            ["2026-05", 12],
            ["2026-06", 14],
            ["2026-07", 13],
            ["2026-08", 90],
        ],
        "row_count": 8,
        "truncated": truncated,
    }


def test_analysis_facts_cover_trend_contribution_and_robust_anomaly() -> None:
    facts, profile = build_analysis_facts(_monthly_result())

    fact_types = {fact["type"] for fact in facts}
    assert {"dataset", "trend", "comparison", "contribution", "anomaly"} <= fact_types
    assert profile["metrics"] == ["amount"]
    assert profile["temporal_dimension"] == "month"
    anomaly = next(fact for fact in facts if fact["type"] == "anomaly")
    assert anomaly["values"]["label"] == "month=2026-08"
    assert anomaly["values"]["value"] == "90"


def test_compound_analysis_modes_keep_both_trend_and_anomaly_evidence() -> None:
    facts, _ = build_analysis_facts(
        _monthly_result(),
        modes=["trend", "anomaly"],
    )

    fact_types = {fact["type"] for fact in facts}
    assert {"dataset", "trend", "comparison", "anomaly", "distribution"} <= (fact_types)
    assert "contribution" not in fact_types


def test_single_metric_result_is_reported_as_a_value_not_a_fake_distribution() -> None:
    facts, _ = build_analysis_facts(
        {"columns": ["total_amount"], "rows": [["123.45"]], "row_count": 1}
    )

    distribution = next(fact for fact in facts if fact["type"] == "distribution")
    assert distribution["summary"] == "total_amount 的当前结果值为 123.45。"
    assert distribution["values"] == {
        "metric": "total_amount",
        "count": 1,
        "value": "123.45",
    }


def test_analysis_uses_only_deterministic_evidence_for_displayed_findings() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return (
            json.dumps(
                {
                    "title": "月度金额分析",
                    "executive_summary": "模型擅自声称业务增长了 999%。",
                    "findings": [
                        {
                            "type": "trend",
                            "statement": "模型擅自声称金额增长了 999%。",
                            "evidence_fact_ids": ["f4"],
                            "confidence": "high",
                        }
                    ],
                    "caveats": [],
                    "suggested_followups": ["按地区拆分金额"],
                },
                ensure_ascii=False,
            ),
            {"input_tokens": 20, "output_tokens": 10},
        )

    report, usage = execute_analysis(
        query_result=_monthly_result(),
        arguments={"modes": ["trend", "anomaly"]},
        binding_config={"max_findings": 3},
        invoke=invoke,
    )

    finding = report["findings"][0]
    referenced_summary = finding["evidence"][0]["summary"]
    assert finding["statement"] == referenced_summary
    assert report["executive_summary"] == referenced_summary
    assert "999" not in json.dumps(report, ensure_ascii=False)
    assert usage == {"input_tokens": 20, "output_tokens": 10}
    assert captured_payload["analysis_modes"] == ["trend", "anomaly"]
    assert {"profile", "facts"} <= captured_payload.keys()
    assert {
        "latest_user_request",
        "effective_question",
        "sql",
        "query_state",
    }.isdisjoint(captured_payload)


def test_analysis_rejects_truncated_results_by_default() -> None:
    with pytest.raises(AnalysisSkipped, match="已截断"):
        execute_analysis(
            query_result=_monthly_result(truncated=True),
            arguments={},
            binding_config={},
            invoke=lambda _messages: ("{}", None),
        )


def test_personal_preferences_and_skill_are_scoped_to_result_analysis() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    report, _ = execute_analysis(
        query_result=_monthly_result(),
        arguments={},
        binding_config={"max_findings": 4},
        runtime_config={
            "user_config": {
                "focus_modes": ["trend", "anomaly"],
                "detail_level": "concise",
                "max_findings": 2,
                "audience_label": "private-team-name",
            },
            "user_skill": {
                "name": "渠道经营分析",
                "version": "1.2.0",
                "instructions": "先总结趋势，再解释贡献度。",
                "analysis_steps": ["定位变化最大的月份"],
                "examples": [{"output": "示例中的数字不能作为事实"}],
            },
        },
        invoke=invoke,
    )

    assert captured_payload["analysis_modes"] == ["trend", "anomaly"]
    assert captured_payload["detail_level"] == "concise"
    assert captured_payload["max_findings"] == 2
    assert captured_payload["personal_skill"]["name"] == "渠道经营分析"
    assert "sql" not in captured_payload
    assert report["personalization"]["detail_level"] == "concise"
    assert report["personalization"]["skill_name"] == "渠道经营分析"
    assert report["personalization"]["skill_version"] == "1.2.0"
    assert report["personalization"]["effective_preference_keys"] == [
        "audience_label",
        "detail_level",
        "focus_modes",
        "max_findings",
    ]
    assert "private-team-name" not in json.dumps(report, ensure_ascii=False)
    assert report["personalization"]["sources"] == {
        "tool_defaults": False,
        "profile": True,
        "skill": True,
        "request_overrides": [],
    }


def test_analysis_preference_precedence_and_agent_limit() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    report, _ = execute_analysis(
        query_result=_monthly_result(),
        arguments={
            "focus_modes": ["comparison"],
            "detail_level": "deep",
            "max_findings": 9,
        },
        binding_config={"max_findings": 5},
        runtime_config={
            "tool_defaults": {
                "focus_modes": ["distribution"],
                "detail_level": "standard",
                "max_findings": 8,
            },
            "user_config": {
                "focus_modes": ["trend"],
                "detail_level": "standard",
                "max_findings": 6,
            },
            "user_skill": {
                "name": "我的分析方法",
                "preferences": {
                    "focus_modes": ["anomaly"],
                    "detail_level": "concise",
                    "max_findings": 4,
                },
                "analysis_steps": ["先比较，再总结"],
            },
        },
        invoke=invoke,
    )

    assert captured_payload["analysis_modes"] == ["comparison"]
    assert captured_payload["detail_level"] == "deep"
    assert captured_payload["max_findings"] == 5
    assert report["personalization"]["sources"] == {
        "tool_defaults": True,
        "profile": True,
        "skill": True,
        "request_overrides": ["detail_level", "focus_modes", "max_findings"],
    }


def test_verified_sql_projection_distinguishes_numeric_dimension_from_metric() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    report, _ = execute_analysis(
        query_result={
            "columns": ["month_number", "total_amount"],
            "rows": [[1, 10], [2, 20]],
            "row_count": 2,
        },
        arguments={},
        binding_config={"max_findings": 3},
        analysis_context={
            "sql": (
                "SELECT MONTH(created_at) AS month_number, "
                "SUM(amount) AS total_amount FROM orders "
                "GROUP BY MONTH(created_at)"
            ),
            "query_state": {},
        },
        invoke=invoke,
    )

    assert captured_payload["profile"]["metrics"] == ["total_amount"]
    assert captured_payload["profile"]["dimensions"] == ["month_number"]
    assert report["source"]["role_hints"] == {
        "metrics": ["total_amount"],
        "dimensions": ["month_number"],
    }


def test_explicit_analysis_modes_override_personal_default() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    execute_analysis(
        query_result=_monthly_result(),
        arguments={"modes": ["distribution"]},
        binding_config={"max_findings": 3},
        runtime_config={"user_config": {"focus_modes": ["anomaly"]}},
        invoke=invoke,
    )

    assert captured_payload["analysis_modes"] == ["distribution"]


def test_personal_skill_drops_oversized_nested_payloads() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    execute_analysis(
        query_result=_monthly_result(),
        arguments={},
        binding_config={"max_findings": 3},
        runtime_config={
            "user_skill": {
                "name": "oversized",
                "output": {"template": "x" * 5_000},
                "examples": [{"output": "y" * 21_000}],
            }
        },
        invoke=invoke,
    )

    assert captured_payload["personal_skill"]["output"] == {}
    assert captured_payload["personal_skill"]["examples"] == []


def test_analysis_does_not_silently_replace_an_invalid_requested_metric() -> None:
    with pytest.raises(AnalysisSkipped, match="缺少足够"):
        execute_analysis(
            query_result=_monthly_result(),
            arguments={"focus_metrics": ["profit_rate"]},
            binding_config={},
            invoke=lambda _messages: ("{}", None),
        )


def test_agent_executor_runs_only_agent_stage_tools() -> None:
    definitions = [
        {
            "name": "analyze_result",
            "executor_key": "analyze_result",
            "execution_stage": "agent_post_query",
            "binding_config": {"max_findings": 2},
        },
        {
            "name": "export_result",
            "executor_key": "export_result",
            "execution_stage": "channel_post_query",
        },
    ]
    calls = [
        {"name": "analyze_result", "arguments": {"modes": ["trend"]}},
        {"name": "export_result", "arguments": {"format": "xlsx"}},
    ]

    results = execute_agent_result_tools(
        calls,
        definitions,
        query_result=_monthly_result(),
        invoke=lambda _messages: ("{}", None),
    )

    assert [result["name"] for result in results] == ["analyze_result"]
    assert results[0]["status"] == "success"
    assert results[0]["output"]["findings"]


def test_agent_executor_applies_only_the_selected_tool_runtime_config() -> None:
    captured_payload = {}

    def invoke(messages):
        captured_payload.update(json.loads(messages[-1]["content"]))
        return "{}", None

    results = execute_agent_result_tools(
        [{"name": "analyze_result", "arguments": {}}],
        [
            {
                "name": "analyze_result",
                "executor_key": "analyze_result",
                "execution_stage": "agent_post_query",
                "binding_config": {"max_findings": 5},
                "runtime_config": {
                    "tool_defaults": {"detail_level": "standard"},
                    "user_config": {"detail_level": "concise"},
                },
            }
        ],
        query_result=_monthly_result(),
        analysis_context={
            "sql": "SELECT month, SUM(amount) AS amount FROM metrics GROUP BY month",
            "query_state": {},
        },
        invoke=invoke,
    )

    assert results[0]["status"] == "success"
    assert captured_payload["detail_level"] == "concise"
    assert results[0]["output"]["source"]["role_hints"] == {
        "metrics": ["amount"],
        "dimensions": ["month"],
    }


def test_agent_executor_does_not_analyze_a_sql_or_system_failure() -> None:
    invoked = False

    def invoke(_messages):
        nonlocal invoked
        invoked = True
        return "{}", None

    results = execute_agent_result_tools(
        [{"name": "analyze_result", "arguments": {"modes": ["anomaly"]}}],
        [
            {
                "name": "analyze_result",
                "executor_key": "analyze_result",
                "execution_stage": "agent_post_query",
            }
        ],
        query_result=None,
        missing_result_error="查询执行失败，不分析系统异常",
        invoke=invoke,
    )

    assert invoked is False
    assert results[0]["status"] == "skipped"
    assert results[0]["error"] == "查询执行失败，不分析系统异常"
