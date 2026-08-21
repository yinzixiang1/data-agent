import json
from types import SimpleNamespace

from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.context_compressor import ContextCompressor, QueryState


class _SessionModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.calls.append(prompt)
        if "查询状态更新助手" in prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_modify",
                        "query_state": {
                            "subject": "订单交易",
                            "time_range": "最近一个月",
                            "filters": ["只保留成功订单", "只保留指定订单类型"],
                            "metrics": ["交易笔数", "交易金额"],
                            "dimensions": ["渠道", "订单类型"],
                            "currency_conversion": "",
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["最近一个月", "渠道", "交易笔数", "交易金额"],
                            "set": ["成功订单", "指定订单类型"],
                            "removed": [],
                        },
                        "effective_question": (
                            "最近一个月按渠道和订单类型统计成功订单的交易笔数和交易金额"
                        ),
                        "interpretation": "保留上一轮结构并追加两个筛选条件。",
                        "confidence": 0.98,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            content="""```sql
SELECT o.channel_code,
       o.order_type,
       COUNT(*) AS order_count,
       SUM(o.source_amount) AS source_amount
FROM analytics.orders AS o
WHERE o.status = 'SUCCESS'
  AND o.order_type = 'configured_value'
  AND o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY o.channel_code, o.order_type
```""",
            usage_metadata=None,
        )


class _SessionRetriever:
    def __init__(self):
        self.kwargs = {}

    def retrieve(self, _query, **kwargs):
        self.kwargs = kwargs
        schema = {
            "table_name": "analytics.orders",
            "table_name_short": "orders",
            "display_name": "订单事实表",
            "columns": [
                {"name": "channel_code"},
                {"name": "order_type"},
                {"name": "status"},
                {"name": "source_amount"},
                {"name": "create_time"},
            ],
        }
        return SimpleNamespace(
            relevant_tables=[
                {
                    "table_name": "analytics.orders",
                    "schema": schema,
                    "score": 0.9,
                    "selected_columns": [
                        "channel_code",
                        "order_type",
                        "status",
                        "source_amount",
                        "create_time",
                    ],
                }
            ],
            relevant_examples=[],
            enum_hits=[],
            value_hits=[],
            business_context="",
            prompt_text="Use the retrieved schema and configured glossary evidence.",
            matched_terms=["configured order type"],
            required_columns=[],
            join_paths=[],
            inferred_biz_line="",
            context_stats={},
            query_intent={"currency_conversion": False, "count_only": False},
            requested_fields=[],
            entity_filters=[],
            unresolved_entities=[],
            rejected_terms=[],
        )


class _ClarificationSessionModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "relation": "follow_up_modify",
                    "query_state": {
                        "subject": "订单交易",
                        "time_range": "最近一个月",
                        "filters": [],
                        "metrics": ["交易次数"],
                        "dimensions": ["渠道"],
                        "currency_conversion": "",
                        "exclusions": [],
                    },
                    "changes": {
                        "kept": ["最近一个月", "交易次数"],
                        "set": ["更换渠道"],
                        "removed": [],
                    },
                    "effective_question": "最近一个月按新渠道统计订单交易次数",
                    "interpretation": "渠道对应的记录类型存在歧义。",
                    "confidence": 0.7,
                    "needs_clarification": True,
                    "clarification": {
                        "question": "要查询哪类渠道记录？",
                        "options": [
                            {
                                "label": "订单记录",
                                "value": "使用订单事实记录",
                            }
                        ],
                    },
                },
                ensure_ascii=False,
            )
        )


class _SqlClarificationModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        return SimpleNamespace(
            content=(
                'NEED_CLARIFY: {"question":"请选择统计维度",'
                '"options":[{"label":"按天","value":"按天统计"}]}'
            ),
            usage_metadata=None,
        )


class _UnavailableExplainValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, _answer):
        self.calls += 1
        return {
            "valid": False,
            "sql": None,
            "error": "connection reset by peer",
            "plan": None,
            "connection_retries": 1,
            "infrastructure_error": True,
        }


def test_follow_up_pipeline_inherits_verified_sql_structure(monkeypatch):
    import app as service

    previous_sql = """
    SELECT o.channel_code,
           COUNT(*) AS order_count,
           SUM(o.source_amount) AS source_amount
    FROM analytics.orders AS o
    WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY o.channel_code
    """
    history = ContextCompressor.build_summary(
        "最近一个月按渠道统计订单交易笔数和交易金额",
        ["analytics.orders"],
        previous_sql,
        query_state=QueryState(
            subject="订单交易",
            time_range="最近一个月",
            metrics=("交易笔数", "交易金额"),
            dimensions=("渠道",),
        ),
    )
    model = _SessionModel()
    retriever = _SessionRetriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
    )

    result = service._run_query_impl(
        "只看成功的，再限制为配置中定义的订单类型",
        config,
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert retriever.kwargs["inherited_tables"] == {"analytics.orders"}
    assert {
        "analytics.orders.channel_code",
        "analytics.orders.source_amount",
        "analytics.orders.create_time",
    } <= retriever.kwargs["inherited_columns"]
    assert "o.status = 'SUCCESS'" in result["sql"]
    assert "GROUP BY o.channel_code, o.order_type" in result["sql"]
    assert "上一轮成功结果（本轮结构基线）" in model.calls[-1]
    assert previous_sql.strip() in model.calls[-1]
    assert len(model.calls) == 2


def test_output_requirement_query_is_separate_from_transport_transcript(monkeypatch):
    import app as service

    model = _SessionModel()
    retriever = _SessionRetriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
    )

    result = service._run_query_impl(
        "原问题\n用户确认：按天展示注册数量趋势",
        config,
        model,
        metadata_context={"output_requirement_query": "原问题\n按天展示注册数量趋势"},
    )

    assert result["is_success"] is True, result["error"]
    assert retriever.kwargs["requested_field_query"] == ("原问题\n按天展示注册数量趋势")


def test_follow_up_clarification_includes_verified_table_references(monkeypatch):
    import app as service

    history = ContextCompressor.build_summary(
        "最近一个月按渠道统计订单交易次数",
        ["analytics.orders"],
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code",
    )
    model = _ClarificationSessionModel()
    retriever = _SessionRetriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
    )

    result = service._run_query_impl(
        "换个渠道",
        config,
        model,
        history_summary=history,
    )

    assert result["needs_clarification"] is True
    assert result["matched_tables"] == ["analytics.orders"]
    assert result["clarification"]["table_references"] == [
        {"name": "analytics.orders", "description": "订单事实表"}
    ]
    assert len(model.calls) == 1

    response = service.QueryResponse(
        session_id="session-1",
        question="换个渠道",
        **{
            key: value
            for key, value in result.items()
            if key in service.QueryResponse.model_fields
        },
    )
    assert response.model_dump()["clarification"]["table_references"] == [
        {"name": "analytics.orders", "description": "订单事实表"}
    ]


def test_clarification_defers_explicit_result_tool_without_planner_call(monkeypatch):
    import app as service

    model = _SqlClarificationModel()
    retriever = _SessionRetriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
        tools=[
            {
                "name": "render_chart",
                "display_name": "生成图表",
                "description": "根据查询结果生成图表",
                "intent_phrases": ["生成图表", "图表"],
                "requires_query_result": True,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        ],
    )

    result = service._run_query_impl(
        "统计最近一个月的用户分布并生成图表",
        config,
        model,
    )

    assert result["needs_clarification"] is True
    assert result["tool_calls"] == [
        {
            "name": "render_chart",
            "arguments": {},
            "requires_query_result": True,
        }
    ]
    assert len(model.calls) == 1
    assert any(
        step["step"] == "tool_intent_deferred" for step in result["trace"]["steps"]
    )
    assert not any(step["step"] == "tool_planning" for step in result["trace"]["steps"])


def test_explain_infrastructure_failure_does_not_trigger_llm_rewrite(monkeypatch):
    import app as service

    model = _SessionModel()
    retriever = _SessionRetriever()
    explain_validator = _UnavailableExplainValidator()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", explain_validator)
    config = AgentRuntimeConfig(
        enable_explain=True,
        enable_execute=False,
        enable_enum_validate=False,
        max_fix_retries=2,
    )

    result = service._run_query_impl(
        "按渠道统计订单",
        config,
        model,
    )

    assert result["is_success"] is False
    assert result["error"] == "connection reset by peer"
    assert explain_validator.calls == 1
    assert len(model.calls) == 1
