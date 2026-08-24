import json
from types import SimpleNamespace

from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.context_compressor import ContextCompressor, QueryState
from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.retriever import SchemaRetriever
from src.retrieval.sql_validator import SQLValidator


class _InvalidMergeModel:
    def invoke(self, _messages):
        return SimpleNamespace(content="not-json")


class _FollowupModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "query_state": {
                            "subject": "订单交易",
                            "time_range": "最近一个月",
                            "filters": [],
                            "metrics": ["交易笔数"],
                            "dimensions": ["渠道", "订单类型"],
                            "currency_conversion": "",
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["最近一个月", "渠道", "交易笔数"],
                            "set": ["增加订单类型维度"],
                            "removed": [],
                        },
                        "effective_question": (
                            "最近一个月按渠道和订单类型统计订单交易笔数"
                        ),
                        "interpretation": "保留上一轮查询并增加订单类型维度。",
                        "confidence": 0.99,
                        "needs_clarification": True,
                        "clarification": {
                            "question": "请确认最终状态口径",
                            "options": [],
                        },
                    },
                    ensure_ascii=False,
                )
            )
        if len(self.calls) == 2:
            return SimpleNamespace(
                content="""```sql
SELECT o.order_type, COUNT(*) AS order_count
FROM analytics.orders AS o
WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY o.order_type
LIMIT 100
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content="""```sql
SELECT o.channel_code,
       o.order_type,
       COUNT(*) AS order_count
FROM analytics.orders AS o
WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
GROUP BY o.channel_code, o.order_type
ORDER BY order_count DESC
LIMIT 100
```""",
            usage_metadata=None,
        )


class _ComplexQuestionModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "new_question",
                        "query_state": {
                            "subject": "退款后的渠道事件和最终状态",
                            "time_range": "[2026-07-01, 2026-08-01)",
                            "filters": ["指定渠道"],
                            "metrics": [],
                            "dimensions": [],
                            "currency_conversion": "",
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": [],
                            "set": ["完整用户问题"],
                            "removed": [],
                        },
                        "effective_question": (
                            "查询指定渠道在 [2026-07-01, 2026-08-01) 内退款后的全部渠道事件，"
                            "判断渠道变化并展示最终状态"
                        ),
                        "interpretation": "保留事件顺序和派生结果要求。",
                        "confidence": 0.98,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            content="""```sql
SELECT t.order_id,
       t.event_type,
       t.event_time,
       t.event_data
FROM analytics.order_timeline AS t
WHERE t.event_time >= '2026-07-01'
  AND t.event_time < '2026-08-01'
ORDER BY t.order_id, t.event_time
LIMIT 100
```""",
            usage_metadata=None,
        )


class _TurnIntentModel:
    def __init__(self, turn_intent: str, direct_response: str = "") -> None:
        self.turn_intent = turn_intent
        self.direct_response = direct_response
        self.calls: list[str] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "turn_intent": self.turn_intent,
                        "query_state": {
                            "subject": "订单交易",
                            "time_range": "最近一个月",
                            "filters": [],
                            "metrics": ["交易笔数"],
                            "dimensions": ["渠道"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["上一轮查询"],
                            "set": [],
                            "removed": [],
                        },
                        "removed_sql_context": {},
                        "effective_question": "最近一个月按渠道统计订单交易笔数",
                        "interpretation": "沿用上一轮查询。",
                        "direct_response": self.direct_response,
                        "confidence": 0.99,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            content="上一轮查询按渠道统计订单笔数。",
            usage_metadata=None,
        )


class _CountOnlyModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "new_question",
                        "turn_intent": "sql_query",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": [],
                            "currency_conversion": "",
                            "result_shape": "count_only",
                            "exclusions": [],
                        },
                        "changes": {"kept": [], "set": ["只查数量"], "removed": []},
                        "removed_sql_context": {},
                        "effective_question": "只查订单总数量",
                        "interpretation": "仅返回一个计数结果。",
                        "direct_response": "",
                        "confidence": 0.99,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        if len(self.calls) == 2:
            return SimpleNamespace(
                content="""```sql
SELECT o.channel_code, COUNT(*) AS order_count
FROM analytics.orders AS o
GROUP BY o.channel_code
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content="""```sql
SELECT COUNT(*) AS order_count
FROM analytics.orders
```""",
            usage_metadata=None,
        )


class _Retriever:
    def __init__(self) -> None:
        self.query = ""
        self.kwargs = {}
        self.table_schemas = {
            "analytics.orders": {
                "database": "analytics",
                "table_name": "analytics.orders",
                "table_name_short": "orders",
                "description": "订单事实表",
                "columns": [
                    {"name": "channel_code", "type": "varchar"},
                    {"name": "order_count", "type": "bigint"},
                    {"name": "order_type", "type": "varchar"},
                    {"name": "create_time", "type": "datetime"},
                ],
            }
        }

    def retrieve(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        schema = {
            "database": "analytics",
            "table_name": "analytics.orders",
            "table_name_short": "orders",
            "description": "订单事实表",
            "columns": [
                {"name": "channel_code", "type": "varchar"},
                {"name": "order_type", "type": "varchar"},
                {"name": "create_time", "type": "datetime"},
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
                        "create_time",
                    ],
                }
            ],
            relevant_examples=[],
            enum_hits=[],
            value_hits=[],
            business_context="",
            prompt_text="Use only the retrieved schema evidence.",
            matched_terms=[],
            required_columns=[],
            join_paths=[],
            inferred_biz_line="",
            context_stats={},
            query_intent={"state": kwargs.get("query_state", {})},
            requested_fields=[],
            entity_filters=[],
            unresolved_entities=[],
            rejected_terms=[],
        )


class _ComplexRetriever:
    def __init__(self) -> None:
        self.query = ""
        self.kwargs = {}

    def retrieve(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        schema = {
            "database": "analytics",
            "table_name": "analytics.order_timeline",
            "table_name_short": "order_timeline",
            "description": "订单事件时间线",
            "columns": [
                {"name": "order_id", "type": "varchar"},
                {"name": "event_type", "type": "varchar"},
                {"name": "event_time", "type": "datetime"},
                {"name": "event_data", "type": "json"},
            ],
        }
        return SimpleNamespace(
            relevant_tables=[
                {
                    "table_name": "analytics.order_timeline",
                    "schema": schema,
                    "score": 0.95,
                    "selected_columns": [
                        "order_id",
                        "event_type",
                        "event_time",
                        "event_data",
                    ],
                }
            ],
            relevant_examples=[],
            enum_hits=[],
            value_hits=[],
            business_context=(
                "测试语义证据：事件先后顺序由 event_time 表达，事件扩展数据位于 event_data。"
            ),
            prompt_text="Use only the retrieved event schema and semantic evidence.",
            matched_terms=["事件顺序"],
            required_columns=[],
            join_paths=[],
            inferred_biz_line="",
            context_stats={},
            query_intent={"state": kwargs.get("query_state", {})},
            requested_fields=[],
            entity_filters=[],
            unresolved_entities=[],
            rejected_terms=[],
        )


class _GlossaryResolver:
    def __init__(self) -> None:
        self.query = ""

    def resolve(self, query, **_kwargs):
        self.query = query
        return {
            "enriched_query": query,
            "business_context": "",
            "related_tables": [],
            "related_columns": [],
            "matched_terms": [],
            "rejected_terms": [],
        }


class _HybridSearcher:
    def _find_explicit_tables(self, _query):
        return set()

    def search(self, _query, **_kwargs):
        return [
            {
                "table_name": "analytics.order_timeline",
                "score": 0.9,
                "schema": {
                    "database": "analytics",
                    "table_name": "analytics.order_timeline",
                    "table_name_short": "order_timeline",
                    "columns": [
                        {"name": "order_id", "type": "varchar"},
                        {"name": "event_type", "type": "varchar"},
                        {"name": "event_time", "type": "datetime"},
                        {"name": "event_data", "type": "json"},
                    ],
                },
            }
        ]

    def search_enums(self, _query, **_kwargs):
        return []


class _ContextPlanner:
    def add_join_bridges(self, candidates, **_kwargs):
        return candidates, []

    def resolve_requested_columns(self, candidates, requested_fields, query=""):
        return SchemaContextPlanner.resolve_requested_columns(
            candidates,
            requested_fields,
            query=query,
        )

    def prune_columns(self, candidates, _query, **_kwargs):
        for candidate in candidates:
            candidate["selected_columns"] = [
                column["name"] for column in candidate["schema"]["columns"]
            ]
        return candidates, {}


class _CapturingContextPlanner(_ContextPlanner):
    def __init__(self) -> None:
        self.required_columns: set[str] = set()

    def prune_columns(self, candidates, _query, **kwargs):
        self.required_columns = set(kwargs.get("required_columns") or set())
        return super().prune_columns(candidates, _query, **kwargs)


class _FewshotSelector:
    def select(self, **_kwargs):
        return []


class _ExampleGuidedFewshotSelector:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def select(self, **kwargs):
        tables = list(kwargs.get("tables") or [])
        self.calls.append(tables)
        if not tables:
            return [
                {
                    "question": "按渠道统计订单",
                    "sql": "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code",
                    "tables": ["analytics.orders"],
                }
            ]
        return []


class _ExampleGuidedSearcher(_HybridSearcher):
    def __init__(self) -> None:
        self.required_tables: set[str] = set()

    def search(self, _query, **kwargs):
        self.required_tables = set(kwargs.get("required_tables") or set())
        table_name = (
            "analytics.orders"
            if "analytics.orders" in self.required_tables
            else "analytics.order_timeline"
        )
        return [
            {
                "table_name": table_name,
                "score": 0.9,
                "pinned": table_name in self.required_tables,
                "schema": {
                    "database": "analytics",
                    "table_name": table_name,
                    "table_name_short": table_name.rsplit(".", 1)[-1],
                    "columns": [
                        {"name": "channel_code", "type": "varchar"},
                        {"name": "create_time", "type": "datetime"},
                    ],
                },
            }
        ]


class _Formatter:
    def format_all(self, **kwargs):
        return kwargs.get("intent_context", "")


def test_merge_fallback_keeps_previous_question_and_state() -> None:
    previous_state = QueryState(
        subject="订单交易",
        time_range="最近一个月",
        metrics=("交易笔数",),
        dimensions=("渠道",),
    )
    history = ContextCompressor.build_summary(
        "最近一个月按渠道统计订单交易笔数",
        ["analytics.orders"],
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code",
        query_state=previous_state,
    )

    result = ContextCompressor(_InvalidMergeModel()).merge(
        history,
        "再增加订单类型",
    )

    assert result.effective_question == (
        "最近一个月按渠道统计订单交易笔数\n用户补充：再增加订单类型"
    )
    assert result.query_state == previous_state
    assert result.changes["kept"] == ["上一轮完整查询"]


def test_additive_followup_contract_detects_dropped_sql_structure() -> None:
    previous_sql = """
    SELECT o.channel_code, COUNT(*) AS order_count
    FROM analytics.orders AS o
    WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY o.channel_code
    ORDER BY order_count DESC
    LIMIT 100
    """
    previous_context = ContextCompressor.extract_sql_context(previous_sql)
    dropped_sql = """
    SELECT o.order_type, COUNT(*) AS order_count
    FROM analytics.orders AS o
    WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY o.order_type
    LIMIT 100
    """

    valid, error, detail = SQLValidator.validate_followup_inheritance(
        dropped_sql,
        previous_context,
        "follow_up_add",
    )

    assert valid is False
    assert "projections" in detail["missing"]
    assert "dimensions" in detail["missing"]
    assert "order_by" in detail["missing"]
    assert "丢失" in error


def test_additive_followup_accepts_alias_qualification_after_join() -> None:
    previous_sql = """
    SELECT account_id, account_name, source_currency, SUM(source_amount) AS amount
    FROM analytics.orders
    WHERE status = 'completed'
    GROUP BY account_id, account_name, source_currency
    ORDER BY account_id, source_currency
    LIMIT 100
    """
    current_sql = """
    SELECT o.account_id, o.account_name, a.legal_entity_name,
           o.source_currency, SUM(o.source_amount) AS amount
    FROM analytics.orders AS o
    JOIN analytics.accounts AS a ON a.account_id = o.account_id
    WHERE o.status = 'completed'
    GROUP BY o.account_id, o.account_name, a.legal_entity_name, o.source_currency
    ORDER BY o.account_id, o.source_currency
    LIMIT 100
    """

    valid, error, detail = SQLValidator.validate_followup_inheritance(
        current_sql,
        ContextCompressor.extract_sql_context(previous_sql),
        "follow_up_add",
    )

    assert valid is True
    assert error == ""
    assert detail["missing"] == {}


def test_modify_followup_without_removal_still_preserves_sql_structure() -> None:
    previous = ContextCompressor.extract_sql_context(
        """
        SELECT account_id, account_name, COUNT(*) AS order_count
        FROM `dwd_bi_banking`.`banking_order_requests`
        GROUP BY account_id, account_name
        """
    )
    current = """
        SELECT account_id, COUNT(*) AS order_count
        FROM `dwd_bi_banking`.`banking_order_requests`
        GROUP BY account_id
    """

    valid, error, detail = SQLValidator.validate_followup_inheritance(
        current,
        previous,
        "follow_up_modify",
    )

    assert valid is False
    assert "account_name" in error
    assert detail["required"] is True


def test_modify_followup_only_allows_explicit_sql_fragment_removal() -> None:
    previous = ContextCompressor.extract_sql_context(
        """
        SELECT channel_code, COUNT(*) AS order_count
        FROM analytics.orders
        WHERE status = 'completed' AND create_time >= '2026-07-01'
        GROUP BY channel_code
        """
    )
    removed_context = {"filters": ["create_time >= '2026-07-01'"]}
    current = """
        SELECT channel_code, COUNT(*) AS order_count
        FROM analytics.orders
        WHERE status = 'completed' AND create_time >= '2026-08-01'
        GROUP BY channel_code
    """

    valid, error, detail = SQLValidator.validate_followup_inheritance(
        current,
        previous,
        "follow_up_modify",
        removed_context,
    )

    assert valid is True
    assert error == ""
    assert detail["allowed_removed"]["filters"] == ["create_time >= '2026-07-01'"]

    invalid, invalid_error, invalid_detail = SQLValidator.validate_followup_inheritance(
        """
        SELECT COUNT(*) AS order_count
        FROM analytics.orders
        WHERE status = 'completed' AND create_time >= '2026-08-01'
        """,
        previous,
        "follow_up_modify",
        removed_context,
    )

    assert invalid is False
    assert "channel_code" in invalid_error
    assert "projections" in invalid_detail["missing"]


def test_query_contract_reads_result_shape_and_currency_from_state() -> None:
    intent = {
        "state": {
            "currency_conversion": "usd",
            "result_shape": "count_only",
        }
    }

    assert SQLValidator.currency_conversion_target(intent) == "USD"
    assert SQLValidator.is_count_only(intent) is True
    assert SQLValidator.validate_metric_projection(
        "SELECT COUNT(*) AS total FROM analytics.orders",
        intent,
    )[0]
    assert not SQLValidator.validate_metric_projection(
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code",
        intent,
    )[0]


def test_pipeline_repairs_count_only_contract_from_nested_query_state(
    monkeypatch,
) -> None:
    import app as service

    model = _CountOnlyModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "只查订单数量",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            max_fix_retries=1,
        ),
        model,
    )

    assert result["is_success"] is True, result["error"]
    assert result["query_state"]["result_shape"] == "count_only"
    assert "SELECT COUNT(*) AS order_count" in result["sql"]
    assert "GROUP BY" not in result["sql"]
    projection_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "requested_projection_validate"
    )
    assert [attempt["valid"] for attempt in projection_step["attempts"]] == [
        False,
        True,
    ]


def test_retrieval_uses_model_state_without_regex_business_decisions(
    monkeypatch,
) -> None:
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_reranker", lambda: None)
    retriever = SchemaRetriever()
    retriever._initialized = True
    retriever.config = AgentRuntimeConfig(enable_reranker=False)
    retriever.glossary_resolver = _GlossaryResolver()
    retriever.value_indexer = None
    retriever.searcher = _HybridSearcher()
    retriever.context_planner = _ContextPlanner()
    retriever.fewshot = _FewshotSelector()
    retriever.formatter = _Formatter()
    state = {
        "subject": "退款后的事件与终态",
        "time_range": "[2026-07-01, 2026-08-01)",
        "filters": ["指定渠道"],
        "metrics": [],
        "dimensions": [],
        "currency_conversion": "",
        "exclusions": [],
    }

    result = retriever.retrieve(
        (
            "在 account_refund 后列出所有 send_to_channel 事件，"
            "不同渠道算换渠道，并展示最终状态"
        ),
        biz_line="banking",
        query_state=state,
    )

    assert result.query_intent == {"state": state}
    assert result.requested_fields == []
    assert result.required_columns == []
    assert "粗粒度查询摘要" in result.prompt_text
    assert "sys_exchange_rate" not in result.prompt_text


def test_retrieval_grounds_terms_with_effective_and_original_question(
    monkeypatch,
) -> None:
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_reranker", lambda: None)
    glossary = _GlossaryResolver()
    retriever = SchemaRetriever()
    retriever._initialized = True
    retriever.config = AgentRuntimeConfig(enable_reranker=False)
    retriever.glossary_resolver = glossary
    retriever.value_indexer = None
    retriever.searcher = _HybridSearcher()
    retriever.context_planner = _ContextPlanner()
    retriever.fewshot = _FewshotSelector()
    retriever.formatter = _Formatter()

    retriever.retrieve(
        "账户类型为公司主体",
        original_query="只筛选公司主体类型的账户",
        biz_line="banking",
    )

    assert glossary.query == (
        "账户类型为公司主体\n本轮用户原话：只筛选公司主体类型的账户"
    )


def test_fewshot_is_selected_after_schema_and_cannot_pin_wrong_table(
    monkeypatch,
) -> None:
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_reranker", lambda: None)
    searcher = _ExampleGuidedSearcher()
    fewshot = _ExampleGuidedFewshotSelector()
    context_planner = _CapturingContextPlanner()
    retriever = SchemaRetriever()
    retriever._initialized = True
    retriever.config = AgentRuntimeConfig(enable_reranker=False)
    retriever.glossary_resolver = _GlossaryResolver()
    retriever.value_indexer = None
    retriever.searcher = searcher
    retriever.context_planner = context_planner
    retriever.fewshot = fewshot
    retriever.formatter = _Formatter()

    result = retriever.retrieve("按渠道统计订单", biz_line="banking")

    assert searcher.required_tables == set()
    assert result.relevant_tables[0]["table_name"] == "analytics.order_timeline"
    assert context_planner.required_columns == set()
    assert fewshot.calls == [["analytics.order_timeline"]]


def test_rebuild_all_rebuilds_every_persisted_collection(monkeypatch) -> None:
    retriever = SchemaRetriever()
    captured: list[list[str]] = []

    def rebuild_partial(collections: list[str]) -> int:
        captured.append(collections)
        return 13

    monkeypatch.setattr(retriever, "rebuild_partial", rebuild_partial)

    assert retriever.rebuild_all() == 13
    assert captured == [["table", "enum", "value", "fewshot", "glossary"]]


def test_query_state_dimensions_become_required_projection_fields(
    monkeypatch,
) -> None:
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_reranker", lambda: None)
    retriever = SchemaRetriever()
    retriever._initialized = True
    retriever.config = AgentRuntimeConfig(enable_reranker=False)
    retriever.glossary_resolver = _GlossaryResolver()
    retriever.value_indexer = None
    retriever.searcher = _HybridSearcher()
    retriever.context_planner = _CapturingContextPlanner()
    retriever.fewshot = _FewshotSelector()
    retriever.formatter = _Formatter()

    result = retriever.retrieve(
        "展示 event_type 和事件数",
        biz_line="banking",
        query_state={"dimensions": ["event_type"]},
    )

    assert result.requested_fields == [
        {
            "field": "event_type",
            "columns": ["analytics.order_timeline.event_type"],
        }
    ]
    assert retriever.context_planner.required_columns == {
        "analytics.order_timeline.event_type"
    }


def test_unmapped_dimension_stays_soft_context() -> None:
    candidates = [
        {
            "table_name": "analytics.orders",
            "schema": {
                "columns": [
                    {"name": "order_id", "type": "varchar"},
                    {"name": "payload", "type": "json"},
                ]
            },
        }
    ]

    assert (
        SchemaContextPlanner.resolve_requested_columns(
            candidates,
            ["LOCAL/SWIFT"],
            query="按 LOCAL/SWIFT 统计订单",
        )
        == []
    )
    assert SQLValidator.validate_requested_projection(
        "SELECT payload FROM analytics.orders",
        [{"field": "LOCAL/SWIFT", "columns": []}],
    ) == (True, "", {"required": False})


def test_requested_field_keeps_all_direct_schema_candidates() -> None:
    candidates = [
        {
            "table_name": "analytics.orders",
            "score": 0.7,
            "schema": {
                "columns": [
                    {
                        "name": "account_name",
                        "display_name": "账户名称",
                    }
                ]
            },
        },
        {
            "table_name": "analytics.balances",
            "score": 0.9,
            "schema": {
                "columns": [
                    {
                        "name": "balance_name",
                        "display_name": "余额子账户名称",
                    }
                ]
            },
        },
    ]

    requirements = SchemaContextPlanner.resolve_requested_columns(
        candidates,
        ["账户名称"],
        query="按账户名称统计订单",
    )

    assert requirements == [
        {
            "field": "账户名称",
            "columns": [
                "analytics.orders.account_name",
                "analytics.balances.balance_name",
            ],
        }
    ]


def test_pipeline_repairs_dropped_context_before_returning_sql(monkeypatch) -> None:
    import app as service

    previous_sql = """
    SELECT o.channel_code, COUNT(*) AS order_count
    FROM analytics.orders AS o
    WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY o.channel_code
    ORDER BY order_count DESC
    LIMIT 100
    """
    history = ContextCompressor.build_summary(
        "最近一个月按渠道统计订单交易笔数",
        ["analytics.orders"],
        previous_sql,
        query_state=QueryState(
            subject="订单交易",
            time_range="最近一个月",
            metrics=("交易笔数",),
            dimensions=("渠道",),
        ),
    )
    model = _FollowupModel()
    retriever = _Retriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
        max_fix_retries=2,
    )

    result = service._run_query_impl(
        "再增加订单类型维度",
        config,
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert "o.channel_code" in result["sql"]
    assert "GROUP BY o.channel_code, o.order_type" in result["sql"]
    assert retriever.query == "最近一个月按渠道和订单类型统计订单交易笔数"
    assert retriever.kwargs["query_state"]["dimensions"] == ["渠道", "订单类型"]
    assert len(model.calls) == 3
    assert "本轮语义变更" in model.calls[1]
    assert "缺失结构" in model.calls[2]
    inheritance_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "followup_inheritance_validate"
    )
    assert [attempt["valid"] for attempt in inheritance_step["attempts"]] == [
        False,
        True,
    ]


def test_complex_derived_requirements_reach_generation_without_field_block(
    monkeypatch,
) -> None:
    import app as service

    model = _ComplexQuestionModel()
    retriever = _ComplexRetriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
    )
    question = (
        "在 account_refund 后列出所有 send_to_channel 事件，"
        "不同渠道算换渠道，并展示最终成功或业务退款状态"
    )

    result = service._run_query_impl(question, config, model)

    assert result["is_success"] is True, result["error"]
    assert result["needs_clarification"] is False
    assert retriever.query == (
        "查询指定渠道在 [2026-07-01, 2026-08-01) 内退款后的全部渠道事件，"
        "判断渠道变化并展示最终状态"
    )
    assert retriever.kwargs["query_state"]["metrics"] == []
    assert len(model.calls) == 2
    assert not any(
        step.get("step") == "unresolved_requested_field"
        for step in result["trace"]["steps"]
    )


def test_result_operation_reuses_previous_sql_without_regeneration(monkeypatch) -> None:
    import app as service

    previous_sql = """
    SELECT o.channel_code, COUNT(*) AS order_count
    FROM analytics.orders AS o
    WHERE o.create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY o.channel_code
    LIMIT 100
    """
    history = ContextCompressor.build_summary(
        "最近一个月按渠道统计订单交易笔数",
        ["analytics.orders"],
        previous_sql,
        query_state=QueryState(
            subject="订单交易",
            time_range="最近一个月",
            filters=("状态有效",),
            metrics=("交易笔数",),
            dimensions=("渠道",),
            result_shape="aggregate",
        ),
    )
    model = _TurnIntentModel("result_operation")
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "把这个结果导出",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
        ),
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert result["turn_intent"] == "result_operation"
    assert result["context_relation"] == "follow_up_add"
    assert "o.channel_code" in result["sql"]
    assert result["query_state"]["filters"] == ["状态有效"]
    assert len(model.calls) == 1
    assert service.retriever.query == ""
    generation_step = next(
        step for step in result["trace"]["steps"] if step["step"] == "llm_generation"
    )
    assert generation_step["calls"][0]["role"] == "reuse_previous_sql"


def test_result_explanation_reuses_sql_and_returns_grounded_summary(
    monkeypatch,
) -> None:
    import app as service

    previous_sql = """
    SELECT o.channel_code, COUNT(*) AS order_count
    FROM analytics.orders AS o
    GROUP BY o.channel_code
    LIMIT 100
    """
    history = ContextCompressor.build_summary(
        "按渠道统计订单交易笔数",
        ["analytics.orders"],
        previous_sql,
        query_state=QueryState(
            subject="订单交易",
            metrics=("交易笔数",),
            dimensions=("渠道",),
            result_shape="aggregate",
        ),
    )
    model = _TurnIntentModel("result_explanation")
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "解释一下这个结果",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
        ),
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert result["summary"] == "上一轮查询按渠道统计订单笔数。"
    assert len(model.calls) == 2
    explanation_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "result_explanation"
    )
    assert explanation_step["has_query_result"] is False


def test_non_query_turn_bypasses_retrieval_and_preserves_context(monkeypatch) -> None:
    import app as service

    class _UnexpectedRetriever:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("non-query turn must not enter retrieval")

    history = ContextCompressor.build_summary(
        "查询订单数量",
        ["analytics.orders"],
        "SELECT COUNT(*) FROM analytics.orders",
    )
    model = _TurnIntentModel(
        "non_query",
        direct_response="我可以继续帮你查询数据库。",
    )
    monkeypatch.setattr(service, "retriever", _UnexpectedRetriever())

    result = service._run_query_impl(
        "你能做什么",
        AgentRuntimeConfig(enable_explain=False, enable_execute=False),
        model,
        history_summary=history,
    )

    assert result["is_success"] is True
    assert result["sql"] == ""
    assert result["summary"] == "我可以继续帮你查询数据库。"
    assert result["context_summary"] == history
