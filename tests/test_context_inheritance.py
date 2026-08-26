import asyncio
import json
from types import SimpleNamespace

import pytest

from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.context_compressor import ContextCompressor, QueryState
from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.retriever import SchemaRetriever
from src.retrieval.sql_validator import SQLValidator


class _InvalidMergeModel:
    def invoke(self, _messages):
        return SimpleNamespace(content="not-json")


class _ClarificationWithoutColonModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "new_question",
                        "turn_intent": "sql_query",
                        "query_state": {
                            "subject": "C2C payout 交易",
                            "time_range": "近6个月",
                            "filters": ["payment scope=swift"],
                            "metrics": ["平均 payout 金额"],
                            "dimensions": [],
                            "currency_conversion": "USD",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": ["internal"],
                        },
                        "changes": {"kept": [], "set": ["完整查询"], "removed": []},
                        "removed_sql_context": {},
                        "effective_question": (
                            "查询近6个月 C2C 的 payout 交易，payment scope=swift，"
                            "排除 internal，按 USD 计算平均金额"
                        ),
                        "interpretation": "查询 C2C SWIFT payout 的美元平均金额。",
                        "direct_response": "",
                        "confidence": 0.99,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            content=(
                'NEED_CLARIFY {"question":"请确认 C2C 与 payout SWIFT 的判定口径。",'
                '"options":[]}'
            ),
            usage_metadata=None,
        )


class _MetricReplacementModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_modify",
                        "turn_intent": "sql_query",
                        "presentation_relation": "inherit",
                        "query_state": {
                            "subject": "入金交易",
                            "time_range": "今年",
                            "filters": ["Visa 渠道"],
                            "metrics": ["入金交易数量"],
                            "dimensions": ["月份"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["今年", "Visa 渠道", "月份"],
                            "set": ["改为入金交易数量"],
                            "removed": ["出金交易数量"],
                        },
                        "removed_sql_context": {"filters": ["order_type = 'payout'"]},
                        "effective_question": (
                            "查询今年 Visa 渠道的入金交易数量，按月份聚合"
                        ),
                        "interpretation": "保留其他条件，将出金改为入金。",
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
SELECT DATE_FORMAT(create_time, '%Y-%m') AS month,
       COUNT(*) AS payout_count
FROM analytics.orders
WHERE order_type = 'deposit'
GROUP BY DATE_FORMAT(create_time, '%Y-%m')
ORDER BY month
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content="""```sql
SELECT DATE_FORMAT(create_time, '%Y-%m') AS month,
       COUNT(*) AS deposit_count
FROM analytics.orders
WHERE order_type = 'deposit'
GROUP BY DATE_FORMAT(create_time, '%Y-%m')
ORDER BY month
```""",
            usage_metadata=None,
        )


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


class _CalendarWindowModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "new_question",
                        "turn_intent": "sql_query",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "最近7天",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": [],
                            "currency_conversion": "",
                            "result_shape": "count_only",
                            "calendar_day_window": 7,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {"kept": [], "set": ["最近7天"], "removed": []},
                        "removed_sql_context": {},
                        "effective_question": "统计最近7天的订单数量",
                        "interpretation": "统计包含今天的最近7个自然日。",
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
SELECT COUNT(*) AS order_count
FROM analytics.orders
WHERE create_time >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
  AND create_time < DATE_ADD(CURDATE(), INTERVAL 1 DAY)
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content="""```sql
SELECT COUNT(*) AS order_count
FROM analytics.orders
WHERE create_time >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
  AND create_time < DATE_ADD(CURDATE(), INTERVAL 1 DAY)
```""",
            usage_metadata=None,
        )


class _AggregateLimitModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
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
                            "dimensions": ["渠道"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {"kept": [], "set": ["按渠道统计"], "removed": []},
                        "removed_sql_context": {},
                        "effective_question": "按渠道统计订单数量",
                        "interpretation": "返回完整渠道分组。",
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
SELECT channel_code, COUNT(*) AS order_count
FROM analytics.orders
GROUP BY channel_code
LIMIT 100
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content="""```sql
SELECT channel_code, COUNT(*) AS order_count
FROM analytics.orders
GROUP BY channel_code
```""",
            usage_metadata=None,
        )


class _MixedChartQueryModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "turn_intent": "sql_query",
                        "presentation_relation": "add",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": ["日期"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["订单数量"],
                            "set": ["日期维度", "折线图"],
                            "removed": [],
                        },
                        "removed_sql_context": {},
                        "effective_question": "按日期统计订单数量，并以折线图呈现",
                        "query_question": "按日期统计订单数量",
                        "interpretation": "增加日期维度并要求折线图呈现。",
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
SELECT DATE_FORMAT(o.create_time, '%Y-%m-%d') AS order_date,
       COUNT(*) AS order_count
FROM analytics.orders AS o
GROUP BY DATE_FORMAT(o.create_time, '%Y-%m-%d')
ORDER BY order_date
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content=(
                '{"actions":[{"name":"render_chart",'
                '"arguments":{"chart_type":"line"}}]}'
            ),
            usage_metadata=None,
        )


class _InheritedChartFollowupModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "turn_intent": "sql_query",
                        "presentation_relation": "inherit",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": ["日期", "订单类型"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["按日期统计", "订单数量", "折线图"],
                            "set": ["订单类型维度"],
                            "removed": [],
                        },
                        "removed_sql_context": {},
                        "effective_question": (
                            "按日期和订单类型统计订单数量，并以折线图呈现"
                        ),
                        "interpretation": "保留折线图并增加订单类型维度。",
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
SELECT DATE_FORMAT(o.create_time, '%Y-%m-%d') AS order_date,
       o.order_type,
       COUNT(*) AS order_count
FROM analytics.orders AS o
GROUP BY DATE_FORMAT(o.create_time, '%Y-%m-%d'), o.order_type
ORDER BY order_date
```""",
                usage_metadata=None,
            )
        return SimpleNamespace(
            content=(
                '{"actions":[{"name":"render_chart",'
                '"arguments":{"chart_type":"line"}}]}'
            ),
            usage_metadata=None,
        )


class _AddDownloadToChartModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "turn_intent": "result_operation",
                        "presentation_relation": "add",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": ["日期"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["上一轮查询", "折线图"],
                            "set": ["下载结果"],
                            "removed": [],
                        },
                        "removed_sql_context": {},
                        "effective_question": "按日期统计订单数量，生成折线图并下载结果",
                        "interpretation": "保留折线图并新增下载结果操作。",
                        "direct_response": "",
                        "confidence": 0.99,
                        "needs_clarification": False,
                        "clarification": {"question": "", "options": []},
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "name": "render_chart",
                            "arguments": {"chart_type": "line"},
                        },
                        {
                            "name": "export_result",
                            "arguments": {"format": "xlsx"},
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            usage_metadata=None,
        )


class _AnalyzePreviousResultModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        if len(self.calls) == 1:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "relation": "follow_up_add",
                        "turn_intent": "result_operation",
                        "presentation_relation": "add",
                        "query_state": {
                            "subject": "订单",
                            "time_range": "",
                            "filters": [],
                            "metrics": ["数量"],
                            "dimensions": ["日期"],
                            "currency_conversion": "",
                            "result_shape": "aggregate",
                            "calendar_day_window": None,
                            "requested_limit": None,
                            "exclusions": [],
                        },
                        "changes": {
                            "kept": ["上一轮完整查询"],
                            "set": ["趋势和数据异常分析"],
                            "removed": [],
                        },
                        "removed_sql_context": {},
                        "effective_question": "按日期统计订单数量并分析趋势和异常",
                        "query_question": "按日期统计订单数量",
                        "interpretation": "分析上一轮结果中的趋势和数据异常。",
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
                content=(
                    '{"actions":[{"name":"analyze_result",'
                    '"arguments":{"modes":["trend","anomaly"]}}]}'
                ),
                usage_metadata=None,
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "title": "订单数量分析",
                    "executive_summary": "",
                    "findings": [
                        {
                            "type": "trend",
                            "statement": "以确定性事实为准",
                            "evidence_fact_ids": ["f4"],
                            "confidence": "high",
                        }
                    ],
                    "caveats": [],
                    "suggested_followups": [],
                },
                ensure_ascii=False,
            ),
            usage_metadata=None,
        )


class _ClearChartModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        self.calls.append(messages[-1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "relation": "follow_up_add",
                    "turn_intent": "result_operation",
                    "presentation_relation": "clear",
                    "query_state": {
                        "subject": "订单",
                        "time_range": "",
                        "filters": [],
                        "metrics": ["数量"],
                        "dimensions": ["日期"],
                        "currency_conversion": "",
                        "result_shape": "aggregate",
                        "calendar_day_window": None,
                        "requested_limit": None,
                        "exclusions": [],
                    },
                    "changes": {
                        "kept": ["上一轮完整查询"],
                        "set": [],
                        "removed": ["折线图"],
                    },
                    "removed_sql_context": {},
                    "effective_question": "按日期统计订单数量",
                    "interpretation": "保留查询，只取消折线图。",
                    "direct_response": "",
                    "confidence": 0.99,
                    "needs_clarification": False,
                    "clarification": {"question": "", "options": []},
                },
                ensure_ascii=False,
            )
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


def _render_chart_tool() -> dict:
    return {
        "name": "render_chart",
        "display_name": "生成图表",
        "description": "将查询结果呈现为图表",
        "execution_stage": "channel_post_query",
        "state_policy": "sticky",
        "intent_phrases": ["生成图表", "折线图", "曲线图"],
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar"],
                }
            },
            "required": ["chart_type"],
            "additionalProperties": False,
        },
        "requires_query_result": True,
    }


def _export_result_tool() -> dict:
    return {
        "name": "export_result",
        "display_name": "下载数据",
        "description": "把当前查询结果生成文件供用户下载",
        "execution_stage": "channel_post_query",
        "state_policy": "one_shot",
        "intent_phrases": ["下载数据", "导出数据"],
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["csv", "xlsx"],
                }
            },
            "required": ["format"],
            "additionalProperties": False,
        },
        "requires_query_result": True,
    }


def _analyze_result_tool() -> dict:
    return {
        "name": "analyze_result",
        "display_name": "智能分析",
        "description": "只分析当前查询结果中的趋势和数据异常",
        "executor_key": "analyze_result",
        "execution_stage": "agent_post_query",
        "state_policy": "sticky",
        "trigger_mode": "intent_auto",
        "intent_phrases": ["分析结果", "趋势", "异常"],
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
        "requires_query_result": True,
        "binding_config": {"max_findings": 3},
    }


def _dated_order_history(*, with_chart: bool) -> str:
    sql = """
    SELECT DATE_FORMAT(o.create_time, '%Y-%m-%d') AS order_date,
           COUNT(*) AS order_count
    FROM analytics.orders AS o
    GROUP BY DATE_FORMAT(o.create_time, '%Y-%m-%d')
    ORDER BY order_date
    """
    presentation_state = (
        {
            "tool_calls": [
                {
                    "name": "render_chart",
                    "arguments": {"chart_type": "line"},
                    "requires_query_result": True,
                }
            ]
        }
        if with_chart
        else None
    )
    return ContextCompressor.build_summary(
        "按日期统计订单数量",
        ["analytics.orders"],
        sql,
        query_state=QueryState(
            subject="订单",
            metrics=("数量",),
            dimensions=("日期",),
            result_shape="aggregate",
        ),
        presentation_state=presentation_state,
    )


def test_clarification_protocol_accepts_colon_or_whitespace_separator() -> None:
    expected = {"question": "请确认查询口径。", "options": []}

    assert (
        SQLValidator.extract_clarification(
            'NEED_CLARIFY: {"question":"请确认查询口径。","options":[]}'
        )
        == expected
    )
    assert (
        SQLValidator.extract_clarification(
            'NEED_CLARIFY {"question":"请确认查询口径。","options":[]}'
        )
        == expected
    )
    assert (
        SQLValidator.extract_clarification(
            'NEED_CLARIFY\n{"question":"请确认查询口径。","options":[]}'
        )
        == expected
    )


def test_pipeline_returns_whitespace_clarification_without_sql_retry(
    monkeypatch,
) -> None:
    import app as service

    model = _ClarificationWithoutColonModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
        max_fix_retries=2,
    )

    result = service._run_query_impl(
        (
            '帮我查一下C2C的交易，且payment scope="swift"，'
            "统计这些交易近6个月的平均payout金额，按美元计算并剔除internal"
        ),
        config,
        model,
    )

    assert result["needs_clarification"] is True
    assert result["clarification"]["question"] == (
        "请确认 C2C 与 payout SWIFT 的判定口径。"
    )
    assert result["sql"] == ""
    assert len(model.calls) == 2
    generation_step = next(
        step for step in result["trace"]["steps"] if step["step"] == "llm_generation"
    )
    assert [call["role"] for call in generation_step["calls"]] == ["initial"]


def test_pipeline_turns_exhausted_schema_grounding_failure_into_clarification(
    monkeypatch,
) -> None:
    import app as service

    class _UngroundedSchemaModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def invoke(self, messages):
            self.calls.append(messages[-1].content)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "relation": "new_question",
                            "turn_intent": "sql_query",
                            "query_state": {
                                "subject": "订单",
                                "time_range": "",
                                "filters": ["未配置的业务范围"],
                                "metrics": ["数量"],
                                "dimensions": ["业务范围"],
                                "currency_conversion": "",
                                "result_shape": "aggregate",
                                "calendar_day_window": None,
                                "requested_limit": None,
                                "exclusions": [],
                            },
                            "changes": {
                                "kept": [],
                                "set": ["完整查询"],
                                "removed": [],
                            },
                            "removed_sql_context": {},
                            "effective_question": "按业务范围统计订单数量",
                            "interpretation": "按业务范围聚合订单数量。",
                            "direct_response": "",
                            "confidence": 0.99,
                            "needs_clarification": False,
                            "clarification": {"question": "", "options": []},
                        },
                        ensure_ascii=False,
                    )
                )
            return SimpleNamespace(
                content="""```sql
SELECT o.unmapped_scope, COUNT(*) AS order_count
FROM analytics.orders AS o
GROUP BY o.unmapped_scope
```""",
                usage_metadata=None,
            )

    model = _UngroundedSchemaModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "按业务范围统计订单数量",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            max_fix_retries=1,
        ),
        model,
    )

    assert result["is_success"] is False
    assert result["needs_clarification"] is True
    assert result["sql"] == ""
    assert result["query_result"] is None
    assert result["error"].startswith("NEED_CLARIFY:")
    assert "o.unmapped_scope" in result["clarification"]["question"]
    assert len(model.calls) == 3
    grounding_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "schema_grounding_validate"
    )
    assert [attempt["valid"] for attempt in grounding_step["attempts"]] == [
        False,
        False,
    ]
    assert grounding_step["needs_clarification"] is True


def test_pipeline_does_not_turn_sql_syntax_failure_into_clarification(
    monkeypatch,
) -> None:
    import app as service

    class _InvalidSqlModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "relation": "new_question",
                            "turn_intent": "sql_query",
                            "query_state": {
                                "subject": "订单",
                                "time_range": "",
                                "filters": [],
                                "metrics": [],
                                "dimensions": [],
                                "currency_conversion": "",
                                "result_shape": "detail",
                                "calendar_day_window": None,
                                "requested_limit": None,
                                "exclusions": [],
                            },
                            "changes": {
                                "kept": [],
                                "set": ["完整查询"],
                                "removed": [],
                            },
                            "removed_sql_context": {},
                            "effective_question": "查询订单",
                            "interpretation": "查询订单明细。",
                            "direct_response": "",
                            "confidence": 0.99,
                            "needs_clarification": False,
                            "clarification": {"question": "", "options": []},
                        },
                        ensure_ascii=False,
                    )
                )
            return SimpleNamespace(
                content="```sql\nSELECT * FROM\n```",
                usage_metadata=None,
            )

    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "查询订单",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            max_fix_retries=1,
        ),
        _InvalidSqlModel(),
    )

    assert result["is_success"] is False
    assert result["needs_clarification"] is False
    assert result["clarification"] is None
    assert not result["error"].startswith("NEED_CLARIFY:")


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


def test_replaced_metrics_reject_stale_aggregate_projection() -> None:
    previous_sql = """
        SELECT DATE_FORMAT(create_time, '%Y-%m') AS month,
               COUNT(*) AS payout_count
        FROM analytics.orders
        WHERE order_type = 'payout'
        GROUP BY DATE_FORMAT(create_time, '%Y-%m')
        ORDER BY month
    """
    current_sql = """
        SELECT DATE_FORMAT(create_time, '%Y-%m') AS month,
               COUNT(*) AS payout_count
        FROM analytics.orders
        WHERE order_type = 'deposit'
        GROUP BY DATE_FORMAT(create_time, '%Y-%m')
        ORDER BY month
    """
    previous_context = ContextCompressor.extract_sql_context(previous_sql)

    valid, error, detail = SQLValidator.validate_followup_inheritance(
        current_sql,
        previous_context,
        "follow_up_modify",
        {
            "projections": ["COUNT(*) AS payout_count"],
            "filters": ["order_type = 'payout'"],
        },
        {"metrics": ["出金交易数量"]},
        {"metrics": ["入金交易数量"]},
    )

    assert valid is False
    assert "指标已替换但仍保留旧聚合投影" in error
    assert detail["stale_metric_projections"] == ["COUNT(*) AS payout_count"]


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


def test_query_state_normalizes_currency_and_explicit_numeric_contracts() -> None:
    state = QueryState.from_value(
        {
            "currency_conversion": "美元（USD）",
            "calendar_day_window": "7",
            "requested_limit": "10",
        }
    )

    assert state.currency_conversion == "USD"
    assert state.calendar_day_window == 7
    assert state.requested_limit == 10


def test_currency_validator_accepts_normalized_code_inside_display_label() -> None:
    sql = """
    SELECT o.source_currency,
           SUM(CASE WHEN o.source_currency = 'USD'
                    THEN o.source_amount
                    ELSE o.source_amount * r.mid END) AS amount_usd,
           SUM(CASE WHEN o.source_currency <> 'USD' AND r.mid IS NULL
                    THEN 1 ELSE 0 END) AS missing_rate_count
    FROM analytics.orders AS o
    LEFT JOIN warehouse_sys.sys_exchange_rate AS r
      ON r.source_currency = o.source_currency
     AND r.target_currency = 'USD'
     AND DATE(r.sync_time) = DATE(o.create_time)
    GROUP BY o.source_currency
    """

    valid, error, detail = SQLValidator.validate_currency_conversion(
        sql,
        {"state": {"currency_conversion": "美元（USD）"}},
    )

    assert valid is True, error
    assert detail["target_currency"] == "USD"


def test_calendar_day_window_includes_today_without_off_by_one() -> None:
    intent = {"state": {"calendar_day_window": 7}}
    valid_sql = """
    SELECT COUNT(*) FROM analytics.orders
    WHERE create_time >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
      AND create_time < DATE_ADD(CURDATE(), INTERVAL 1 DAY)
    """
    invalid_sql = valid_sql.replace("INTERVAL 6 DAY", "INTERVAL 7 DAY")

    assert SQLValidator.validate_calendar_day_window(valid_sql, intent)[0] is True
    invalid, error, detail = SQLValidator.validate_calendar_day_window(
        invalid_sql,
        intent,
    )
    assert invalid is False
    assert "INTERVAL 6 DAY" in error
    assert detail["includes_today"] is True


def test_aggregate_limit_requires_explicit_user_request() -> None:
    aggregate = {"state": {"result_shape": "aggregate", "requested_limit": None}}
    explicit = {"state": {"result_shape": "aggregate", "requested_limit": 10}}

    assert SQLValidator.validate_result_limit(
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code",
        aggregate,
    )[0]
    assert not SQLValidator.validate_result_limit(
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code LIMIT 100",
        aggregate,
    )[0]
    assert SQLValidator.validate_result_limit(
        "SELECT channel_code, COUNT(*) FROM analytics.orders GROUP BY channel_code LIMIT 10",
        explicit,
    )[0]


def test_pipeline_repairs_calendar_day_window(monkeypatch) -> None:
    import app as service

    model = _CalendarWindowModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "统计最近7天的订单数量",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            max_fix_retries=1,
        ),
        model,
    )

    assert result["is_success"] is True, result["error"]
    assert "INTERVAL 6 DAY" in result["sql"]
    assert "INTERVAL 7 DAY" not in result["sql"]
    time_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "calendar_day_window_validate"
    )
    assert [attempt["valid"] for attempt in time_step["attempts"]] == [False, True]


def test_pipeline_removes_unrequested_aggregate_limit(monkeypatch) -> None:
    import app as service

    model = _AggregateLimitModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "按渠道统计订单数量",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            max_fix_retries=1,
        ),
        model,
    )

    assert result["is_success"] is True, result["error"]
    assert "LIMIT" not in result["sql"]
    limit_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "result_limit_validate"
    )
    assert [attempt["valid"] for attempt in limit_step["attempts"]] == [False, True]


def test_prepared_turn_is_reused_without_running_context_merge_again(
    monkeypatch,
) -> None:
    import app as service

    model = _AggregateLimitModel()
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
        max_fix_retries=1,
    )
    prepared = service.prepare_query_context(
        "按渠道统计订单数量",
        config,
        model,
    )
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    def fail_duplicate_merge(*_args, **_kwargs):
        raise AssertionError("prepared context must skip duplicate turn recognition")

    monkeypatch.setattr(ContextCompressor, "merge", fail_duplicate_merge)

    result = service._run_query_impl(
        "按渠道统计订单数量",
        config,
        model,
        prepared_context=prepared,
    )

    assert result["is_success"] is True, result["error"]
    assert result["context_relation"] == "new_question"
    assert len(model.calls) == 3


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


def test_pipeline_repairs_stale_metric_alias_after_metric_replacement(
    monkeypatch,
) -> None:
    import app as service

    previous_sql = """
    SELECT DATE_FORMAT(create_time, '%Y-%m') AS month,
           COUNT(*) AS payout_count
    FROM analytics.orders
    WHERE order_type = 'payout'
    GROUP BY DATE_FORMAT(create_time, '%Y-%m')
    ORDER BY month
    """
    history = ContextCompressor.build_summary(
        "查询今年 Visa 渠道的出金交易数量，按月份聚合",
        ["analytics.orders"],
        previous_sql,
        query_state=QueryState(
            subject="出金交易",
            time_range="今年",
            filters=("Visa 渠道",),
            metrics=("出金交易数量",),
            dimensions=("月份",),
            result_shape="aggregate",
        ),
    )
    model = _MetricReplacementModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)
    config = AgentRuntimeConfig(
        enable_explain=False,
        enable_execute=False,
        enable_enum_validate=False,
        max_fix_retries=2,
    )

    result = service._run_query_impl(
        "查询入金吧",
        config,
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert "COUNT(*) AS deposit_count" in result["sql"]
    assert "payout_count" not in result["sql"]
    assert len(model.calls) == 3
    context_step = next(
        step for step in result["trace"]["steps"] if step["step"] == "context_compress"
    )
    assert context_step["removed_sql_context"]["projections"] == [
        "COUNT(*) AS payout_count"
    ]
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


def test_mixed_query_change_and_chart_generates_sql_and_persists_chart(
    monkeypatch,
) -> None:
    import app as service

    history = ContextCompressor.build_summary(
        "统计订单数量",
        ["analytics.orders"],
        "SELECT COUNT(*) AS order_count FROM analytics.orders",
        query_state=QueryState(
            subject="订单",
            metrics=("数量",),
            result_shape="aggregate",
        ),
    )
    model = _MixedChartQueryModel()
    retriever = _Retriever()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "按照日期维度生成折线图",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            tools=[_render_chart_tool()],
        ),
        model,
        history_summary=history,
    )

    assert result["is_success"] is True, result["error"]
    assert result["turn_intent"] == "sql_query"
    assert "GROUP BY DATE_FORMAT" in result["sql"]
    assert result["tool_calls"][0]["name"] == "render_chart"
    summary = ContextCompressor.parse_summary(result["context_summary"])
    assert summary["query_state"]["dimensions"] == ["日期"]
    assert summary["presentation_state"]["tool_calls"] == result["tool_calls"]
    assert retriever.query == "按日期统计订单数量"
    assert retriever.kwargs["original_query"] == "按日期统计订单数量"
    assert ContextCompressor.parse_summary(result["context_summary"])["question"] == (
        "按日期统计订单数量"
    )
    assert result["trace"]["effective_question"] == (
        "按日期统计订单数量，并以折线图呈现"
    )
    assert result["trace"]["query_question"] == "按日期统计订单数量"


def test_prepare_accepts_additive_presentation_relation() -> None:
    import app as service

    history = ContextCompressor.build_summary(
        "统计订单数量",
        ["analytics.orders"],
        "SELECT COUNT(*) AS order_count FROM analytics.orders",
        query_state=QueryState(
            subject="订单",
            metrics=("数量",),
            result_shape="aggregate",
        ),
    )

    prepared = service.prepare_query_context(
        "按照日期维度生成折线图",
        AgentRuntimeConfig(),
        _MixedChartQueryModel(),
        history_summary=history,
    )

    assert prepared["relation"] == "follow_up_add"
    assert prepared["turn_intent"] == "sql_query"
    assert prepared["presentation_relation"] == "add"
    assert prepared["effective_question"] == "按日期统计订单数量，并以折线图呈现"
    assert prepared["query_question"] == "按日期统计订单数量"


def test_prepare_failure_creates_failed_query_log(monkeypatch) -> None:
    import app as service

    class _QueryLogger:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def log(self, **kwargs) -> int:
            self.calls.append(kwargs)
            return 920

    async def fail_prepare(*_args, **_kwargs):
        raise ValueError("invalid prepared context")

    config = AgentRuntimeConfig(agent_id=1, token="token")
    query_logger = _QueryLogger()
    monkeypatch.setattr(service, "agent_config", config)
    monkeypatch.setattr(service, "llm_client", object())
    monkeypatch.setattr(service, "query_logger", query_logger)
    monkeypatch.setattr(service, "_verify_token", lambda *_args: None)
    monkeypatch.setattr(service, "_prepare_context_in_worker", fail_prepare)

    request = service.QueryPrepareRequest(
        question="按天为维度生成图表",
        agent_id=1,
        session_id="7017d5cc0db4db3d3a6ebc9bee7e4c59",
        metadata=service.QueryMetadata(
            caller="lark",
            user_id="ou-1",
            user_name="测试用户",
            trace_id="evt-prepare-failure",
            filter={"scenario": "bi", "business": "banking"},
        ),
    )

    with pytest.raises(ValueError, match="invalid prepared context"):
        asyncio.run(service.prepare_query(request, SimpleNamespace(headers={})))

    assert query_logger.calls == [
        {
            "session_id": "7017d5cc0db4db3d3a6ebc9bee7e4c59",
            "user_query": "按天为维度生成图表",
            "intent": "prepare_failed",
            "execution_result": {
                "stage": "query_prepare",
                "error_type": "ValueError",
                "error": "invalid prepared context",
            },
            "is_success": False,
            "agent_id": 1,
            "scenario": "bi",
            "business": "banking",
            "caller": "lark",
            "user_id": "ou-1",
            "user_name": "测试用户",
            "trace_id": "evt-prepare-failure:prepare",
            "trace_detail": {
                "stage": "query_prepare",
                "error_type": "ValueError",
                "error": "invalid prepared context",
            },
        }
    ]


def test_query_followup_replans_inherited_chart_against_new_projection(
    monkeypatch,
) -> None:
    import app as service

    model = _InheritedChartFollowupModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "再加订单类型维度",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            tools=[_render_chart_tool()],
        ),
        model,
        history_summary=_dated_order_history(with_chart=True),
    )

    assert result["is_success"] is True, result["error"]
    assert "o.order_type" in result["sql"]
    assert result["tool_calls"] == [
        {
            "name": "render_chart",
            "arguments": {"chart_type": "line"},
            "requires_query_result": True,
        }
    ]
    planner_request = json.loads(model.calls[-1])
    assert planner_request["active_actions"] == [
        {"name": "render_chart", "arguments": {"chart_type": "line"}}
    ]
    tool_step = next(
        step for step in result["trace"]["steps"] if step["step"] == "tool_planning"
    )
    assert tool_step["inherited_tools"] == ["render_chart"]
    assert tool_step["selected_tools"] == ["render_chart"]
    summary = ContextCompressor.parse_summary(result["context_summary"])
    assert summary["presentation_state"]["tool_calls"] == result["tool_calls"]


def test_result_operation_can_add_download_without_hiding_inherited_chart(
    monkeypatch,
) -> None:
    import app as service

    model = _AddDownloadToChartModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "下载结果",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            tools=[_render_chart_tool(), _export_result_tool()],
        ),
        model,
        history_summary=_dated_order_history(with_chart=True),
    )

    assert result["is_success"] is True, result["error"]
    assert [call["name"] for call in result["tool_calls"]] == [
        "render_chart",
        "export_result",
    ]
    planner_request = json.loads(model.calls[-1])
    assert [item["name"] for item in planner_request["available_actions"]] == [
        "render_chart",
        "export_result",
    ]
    tool_step = next(
        step for step in result["trace"]["steps"] if step["step"] == "tool_planning"
    )
    assert tool_step["inherited_tools"] == ["render_chart"]
    assert tool_step["agent_bound_tools"] == ["render_chart", "export_result"]
    assert tool_step["channel_allowed_tools"] == ["render_chart", "export_result"]
    summary = ContextCompressor.parse_summary(result["context_summary"])
    assert [call["name"] for call in summary["presentation_state"]["tool_calls"]] == [
        "render_chart"
    ]


def test_analysis_followup_uses_the_previous_result_without_rerunning_sql(
    monkeypatch,
) -> None:
    import app as service

    class _DatabaseMustNotBeTouched:
        def validate(self, *_args, **_kwargs):
            raise AssertionError("result analysis must not run EXPLAIN")

        def explain(self, *_args, **_kwargs):
            raise AssertionError("result analysis must not run EXPLAIN")

    previous_result = {
        "columns": ["order_date", "order_count"],
        "rows": [
            ["2026-01-01", 10],
            ["2026-01-02", 12],
            ["2026-01-03", 11],
            ["2026-01-04", 13],
            ["2026-01-05", 12],
            ["2026-01-06", 14],
            ["2026-01-07", 13],
            ["2026-01-08", 90],
        ],
        "row_count": 8,
        "truncated": False,
    }
    model = _AnalyzePreviousResultModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", _DatabaseMustNotBeTouched())

    result = service._run_query_impl(
        "分析一下趋势和异常",
        AgentRuntimeConfig(
            enable_explain=True,
            enable_execute=True,
            enable_enum_validate=True,
            tools=[_analyze_result_tool()],
        ),
        model,
        history_summary=_dated_order_history(with_chart=False),
        previous_query_result=previous_result,
    )

    assert result["is_success"] is True, result["error"]
    assert result["query_result"] == previous_result
    assert result["tool_results"][0]["status"] == "success"
    assert result["tool_results"][0]["output"]["findings"]
    snapshot_step = next(
        step
        for step in result["trace"]["steps"]
        if step["step"] == "result_snapshot_reuse"
    )
    assert snapshot_step == {
        "step": "result_snapshot_reuse",
        "available": True,
        "sql_reexecuted": False,
    }


def test_analysis_followup_with_missing_snapshot_is_skipped_without_sql_execution(
    monkeypatch,
) -> None:
    import app as service

    model = _AnalyzePreviousResultModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "分析一下趋势和异常",
        AgentRuntimeConfig(
            enable_explain=True,
            enable_execute=True,
            enable_enum_validate=True,
            tools=[_analyze_result_tool()],
        ),
        model,
        history_summary=_dated_order_history(with_chart=False),
    )

    assert result["is_success"] is True, result["error"]
    assert result["query_result"] is None
    assert len(result["tool_results"]) == 1
    tool_result = result["tool_results"][0]
    assert tool_result["name"] == "analyze_result"
    assert tool_result["status"] == "skipped"
    assert tool_result["output"] == {}
    assert tool_result["error"] == (
        "上一轮查询结果快照不存在或已过期，请重新执行数据查询后再分析"
    )
    assert tool_result["duration_ms"] >= 0
    assert len(model.calls) == 2


def test_explicit_presentation_clear_stops_and_forgets_chart(monkeypatch) -> None:
    import app as service

    model = _ClearChartModel()
    monkeypatch.setattr(service, "retriever", _Retriever())
    monkeypatch.setattr(service, "validator", None)

    result = service._run_query_impl(
        "只返回数据，不要图表",
        AgentRuntimeConfig(
            enable_explain=False,
            enable_execute=False,
            enable_enum_validate=False,
            tools=[_render_chart_tool()],
        ),
        model,
        history_summary=_dated_order_history(with_chart=True),
    )

    assert result["is_success"] is True, result["error"]
    assert result["turn_intent"] == "result_operation"
    assert result["tool_calls"] == []
    assert len(model.calls) == 1
    summary = ContextCompressor.parse_summary(result["context_summary"])
    assert summary["presentation_state"] == {"tool_calls": []}


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
