import json
from types import SimpleNamespace

from src.retrieval.context_compressor import ContextCompressor


def test_context_summary_roundtrip_preserves_verified_sql():
    summary = ContextCompressor.build_summary(
        "查询最近一个月交易",
        ["banking.transactions"],
        "SELECT transaction_type, COUNT(*) FROM banking.transactions",
    )

    payload = ContextCompressor.parse_summary(summary)

    assert json.loads(summary)["question"] == "查询最近一个月交易"
    assert payload == {
        "question": "查询最近一个月交易",
        "tables": ["banking.transactions"],
        "sql": "SELECT transaction_type, COUNT(*) FROM banking.transactions",
        "sql_context": {
            "tables": ["banking.transactions"],
            "columns": ["banking.transactions.transaction_type"],
            "projections": ["transaction_type", "COUNT(*)"],
            "dimensions": [],
            "filters": [],
        },
        "query_state": {
            "subject": "",
            "time_range": "最近一个月",
            "filters": [],
            "metrics": [],
            "dimensions": [],
            "currency_conversion": "",
            "exclusions": [],
        },
    }


def test_extract_sql_context_assigns_unqualified_columns_for_single_table():
    context = ContextCompressor.extract_sql_context(
        "SELECT channel, COUNT(*) FROM analytics.orders "
        "WHERE status = 'SUCCESS' GROUP BY channel"
    )

    assert context["tables"] == ["analytics.orders"]
    assert {
        "analytics.orders.channel",
        "analytics.orders.status",
    } <= set(context["columns"])


def test_context_summary_parser_supports_legacy_format():
    payload = ContextCompressor.parse_summary(
        "查询最近一个月交易|||banking.transactions,banking.accounts"
    )

    assert payload == {
        "question": "查询最近一个月交易",
        "tables": ["banking.transactions", "banking.accounts"],
        "sql": "",
        "sql_context": {
            "tables": [],
            "columns": [],
            "projections": [],
            "dimensions": [],
            "filters": [],
        },
        "query_state": {
            "subject": "",
            "time_range": "",
            "filters": [],
            "metrics": [],
            "dimensions": [],
            "currency_conversion": "",
            "exclusions": [],
        },
    }


class _FakeModel:
    def __init__(self, payload: dict):
        self.payload = payload

    def invoke(self, _messages):
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def test_count_only_correction_removes_previous_amount_and_currency_state():
    history = ContextCompressor.build_summary(
        "统一最近一个月的交易金额并折美元",
        ["dwd_bi_banking.pmt_finance_transactions"],
        "SELECT currency, SUM(amount) FROM transactions GROUP BY currency",
    )
    model = _FakeModel(
        {
            "relation": "correction_override",
            "query_state": {
                "subject": "出金交易",
                "time_range": "最近一个月",
                "filters": ["交易类型为出金"],
                "metrics": ["count"],
                "dimensions": [],
                "currency_conversion": "",
                "exclusions": ["金额", "币种维度", "币种换算"],
            },
            "changes": {
                "kept": ["最近一个月", "交易类型为出金"],
                "set": ["仅统计交易次数"],
                "removed": ["金额", "币种维度", "币种换算"],
            },
            "effective_question": (
                "最近一个月只统计出金交易次数，不返回金额，不按币种分组，不进行币种换算"
            ),
            "interpretation": "保留时间和出金筛选，改为只统计次数。",
            "confidence": 0.95,
            "needs_clarification": False,
            "clarification": {"question": "", "options": []},
        }
    )

    result = ContextCompressor(model).merge(history, "只查询出金的交易次数")

    assert result.relation == "correction_override"
    assert result.query_state.metrics == ("count",)
    assert result.query_state.dimensions == ()
    assert result.query_state.currency_conversion == ""
    assert "最近一个月" in result.effective_question
    assert "不返回金额" in result.effective_question
    assert "不按币种分组" in result.effective_question
    assert result.needs_clarification is False
    assert result.changes["removed"] == ["金额", "币种维度", "币种换算"]


def test_ambiguous_semantic_change_returns_structured_clarification():
    history = ContextCompressor.build_summary(
        "最近一个月按币种统计交易金额",
        ["banking.transactions"],
        "SELECT currency, SUM(amount) FROM banking.transactions GROUP BY currency",
    )
    model = _FakeModel(
        {
            "relation": "follow_up_modify",
            "query_state": {},
            "changes": {"kept": [], "set": [], "removed": []},
            "effective_question": "改成出金",
            "interpretation": "出金范围明确，但统计口径不明确。",
            "confidence": 0.45,
            "needs_clarification": True,
            "clarification": {
                "question": "是否保留金额和美元换算？",
                "options": [
                    {"label": "保留", "value": "保留金额和美元换算"},
                    {"label": "仅次数", "value": "仅统计出金次数"},
                ],
            },
        }
    )

    result = ContextCompressor(model).merge(history, "改成出金")

    assert result.needs_clarification is True
    assert result.clarification["question"] == "是否保留金额和美元换算？"
    assert len(result.clarification["options"]) == 2
