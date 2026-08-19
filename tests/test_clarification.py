from src.retrieval.sql_validator import SQLValidator


def test_plain_clarification_remains_backward_compatible():
    assert SQLValidator.extract_clarification(
        "NEED_CLARIFY: 你所说的销售额采用哪个口径？"
    ) == {
        "question": "你所说的销售额采用哪个口径？",
        "options": [],
    }


def test_structured_clarification_limits_and_normalizes_options():
    response = (
        'NEED_CLARIFY: {"question":"请选择销售额口径",'
        '"options":[{"label":"含税","value":"使用含税订单金额"},'
        '{"label":"不含税","value":"使用不含税订单金额"}]}'
    )

    assert SQLValidator.extract_clarification(response) == {
        "question": "请选择销售额口径",
        "options": [
            {"label": "含税", "value": "使用含税订单金额"},
            {"label": "不含税", "value": "使用不含税订单金额"},
        ],
    }


def test_non_clarification_returns_none():
    assert SQLValidator.extract_clarification("```sql\nSELECT 1\n```") is None


def test_structured_clarification_allows_trailing_model_text():
    response = (
        'NEED_CLARIFY: {"question":"请选择口径",'
        '"options":[{"label":"支付成功","value":"按支付成功统计"}]}\n'
        "请等待用户确认。"
    )

    assert SQLValidator.extract_clarification(response) == {
        "question": "请选择口径",
        "options": [{"label": "支付成功", "value": "按支付成功统计"}],
    }


def test_clarification_signal_wins_when_response_also_contains_sql():
    response = "```sql\nSELECT 1\n```\nNEED_CLARIFY: 请确认统计口径"

    assert SQLValidator.extract_clarification(response) == {
        "question": "请确认统计口径",
        "options": [],
    }


def test_clarification_marker_inside_sql_line_is_not_protocol_signal():
    response = "```sql\nSELECT 'NEED_CLARIFY: literal' AS message\n```"

    assert SQLValidator.extract_clarification(response) is None
