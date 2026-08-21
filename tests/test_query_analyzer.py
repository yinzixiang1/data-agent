from src.retrieval.query_analyzer import QueryAnalyzer


def test_query_analyzer_extracts_ranking_aggregation_and_time_grain():
    analysis = QueryAnalyzer.analyze("按月统计交易总额最高的前十个客户")

    assert analysis.intents == ["ranking", "trend", "aggregate"]
    assert set(analysis.aggregations) == {"sum", "max"}
    assert analysis.time_grain == "month"


def test_query_analyzer_defaults_to_detail():
    analysis = QueryAnalyzer.analyze("查询客户资料")

    assert analysis.intents == ["detail"]


def test_query_analyzer_understands_english_time_aggregation():
    analysis = QueryAnalyzer.analyze(
        "Count banking orders by order type for each of the last 7 days."
    )

    assert "aggregate" in analysis.intents
    assert "trend" in analysis.intents
    assert analysis.aggregations == ["count"]
    assert analysis.time_grain == "day"
    assert analysis.has_time_filter is True


def test_query_analyzer_detects_chinese_currency_conversion_target():
    analysis = QueryAnalyzer.analyze("查询最近一个月的入金，转换成美元")

    assert analysis.currency_conversion is True
    assert analysis.requires_exchange_rate is True
    assert analysis.target_currency == "USD"
    assert "currency_conversion" in analysis.intents
    assert "warehouse_sys.sys_exchange_rate" in analysis.to_prompt_context()
    assert "原金额*mid" in analysis.to_prompt_context()
    assert "COALESCE/IFNULL" in analysis.to_prompt_context()
    assert "缺失汇率笔数" in analysis.to_prompt_context()


def test_query_analyzer_detects_short_chinese_conversion_phrase():
    analysis = QueryAnalyzer.analyze("查询最近一个月的 transfer 交易，转美元")

    assert analysis.currency_conversion is True
    assert analysis.requires_exchange_rate is True
    assert analysis.target_currency == "USD"


def test_query_analyzer_detects_currency_reporting_phrase():
    analysis = QueryAnalyzer.analyze("最近一个月的入金按新加坡元统计")

    assert analysis.currency_conversion is True
    assert analysis.target_currency == "SGD"


def test_query_analyzer_detects_exchange_rate_lookup_without_conversion():
    analysis = QueryAnalyzer.analyze("查询今天美元兑新币汇率")

    assert analysis.requires_exchange_rate is True
    assert analysis.currency_conversion is False
    assert analysis.target_currency == ""
    assert "exchange_rate" in analysis.intents


def test_query_analyzer_does_not_treat_currency_filter_as_conversion():
    analysis = QueryAnalyzer.analyze("查询最近一个月的 USD 入金")

    assert analysis.requires_exchange_rate is False
    assert analysis.currency_conversion is False


def test_query_analyzer_extracts_explicit_result_fields_from_followup():
    analysis = QueryAnalyzer.analyze(
        "查询account_name为 yzx的用户最近一个月的交易情况，并展示其邮箱。"
    )

    assert analysis.requested_fields == ["邮箱"]
    assert "必须展示字段: 邮箱" in analysis.to_prompt_context()


def test_query_analyzer_extracts_multiple_result_fields():
    analysis = QueryAnalyzer.analyze("显示开户时间和手机号")

    assert analysis.requested_fields == ["开户时间", "手机号"]


def test_query_analyzer_preserves_action_words_inside_field_names():
    analysis = QueryAnalyzer.analyze(
        "show report_date, analysis_id and distribution_channel"
    )

    assert analysis.requested_fields == [
        "report_date",
        "analysis_id",
        "distribution_channel",
    ]
    assert analysis.presentation_actions == []


def test_query_analyzer_separates_derived_metric_and_presentation_actions():
    analysis = QueryAnalyzer.analyze("按天展示注册数量趋势并生成图表")

    assert analysis.requested_fields == []
    assert analysis.derived_metrics == ["注册数量"]
    assert analysis.presentation_actions == ["趋势", "图表"]
    assert "派生指标（由聚合计算，不是物理字段）: 注册数量" in (
        analysis.to_prompt_context()
    )
    assert "结果展示动作（不作为数据库字段）: 趋势, 图表" in (
        analysis.to_prompt_context()
    )


def test_query_analyzer_does_not_parse_clarification_transport_labels_as_fields():
    analysis = QueryAnalyzer.analyze(
        "原问题\n用户补充：创建时间\n用户补充：趋势 是图表分析"
    )

    assert analysis.requested_fields == []
    assert analysis.derived_metrics == []
    assert analysis.presentation_actions == ["趋势", "图表", "分析"]


def test_query_analyzer_marks_exclusive_count_as_result_contract():
    analysis = QueryAnalyzer.analyze(
        "最近一个月只查询出金交易次数，不返回金额，不按币种分组"
    )

    assert analysis.aggregations == ["count"]
    assert analysis.count_only is True
    assert "仅返回 COUNT 次数" in analysis.to_prompt_context()


def test_query_analyzer_detects_relative_month_without_table_specific_intent():
    analysis = QueryAnalyzer.analyze("查询visa渠道这一个月的交易金额")

    assert analysis.has_time_filter is True
    assert "transaction_amount" not in analysis.to_dict()
    assert "channel_scoped" not in analysis.to_dict()
