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
