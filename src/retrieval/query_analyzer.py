"""轻量查询分析器：识别 NL2SQL 查询结构，不依赖额外 LLM 调用。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class QueryAnalysis:
    intents: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    time_grain: str = ""
    has_time_filter: bool = False
    asks_distinct: bool = False
    requires_exchange_rate: bool = False
    currency_conversion: bool = False
    target_currency: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        parts = []
        if self.intents:
            parts.append("查询类型: " + ", ".join(self.intents))
        if self.aggregations:
            parts.append("聚合意图: " + ", ".join(self.aggregations))
        if self.time_grain:
            parts.append("时间粒度: " + self.time_grain)
        if self.asks_distinct:
            parts.append("需要去重")
        if self.requires_exchange_rate:
            parts.append("需要汇率上下文: warehouse_sys.sys_exchange_rate")
        if self.currency_conversion:
            target = self.target_currency or "用户指定的目标币种"
            parts.extend(
                (
                    f"货币换算目标: {target}",
                    "换算规则: source_currency=原币种，target_currency=目标币种，"
                    "DATE(sync_time)=DATE(交易时间)，目标金额=原金额*mid；"
                    "原币种等于目标币种时汇率按1处理",
                    "禁止用 CASE WHEN 只保留目标币种、把其他币种金额置为0",
                    "非目标币种缺失汇率时禁止用 COALESCE/IFNULL 按1或0兜底；"
                    "换算结果保留 NULL，并显式返回缺失汇率笔数",
                )
            )
        return "\n".join(parts)


class QueryAnalyzer:
    """用确定性规则提取会影响 SQL 结构的意图。"""

    _INTENT_RULES = {
        "ranking": r"排名|排行|top\s*\d*|最高|最低|最多|最少|ranking|highest|lowest",
        "trend": (
            r"趋势|同比|环比|按(?:日|天|周|月|季|年)|每天|每月|"
            r"trend|by\s+(?:day|week|month|quarter|year)|daily|weekly|monthly|"
            r"each\s+of\s+the\s+last\s+\d+\s+days?"
        ),
        "detail": r"明细|列表|哪些|逐笔|每一笔|详情|detail|list|which",
        "aggregate": (
            r"多少|数量|总计|合计|总额|平均|均值|汇总|统计|"
            r"count|total|sum|average|aggregate"
        ),
    }
    _AGGREGATION_RULES = {
        "count": r"多少|数量|笔数|个数|count",
        "sum": r"总计|合计|总额|总金额|sum",
        "avg": r"平均|均值|avg",
        "max": r"最高|最大|max",
        "min": r"最低|最小|min",
    }
    _TIME_GRAINS = (
        (
            "day",
            r"按日|每天|每日|by\s+day|daily|each\s+day|"
            r"each\s+of\s+the\s+last\s+\d+\s+days?",
        ),
        ("week", r"按周|每周|by\s+week|weekly|each\s+week"),
        ("month", r"按月|每月|月度|by\s+month|monthly|each\s+month"),
        ("quarter", r"按季|季度|by\s+quarter|quarterly"),
        ("year", r"按年|每年|年度|by\s+year|yearly|annually"),
    )
    _TIME_FILTER_RULE = (
        r"最近|近\s*\d+|过去|今日|今天|昨日|昨天|本(?:周|月|季|年)|"
        r"上(?:周|月|季|年)|last\s+\d+|past\s+\d+|recent|today|yesterday|"
        r"this\s+(?:week|month|quarter|year)|last\s+(?:week|month|quarter|year)"
    )
    _CURRENCY_ALIASES = {
        "美元": "USD",
        "美金": "USD",
        "人民币": "CNY",
        "新加坡元": "SGD",
        "新币": "SGD",
        "欧元": "EUR",
        "英镑": "GBP",
        "港币": "HKD",
        "日元": "JPY",
        "澳元": "AUD",
        "加元": "CAD",
    }
    _CURRENCY_TARGET = (
        r"(?:USD|CNY|SGD|EUR|GBP|HKD|JPY|AUD|CAD|"
        + "|".join(sorted(_CURRENCY_ALIASES, key=len, reverse=True))
        + r")"
    )
    _CONVERSION_TARGET_RULES = (
        re.compile(
            rf"(?:转换|换算|折算|兑换|转成|换成|统一(?:为|成))\s*"
            rf"(?:为|成|到)?\s*(?P<target>{_CURRENCY_TARGET})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"按\s*(?P<target>{_CURRENCY_TARGET})\s*"
            r"(?:统计|计算|计价|展示|汇总|口径)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:convert|converted|conversion|express|report)\b.*?\b(?:to|in)\s+"
            rf"(?P<target>{_CURRENCY_TARGET})\b",
            re.IGNORECASE,
        ),
    )
    _EXCHANGE_RATE_RULE = re.compile(
        r"汇率|兑换率|外汇牌价|exchange\s+rate|\bfx\s+rate\b",
        re.IGNORECASE,
    )

    @classmethod
    def _normalize_currency(cls, value: str) -> str:
        normalized = value.strip()
        return cls._CURRENCY_ALIASES.get(normalized, normalized.upper())

    @classmethod
    def _currency_intent(cls, query: str) -> tuple[bool, bool, str]:
        for pattern in cls._CONVERSION_TARGET_RULES:
            match = pattern.search(query)
            if match:
                target = cls._normalize_currency(match.group("target"))
                return True, True, target
        requires_rate = bool(cls._EXCHANGE_RATE_RULE.search(query))
        return requires_rate, False, ""

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        requires_exchange_rate, currency_conversion, target_currency = (
            cls._currency_intent(query)
        )
        intents = [
            name
            for name, pattern in cls._INTENT_RULES.items()
            if re.search(pattern, query, re.IGNORECASE)
        ]
        if currency_conversion:
            intents.append("currency_conversion")
        elif requires_exchange_rate:
            intents.append("exchange_rate")
        aggregations = [
            name
            for name, pattern in cls._AGGREGATION_RULES.items()
            if re.search(pattern, query, re.IGNORECASE)
        ]
        time_grain = next(
            (
                grain
                for grain, pattern in cls._TIME_GRAINS
                if re.search(pattern, query, re.IGNORECASE)
            ),
            "",
        )
        return QueryAnalysis(
            intents=intents or ["detail"],
            aggregations=aggregations,
            time_grain=time_grain,
            has_time_filter=bool(
                time_grain or re.search(cls._TIME_FILTER_RULE, query, re.IGNORECASE)
            ),
            asks_distinct=bool(re.search(r"去重|不同|唯一|distinct", query, re.I)),
            requires_exchange_rate=requires_exchange_rate,
            currency_conversion=currency_conversion,
            target_currency=target_currency,
        )
