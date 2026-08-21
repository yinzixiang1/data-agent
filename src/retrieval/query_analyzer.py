"""轻量查询分析器：识别 NL2SQL 查询结构，不依赖额外 LLM 调用。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import ClassVar


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
    requested_fields: list[str] = field(default_factory=list)
    derived_metrics: list[str] = field(default_factory=list)
    presentation_actions: list[str] = field(default_factory=list)
    count_only: bool = False

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
                    (
                        "换算规则: source_currency=原币种，target_currency=目标币种，"
                        "DATE(sync_time)=DATE(交易时间)，目标金额=原金额*mid；"
                        "原币种等于目标币种时汇率按1处理"
                    ),
                    "禁止用 CASE WHEN 只保留目标币种、把其他币种金额置为0",
                    (
                        "非目标币种缺失汇率时禁止用 COALESCE/IFNULL 按1或0兜底；"
                        "换算结果保留 NULL，并显式返回缺失汇率笔数"
                    ),
                )
            )
        if self.requested_fields:
            parts.append("必须展示字段: " + ", ".join(self.requested_fields))
        if self.derived_metrics:
            parts.append(
                "派生指标（由聚合计算，不是物理字段）: "
                + ", ".join(self.derived_metrics)
            )
        if self.presentation_actions:
            parts.append(
                "结果展示动作（不作为数据库字段）: "
                + ", ".join(self.presentation_actions)
            )
        if self.count_only:
            parts.append(
                "结果契约: 仅返回 COUNT 次数；禁止返回金额、币种、汇率或其他展示维度"
            )
        return "\n".join(parts)


class QueryAnalyzer:
    """用确定性规则提取会影响 SQL 结构的意图。"""

    _INTENT_RULES: ClassVar[dict[str, str]] = {
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
    _AGGREGATION_RULES: ClassVar[dict[str, str]] = {
        "count": r"多少|数量|笔数|次数|个数|count",
        "sum": r"总计|合计|总额|总金额|sum",
        "avg": r"平均|均值|avg",
        "max": r"最高|最大|max",
        "min": r"最低|最小|min",
    }
    _TIME_GRAINS = (
        (
            "day",
            (
                r"按日|每天|每日|by\s+day|daily|each\s+day|"
                r"each\s+of\s+the\s+last\s+\d+\s+days?"
            ),
        ),
        ("week", r"按周|每周|by\s+week|weekly|each\s+week"),
        ("month", r"按月|每月|月度|by\s+month|monthly|each\s+month"),
        ("quarter", r"按季|季度|by\s+quarter|quarterly"),
        ("year", r"按年|每年|年度|by\s+year|yearly|annually"),
    )
    _TIME_FILTER_RULE = (
        r"最近|近\s*\d+|过去|今日|今天|昨日|昨天|本(?:周|月|季|年)|"
        r"这(?:一|个|一个)?(?:天|周|月|个月|季度|年)|"
        r"上(?:周|月|季|年)|last\s+\d+|past\s+\d+|recent|today|yesterday|"
        r"this\s+(?:week|month|quarter|year)|last\s+(?:week|month|quarter|year)"
    )
    _CURRENCY_ALIASES: ClassVar[dict[str, str]] = {
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
            rf"(?:转换|换算|折算|兑换|转成|换成|转|统一(?:为|成))\s*"
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
    _REQUESTED_FIELD_RULES = (
        re.compile(
            r"(?:展示|显示|列出|返回|带上|加上|查看)(?:一下|下)?"
            r"(?P<fields>[^，。；！？\n]{1,40})",
            re.IGNORECASE,
        ),
        re.compile(
            r"补充(?!\s*[：:])(?:一下|下)?(?P<fields>[^，。；！？\n]{1,40})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:show|display|include|return|add)\s+"
            r"(?P<fields>[^.;!?\n]{1,60})",
            re.IGNORECASE,
        ),
    )
    _PRESENTATION_ACTION_RULES = (
        ("趋势", re.compile(r"趋势|折线图|line\s*chart", re.IGNORECASE)),
        ("分布", re.compile(r"分布|\bdistribution\b", re.IGNORECASE)),
        ("对比", re.compile(r"对比|比较|\bcomparison\b", re.IGNORECASE)),
        (
            "图表",
            re.compile(
                r"图表|可视化|柱状图|条形图|饼图|散点图|"
                r"\bchart\b|\bvisuali[sz]e\b",
                re.IGNORECASE,
            ),
        ),
        ("看板", re.compile(r"看板|\bdashboard\b", re.IGNORECASE)),
        ("报表", re.compile(r"报表|\breport\b", re.IGNORECASE)),
        ("分析", re.compile(r"分析|\banalysis\b", re.IGNORECASE)),
    )
    _PRESENTATION_TERMS = re.compile(
        r"趋势|分布|对比|比较|图表|可视化|折线图|柱状图|条形图|饼图|散点图|"
        r"看板|报表|分析|\bline\s*chart\b|\bchart\b|\bdashboard\b|"
        r"\breport\b|\banalysis\b|\bvisuali[sz]e\b|\bdistribution\b|"
        r"\bcomparison\b",
        re.IGNORECASE,
    )
    _DERIVED_METRIC_RULE = re.compile(
        r"(?:数量|次数|笔数|个数|总数|总额|总金额|合计|平均值?|均值|最大值?|"
        r"最小值?|占比|比例|成功率|转化率)$",
        re.IGNORECASE,
    )
    _COUNT_ONLY_RULE = re.compile(
        r"(?:只|仅)(?:需|要|查询|查看|统计|返回|展示)?[^，。；！？\n]{0,24}"
        r"(?:次数|笔数|数量|个数)|"
        r"(?:次数|笔数|数量|个数)[^，。；！？\n]{0,12}(?:即可|就行)|"
        r"\bonly\b[^,.;!?\n]{0,30}\b(?:count|number)\b",
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
    def _output_requirements(cls, query: str) -> tuple[list[str], list[str]]:
        """区分必须返回的物理字段与需要计算的派生指标。"""
        fields: list[str] = []
        derived_metrics: list[str] = []
        for pattern in cls._REQUESTED_FIELD_RULES:
            for match in pattern.finditer(query):
                phrase = match.group("fields").strip(" \t:：")
                phrase = re.sub(
                    r"^(?:一下|其|对应的?|相关的?|用户的?|账户的?)",
                    "",
                    phrase,
                    flags=re.IGNORECASE,
                ).strip()
                for value in re.split(
                    r"\s*(?:、|和|以及|并且|并|与|及|,|&)\s*", phrase
                ):
                    value = re.sub(
                        r"(?:字段|信息)$", "", value.strip(), flags=re.IGNORECASE
                    ).strip(" \t:：")
                    if not value:
                        continue
                    analytical_value = cls._PRESENTATION_TERMS.sub("", value)
                    analytical_value = re.sub(
                        r"^(?:生成|做成|作为|使用|用|是|为)+|"
                        r"(?:生成|做成|作为|使用|用|是|为)+$",
                        "",
                        analytical_value,
                        flags=re.IGNORECASE,
                    ).strip(" \t:：")
                    if not analytical_value:
                        continue
                    target = (
                        derived_metrics
                        if cls._DERIVED_METRIC_RULE.search(analytical_value)
                        else fields
                    )
                    if (
                        1 < len(analytical_value) <= 30
                        and analytical_value not in target
                    ):
                        target.append(analytical_value)
        return fields, derived_metrics

    @classmethod
    def _presentation_actions(cls, query: str) -> list[str]:
        return [
            action
            for action, pattern in cls._PRESENTATION_ACTION_RULES
            if pattern.search(query)
        ]

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
        count_only = "count" in aggregations and bool(
            cls._COUNT_ONLY_RULE.search(query)
        )
        time_grain = next(
            (
                grain
                for grain, pattern in cls._TIME_GRAINS
                if re.search(pattern, query, re.IGNORECASE)
            ),
            "",
        )
        requested_fields, derived_metrics = cls._output_requirements(query)
        return QueryAnalysis(
            intents=intents or ["detail"],
            aggregations=aggregations,
            time_grain=time_grain,
            has_time_filter=bool(
                time_grain or re.search(cls._TIME_FILTER_RULE, query, re.IGNORECASE)
            ),
            asks_distinct=bool(
                re.search(r"去重|不同|唯一|distinct", query, re.IGNORECASE)
            ),
            requires_exchange_rate=requires_exchange_rate,
            currency_conversion=currency_conversion,
            target_currency=target_currency,
            requested_fields=requested_fields,
            derived_metrics=derived_metrics,
            presentation_actions=cls._presentation_actions(query),
            count_only=count_only,
        )
