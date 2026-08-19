"""
多轮对话上下文压缩 — 将历史问答 + 当前补充合并为一个完整的自然语言问题。

解决的问题:
    用户第1轮: "张3今天有多少笔交易"
    用户第2轮: "不包含手续费，一次兑换属于一笔交易"
    → 压缩为: "张3今天有多少笔交易，不包含手续费，一次兑换属于一笔交易"

这样 RAG 检索时用完整问题去检索，命中率更高。

使用示例::

    compressor = ContextCompressor(client, model="deepseek-chat")
    merged = compressor.compress(
        history_summary="张3今天有多少笔交易|||pmt_finance_transactions,pmt_account",
        current_question="不包含手续费，一次兑换属于一笔交易",
    )
    # "张3今天有多少笔交易，不包含手续费，一次兑换属于一笔交易"
"""

import logging
import json
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """你是一个查询状态更新助手。不要机械拼接历史文本，而要理解用户是在追加、修改、纠正还是开始新问题。

请根据历史查询状态和本轮输入，输出一个 JSON 对象。禁止输出 Markdown 或解释。

规则:
1. 查询状态分为：查询对象、时间范围、筛选条件、指标、展示维度、币种换算、排除项。
2. “再展示/加上/带上”表示增量追加；“只要/仅查询/不要/去掉/改为”表示替换或移除对应状态。
3. 修改指标或展示维度时，可以保留仍适用的时间和筛选条件，但不能保留用户明确移除的金额、币种、换算或维度。
4. 新问题与历史无关时 relation=new_question，只使用本轮输入。
5. 只有存在会实质改变 SQL 的歧义或冲突时 needs_clarification=true；明确时直接给出完整状态。
6. effective_question 必须与 query_state 完全一致，不能包含已移除的指标或维度。

输出格式:
{{
  "relation": "follow_up_add|follow_up_modify|correction_override|new_question",
  "query_state": {{
    "subject": "查询对象",
    "time_range": "时间范围",
    "filters": ["筛选条件"],
    "metrics": ["统计指标"],
    "dimensions": ["展示维度"],
    "currency_conversion": "目标币种或空字符串",
    "exclusions": ["明确不需要的内容"]
  }},
  "changes": {{"kept": ["保留项"], "set": ["新增或修改项"], "removed": ["移除项"]}},
  "effective_question": "合并后完整、独立且无歧义的问题",
  "interpretation": "一句话说明本轮保留、修改和移除了什么",
  "confidence": 0.0,
  "needs_clarification": false,
  "clarification": {{"question": "", "options": []}}
}}

## 历史查询
{history}

## 用户新补充
{current}

## 查询状态更新结果 JSON"""

_RELATIONS = {
    "follow_up_add",
    "follow_up_modify",
    "correction_override",
    "new_question",
}
_COUNT_ONLY_RE = re.compile(
    r"(?:只|仅)(?:需|要|查询|查看|统计|返回|展示)?[^，。；！？\n]{0,20}"
    r"(?:次数|笔数|数量|个数)|\bonly\b[^,.;!?\n]{0,30}\b(?:count|number)\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(r"次数|笔数|数量|个数|\bcount\b|\bnumber\b", re.IGNORECASE)
_AMOUNT_OR_CURRENCY_RE = re.compile(
    r"金额|币种|货币|汇率|折(?:算)?(?:美元|美金)|换算|兑换|"
    r"\bamount\b|\bcurrency\b|exchange\s+rate",
    re.IGNORECASE,
)
_REMOVE_AMOUNT_RE = re.compile(
    r"(?:不要|不需要|无需|去掉|移除|不返回|不展示)[^，。；！？\n]{0,12}"
    r"(?:金额|币种|货币|汇率|换算)|without\s+(?:amount|currency)",
    re.IGNORECASE,
)
_TIME_PATTERNS = (
    re.compile(
        r"(?:最近|近|过去)\s*"
        r"(?:\d+|[一二两三四五六七八九十百]+个?)?\s*"
        r"(?:天|周|个月|月|季度|年)"
    ),
    re.compile(r"这(?:一|个|一个)?(?:天|周|月|个月|季度|年)"),
    re.compile(r"(?:今天|今日|昨天|昨日|本周|本月|本季度|本年|今年|上周|上月|去年)"),
    re.compile(
        r"\b(?:last|past|recent|this)\s+(?:\d+\s+)?"
        r"(?:day|week|month|quarter|year)s?\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class QueryState:
    subject: str = ""
    time_range: str = ""
    filters: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    currency_conversion: str = ""
    exclusions: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> "QueryState":
        payload = value if isinstance(value, dict) else {}

        def strings(key: str) -> tuple[str, ...]:
            values = payload.get(key) or []
            if not isinstance(values, list):
                return ()
            return tuple(
                str(item).strip()[:100] for item in values if str(item).strip()
            )

        return cls(
            subject=str(payload.get("subject") or "").strip()[:200],
            time_range=str(payload.get("time_range") or "").strip()[:100],
            filters=strings("filters"),
            metrics=strings("metrics"),
            dimensions=strings("dimensions"),
            currency_conversion=str(payload.get("currency_conversion") or "").strip()[
                :32
            ],
            exclusions=strings("exclusions"),
        )

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "time_range": self.time_range,
            "filters": list(self.filters),
            "metrics": list(self.metrics),
            "dimensions": list(self.dimensions),
            "currency_conversion": self.currency_conversion,
            "exclusions": list(self.exclusions),
        }


@dataclass(frozen=True)
class ContextMergeResult:
    effective_question: str
    query_state: QueryState
    relation: str = "follow_up_modify"
    interpretation: str = ""
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)


class ContextCompressor:
    """多轮对话上下文压缩器。"""

    def __init__(self, model: BaseChatModel, custom_prompt: str = ""):
        self.model = model
        self.prompt_template = (
            custom_prompt.strip() if custom_prompt else COMPRESS_PROMPT
        )

    def compress(self, history_summary: str, current_question: str) -> str:
        """
        将历史摘要 + 当前问题压缩为一个完整问题。

        Args:
            history_summary: 上一轮的摘要，格式 "question|||table1,table2,..."
            current_question: 当前用户输入

        Returns:
            合并后的完整自然语言问题
        """
        return self.merge(history_summary, current_question).effective_question

    def merge(
        self,
        history_summary: str,
        current_question: str,
    ) -> ContextMergeResult:
        """Apply the current utterance as a semantic patch to the previous state."""
        summary = self.parse_summary(history_summary)
        prev_question = summary["question"]
        prev_tables = ",".join(summary["tables"])
        prev_sql = summary["sql"]
        previous_query_state = QueryState.from_value(summary.get("query_state"))

        history_text = f"问题: {prev_question}"
        history_text += "\n结构化状态: " + json.dumps(
            previous_query_state.to_dict(), ensure_ascii=False
        )
        if prev_tables:
            history_text += f"\n涉及表: {prev_tables}"
        if prev_sql:
            history_text += f"\n上一轮已验证 SQL:\n```sql\n{prev_sql}\n```"

        prompt = self.prompt_template.format(
            history=history_text, current=current_question
        )

        try:
            resp = self.model.invoke([HumanMessage(content=prompt)])
            result = self._parse_merge_response(
                str(resp.content or ""),
                current_question=current_question,
                previous_question=prev_question,
            )
            logger.info(
                "context state merged",
                extra={
                    "relation": result.relation,
                    "confidence": result.confidence,
                    "needs_clarification": result.needs_clarification,
                },
            )
            return result
        except Exception as exc:
            logger.warning(
                "context state merge fallback",
                extra={"error": str(exc)},
            )
            return self._fallback_result(current_question, prev_question)

    @classmethod
    def _parse_merge_response(
        cls,
        content: str,
        *,
        current_question: str,
        previous_question: str,
    ) -> ContextMergeResult:
        stripped = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1).strip()
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            result = ContextMergeResult(
                effective_question=stripped or current_question,
                query_state=cls.infer_state(stripped or current_question),
            )
            return cls._apply_deterministic_overrides(
                result,
                current_question=current_question,
                previous_question=previous_question,
            )
        if not isinstance(payload, dict):
            return cls._fallback_result(current_question, previous_question)

        effective_question = str(
            payload.get("effective_question") or current_question
        ).strip()[:2000]
        relation = str(payload.get("relation") or "follow_up_modify")
        if relation not in _RELATIONS:
            relation = "follow_up_modify"
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 1.0))))
        except (TypeError, ValueError):
            confidence = 1.0
        clarification = cls._normalize_clarification(payload.get("clarification"))
        needs_clarification = bool(payload.get("needs_clarification"))
        if needs_clarification and not clarification.get("question"):
            clarification = {
                "question": "你希望保留上一轮哪些查询条件？",
                "options": [],
            }
        result = ContextMergeResult(
            effective_question=effective_question,
            query_state=QueryState.from_value(payload.get("query_state")),
            relation=relation,
            interpretation=str(payload.get("interpretation") or "").strip()[:500],
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification=clarification,
            changes=cls._normalize_changes(payload.get("changes")),
        )
        return cls._apply_deterministic_overrides(
            result,
            current_question=current_question,
            previous_question=previous_question,
        )

    @staticmethod
    def _normalize_clarification(value: object) -> dict:
        payload = value if isinstance(value, dict) else {}
        question = str(payload.get("question") or "").strip()[:1000]
        options = []
        for option in payload.get("options") or []:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").strip()[:80]
            option_value = str(option.get("value") or "").strip()[:500]
            if label and option_value:
                options.append({"label": label, "value": option_value})
            if len(options) == 4:
                break
        return {"question": question, "options": options}

    @staticmethod
    def _normalize_changes(value: object) -> dict:
        payload = value if isinstance(value, dict) else {}
        normalized = {}
        for key in ("kept", "set", "removed"):
            values = payload.get(key)
            if not isinstance(values, list):
                values = []
            normalized[key] = [
                str(item).strip()[:100] for item in values if str(item).strip()
            ]
        return normalized

    @classmethod
    def _fallback_result(
        cls,
        current_question: str,
        previous_question: str,
    ) -> ContextMergeResult:
        result = ContextMergeResult(
            effective_question=current_question,
            query_state=cls.infer_state(current_question),
            relation="follow_up_modify",
            confidence=0.5,
        )
        return cls._apply_deterministic_overrides(
            result,
            current_question=current_question,
            previous_question=previous_question,
        )

    @classmethod
    def infer_state(cls, question: str) -> QueryState:
        from src.retrieval.query_analyzer import QueryAnalyzer

        analysis = QueryAnalyzer.analyze(question)
        return QueryState(
            time_range=cls._extract_time_range(question),
            metrics=tuple(analysis.aggregations),
            dimensions=tuple(analysis.requested_fields),
            currency_conversion=(
                analysis.target_currency if analysis.currency_conversion else ""
            ),
        )

    @staticmethod
    def _extract_time_range(question: str) -> str:
        for pattern in _TIME_PATTERNS:
            if match := pattern.search(question):
                return match.group(0).strip()
        return ""

    @classmethod
    def _apply_deterministic_overrides(
        cls,
        result: ContextMergeResult,
        *,
        current_question: str,
        previous_question: str,
    ) -> ContextMergeResult:
        has_count = bool(_COUNT_RE.search(current_question))
        explicit_count_only = bool(_COUNT_ONLY_RE.search(current_question))
        mentions_amount = bool(_AMOUNT_OR_CURRENCY_RE.search(current_question))
        removes_amount = bool(_REMOVE_AMOUNT_RE.search(current_question))
        if not (
            has_count
            and explicit_count_only
            and (not mentions_amount or removes_amount)
        ):
            return result

        time_range = (
            result.query_state.time_range
            or cls._extract_time_range(current_question)
            or cls._extract_time_range(previous_question)
        )
        exclusions = tuple(
            dict.fromkeys(
                (*result.query_state.exclusions, "金额", "币种维度", "币种换算")
            )
        )
        query_state = QueryState(
            subject=result.query_state.subject,
            time_range=time_range,
            filters=result.query_state.filters,
            metrics=("count",),
            dimensions=(),
            currency_conversion="",
            exclusions=exclusions,
        )
        time_prefix = ""
        if time_range and time_range not in current_question:
            time_prefix = f"{time_range}，"
        effective_question = (
            f"{time_prefix}{current_question.strip('，。； ')}；"
            "仅返回交易次数，不返回金额，不按币种分组，不进行币种换算"
        )
        return ContextMergeResult(
            effective_question=effective_question[:2000],
            query_state=query_state,
            relation="correction_override",
            interpretation=(f"已保留“{time_range}”；" if time_range else "")
            + "已改为仅统计交易次数，并移除金额、币种维度和币种换算。",
            confidence=max(result.confidence, 0.95),
            needs_clarification=False,
            clarification={},
            changes={
                "kept": [time_range] if time_range else [],
                "set": ["仅统计交易次数"],
                "removed": ["金额", "币种维度", "币种换算"],
            },
        )

    @staticmethod
    def build_summary(
        question: str,
        tables: list[str],
        sql: str = "",
        query_state: QueryState | dict | None = None,
    ) -> str:
        """
        构建本轮摘要，供下一轮使用。

        Args:
            question: 本轮的完整问题（可能已经过压缩）
            tables: 本轮命中的表名列表

        Returns:
            JSON 摘要字符串；读取端仍兼容旧版 question|||table1,table2 格式。
        """
        state = (
            query_state
            if isinstance(query_state, QueryState)
            else QueryState.from_value(query_state)
        )
        if not any(state.to_dict().values()):
            state = ContextCompressor.infer_state(question)
        return json.dumps(
            {
                "question": question,
                "tables": tables,
                "sql": sql,
                "query_state": state.to_dict(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def parse_summary(history_summary: str) -> dict:
        """读取结构化摘要，并兼容已保存的旧版分隔符格式。"""
        try:
            payload = json.loads(history_summary)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            tables = payload.get("tables") or []
            return {
                "question": str(payload.get("question") or ""),
                "tables": [str(table) for table in tables if table],
                "sql": str(payload.get("sql") or ""),
                "query_state": QueryState.from_value(
                    payload.get("query_state")
                ).to_dict(),
            }

        parts = history_summary.split("|||", 1)
        return {
            "question": parts[0] if parts else "",
            "tables": [
                table
                for table in (parts[1].split(",") if len(parts) > 1 else [])
                if table
            ],
            "sql": "",
            "query_state": QueryState().to_dict(),
        }
