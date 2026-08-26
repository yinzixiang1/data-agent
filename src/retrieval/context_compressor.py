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

import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """你是一个查询状态更新与轮次意图识别助手。不要机械拼接历史文本，而要理解用户是在追加、修改、纠正、操作上一轮结果，还是开始新问题。

请根据历史查询状态和本轮输入，输出一个 JSON 对象。禁止输出 Markdown 或解释。

规则:
1. query_state 只是查询对象、时间范围、筛选条件、指标、展示维度、币种换算和排除项的粗粒度摘要，不是完整需求清单。
   事件顺序、派生结果、JSON 内容、终态判断和其他复杂要求必须完整保留在 effective_question 中，不得为了适配 query_state 而删减。
2. 上一轮已验证 SQL 是本轮修改的结构基线；除非用户明确替换、删除或开始新问题，必须继承其中的表、字段、展示维度、聚合方式和过滤条件。
3. “再展示/加上/带上”表示增量追加；“只要/仅查询/不要/去掉/改为”表示替换或移除对应状态。
4. 修改指标或展示维度时，可以保留仍适用的时间和筛选条件，但不能保留用户明确移除的金额、币种、换算或维度。
5. 新问题与历史无关时 relation=new_question，只使用本轮输入。没有历史时必须返回 new_question。
6. 只有存在会实质改变 SQL 的歧义或冲突时 needs_clarification=true；明确时直接给出完整状态。
   本阶段尚未看到数据库 Schema、枚举和业务术语，因此不得因为“成功”“完成”“渠道”“主体”等业务词如何映射到物理字段而澄清；
   应原样保留这些词，由后续 RAG 和 SQL 生成阶段依据证据解释。只有指代不清、与上一轮要求互相冲突且无法形成完整问题时才澄清。
7. effective_question 是完整需求的唯一文本事实来源，必须包含所有保留项和新增项，不能包含已移除项。
   query_question 是只供数据检索和 SQL 生成使用的查询需求：必须保留取得基础结果所需的对象、指标、维度、筛选、排序和限制，
   但移除只对结果执行的分析、解释、图表、导出、下载或发布动作。不得通过关键词机械删减；例如“按月统计交易额并分析趋势和异常”
   的 query_question 应是“按月统计交易额”，因为月份维度是后续趋势分析所需的基础结果结构。
8. relation=follow_up_add 仅表示增加内容且不改变任何已有结构；只要替换或删除已有条件、字段、维度或指标，就必须使用 follow_up_modify 或 correction_override，并在 changes.removed 中列出被替换或删除的内容。
9. turn_intent 用于通用流程路由：
   - sql_query：首次查询、查询追问、纠正或澄清回答；
   - result_explanation：只解释上一轮查询口径或结果，不改变查询语义；
   - result_operation：只基于上一轮结果生成图表、导出或执行其他操作，不改变查询语义；
   - non_query：与数据查询无关的闲聊或能力咨询。
   要求增删筛选、字段、维度、指标、排序或限制行时仍是 sql_query；不得根据具体业务词、表名或字段名决定 turn_intent。
   result_explanation/result_operation 有历史时 relation=follow_up_add，query_state 完全保留上一轮状态。
10. 结果展示动作与 SQL 查询语义是两个正交状态。presentation_relation 表示如何更新上一轮已确认的图表、导出或发布动作：
    - inherit：本轮没有提及展示方式，继续沿用；
    - add：在已有展示动作上新增动作；
    - replace：明确换成另一种展示或交付方式；
    - clear：明确不要原展示动作，或开始一个没有展示要求的新问题。
    同一句同时要求查询结构变化和结果展示时，turn_intent 必须是 sql_query，并通过 presentation_relation 记录展示动作；例如“增加日期维度并生成折线图”会改变 SQL 维度，不能归类为纯 result_operation。
    只有完全不改变筛选、字段、维度、指标、排序、限制和其他 SQL 语义时，才使用 result_operation。
11. result_shape 描述最终结果结构：count_only 表示最终只能返回一个 COUNT 结果且不得分组；aggregate 表示聚合结果；detail 表示明细；mixed 表示混合结果；无法判断时为 unknown。
    currency_conversion 使用目标币种的 ISO 4217 三位代码（例如 USD）；没有换汇要求时为空字符串。
    calendar_day_window 仅用于“最近/近 N 天”这类滚动自然日窗口，填写正整数 N；默认包含今天，因此 SQL 下界是 CURDATE() 减 N-1 天，上界是明天零点。其他时间表达填写 null。
    requested_limit 仅记录用户明确要求的 Top N、前 N 条或数量限制；系统默认限制不算用户要求，没有明确限制时填写 null。
12. removed_sql_context 是对上一轮 SQL 结构的精确删除授权。仅当用户明确删除或替换上一轮结构时填写，并从【上一轮 SQL 结构】中逐字复制对应片段；没有删除时各项必须为空。未列入其中的上一轮结构必须继续保留。
13. turn_intent=non_query 时 direct_response 给出简短回复且不得虚构查询结果；其他意图 direct_response 为空字符串。

输出格式:
{{
  "relation": "follow_up_add|follow_up_modify|correction_override|new_question",
  "turn_intent": "sql_query|result_explanation|result_operation|non_query",
  "presentation_relation": "inherit|add|replace|clear",
  "query_state": {{
    "subject": "查询对象",
    "time_range": "时间范围",
    "filters": ["筛选条件"],
    "metrics": ["统计指标"],
    "dimensions": ["展示维度"],
    "currency_conversion": "目标币种或空字符串",
    "result_shape": "count_only|aggregate|detail|mixed|unknown",
    "calendar_day_window": null,
    "requested_limit": null,
    "exclusions": ["明确不需要的内容"]
  }},
  "changes": {{"kept": ["保留项"], "set": ["新增或修改项"], "removed": ["移除项"]}},
  "removed_sql_context": {{
    "tables": [], "projections": [], "dimensions": [], "filters": [],
    "joins": [], "order_by": [], "limit": []
  }},
  "effective_question": "合并后完整、独立且无歧义的问题",
  "query_question": "仅包含数据检索和 SQL 生成语义的问题",
  "interpretation": "一句话说明本轮保留、修改和移除了什么",
  "direct_response": "仅 non_query 时填写",
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

_TURN_INTENTS = {
    "sql_query",
    "result_explanation",
    "result_operation",
    "non_query",
}

_PRESENTATION_RELATIONS = {
    "inherit",
    "add",
    "replace",
    "clear",
}

_RESULT_SHAPES = {
    "count_only",
    "aggregate",
    "detail",
    "mixed",
    "unknown",
}

_SQL_CONTEXT_SECTIONS = (
    "tables",
    "columns",
    "projections",
    "dimensions",
    "filters",
    "joins",
    "order_by",
    "limit",
)


@dataclass(frozen=True)
class QueryState:
    subject: str = ""
    time_range: str = ""
    filters: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    currency_conversion: str = ""
    result_shape: str = "unknown"
    calendar_day_window: int | None = None
    requested_limit: int | None = None
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

        result_shape = str(payload.get("result_shape") or "unknown").strip()
        if result_shape not in _RESULT_SHAPES:
            result_shape = "unknown"

        def positive_int(key: str, maximum: int) -> int | None:
            value = payload.get(key)
            if isinstance(value, bool):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if 1 <= parsed <= maximum else None

        currency_conversion = str(payload.get("currency_conversion") or "").strip()[:32]
        currency_codes = re.findall(
            r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])",
            currency_conversion,
        )
        if currency_codes:
            currency_conversion = currency_codes[-1].upper()

        return cls(
            subject=str(payload.get("subject") or "").strip()[:200],
            time_range=str(payload.get("time_range") or "").strip()[:100],
            filters=strings("filters"),
            metrics=strings("metrics"),
            dimensions=strings("dimensions"),
            currency_conversion=currency_conversion,
            result_shape=result_shape,
            calendar_day_window=positive_int("calendar_day_window", 3660),
            requested_limit=positive_int("requested_limit", 100000),
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
            "result_shape": self.result_shape,
            "calendar_day_window": self.calendar_day_window,
            "requested_limit": self.requested_limit,
            "exclusions": list(self.exclusions),
        }


@dataclass(frozen=True)
class ContextMergeResult:
    effective_question: str
    query_question: str
    query_state: QueryState
    relation: str = "follow_up_modify"
    turn_intent: str = "sql_query"
    presentation_relation: str = "inherit"
    interpretation: str = ""
    direct_response: str = ""
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)
    removed_sql_context: dict = field(default_factory=dict)


class ContextCompressor:
    """多轮对话上下文压缩器。"""

    def __init__(self, model: BaseChatModel, custom_prompt: str = ""):
        self.model = model
        custom_instructions = custom_prompt.strip()
        self.prompt_template = (
            f"{custom_instructions}\n\n{COMPRESS_PROMPT}"
            if custom_instructions
            else COMPRESS_PROMPT
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
        prev_sql_context = summary["sql_context"]
        previous_query_state = QueryState.from_value(summary.get("query_state"))
        previous_presentation_state = self.normalize_presentation_state(
            summary.get("presentation_state")
        )

        history_text = f"问题: {prev_question}" if prev_question else "无（这是新问题）"
        history_text += "\n结构化状态: " + json.dumps(
            previous_query_state.to_dict(), ensure_ascii=False
        )
        history_text += "\n结果展示状态: " + json.dumps(
            previous_presentation_state,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if prev_tables:
            history_text += f"\n涉及表: {prev_tables}"
        if prev_sql:
            history_text += f"\n上一轮已验证 SQL:\n```sql\n{prev_sql}\n```"
        if any(prev_sql_context.values()):
            history_text += "\n上一轮 SQL 结构: " + json.dumps(
                prev_sql_context,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        prompt = self.prompt_template.format(
            history=history_text, current=current_question
        )

        try:
            resp = self.model.invoke([HumanMessage(content=prompt)])
            result = self._parse_merge_response(
                str(resp.content or ""),
                current_question=current_question,
                previous_question=prev_question,
                previous_query_state=previous_query_state,
                previous_presentation_state=previous_presentation_state,
                previous_sql_context=prev_sql_context,
            )
            if not prev_question and result.relation != "new_question":
                result = ContextMergeResult(
                    effective_question=result.effective_question,
                    query_question=result.query_question,
                    query_state=result.query_state,
                    relation="new_question",
                    turn_intent=result.turn_intent,
                    presentation_relation=(
                        "clear"
                        if result.presentation_relation == "inherit"
                        else result.presentation_relation
                    ),
                    interpretation=result.interpretation,
                    direct_response=result.direct_response,
                    confidence=result.confidence,
                    needs_clarification=result.needs_clarification,
                    clarification=result.clarification,
                    changes=result.changes,
                    removed_sql_context=result.removed_sql_context,
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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "context state merge fallback",
                extra={"error": str(exc)},
            )
            return self._fallback_result(
                current_question,
                prev_question,
                previous_query_state,
            )

    @classmethod
    def _parse_merge_response(
        cls,
        content: str,
        *,
        current_question: str,
        previous_question: str,
        previous_query_state: QueryState,
        previous_presentation_state: dict,
        previous_sql_context: dict[str, list[str]],
    ) -> ContextMergeResult:
        stripped = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1).strip()
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return cls._fallback_result(
                current_question,
                previous_question,
                previous_query_state,
            )
        if not isinstance(payload, dict):
            return cls._fallback_result(
                current_question,
                previous_question,
                previous_query_state,
            )

        effective_question = str(
            payload.get("effective_question") or current_question
        ).strip()[:2000]
        query_question = str(
            payload.get("query_question") or effective_question
        ).strip()[:2000]
        relation = str(payload.get("relation") or "follow_up_modify")
        if relation not in _RELATIONS:
            relation = "follow_up_modify"
        turn_intent = str(payload.get("turn_intent") or "sql_query")
        if turn_intent not in _TURN_INTENTS:
            turn_intent = "sql_query"
        presentation_relation = str(payload.get("presentation_relation") or "").strip()
        if presentation_relation not in _PRESENTATION_RELATIONS:
            presentation_relation = "inherit" if relation != "new_question" else "clear"
        if relation == "new_question" and presentation_relation == "inherit":
            presentation_relation = "clear"
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
        return ContextMergeResult(
            effective_question=effective_question,
            query_question=query_question,
            query_state=QueryState.from_value(payload.get("query_state")),
            relation=relation,
            turn_intent=turn_intent,
            presentation_relation=presentation_relation,
            interpretation=str(payload.get("interpretation") or "").strip()[:500],
            direct_response=str(payload.get("direct_response") or "").strip()[:1000],
            confidence=confidence,
            needs_clarification=needs_clarification,
            clarification=clarification,
            changes=cls._normalize_changes(payload.get("changes")),
            removed_sql_context=cls._normalize_removed_sql_context(
                payload.get("removed_sql_context"),
                previous_sql_context,
            ),
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

    @staticmethod
    def _normalize_removed_sql_context(
        value: object,
        previous_sql_context: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Accept only exact previous SQL fragments as removal authorization."""
        payload = value if isinstance(value, dict) else {}

        def normalized(fragment: object) -> str:
            return re.sub(
                r"\s+", " ", str(fragment).replace("`", "").strip()
            ).casefold()

        result: dict[str, list[str]] = {}
        for section in _SQL_CONTEXT_SECTIONS:
            previous_values = previous_sql_context.get(section) or []
            previous_by_normalized = {
                normalized(fragment): str(fragment)
                for fragment in previous_values
                if str(fragment).strip()
            }
            requested = payload.get(section)
            requested = requested if isinstance(requested, list) else []
            result[section] = list(
                dict.fromkeys(
                    previous_by_normalized[key]
                    for fragment in requested
                    if (key := normalized(fragment)) in previous_by_normalized
                )
            )
        return result

    @classmethod
    def _fallback_result(
        cls,
        current_question: str,
        previous_question: str,
        previous_query_state: QueryState | None = None,
    ) -> ContextMergeResult:
        effective_question = current_question
        relation = "new_question"
        query_state = QueryState()
        if previous_question:
            effective_question = f"{previous_question}\n用户补充：{current_question}"
            relation = "follow_up_modify"
            query_state = previous_query_state or QueryState()
        return ContextMergeResult(
            effective_question=effective_question,
            query_question=effective_question,
            query_state=query_state,
            relation=relation,
            turn_intent="sql_query",
            presentation_relation="inherit" if previous_question else "clear",
            confidence=0.5,
            changes={
                "kept": ["上一轮完整查询"] if previous_question else [],
                "set": [current_question],
                "removed": [],
            },
            removed_sql_context={section: [] for section in _SQL_CONTEXT_SECTIONS},
        )

    @classmethod
    def infer_state(cls, question: str) -> QueryState:
        del question
        return QueryState()

    @staticmethod
    def extract_sql_context(sql: str) -> dict[str, list[str]]:
        """Extract reusable SQL structure without interpreting business meaning."""
        empty = {
            "tables": [],
            "columns": [],
            "projections": [],
            "dimensions": [],
            "filters": [],
            "joins": [],
            "order_by": [],
            "limit": [],
        }
        if not sql.strip():
            return empty
        try:
            import sqlglot
            from sqlglot import expressions as exp

            tree = sqlglot.parse_one(sql, read="mysql")
        except (ImportError, ValueError):
            return empty
        except sqlglot.errors.SqlglotError as exc:
            logger.info(
                "SQL context extraction skipped",
                extra={"error_type": type(exc).__name__},
            )
            return empty

        aliases: dict[str, str] = {}
        tables: list[str] = []
        for table in tree.find_all(exp.Table):
            if not table.db:
                continue
            full_name = f"{table.db}.{table.name}"
            tables.append(full_name)
            aliases[table.alias_or_name.casefold()] = full_name
            aliases[table.name.casefold()] = full_name
        tables = list(dict.fromkeys(tables))
        single_table = tables[0] if len(tables) == 1 else None

        columns: list[str] = []
        for column in tree.find_all(exp.Column):
            owner = (
                aliases.get(str(column.table).casefold())
                if column.table
                else single_table
            )
            if owner:
                columns.append(f"{owner}.{column.name}")

        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        group = select.args.get("group") if select is not None else None
        where = select.args.get("where") if select is not None else None
        order = select.args.get("order") if select is not None else None
        limit = select.args.get("limit") if select is not None else None

        def split_filters(expression: exp.Expression) -> list[exp.Expression]:
            if isinstance(expression, exp.And):
                return split_filters(expression.left) + split_filters(expression.right)
            return [expression]

        return {
            "tables": tables,
            "columns": list(dict.fromkeys(columns)),
            "projections": [
                expression.sql(dialect="mysql")
                for expression in (select.expressions if select is not None else [])
            ],
            "dimensions": [
                expression.sql(dialect="mysql")
                for expression in (group.expressions if group else [])
            ],
            "filters": [
                expression.sql(dialect="mysql")
                for expression in (split_filters(where.this) if where else [])
            ],
            "joins": [
                join.sql(dialect="mysql")
                for join in (select.args.get("joins") or [] if select else [])
            ],
            "order_by": [
                expression.sql(dialect="mysql")
                for expression in (order.expressions if order else [])
            ],
            "limit": [limit.sql(dialect="mysql")] if limit else [],
        }

    @staticmethod
    def build_summary(
        question: str,
        tables: list[str],
        sql: str = "",
        query_state: QueryState | dict | None = None,
        presentation_state: object = None,
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
        return json.dumps(
            {
                "question": question,
                "tables": tables,
                "sql": sql,
                "sql_context": ContextCompressor.extract_sql_context(sql),
                "query_state": state.to_dict(),
                "presentation_state": ContextCompressor.normalize_presentation_state(
                    presentation_state
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def parse_summary(history_summary: str) -> dict:
        """读取当前结构化摘要；无效或旧格式按空状态处理。"""
        try:
            payload = json.loads(history_summary)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            tables = payload.get("tables") or []
            sql = str(payload.get("sql") or "")
            sql_context = payload.get("sql_context")
            if not isinstance(sql_context, dict):
                sql_context = ContextCompressor.extract_sql_context(sql)
            return {
                "question": str(payload.get("question") or ""),
                "tables": [str(table) for table in tables if table],
                "sql": sql,
                "sql_context": {
                    key: [str(item) for item in sql_context.get(key, []) if item]
                    for key in _SQL_CONTEXT_SECTIONS
                },
                "query_state": QueryState.from_value(
                    payload.get("query_state")
                ).to_dict(),
                "presentation_state": ContextCompressor.normalize_presentation_state(
                    payload.get("presentation_state")
                ),
            }

        return {
            "question": "",
            "tables": [],
            "sql": "",
            "sql_context": ContextCompressor.extract_sql_context(""),
            "query_state": QueryState().to_dict(),
            "presentation_state": ContextCompressor.normalize_presentation_state(None),
        }

    @staticmethod
    def normalize_presentation_state(value: object) -> dict:
        """Keep only serializable, validated-result action state."""
        payload = value if isinstance(value, dict) else {}
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list):
            raw_calls = []

        calls = []
        remaining_argument_chars = 6000
        for raw_call in raw_calls[:5]:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()[:128]
            arguments = raw_call.get("arguments")
            if not name or not isinstance(arguments, dict):
                continue
            try:
                rendered_arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(rendered_arguments) > remaining_argument_chars:
                    continue
                normalized_arguments = json.loads(rendered_arguments)
            except (TypeError, ValueError):
                continue
            calls.append(
                {
                    "name": name,
                    "arguments": normalized_arguments,
                    "requires_query_result": bool(
                        raw_call.get("requires_query_result")
                    ),
                }
            )
            remaining_argument_chars -= len(rendered_arguments)
        return {"tool_calls": calls}

    @staticmethod
    def update_presentation_state(
        previous_state: object,
        current_tool_calls: object,
        *,
        relation: str,
        context_relation: str,
    ) -> dict:
        """Apply the current presentation patch after tool-call validation."""
        previous = ContextCompressor.normalize_presentation_state(previous_state)
        current = ContextCompressor.normalize_presentation_state(
            {"tool_calls": current_tool_calls}
        )
        if relation == "clear" and context_relation != "new_question":
            return {"tool_calls": []}

        base_calls = (
            []
            if context_relation == "new_question" or relation == "replace"
            else previous["tool_calls"]
        )
        calls_by_name = {call["name"]: call for call in base_calls}
        for call in current["tool_calls"]:
            calls_by_name[call["name"]] = call
        return {"tool_calls": list(calls_by_name.values())[:5]}
