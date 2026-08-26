"""Evidence-grounded analysis for an already validated query result."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from statistics import median
from typing import Any


class AnalysisSkipped(ValueError):
    """The current result cannot be analyzed without misleading the user."""


AnalysisInvoker = Callable[[list[dict[str, str]]], tuple[str, dict | None]]

_ANALYSIS_SYSTEM_PROMPT = """你是数据库查询结果分析器。
只能使用输入中的 facts，不得补造数值、业务原因、维度、口径或因果关系。
anomaly 只表示返回数据中的数值或时间序列异常点；SQL 错误、超时、连接失败和系统异常不属于数据异常，也不会作为分析输入。
每条 finding 必须引用至少一个真实 fact_id；证据不足时把限制写入 caveats。
只返回 JSON 对象：
{"title":"", "executive_summary":"", "findings":[{"type":"trend|comparison|distribution|contribution|anomaly","statement":"","evidence_fact_ids":["f1"],"confidence":"high|medium|low"}], "caveats":[], "suggested_followups":[]}。
最多返回请求指定数量的 findings 和 3 个 suggested_followups。"""

_TEMPORAL_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?"),
)


def execute_analysis(
    *,
    query_result: dict[str, Any],
    arguments: dict[str, Any],
    binding_config: dict[str, Any],
    invoke: AnalysisInvoker,
) -> tuple[dict[str, Any], dict | None]:
    """Build deterministic facts and let the LLM select grounded insights."""
    rows = query_result.get("rows")
    columns = query_result.get("columns")
    if not isinstance(columns, list) or not columns:
        raise AnalysisSkipped("查询结果没有可分析的字段")
    if not isinstance(rows, list) or not rows:
        raise AnalysisSkipped("查询结果为空，没有可分析的数据")

    allow_truncated = bool(binding_config.get("allow_truncated_result", False))
    if query_result.get("truncated") and not allow_truncated:
        raise AnalysisSkipped(
            "查询结果已截断；为避免基于不完整样本得出结论，本次未分析"
        )

    max_rows = _bounded_int(binding_config.get("max_input_rows"), 1000, 10, 5000)
    max_findings = _bounded_int(binding_config.get("max_findings"), 5, 1, 10)
    modes = _analysis_modes(
        arguments.get("modes"),
        default_mode=binding_config.get("default_mode"),
    )
    focus_metrics = _string_list(arguments.get("focus_metrics"), limit=5)
    group_by = _string_list(arguments.get("group_by"), limit=3)
    facts, profile = build_analysis_facts(
        query_result,
        modes=modes,
        focus_metrics=focus_metrics,
        group_by=group_by,
        max_rows=max_rows,
        max_facts=max_findings * 3 + 3,
    )
    insight_facts = [fact for fact in facts if fact["type"] != "dataset"]
    if not insight_facts:
        raise AnalysisSkipped("当前结果缺少足够的数值或时间序列，无法生成可靠洞察")

    requested_title = str(arguments.get("title") or "").strip()[:120]
    payload = {
        "analysis_modes": modes,
        "requested_title": requested_title,
        "max_findings": max_findings,
        "profile": profile,
        "facts": facts,
    }
    answer, usage = invoke(
        [
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), default=str
                ),
            },
        ]
    )
    report = _validated_report(
        answer,
        facts=facts,
        max_findings=max_findings,
        fallback_title=requested_title or "查询结果智能分析",
    )
    caveats = list(report["caveats"])
    if query_result.get("truncated"):
        caveats.insert(0, "查询结果已截断，结论仅适用于返回的数据范围。")
    if profile["used_row_count"] < profile["input_row_count"]:
        caveats.insert(
            0,
            f"分析使用前 {profile['used_row_count']} 行，原结果包含 "
            f"{profile['input_row_count']} 行。",
        )
    report["caveats"] = list(dict.fromkeys(caveats))[:8]
    report["facts"] = facts
    report["source"] = {
        "row_count": int(query_result.get("row_count") or len(rows)),
        "used_row_count": profile["used_row_count"],
        "truncated": bool(query_result.get("truncated")),
        "columns": profile["columns"],
        "metrics": profile["metrics"],
        "dimensions": profile["dimensions"],
    }
    return report, usage


def build_analysis_facts(
    query_result: dict[str, Any],
    *,
    modes: list[str] | None = None,
    focus_metrics: list[str] | None = None,
    group_by: list[str] | None = None,
    max_rows: int = 1000,
    max_facts: int = 18,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create compact statistical facts without inferring business semantics."""
    raw_columns = query_result.get("columns")
    raw_rows = query_result.get("rows")
    columns = (
        [str(column) for column in raw_columns] if isinstance(raw_columns, list) else []
    )
    rows = raw_rows[:max_rows] if isinstance(raw_rows, list) else []
    normalized_rows = [_normalize_row(row, columns) for row in rows]

    numeric_values: dict[str, list[tuple[int, Decimal]]] = {}
    for column in columns:
        values = []
        nonempty = 0
        for index, row in enumerate(normalized_rows):
            value = row.get(column)
            if value not in (None, ""):
                nonempty += 1
            number = _as_decimal(value)
            if number is not None:
                values.append((index, number))
        if nonempty and len(values) == nonempty:
            numeric_values[column] = values

    requested_metrics = [
        column for column in (focus_metrics or []) if column in numeric_values
    ]
    metrics = requested_metrics if focus_metrics else list(numeric_values)
    requested_dimensions = [
        column
        for column in (group_by or [])
        if column in columns and column not in metrics
    ]
    dimensions = (
        requested_dimensions
        if group_by
        else [column for column in columns if column not in numeric_values]
    )
    temporal_dimension = next(
        (
            column
            for column in dimensions
            if _temporal_values([row.get(column) for row in normalized_rows])
        ),
        None,
    )

    facts: list[dict[str, Any]] = []

    def add_fact(fact_type: str, summary: str, values: dict[str, Any]) -> None:
        if len(facts) >= max_facts:
            return
        facts.append(
            {
                "fact_id": f"f{len(facts) + 1}",
                "type": fact_type,
                "summary": summary,
                "values": values,
            }
        )

    add_fact(
        "dataset",
        f"当前结果包含 {len(normalized_rows)} 行、{len(columns)} 列。",
        {
            "row_count": len(normalized_rows),
            "column_count": len(columns),
            "truncated": bool(query_result.get("truncated")),
        },
    )

    allowed_types = _fact_types_for_modes(modes)
    for metric in metrics:
        points = numeric_values[metric]
        if not points:
            continue
        numbers = [number for _, number in points]
        if "distribution" in allowed_types:
            if len(numbers) == 1:
                add_fact(
                    "distribution",
                    f"{metric} 的当前结果值为 {_format_number(numbers[0])}。",
                    {
                        "metric": metric,
                        "count": 1,
                        "value": _format_number(numbers[0]),
                    },
                )
            else:
                mean = sum(numbers) / Decimal(len(numbers))
                med = Decimal(str(median(numbers)))
                add_fact(
                    "distribution",
                    f"{metric} 的均值为 {_format_number(mean)}，中位数为 "
                    f"{_format_number(med)}，范围为 {_format_number(min(numbers))} 至 "
                    f"{_format_number(max(numbers))}。",
                    {
                        "metric": metric,
                        "count": len(numbers),
                        "mean": _format_number(mean),
                        "median": _format_number(med),
                        "min": _format_number(min(numbers)),
                        "max": _format_number(max(numbers)),
                    },
                )

        if len(points) >= 2 and "comparison" in allowed_types:
            minimum_index, minimum = min(points, key=lambda item: item[1])
            maximum_index, maximum = max(points, key=lambda item: item[1])
            minimum_label = _row_label(
                normalized_rows[minimum_index], dimensions, minimum_index
            )
            maximum_label = _row_label(
                normalized_rows[maximum_index], dimensions, maximum_index
            )
            add_fact(
                "comparison",
                f"{metric} 最高点是 {maximum_label}（{_format_number(maximum)}），"
                f"最低点是 {minimum_label}（{_format_number(minimum)}）。",
                {
                    "metric": metric,
                    "maximum_label": maximum_label,
                    "maximum": _format_number(maximum),
                    "minimum_label": minimum_label,
                    "minimum": _format_number(minimum),
                },
            )

        if "contribution" in allowed_types and len(points) >= 2:
            total = sum(numbers)
            if total > 0 and all(number >= 0 for number in numbers):
                maximum_index, maximum = max(points, key=lambda item: item[1])
                label = _row_label(
                    normalized_rows[maximum_index], dimensions, maximum_index
                )
                share = maximum / total * Decimal(100)
                add_fact(
                    "contribution",
                    f"{label} 对 {metric} 的占比最高，为 {_format_number(share)}%。",
                    {
                        "metric": metric,
                        "label": label,
                        "value": _format_number(maximum),
                        "total": _format_number(total),
                        "share_percent": _format_number(share),
                    },
                )

        if temporal_dimension and "trend" in allowed_types:
            ordered = _ordered_temporal_points(
                normalized_rows, points, temporal_dimension
            )
            if len(ordered) >= 2:
                first_label, first_value = ordered[0]
                last_label, last_value = ordered[-1]
                change = last_value - first_value
                change_percent = (
                    change / abs(first_value) * Decimal(100)
                    if first_value != 0
                    else None
                )
                percent_summary = (
                    f"，变化率为 {_format_number(change_percent)}%"
                    if change_percent is not None
                    else "，起始值为 0，无法计算变化率"
                )
                add_fact(
                    "trend",
                    f"{metric} 从 {first_label} 的 {_format_number(first_value)} 变为 "
                    f"{last_label} 的 {_format_number(last_value)}，变化量为 "
                    f"{_format_number(change)}{percent_summary}。",
                    {
                        "metric": metric,
                        "first_label": first_label,
                        "first_value": _format_number(first_value),
                        "last_label": last_label,
                        "last_value": _format_number(last_value),
                        "change": _format_number(change),
                        "change_percent": (
                            _format_number(change_percent)
                            if change_percent is not None
                            else None
                        ),
                    },
                )

        if "anomaly" in allowed_types and len(points) >= 8:
            med = Decimal(str(median(numbers)))
            deviations = [abs(number - med) for number in numbers]
            mad = Decimal(str(median(deviations)))
            if mad > 0:
                candidates = []
                for index, number in points:
                    score = Decimal("0.6745") * abs(number - med) / mad
                    if score > Decimal("3.5"):
                        candidates.append((score, index, number))
                if candidates:
                    score, index, number = max(candidates)
                    label = _row_label(normalized_rows[index], dimensions, index)
                    add_fact(
                        "anomaly",
                        f"{metric} 在 {label} 出现稳健异常点：值为 "
                        f"{_format_number(number)}，MAD 得分为 {_format_number(score)}。",
                        {
                            "metric": metric,
                            "label": label,
                            "value": _format_number(number),
                            "mad_score": _format_number(score),
                            "threshold": "3.5",
                        },
                    )

    profile = {
        "columns": columns,
        "metrics": metrics,
        "dimensions": dimensions,
        "temporal_dimension": temporal_dimension,
        "input_row_count": len(raw_rows) if isinstance(raw_rows, list) else 0,
        "used_row_count": len(normalized_rows),
    }
    return facts, profile


def _validated_report(
    answer: str,
    *,
    facts: list[dict[str, Any]],
    max_findings: int,
    fallback_title: str,
) -> dict[str, Any]:
    payload = _decode_json_object(answer)
    fact_by_id = {str(fact["fact_id"]): fact for fact in facts}
    findings = []
    if isinstance(payload, dict):
        raw_findings = payload.get("findings")
        if isinstance(raw_findings, list):
            for item in raw_findings[:max_findings]:
                if not isinstance(item, dict):
                    continue
                evidence_ids = [
                    value
                    for value in _string_list(item.get("evidence_fact_ids"), limit=6)
                    if value in fact_by_id
                ]
                statement = str(item.get("statement") or "").strip()[:500]
                if not evidence_ids or not statement:
                    continue
                finding_type = str(item.get("type") or "distribution").strip()
                if finding_type not in {
                    "trend",
                    "comparison",
                    "distribution",
                    "contribution",
                    "anomaly",
                }:
                    finding_type = "distribution"
                confidence = str(item.get("confidence") or "medium").strip()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "medium"
                findings.append(
                    {
                        "type": finding_type,
                        "statement": "；".join(
                            fact_by_id[value]["summary"] for value in evidence_ids
                        ),
                        "evidence_fact_ids": evidence_ids,
                        "evidence": [fact_by_id[value] for value in evidence_ids],
                        "confidence": confidence,
                    }
                )

    if not findings:
        for fact in facts:
            if fact["type"] == "dataset":
                continue
            findings.append(
                {
                    "type": fact["type"],
                    "statement": fact["summary"],
                    "evidence_fact_ids": [fact["fact_id"]],
                    "evidence": [fact],
                    "confidence": "high",
                }
            )
            if len(findings) >= max_findings:
                break

    title = fallback_title
    summary = "；".join(item["statement"] for item in findings[:3])
    if not summary:
        summary = "当前结果没有可输出的统计洞察。"
    caveats: list[str] = []
    followups: list[str] = []
    if isinstance(payload, dict):
        title = str(payload.get("title") or title).strip()[:120] or title
        caveats = _string_list(payload.get("caveats"), limit=8, item_limit=300)
        followups = _string_list(
            payload.get("suggested_followups"), limit=3, item_limit=300
        )
    return {
        "title": title,
        "executive_summary": summary,
        "findings": findings,
        "caveats": caveats,
        "suggested_followups": followups,
    }


def _decode_json_object(answer: str) -> dict[str, Any] | None:
    rendered = str(answer or "").strip()
    if rendered.startswith("```"):
        rendered = re.sub(r"^```(?:json)?\s*", "", rendered, flags=re.IGNORECASE)
        rendered = re.sub(r"\s*```$", "", rendered)
    start = rendered.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(rendered[start:])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _normalize_row(row: Any, columns: list[str]) -> dict[str, Any]:
    if isinstance(row, dict):
        return {column: row.get(column) for column in columns}
    if isinstance(row, (list, tuple)):
        return {
            column: row[index] if index < len(row) else None
            for index, column in enumerate(columns)
        }
    return {column: None for column in columns}


def _as_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    rendered = str(value).strip().replace(",", "")
    if not rendered:
        return None
    try:
        number = Decimal(rendered)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _temporal_values(values: list[Any]) -> bool:
    rendered = [str(value).strip() for value in values if str(value or "").strip()]
    return bool(rendered) and all(
        _parse_temporal(value) is not None for value in rendered
    )


def _parse_temporal(value: Any) -> datetime | None:
    rendered = str(value or "").strip()
    if not rendered or not any(
        pattern.match(rendered) for pattern in _TEMPORAL_PATTERNS
    ):
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}", rendered):
            year, month = rendered.split("-")
            return datetime(int(year), int(month), 1, tzinfo=UTC)
        parsed = datetime.fromisoformat(rendered.replace(" ", "T"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _ordered_temporal_points(
    rows: list[dict[str, Any]],
    points: list[tuple[int, Decimal]],
    temporal_dimension: str,
) -> list[tuple[str, Decimal]]:
    ordered = []
    seen = set()
    for index, number in points:
        label = str(rows[index].get(temporal_dimension) or "").strip()
        parsed = _parse_temporal(label)
        if parsed is None or parsed in seen:
            return []
        seen.add(parsed)
        ordered.append((parsed, label, number))
    ordered.sort(key=lambda item: item[0])
    return [(label, number) for _, label, number in ordered]


def _row_label(row: dict[str, Any], dimensions: list[str], index: int) -> str:
    parts = [
        f"{column}={str(row.get(column) or '').strip()}"
        for column in dimensions
        if str(row.get(column) or "").strip()
    ]
    return "、".join(parts[:3]) or f"第 {index + 1} 行"


def _fact_types_for_modes(modes: list[str] | None) -> set[str]:
    mapping = {
        "trend": {"trend", "comparison"},
        "comparison": {"comparison", "distribution"},
        "distribution": {"distribution", "comparison"},
        "contribution": {"contribution", "comparison"},
        "anomaly": {"anomaly", "distribution"},
    }
    requested = _analysis_modes(modes)
    if "auto" in requested:
        return {"trend", "comparison", "distribution", "contribution", "anomaly"}
    return set().union(*(mapping[mode] for mode in requested))


def _analysis_modes(value: Any, *, default_mode: Any = None) -> list[str]:
    allowed = {
        "auto",
        "trend",
        "comparison",
        "distribution",
        "contribution",
        "anomaly",
    }
    raw_modes = value if isinstance(value, list) else [default_mode or "auto"]
    modes = [str(mode).strip() for mode in raw_modes if str(mode).strip() in allowed]
    return list(dict.fromkeys(modes)) or ["auto"]


def _format_number(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + 8)
        quantized = (
            value.quantize(Decimal("0.0001")) if value != value.to_integral() else value
        )
    rendered = format(quantized, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _string_list(
    value: Any,
    *,
    limit: int,
    item_limit: int = 128,
) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        rendered
        for item in value[:limit]
        if (rendered := str(item or "").strip()[:item_limit])
    ]
