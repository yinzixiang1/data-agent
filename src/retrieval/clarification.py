"""Schema-grounded clarification construction."""


def table_references(relevant_tables: list[dict]) -> list[dict[str, str]]:
    """Build verified table labels from retrieval results for clarification cards."""
    references = []
    seen_table_names = set()
    for table in relevant_tables:
        if not isinstance(table, dict):
            continue
        schema = table.get("schema")
        schema = schema if isinstance(schema, dict) else {}
        table_name = str(
            table.get("table_name") or schema.get("table_name") or ""
        ).strip()
        if not table_name or table_name in seen_table_names:
            continue
        seen_table_names.add(table_name)
        description = str(
            schema.get("display_name")
            or schema.get("description")
            or schema.get("table_comment")
            or ""
        ).strip()
        references.append(
            {
                "name": table_name[:200],
                "description": description[:200],
            }
        )
        if len(references) == 6:
            break
    return references


def schema_grounding_clarification(
    effective_question: str,
    validation_detail: dict,
) -> dict[str, object]:
    """Turn exhausted Schema-grounding failures into an actionable question."""
    unsupported_references = [
        str(value)
        for value in (
            validation_detail.get("invalid_columns")
            or validation_detail.get("invalid_tables")
            or []
        )
        if str(value).strip()
    ]
    reference_hint = (
        "（尚未获得证据支持：" + "、".join(unsupported_references[:6]) + "）"
        if unsupported_references
        else ""
    )
    question = str(effective_question or "当前查询").strip()[:300]
    return {
        "question": (
            f"当前语义资料不足以把“{question}”完整映射到真实数据字段"
            f"{reference_hint}。请补充缺失条件的业务定义，或指定对应的表、字段及取值。"
        )[:1000],
        "options": [],
    }


def is_schema_grounding_failure(validation_detail: dict) -> bool:
    return validation_detail.get("failure_type") in {
        "table_outside_context",
        "column_outside_context",
    }
