"""Safe builders for Milvus scalar filter expressions."""

import json
import math
import re


_METADATA_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _literal(value: str | int | float | bool) -> str:
    """Serialize a supported Python scalar as a Milvus expression literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metadata filter value 不允许 NaN 或 Infinity")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"不支持的 metadata filter value 类型: {type(value).__name__}")


def build_metadata_filter(
    biz_line: str = "",
    metadata_filter: dict | None = None,
) -> str | None:
    """Build a filter from validated field names and escaped scalar values."""
    parts: list[str] = []
    if biz_line:
        parts.append(
            f'(biz_line == {_literal(biz_line)} or biz_line == "sys" or biz_line == "")'
        )
    for key, value in (metadata_filter or {}).items():
        if not isinstance(key, str) or not _METADATA_KEY_RE.fullmatch(key):
            raise ValueError(f"非法 metadata filter key: {key!r}")
        parts.append(
            f'(metadata["{key}"] == {_literal(value)} or not exists metadata["{key}"])'
        )
    return " and ".join(parts) if parts else None


def add_table_name_filter(
    filter_expr: str | None,
    table_names: set[str],
) -> str | None:
    """Constrain an existing Milvus expression to an internal table-name set."""
    if not table_names:
        return filter_expr
    table_literals = ", ".join(_literal(name) for name in sorted(table_names))
    table_filter = f"table_name in [{table_literals}]"
    if filter_expr:
        return f"({filter_expr}) and ({table_filter})"
    return table_filter
