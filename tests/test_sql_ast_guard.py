import pytest

pytest.importorskip("sqlglot")

from src.retrieval.sql_validator import SQLValidator


SCHEMAS = [
    {
        "table_name": "sales.orders",
        "table_name_short": "orders",
        "columns": [{"name": "order_id"}, {"name": "amount"}],
    }
]


def test_ast_guard_rejects_cross_join():
    valid, reason = SQLValidator.validate_read_only(
        "SELECT * FROM sales.orders CROSS JOIN sales.customers"
    )

    assert valid is False
    assert "CROSS JOIN" in reason


def test_schema_guard_rejects_unretrieved_table():
    valid, reason, _ = SQLValidator.validate_schema_references(
        "SELECT c.customer_id FROM sales.customers c", SCHEMAS
    )

    assert valid is False
    assert "上下文之外的表" in reason


def test_schema_guard_rejects_qualified_unselected_column():
    valid, reason, detail = SQLValidator.validate_schema_references(
        "SELECT o.secret_note FROM sales.orders o", SCHEMAS
    )

    assert valid is False
    assert detail["invalid_columns"] == ["o.secret_note"]
