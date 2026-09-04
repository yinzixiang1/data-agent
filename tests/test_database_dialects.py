"""Execution database adapters must preserve dialect and authorization contracts."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from src.retrieval.context_compressor import ContextCompressor
from src.retrieval.schema_formatter import SchemaFormatter
from src.retrieval.sql_validator import SQLValidator
from src.runtime.database import _redshift_connect_kwargs, _runtime_config
from src.runtime.sql_dialect import RedshiftDialectAdapter, get_sql_dialect


@pytest.fixture(autouse=True)
def restore_default_dialect():
    yield
    SQLValidator._DEFAULT_SQLGLOT_DIALECT = "mysql"
    ContextCompressor.configure_sql_dialect("mysql")


class _Result:
    def __init__(self, rows=(), columns=()):
        self._rows = list(rows)
        self._columns = list(columns)

    def fetchall(self):
        return self._rows

    def keys(self):
        return self._columns


class _Connection:
    def __init__(self):
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, statement, _params=None):
        rendered = str(statement)
        self.statements.append(rendered)
        if rendered.startswith("SET statement_timeout"):
            return _Result()
        return _Result(rows=[(1,)], columns=["count"])


class _Engine:
    def __init__(self):
        self.connection = _Connection()

    def connect(self):
        return self.connection

    def dispose(self):
        return None


def test_redshift_password_and_iam_connection_arguments() -> None:
    password = _redshift_connect_kwargs(
        {
            "host": "example.redshift.amazonaws.com",
            "port": 5439,
            "database": "dev",
            "user": "analyst",
            "password": "secret",
            "auth_mode": "password",
        }
    )
    assert password["user"] == "analyst"
    assert password["password"] == "secret"
    assert "iam" not in password

    iam = _redshift_connect_kwargs(
        {
            "host": "example.redshift-serverless.amazonaws.com",
            "port": 5439,
            "database": "dev",
            "user": "analyst",
            "auth_mode": "iam",
            "region": "ap-southeast-1",
            "cluster_type": "serverless",
            "serverless_acct_id": "123456789012",
            "serverless_work_group": "analytics",
            "credential_source": "static",
            "access_key_id": "access",
            "secret_access_key": "secret",
            "session_token": "token",
        }
    )
    assert iam["iam"] is True
    assert iam["is_serverless"] is True
    assert iam["serverless_work_group"] == "analytics"
    assert iam["access_key_id"] == "access"
    assert iam["session_token"] == "token"


def test_runtime_decrypts_database_credentials(monkeypatch) -> None:
    key = Fernet.generate_key()
    cipher = Fernet(key)
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", key.decode("ascii"))
    runtime = _runtime_config(
        {
            "auth_mode": "password",
            "password_ciphertext": "fernet:v1:"
            + cipher.encrypt(b"secret").decode("ascii"),
        }
    )
    assert runtime["password"] == "secret"


def test_redshift_formatter_and_authorization_use_schema_qualification() -> None:
    dialect = get_sql_dialect("redshift")
    prompt = SchemaFormatter(dialect).format_all(
        tables=[
            {
                "schema": {
                    "database": "analytics",
                    "table_name": "analytics.orders",
                    "table_name_short": "orders",
                    "columns": [{"name": "created_at", "type": "timestamp"}],
                }
            }
        ],
        examples=[],
        business_context="",
        question="按月统计订单",
    )
    assert '【表】"analytics"."orders"' in prompt
    assert "Amazon Redshift" in prompt
    assert "不得使用 Doris/MySQL 的反引号" in prompt

    SQLValidator(_Engine(), dialect)
    sql = 'SELECT COUNT(*) FROM "analytics"."orders"'
    allowed, reason, namespaces = SQLValidator.validate_database_access(
        sql, {"analytics"}
    )
    assert allowed is True, reason
    assert namespaces == {"analytics"}


def test_redshift_execution_and_followup_context_preserve_dialect() -> None:
    engine = _Engine()
    validator = SQLValidator(engine, RedshiftDialectAdapter())
    sql = (
        'SELECT DATE_TRUNC(\'day\', "created_at") AS "day", COUNT(*) AS "count" '
        'FROM "analytics"."orders" '
        'WHERE "created_at" >= DATEADD(day, -6, CURRENT_DATE) '
        'AND "created_at" < DATEADD(day, 1, CURRENT_DATE) '
        "GROUP BY DATE_TRUNC('day', \"created_at\")"
    )
    valid, reason, _ = validator.validate_calendar_day_window(
        sql, {"state": {"calendar_day_window": 7}}
    )
    assert valid is True, reason

    result = validator.execute(sql, row_limit=20, timeout=12)
    assert result["success"] is True
    assert engine.connection.statements[0] == "SET statement_timeout TO 12000"
    assert "SET_VAR" not in engine.connection.statements[1]
    assert engine.connection.statements[1].endswith("LIMIT 21")

    context = ContextCompressor.extract_sql_context(sql)
    assert context["tables"] == ["analytics.orders"]
    assert any("DATE_TRUNC" in item.upper() for item in context["projections"])
