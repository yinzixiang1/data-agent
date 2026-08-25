import pytest
from fastapi import HTTPException
from starlette.requests import Request

import app as service
from src.retrieval.config import StartupConfigurationError, validate_startup_config


def _request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def test_admin_endpoint_uses_a_token_separate_from_query_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "AGENT_ADMIN_TOKEN", "admin-token")

    service._verify_admin_token(_request("admin-token"))

    with pytest.raises(HTTPException) as exc_info:
        service._verify_admin_token(_request("query-token"))

    assert exc_info.value.status_code == 401


def test_production_requires_a_strong_agent_admin_token() -> None:
    config = {
        "NL2SQL_ENV": "prod",
        "CONFIG_SOURCE": "mysql",
        "CONFIG_PROFILE": "1",
        "MYSQL_HOST": "mysql.internal",
        "MYSQL_USER": "agent",
        "MYSQL_PASSWORD": "secret",
        "MYSQL_DATABASE": "lumen",
    }

    with pytest.raises(StartupConfigurationError, match="AGENT_ADMIN_TOKEN"):
        validate_startup_config(config)

    validate_startup_config({**config, "AGENT_ADMIN_TOKEN": "a" * 32})


def test_production_rejects_reused_query_and_admin_tokens() -> None:
    shared_token = "s" * 32
    config = {
        "NL2SQL_ENV": "prod",
        "CONFIG_SOURCE": "mysql",
        "CONFIG_PROFILE": "1",
        "MYSQL_HOST": "mysql.internal",
        "MYSQL_USER": "agent",
        "MYSQL_PASSWORD": "secret",
        "MYSQL_DATABASE": "lumen",
        "DEFAULT_AGENT_TOKEN": shared_token,
        "AGENT_ADMIN_TOKEN": shared_token,
    }

    with pytest.raises(StartupConfigurationError, match="must not reuse"):
        validate_startup_config(config)
