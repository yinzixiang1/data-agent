from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.deployment_preflight import (
    DeploymentPreflightError,
    run_candidate_preflight,
)


def _production_environment(**overrides: str) -> dict[str, str]:
    environ = {
        "NL2SQL_ENV": "prod",
        "CONFIG_SOURCE": "mysql",
        "CONFIG_PROFILE": "1",
        "MYSQL_HOST": "db.internal",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "agent",
        "MYSQL_PASSWORD": "do-not-print-this",
        "MYSQL_DATABASE": "uqpay_infra_lumenv10",
    }
    environ.update(overrides)
    return environ


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one(self) -> Any:
        return self.value

    def mappings(self) -> "_FakeResult":
        return self

    def one_or_none(self) -> Any:
        return self.value


class _FakeConnection(AbstractContextManager["_FakeConnection"]):
    def __init__(self, *, database: str, agent_row: dict[str, Any] | None) -> None:
        self.database = database
        self.agent_row = agent_row
        self.execute_count = 0

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, *_args: object, **_kwargs: object) -> _FakeResult:
        self.execute_count += 1
        if self.execute_count == 1:
            return _FakeResult(self.database)
        return _FakeResult(self.agent_row)


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection_instance = connection
        self.disposed = False

    def connect(self) -> _FakeConnection:
        return self.connection_instance

    def dispose(self) -> None:
        self.disposed = True


class _FakeLoader:
    def __init__(self, runtime_config: AgentRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.engine = SimpleNamespace(dispose=lambda: None)

    def load(self, agent_id: int | None = None) -> AgentRuntimeConfig:
        assert agent_id == 1
        return self.runtime_config


def _runtime_config() -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        agent_id=1,
        agent_name="BI Agent",
        token="agent-token",
        llm_model="gpt-5.6-terra",
    )


def test_candidate_preflight_checks_database_and_agent() -> None:
    engine = _FakeEngine(
        _FakeConnection(
            database="uqpay_infra_lumenv10",
            agent_row={"id": 1, "status": 1},
        )
    )

    database, profile_id = run_candidate_preflight(
        _production_environment(),
        engine_factory=lambda *_args, **_kwargs: engine,
        loader_factory=lambda: _FakeLoader(_runtime_config()),
    )

    assert database == "uqpay_infra_lumenv10"
    assert profile_id == 1
    assert engine.disposed is True


def test_candidate_preflight_rejects_database_mismatch_before_connecting() -> None:
    with pytest.raises(DeploymentPreflightError, match="does not match"):
        run_candidate_preflight(
            _production_environment(MYSQL_DATABASE="wrong_database"),
            engine_factory=lambda *_args, **_kwargs: pytest.fail(
                "engine must not be created"
            ),
        )


def test_candidate_preflight_rejects_disabled_agent() -> None:
    engine = _FakeEngine(
        _FakeConnection(
            database="uqpay_infra_lumenv10",
            agent_row={"id": 1, "status": 0},
        )
    )

    with pytest.raises(DeploymentPreflightError, match="is not enabled"):
        run_candidate_preflight(
            _production_environment(),
            engine_factory=lambda *_args, **_kwargs: engine,
            loader_factory=lambda: _FakeLoader(_runtime_config()),
        )


def test_candidate_preflight_does_not_leak_database_error_details() -> None:
    class _BrokenEngine(_FakeEngine):
        def connect(self) -> _FakeConnection:
            raise OperationalError(
                "connect",
                {},
                RuntimeError("password=do-not-print-this"),
            )

    engine = _BrokenEngine(
        _FakeConnection(database="uqpay_infra_lumenv10", agent_row=None)
    )

    with pytest.raises(DeploymentPreflightError) as raised:
        run_candidate_preflight(
            _production_environment(),
            engine_factory=lambda *_args, **_kwargs: engine,
            loader_factory=lambda: _FakeLoader(_runtime_config()),
        )

    assert "do-not-print-this" not in str(raised.value)
    assert engine.disposed is True


@pytest.mark.parametrize(
    "runtime_config, expected_message",
    (
        (AgentRuntimeConfig(agent_id=None), "could not be loaded"),
        (AgentRuntimeConfig(agent_id=1, token="token"), "display name"),
        (AgentRuntimeConfig(agent_id=1, agent_name="Agent"), "request token"),
    ),
)
def test_candidate_preflight_rejects_unusable_runtime_config(
    runtime_config: AgentRuntimeConfig, expected_message: str
) -> None:
    engine = _FakeEngine(
        _FakeConnection(
            database="uqpay_infra_lumenv10",
            agent_row={"id": 1, "status": 1},
        )
    )

    with pytest.raises(DeploymentPreflightError, match=expected_message):
        run_candidate_preflight(
            _production_environment(),
            engine_factory=lambda *_args, **_kwargs: engine,
            loader_factory=lambda: _FakeLoader(runtime_config),
        )
