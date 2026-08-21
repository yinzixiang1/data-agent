"""Deployment preflight checks for a candidate Agent image."""

import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.config import (
    StartupConfigurationError,
    validate_startup_config,
)

EXPECTED_DATABASE_DEFAULT = "uqpay_infra_lumenv10"


class DeploymentPreflightError(RuntimeError):
    """Raised when a candidate image cannot safely replace the live Agent."""


def _agent_profile_id(environ: Mapping[str, str]) -> int:
    raw_profile = (
        environ.get("CONFIG_PROFILE", "").strip()
        or environ.get("DEFAULT_AGENT_ID", "").strip()
    )
    try:
        profile_id = int(raw_profile)
    except (TypeError, ValueError) as exc:
        raise DeploymentPreflightError(
            "Agent CONFIG_PROFILE must be a positive integer"
        ) from exc
    if profile_id <= 0:
        raise DeploymentPreflightError(
            "Agent CONFIG_PROFILE must be a positive integer"
        )
    return profile_id


def _is_enabled(status: object) -> bool:
    if isinstance(status, bool):
        return status
    if isinstance(status, int):
        return status == 1
    return str(status).strip().lower() in {"1", "active", "enabled"}


def _validate_runtime_config(
    runtime_config: AgentRuntimeConfig, profile_id: int
) -> None:
    if runtime_config.agent_id != profile_id:
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} could not be loaded"
        )
    if not runtime_config.agent_name.strip():
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} has no display name"
        )
    if not runtime_config.token.strip():
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} has no request token"
        )
    if not runtime_config.llm_model.strip():
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} has no LLM model"
        )


def run_candidate_preflight(
    environ: Mapping[str, str] | None = None,
    *,
    engine_factory: Callable[..., Any] | None = None,
    loader_factory: Callable[[], AgentConfigLoader] | None = None,
) -> tuple[str, int]:
    """Validate database access and the selected Agent configuration."""
    values = os.environ if environ is None else environ
    validate_startup_config(values)

    configured_database = values.get("MYSQL_DATABASE", "").strip()
    expected_database = values.get(
        "EXPECTED_ADMIN_DATABASE", EXPECTED_DATABASE_DEFAULT
    ).strip()
    if not expected_database:
        raise DeploymentPreflightError("EXPECTED_ADMIN_DATABASE must not be empty")
    if configured_database != expected_database:
        raise DeploymentPreflightError(
            "Configured MySQL database does not match EXPECTED_ADMIN_DATABASE "
            f"(configured={configured_database!r}, expected={expected_database!r})"
        )

    profile_id = _agent_profile_id(values)
    database_url = URL.create(
        "mysql+pymysql",
        username=values["MYSQL_USER"],
        password=values["MYSQL_PASSWORD"],
        host=values["MYSQL_HOST"],
        port=int(values.get("MYSQL_PORT", "3306")),
        database=configured_database,
        query={"charset": "utf8mb4"},
    )
    make_engine = create_engine if engine_factory is None else engine_factory
    engine = make_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    try:
        with engine.connect() as connection:
            active_database = connection.execute(text("SELECT DATABASE()")).scalar_one()
            agent_row = (
                connection.execute(
                    text("SELECT id, status FROM da_agent WHERE id = :agent_id"),
                    {"agent_id": profile_id},
                )
                .mappings()
                .one_or_none()
            )
    except (OSError, SQLAlchemyError, ValueError) as exc:
        raise DeploymentPreflightError(
            "MySQL connection or login validation did not pass"
        ) from exc
    finally:
        engine.dispose()

    if active_database != expected_database:
        raise DeploymentPreflightError(
            "Connected MySQL database does not match EXPECTED_ADMIN_DATABASE"
        )
    if agent_row is None:
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} does not exist"
        )
    if not _is_enabled(agent_row["status"]):
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} is not enabled"
        )

    make_loader = AgentConfigLoader if loader_factory is None else loader_factory
    loader = make_loader()
    try:
        runtime_config = loader.load(agent_id=profile_id)
    except (OSError, SQLAlchemyError, TypeError, ValueError) as exc:
        raise DeploymentPreflightError(
            f"Agent CONFIG_PROFILE {profile_id} could not be loaded"
        ) from exc
    finally:
        if loader.engine is not None:
            loader.engine.dispose()
    _validate_runtime_config(runtime_config, profile_id)
    return expected_database, profile_id


def main() -> int:
    try:
        database, profile_id = run_candidate_preflight()
    except (StartupConfigurationError, DeploymentPreflightError) as exc:
        print(f"Candidate image preflight rejected: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(
            "Candidate image preflight rejected due to an unexpected "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
        return 1

    print(
        "Candidate image preflight passed: "
        f"database={database}, agent_profile={profile_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
