import pytest

from src.retrieval.config import (
    StartupConfigurationError,
    validate_startup_config,
)


def _production_environment(**overrides: str) -> dict[str, str]:
    environ = {
        "NL2SQL_ENV": "prod",
        "CONFIG_SOURCE": "mysql",
        "CONFIG_PROFILE": "1",
        "MYSQL_HOST": "db.internal",
        "MYSQL_USER": "agent",
        "MYSQL_PASSWORD": "secret",
        "MYSQL_DATABASE": "admin",
    }
    environ.update(overrides)
    return environ


def test_development_config_does_not_require_production_credentials() -> None:
    validate_startup_config({"NL2SQL_ENV": "dev"})


@pytest.mark.parametrize(
    "missing_field",
    ("MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"),
)
def test_production_mysql_config_rejects_missing_fields(
    missing_field: str,
) -> None:
    environ = _production_environment()
    environ[missing_field] = ""

    with pytest.raises(StartupConfigurationError, match=missing_field):
        validate_startup_config(environ)


@pytest.mark.parametrize("profile", ("", "zero", "0", "-1"))
def test_production_mysql_config_requires_positive_agent_id(profile: str) -> None:
    environ = _production_environment(CONFIG_PROFILE=profile)

    with pytest.raises(StartupConfigurationError, match="positive Agent ID"):
        validate_startup_config(environ)


def test_production_local_config_requires_profile() -> None:
    environ = _production_environment(CONFIG_SOURCE="local", CONFIG_PROFILE="")

    with pytest.raises(StartupConfigurationError, match="CONFIG_PROFILE"):
        validate_startup_config(environ)


def test_complete_production_mysql_config_passes() -> None:
    validate_startup_config(_production_environment())
