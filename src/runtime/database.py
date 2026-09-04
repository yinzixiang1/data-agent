"""Agent-scoped metadata access and execution database construction."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.config import (
    CONFIG_SOURCE,
    DORIS_HOST,
    DORIS_PASSWORD,
    DORIS_PORT,
    DORIS_USER,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD_URL,
    MYSQL_PORT,
    MYSQL_USER,
)
from src.runtime.sql_dialect import SQLDialectAdapter, get_sql_dialect

logger = logging.getLogger(__name__)


class AgentDatasourceNotConfiguredError(RuntimeError):
    """Agent has no usable execution database binding."""


@dataclass(slots=True)
class DatabaseRuntime:
    engine: Engine
    dialect: SQLDialectAdapter
    config: dict[str, Any]
    resource_id: int | None = None
    resource_name: str = ""

    def dispose(self) -> None:
        self.engine.dispose()


def create_metadata_engine() -> Engine:
    """Create a short-lived engine for the Lumen metadata database."""
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def _decrypt_secret(value: str) -> str:
    if not value:
        return ""
    prefix = "fernet:v1:"
    if not value.startswith(prefix):
        raise RuntimeError("数据库凭据不是受支持的密文格式")
    root_key = os.getenv("INTEGRATION_ENCRYPTION_KEY", "").strip()
    if not root_key:
        raise RuntimeError("未配置 INTEGRATION_ENCRYPTION_KEY，无法读取数据库凭据")
    try:
        from cryptography.fernet import Fernet

        return (
            Fernet(root_key.encode("ascii"))
            .decrypt(value.removeprefix(prefix).encode("ascii"))
            .decode("utf-8")
        )
    except Exception as exc:
        raise RuntimeError("数据库凭据无法解密，请检查根密钥") from exc


def _runtime_config(stored: dict[str, Any]) -> dict[str, Any]:
    config = dict(stored)
    auth_mode = str(config.get("auth_mode") or "password")
    if auth_mode == "password":
        config["password"] = (
            _decrypt_secret(str(config.get("password_ciphertext") or ""))
            if config.get("password_ciphertext")
            else str(config.get("password") or "")
        )
    elif str(config.get("credential_source") or "default_chain") == "static":
        config["access_key_id"] = _decrypt_secret(
            str(config.get("access_key_id_ciphertext") or "")
        )
        config["secret_access_key"] = _decrypt_secret(
            str(config.get("secret_access_key_ciphertext") or "")
        )
        config["session_token"] = _decrypt_secret(
            str(config.get("session_token_ciphertext") or "")
        )
    return config


def _local_execution_config() -> dict[str, Any]:
    return {
        "db_type": "doris",
        "host": DORIS_HOST,
        "port": int(DORIS_PORT),
        "user": DORIS_USER,
        "password": DORIS_PASSWORD,
        "auth_mode": "password",
        "source": "config.yaml (local)",
    }


def load_execution_config(agent_id: int | None) -> dict[str, Any] | None:
    """Load the single execution resource bound to an Agent."""
    if CONFIG_SOURCE == "local":
        return _local_execution_config()
    if not agent_id:
        return None

    engine = create_metadata_engine()
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT DISTINCT r.id, r.name, r.resource_type, r.config_json "
                    "FROM da_agent_exec_db b "
                    "JOIN sys_resource r ON r.id = b.resource_id "
                    "WHERE b.agent_id = :agent_id AND b.status = 1 AND r.status = 1 "
                    "ORDER BY r.id"
                ),
                {"agent_id": agent_id},
            ).fetchall()
    finally:
        engine.dispose()
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeError("一个 Agent 只能绑定一个执行数据库资源")

    resource_id, resource_name, resource_type, raw_config = rows[0]
    if resource_type != "database":
        raise RuntimeError(f"资源 {resource_name}(id={resource_id}) 不是数据库资源")
    try:
        stored = json.loads(raw_config) if raw_config else {}
        config = _runtime_config(stored)
        config["db_type"] = str(config.get("db_type") or "doris").lower()
        config["host"] = str(config["host"]).strip()
        config["port"] = int(config["port"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"数据库资源 {resource_name}(id={resource_id}) 配置不完整"
        ) from exc
    get_sql_dialect(config["db_type"])
    if config["db_type"] == "redshift" and not str(config.get("database") or ""):
        raise RuntimeError(
            f"数据库资源 {resource_name}(id={resource_id}) 缺少 database"
        )
    config.update(
        {
            "resource_id": int(resource_id),
            "resource_name": str(resource_name),
            "source": f"全局资源:{resource_name}(id={resource_id})",
        }
    )
    logger.info(
        "loaded Agent execution resource",
        extra={
            "agent_id": agent_id,
            "resource_id": resource_id,
            "resource_name": resource_name,
            "db_type": config["db_type"],
            "host": config["host"],
            "port": config["port"],
        },
    )
    return config


def _redshift_connect_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "host": config["host"],
        "port": int(config.get("port") or 5439),
        "database": config["database"],
        "user": config.get("user") or None,
        "ssl": bool(config.get("ssl", True)),
        "timeout": 10,
        "application_name": "lumen-nl2sql-agent",
    }
    if config.get("auth_mode") == "password":
        kwargs["password"] = config.get("password") or ""
        return kwargs
    kwargs.update(
        {
            "iam": True,
            "region": config.get("region"),
            "is_serverless": config.get("cluster_type") == "serverless",
        }
    )
    if config.get("cluster_type") == "serverless":
        kwargs["serverless_acct_id"] = config.get("serverless_acct_id")
        kwargs["serverless_work_group"] = config.get("serverless_work_group")
    else:
        kwargs["cluster_identifier"] = config.get("cluster_identifier")
    credential_source = config.get("credential_source") or "default_chain"
    if credential_source == "profile":
        kwargs["profile"] = config.get("profile")
    elif credential_source == "static":
        kwargs["access_key_id"] = config.get("access_key_id")
        kwargs["secret_access_key"] = config.get("secret_access_key")
        if config.get("session_token"):
            kwargs["session_token"] = config["session_token"]
    return kwargs


def _create_execution_engine(config: dict[str, Any]) -> Engine:
    if config["db_type"] == "redshift":
        import redshift_connector

        kwargs = _redshift_connect_kwargs(config)
        return create_engine(
            "redshift+redshift_connector://",
            creator=lambda: redshift_connector.connect(**kwargs),
            pool_size=2,
            pool_recycle=1800,
            pool_pre_ping=True,
        )
    password_url = quote_plus(str(config.get("password") or ""))
    url = (
        f"mysql+pymysql://{config.get('user') or 'root'}:{password_url}"
        f"@{config['host']}:{config['port']}/information_schema?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600, pool_pre_ping=True)


def create_database_runtime(agent_id: int | None = None) -> DatabaseRuntime:
    """Create the configured execution engine and its SQL dialect adapter."""
    config = load_execution_config(agent_id)
    if config is None:
        raise AgentDatasourceNotConfiguredError(f"Agent {agent_id} 尚未绑定执行数据库")
    return DatabaseRuntime(
        engine=_create_execution_engine(config),
        dialect=get_sql_dialect(config["db_type"]),
        config=config,
        resource_id=config.get("resource_id"),
        resource_name=str(config.get("resource_name") or ""),
    )


def load_agent_databases(
    agent_id: int | None, metadata_filter: dict | None = None
) -> set[str]:
    """Load SQL namespaces an Agent may query for its configured dialect."""
    if not agent_id:
        raise RuntimeError("执行 SQL 必须绑定 Agent")
    try:
        engine = create_metadata_engine()
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT b.database_name, b.schema_name, b.meta_json, r.config_json "
                    "FROM da_agent_exec_db b "
                    "JOIN sys_resource r ON r.id = b.resource_id "
                    "WHERE b.agent_id = :agent_id AND b.status = 1 AND r.status = 1"
                ),
                {"agent_id": agent_id},
            ).fetchall()
        engine.dispose()
        result: set[str] = set()
        for database_name, schema_name, raw_meta, raw_resource in rows:
            try:
                metadata = json.loads(raw_meta) if raw_meta else {}
                resource = json.loads(raw_resource) if raw_resource else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if metadata_filter and not all(
                key not in metadata or metadata[key] == value
                for key, value in metadata_filter.items()
            ):
                continue
            db_type = str(resource.get("db_type") or "doris").lower()
            namespace = schema_name if db_type == "redshift" else database_name
            if namespace:
                result.add(str(namespace).lower())
        logger.info(
            "loaded Agent database grants",
            extra={
                "agent_id": agent_id,
                "namespace_count": len(result),
                "has_metadata_filter": bool(metadata_filter),
            },
        )
        return result
    except (OSError, SQLAlchemyError, ValueError) as exc:
        logger.error(
            "failed to load Agent database grants",
            extra={"agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise RuntimeError("无法确认 Agent 的数据库授权范围") from exc
