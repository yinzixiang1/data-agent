"""Agent-scoped MySQL metadata access and Doris engine construction."""

import json
import logging
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.config import (
    CONFIG_SOURCE,
    DORIS_HOST,
    DORIS_PASSWORD,
    DORIS_PASSWORD_URL,
    DORIS_PORT,
    DORIS_USER,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD_URL,
    MYSQL_PORT,
    MYSQL_USER,
)
from src.retrieval.schema_loader import AgentDatasourceNotConfiguredError

logger = logging.getLogger(__name__)


def create_metadata_engine() -> Engine:
    """Create a short-lived engine for the Lumen metadata database."""
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)


def load_agent_databases(
    agent_id: int | None, metadata_filter: dict | None = None
) -> set[str]:
    """Load the databases an Agent may query, honoring public metadata rows."""
    if not agent_id:
        raise RuntimeError("执行 SQL 必须绑定 Agent")
    try:
        engine = create_metadata_engine()
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT DISTINCT database_name, meta_json FROM da_agent_exec_db "
                    "WHERE agent_id = :agent_id AND status = 1"
                ),
                {"agent_id": agent_id},
            ).fetchall()
        engine.dispose()

        if not metadata_filter:
            result = {row[0].lower() for row in rows if row[0]}
        else:
            result = set()
            for row in rows:
                if not row[0]:
                    continue
                try:
                    metadata = json.loads(row[1]) if row[1] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                if all(
                    key not in metadata or metadata[key] == value
                    for key, value in metadata_filter.items()
                ):
                    result.add(row[0].lower())

        logger.info(
            "loaded Agent database grants",
            extra={
                "agent_id": agent_id,
                "database_count": len(result),
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


def load_doris_config(agent_id: int | None) -> dict[str, str | int] | None:
    """Load the Doris resource bound to an Agent."""
    if CONFIG_SOURCE == "local":
        return {
            "host": DORIS_HOST,
            "port": int(DORIS_PORT),
            "user": DORIS_USER,
            "password": DORIS_PASSWORD,
            "password_url": DORIS_PASSWORD_URL,
            "source": "config.yaml (local)",
        }
    if not agent_id:
        return None

    engine = create_metadata_engine()
    try:
        with engine.connect() as connection:
            datasource = connection.execute(
                text(
                    "SELECT resource_id FROM da_agent_exec_db "
                    "WHERE agent_id = :agent_id AND status = 1 "
                    "ORDER BY sort_order LIMIT 1"
                ),
                {"agent_id": agent_id},
            ).fetchone()
            if not datasource:
                return None

            resource_id = datasource[0]
            resource = connection.execute(
                text(
                    "SELECT name, resource_type, config_json FROM sys_resource "
                    "WHERE id = :resource_id AND status = 1"
                ),
                {"resource_id": resource_id},
            ).fetchone()
    finally:
        engine.dispose()

    if not resource:
        raise RuntimeError(
            f"Agent {agent_id} 绑定的数据库资源 {resource_id} 不存在或未启用"
        )
    resource_name, resource_type, raw_config = resource
    if resource_type != "database":
        raise RuntimeError(f"资源 {resource_name}(id={resource_id}) 不是数据库资源")

    try:
        config = json.loads(raw_config) if raw_config else {}
        host = str(config["host"]).strip()
        port = int(config["port"])
        user = str(config["user"]).strip()
        password = str(config.get("password", ""))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"数据库资源 {resource_name}(id={resource_id}) 配置不完整"
        ) from exc
    if not host or not user:
        raise RuntimeError(f"数据库资源 {resource_name}(id={resource_id}) 配置不完整")

    logger.info(
        "loaded Agent Doris resource",
        extra={
            "agent_id": agent_id,
            "resource_id": resource_id,
            "resource_name": resource_name,
            "host": host,
            "port": port,
        },
    )
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "password_url": quote_plus(password),
        "source": f"全局资源:{resource_name}(id={resource_id})",
    }


def create_doris_engine(agent_id: int | None = None) -> Engine:
    """Create a Doris engine from the database resource bound to an Agent."""
    config = load_doris_config(agent_id)
    if config is None:
        raise AgentDatasourceNotConfiguredError(f"Agent {agent_id} 尚未绑定执行数据库")
    url = (
        f"mysql+pymysql://{config['user']}:{config['password_url']}"
        f"@{config['host']}:{config['port']}/information_schema?charset=utf8mb4"
    )
    return create_engine(url, pool_size=2, pool_recycle=3600)
