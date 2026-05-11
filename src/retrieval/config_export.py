"""
配置导出工具 — 从 MySQL 导出 Agent 配置到本地 JSON 文件。

导出内容包括 sys_config 全局参数、Agent 分段配置、Provider 资源信息。
导出的文件可用于 CONFIG_SOURCE=local 模式启动，无需查询配置相关表。

注意：语义层数据（表/列/术语/枚举/fewshot）不在导出范围内，仍从 MySQL 加载。

使用示例::

    # 导出 Agent 1 的配置到默认路径
    python -m src.retrieval.config_export --agent-id 1

    # 导出到指定路径
    python -m src.retrieval.config_export --agent-id 1 --output config/agent_config.json

    # 使用导出的配置启动
    CONFIG_SOURCE=local CONFIG_PROFILE=config/agent_config.json uvicorn app:app --port 9090
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text

from src.retrieval.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "config/agent_config.json"


def export_config(agent_id: int, output_path: str):
    """从 MySQL 导出 Agent 配置到 JSON 文件。"""
    conn_str = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    engine = create_engine(conn_str)

    result = {
        "export_version": "1.0",
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "agent": {},
        "agent_config": {},
        "sys_config": {},
        "providers": [],
    }

    with engine.connect() as conn:
        # 1. Agent 基础信息
        row = conn.execute(text(
            "SELECT id, name, handle, token FROM da_agent WHERE id = :id"
        ), {"id": agent_id}).fetchone()
        if not row:
            logger.error(f"Agent {agent_id} 不存在")
            return
        result["agent"] = {
            "id": row[0],
            "name": row[1],
            "handle": row[2],
            "token": row[3] or "",
        }
        logger.info(f"Agent: id={row[0]}, name={row[1]}")

        # 2. Agent 分段配置
        rows = conn.execute(text(
            "SELECT section, config_json FROM da_agent_config WHERE agent_id = :id"
        ), {"id": agent_id}).fetchall()
        for row in rows:
            try:
                result["agent_config"][row[0]] = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                result["agent_config"][row[0]] = {}
        logger.info(f"Agent 配置段: {list(result['agent_config'].keys())}")

        # 3. sys_config 全局参数
        rows = conn.execute(text(
            "SELECT config_key, config_value FROM sys_config"
        )).fetchall()
        for row in rows:
            key, value = row[0], row[1]
            # 尝试解析 JSON 值
            try:
                result["sys_config"][key] = json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                result["sys_config"][key] = value
        logger.info(f"sys_config: {len(result['sys_config'])} 项")

        # 4. Provider 资源（Agent 引用的 + 全局活跃的）
        # 先查 Agent 引用的 provider
        ref_rows = conn.execute(text(
            "SELECT resource_key FROM da_agent_ref "
            "WHERE agent_id = :id AND resource_type = 'provider'"
        ), {"id": agent_id}).fetchall()
        ref_keys = [r[0] for r in ref_rows]

        # 查所有活跃 provider
        prov_rows = conn.execute(text(
            "SELECT name, display_name, config_json FROM sys_resource "
            "WHERE resource_type = 'provider' AND status = 1"
        )).fetchall()
        for row in prov_rows:
            try:
                cfg = json.loads(row[2]) if row[2] else {}
            except (json.JSONDecodeError, TypeError):
                cfg = {}
            result["providers"].append({
                "name": row[0],
                "display_name": row[1],
                "base_url": cfg.get("base_url", ""),
                "api_key": cfg.get("api_key", ""),
                "referenced": row[0] in ref_keys,
            })
        logger.info(f"Providers: {len(result['providers'])} 个")

    engine.dispose()

    # 写文件
    out = Path(output_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"配置已导出到: {out}")
    logger.info(f"启动命令: CONFIG_SOURCE=local CONFIG_PROFILE={output_path} uvicorn app:app --port 9090")


def main():
    parser = argparse.ArgumentParser(description="从 MySQL 导出 Agent 配置到本地 JSON 文件")
    parser.add_argument("--agent-id", type=int, required=True, help="Agent ID")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"输出文件路径（默认 {DEFAULT_OUTPUT}）")
    args = parser.parse_args()

    export_config(args.agent_id, args.output)


if __name__ == "__main__":
    main()
