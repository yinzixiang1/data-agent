"""请求参数定义 — 引擎侧唯一来源。

启动时根据 PARAM_SOURCE 环境变量选择模式：
  - local: 本地定义 → 注册进 admin DB（适合首次部署/升级）
  - admin: 从 admin DB 读取，缺的用本地补（适合生产运行）

运行时提供:
  - ALLOWED_KEYS: 允许通过 expand_info 覆盖的参数 key 集合
  - PARAM_DEFAULTS: 各参数的默认值字典
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── 本地定义（始终存在，作为 local 模式来源 + admin 模式兜底）──

REQUEST_PARAMS: list[dict[str, Any]] = [
    # ═══ SQL 执行 ═══
    {
        "key": "enable_execute",
        "type": "boolean",
        "default": False,
        "label": "启用 SQL 执行",
        "hint": "生成 SQL 后自动执行并返回查询结果",
        "group": "SQL 执行",
        "section": "flow",
        "scope": ["nl2sql"],
    },
    {
        "key": "execute_row_limit",
        "type": "int",
        "default": 200,
        "label": "最大返回行数",
        "hint": "SQL 执行结果的最大行数限制",
        "group": "SQL 执行",
        "section": "flow",
        "scope": ["nl2sql"],
        "min": 1,
        "max": 10000,
        "depends_on": {"key": "enable_execute", "value": True},
    },
    {
        "key": "execute_timeout",
        "type": "int",
        "default": 30,
        "label": "执行超时 (秒)",
        "hint": "SQL 执行超时时间，超时后触发降级",
        "group": "SQL 执行",
        "section": "flow",
        "scope": ["nl2sql"],
        "min": 1,
        "max": 300,
        "depends_on": {"key": "enable_execute", "value": True},
    },
    # ═══ 纠错增强 ═══
    {
        "key": "max_execute_fix_retries",
        "type": "int",
        "default": 2,
        "label": "执行纠错次数",
        "hint": "SQL 执行失败后的最大自动修复重试次数",
        "group": "纠错增强",
        "section": "flow",
        "scope": ["nl2sql"],
        "min": 0,
        "max": 5,
    },
    {
        "key": "enable_enum_validate",
        "type": "boolean",
        "default": True,
        "label": "枚举值校验",
        "hint": "执行前检查 SQL 中的枚举值是否合法",
        "group": "纠错增强",
        "section": "flow",
        "scope": ["nl2sql"],
    },
    {
        "key": "enable_timeout_fallback",
        "type": "boolean",
        "default": False,
        "label": "超时降级",
        "hint": "SQL 超时后自动简化查询条件重试（缩短时间范围、去除排序等）",
        "group": "纠错增强",
        "section": "flow",
        "scope": ["nl2sql"],
    },
    # ═══ SQL 生成 ═══
    {
        "key": "max_fix_retries",
        "type": "int",
        "default": 5,
        "label": "SQL 修复重试",
        "hint": "SQL 语法检查失败后自动修复次数",
        "group": "SQL 生成",
        "section": "flow",
        "scope": ["nl2sql"],
        "min": 0,
        "max": 10,
    },
    # ═══ 检索增强 ═══
    {
        "key": "exchange_rate_injection",
        "type": "boolean",
        "default": True,
        "label": "汇率表意图注入",
        "hint": "检测到汇率相关意图时，自动注入汇率表到检索结果",
        "group": "检索增强",
        "section": "flow",
        "scope": ["nl2sql"],
    },
    {
        "key": "value_exact_match_boost",
        "type": "float",
        "default": 2.0,
        "label": "实体值精确匹配加权",
        "hint": "BM25 实体值搜索中精确匹配的加权倍数",
        "group": "检索增强",
        "section": "flow",
        "scope": ["nl2sql"],
        "min": 0,
        "max": 10,
    },
]

LOCAL_MAP: dict[str, dict] = {p["key"]: p for p in REQUEST_PARAMS}

# ── 运行时变量（init_request_params 后可用）──

ALLOWED_KEYS: frozenset[str] = frozenset(p["key"] for p in REQUEST_PARAMS)
PARAM_DEFAULTS: dict[str, Any] = {p["key"]: p["default"] for p in REQUEST_PARAMS}


def init_request_params(mode: str, agent_id: int | None = None):
    """根据启动模式初始化请求参数，更新 ALLOWED_KEYS 和 PARAM_DEFAULTS。

    Args:
        mode: "local" 或 "admin"
        agent_id: Agent ID（用于日志，local 模式下注册时标记）
    """
    global ALLOWED_KEYS, PARAM_DEFAULTS

    if mode == "local":
        _init_local(agent_id)
    elif mode == "admin":
        _init_admin(agent_id)
    else:
        logger.warning("未知 PARAM_SOURCE=%s，使用本地定义", mode)
        ALLOWED_KEYS = frozenset(p["key"] for p in REQUEST_PARAMS)
        PARAM_DEFAULTS = {p["key"]: p["default"] for p in REQUEST_PARAMS}

    logger.info(
        "请求参数初始化完成: mode=%s, allowed_keys=%d",
        mode,
        len(ALLOWED_KEYS),
    )


def _init_local(agent_id: int | None):
    """local 模式：本地定义 → 注册进 admin DB。"""
    global ALLOWED_KEYS, PARAM_DEFAULTS

    # 注册到 DB
    _register_to_db(REQUEST_PARAMS)

    # 用本地定义构建运行时
    ALLOWED_KEYS = frozenset(p["key"] for p in REQUEST_PARAMS)
    PARAM_DEFAULTS = {p["key"]: p["default"] for p in REQUEST_PARAMS}


def _init_admin(agent_id: int | None):
    """admin 模式：从 DB 拉取，缺的用本地补。"""
    global ALLOWED_KEYS, PARAM_DEFAULTS

    admin_params = _fetch_from_db()

    if not admin_params:
        logger.warning("从 DB 拉取请求参数失败或为空，回退到本地定义")
        _init_local(agent_id)
        return

    admin_map = {p["key"]: p for p in admin_params}
    merged = []

    # 本地有的：admin 有用 admin，admin 没有用本地 + 补注册
    missing_in_admin = []
    for key, local in LOCAL_MAP.items():
        if key in admin_map:
            merged.append(admin_map[key])
        else:
            merged.append(local)
            missing_in_admin.append(local)

    # admin 有但本地没有的（手动添加的参数）
    for key, ap in admin_map.items():
        if key not in LOCAL_MAP:
            merged.append(ap)

    # 补注册缺失的参数
    if missing_in_admin:
        _register_to_db(missing_in_admin)
        logger.info("补注册 %d 个本地新增参数到 DB", len(missing_in_admin))

    # 构建运行时（跳过 is_active=0 的）
    active = [p for p in merged if p.get("is_active", 1)]
    ALLOWED_KEYS = frozenset(p["key"] for p in active)
    PARAM_DEFAULTS = {
        p["key"]: p.get("default", p.get("default_value")) for p in active
    }


def _get_db_engine():
    """创建临时 DB 连接引擎。"""
    from sqlalchemy import create_engine
    from src.retrieval.config import (
        MYSQL_USER,
        MYSQL_PASSWORD_URL,
        MYSQL_HOST,
        MYSQL_PORT,
        MYSQL_DATABASE,
    )

    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    return create_engine(url, pool_size=1, pool_recycle=3600)


def _register_to_db(params: list[dict]):
    """全量推送参数定义到 sys_config_param 表（upsert + 软下线）。"""
    from sqlalchemy import text

    eng = _get_db_engine()
    try:
        with eng.connect() as conn:
            incoming_keys = {p["key"] for p in params}

            for i, p in enumerate(params):
                row = conn.execute(
                    text(
                        "SELECT id, source FROM sys_config_param WHERE `key` = :key AND section = :section"
                    ),
                    {"key": p["key"], "section": p.get("section", "flow")},
                ).fetchone()

                if row is None:
                    conn.execute(
                        text("""
                            INSERT INTO sys_config_param
                            (`key`, `type`, default_value, label, hint, group_name,
                             section, level, scope, depends_on, min_val, max_val,
                             options, sort_order, source, is_active)
                            VALUES
                            (:key, :type, :default_value, :label, :hint, :group_name,
                             :section, :level, :scope, :depends_on, :min_val, :max_val,
                             :options, :sort_order, 'engine', 1)
                        """),
                        _param_to_row(p, i),
                    )
                elif row[1] == "engine":
                    # engine 源参数：更新元数据
                    conn.execute(
                        text("""
                            UPDATE sys_config_param SET
                                `type` = :type, default_value = :default_value,
                                label = :label, hint = :hint, group_name = :group_name,
                                scope = :scope, depends_on = :depends_on,
                                min_val = :min_val, max_val = :max_val,
                                options = :options, sort_order = :sort_order,
                                is_active = 1
                            WHERE `key` = :key AND section = :section
                        """),
                        _param_to_row(p, i),
                    )
                # source=manual 的不动

            # 软下线：引擎不再上报的 engine 源参数
            if incoming_keys:
                placeholders = ",".join(f":k{i}" for i in range(len(incoming_keys)))
                params_dict = {f"k{i}": k for i, k in enumerate(incoming_keys)}
                params_dict["section"] = "flow"
                conn.execute(
                    text(f"""
                        UPDATE sys_config_param SET is_active = 0
                        WHERE source = 'engine' AND level = 'request'
                          AND section = :section
                          AND `key` NOT IN ({placeholders})
                    """),
                    params_dict,
                )

            conn.commit()
            logger.info("请求参数注册完成: %d 个参数", len(params))
    except Exception as e:
        logger.warning("请求参数注册失败 (非致命): %s", e)
    finally:
        eng.dispose()


def _fetch_from_db() -> list[dict]:
    """从 sys_config_param 拉取 level=request 的活跃参数。"""
    from sqlalchemy import text

    eng = _get_db_engine()
    try:
        with eng.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT `key`, `type`, default_value, label, hint, group_name,
                           section, scope, depends_on, min_val, max_val, options, is_active
                    FROM sys_config_param
                    WHERE level = 'request'
                    ORDER BY section, sort_order, id
                """)
            ).fetchall()

        result = []
        for row in rows:
            p = {
                "key": row[0],
                "type": row[1],
                "default": _safe_json(row[2]),
                "label": row[3],
                "hint": row[4],
                "group": row[5],
                "section": row[6],
                "scope": _safe_json(row[7], []),
                "is_active": row[12],
            }
            if row[8]:
                p["depends_on"] = _safe_json(row[8])
            if row[9] is not None:
                p["min"] = row[9]
            if row[10] is not None:
                p["max"] = row[10]
            if row[11]:
                p["options"] = _safe_json(row[11])
            result.append(p)
        return result
    except Exception as e:
        logger.warning("从 DB 拉取请求参数失败: %s", e)
        return []
    finally:
        eng.dispose()


def _param_to_row(p: dict, sort_idx: int) -> dict:
    """将参数定义转为 DB 行参数。"""
    return {
        "key": p["key"],
        "type": p["type"],
        "default_value": json.dumps(p.get("default"), ensure_ascii=False),
        "label": p.get("label", ""),
        "hint": p.get("hint", ""),
        "group_name": p.get("group", ""),
        "section": p.get("section", "flow"),
        "level": "request",
        "scope": json.dumps(p.get("scope", []), ensure_ascii=False),
        "depends_on": json.dumps(p["depends_on"], ensure_ascii=False)
        if p.get("depends_on")
        else None,
        "min_val": p.get("min"),
        "max_val": p.get("max"),
        "options": json.dumps(p["options"], ensure_ascii=False)
        if p.get("options")
        else None,
        "sort_order": sort_idx * 10,
    }


def _safe_json(raw, fallback=None):
    """安全解析 JSON 字符串。"""
    if raw is None:
        return fallback
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
