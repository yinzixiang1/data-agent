"""
Agent 动态配置加载 — 支持 MySQL 和本地文件两种来源。

配置来源由 CONFIG_SOURCE 环境变量控制：
  - mysql: 从 da_agent + da_agent_config + sys_config 等表加载（默认）
  - local: 从指定的本地 JSON 文件加载

CONFIG_PROFILE 指定加载目标：
  - mysql 模式: Agent ID（如 "1"）
  - local 模式: 配置文件路径（如 "config/agent_config.json"）

使用示例::

    loader = AgentConfigLoader()
    config = loader.load(agent_id=1)
    print(config.llm_base_url)
    print(config.system_prompt)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.config import (
    COLUMN_SEARCH_TOP_K,
    CONFIG_PROFILE,
    CONFIG_SOURCE,
    DEFAULT_AGENT_TOKEN,
    DENSE_DIM,
    DENSE_MODEL,
    ENABLE_RERANKER,
    FEWSHOT_TOP_K,
    MMR_LAMBDA,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD_URL,
    MYSQL_PORT,
    MYSQL_USER,
    NL2SQL_ENV,
    RECALL_TOP_K,
    RERANK_INPUT_TOP_K,
    RERANKER_MODEL,
    RRF_K,
    TABLE_SEARCH_TOP_K,
)

logger = logging.getLogger(__name__)

CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
CODEX_TIMEOUT_MIN_SECONDS = 10
CODEX_TIMEOUT_MAX_SECONDS = 110
CODEX_MAX_CONCURRENCY_LIMIT = 8
CODEX_DEFAULT_MAX_CONCURRENCY = 4


# ── expand 分区：校验规则 + 读取函数 ──

EXPAND_VALIDATORS = {
    "max_execute_fix_retries": lambda v: isinstance(v, int) and 0 <= v <= 5,
    "execute_row_limit": lambda v: isinstance(v, int) and 1 <= v <= 10000,
    "execute_timeout": lambda v: isinstance(v, int) and 1 <= v <= 300,
    "max_fix_retries": lambda v: isinstance(v, int) and 0 <= v <= 10,
    "max_tokens": lambda v: isinstance(v, int) and 1 <= v <= 32768,
    "top_p": lambda v: isinstance(v, (int, float)) and 0 <= v <= 1,
    "llm_temperature": lambda v: isinstance(v, (int, float)) and 0 <= v <= 2,
    "value_exact_match_boost": lambda v: isinstance(v, (int, float)) and v >= 0,
    "table_search_top_k": lambda v: isinstance(v, int) and 1 <= v <= 100,
    "column_search_top_k": lambda v: isinstance(v, int) and 1 <= v <= 200,
    "recall_top_k": lambda v: isinstance(v, int) and 1 <= v <= 100,
    "fewshot_top_k": lambda v: isinstance(v, int) and 1 <= v <= 50,
    "max_context_tables": lambda v: isinstance(v, int) and 1 <= v <= 30,
    "max_relation_hops": lambda v: isinstance(v, int) and 0 <= v <= 4,
    "max_columns_per_table": lambda v: isinstance(v, int) and 4 <= v <= 100,
    "max_explain_scan_rows": lambda v: isinstance(v, int) and v >= 0,
}


def get_expand(expand_cfg: dict, key: str, default=None, cast=None):
    """从 expand 分区读取扩展配置，支持类型转换和校验。

    Args:
        expand_cfg: expand 分区的完整 dict
        key: 配置键名
        default: 默认值
        cast: 类型转换函数（int, float, bool 等）
    """
    value = expand_cfg.get(key, default)
    if value is None:
        return default

    # 类型转换
    if cast is not None:
        try:
            if cast is bool:
                value = (
                    value
                    if isinstance(value, bool)
                    else str(value).lower() in ("true", "1")
                )
            else:
                value = cast(value)
        except (ValueError, TypeError):
            logger.warning(
                f"expand 配置 {key}={value} 类型转换失败，使用默认值 {default}"
            )
            return default

    # 值校验
    validator = EXPAND_VALIDATORS.get(key)
    if validator and not validator(value):
        logger.warning(f"expand 配置 {key}={value} 不合法，使用默认值 {default}")
        return default

    return value


@dataclass
class AgentRuntimeConfig:
    """Agent 运行时配置（合并 agent_config + resource + sys_config）。"""

    # Agent 基础信息
    agent_id: int | None = None
    agent_name: str = ""
    agent_handle: str = ""
    # LLM 配置（来自 agent_config.model 分区 + sys_resource provider）
    llm_provider: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    codex_reasoning_effort: str = "low"
    codex_timeout_seconds: int = 90
    codex_max_concurrency: int = CODEX_DEFAULT_MAX_CONCURRENCY

    # Prompt 配置（来自 agent_config.prompt 分区）
    system_prompt: str = ""
    compress_prompt: str = ""
    output_rules: str = ""

    # 检索参数（来自 agent_config.retrieval 分区，fallback 到 sys_config / 启动配置）
    table_search_top_k: int = TABLE_SEARCH_TOP_K
    column_search_top_k: int = COLUMN_SEARCH_TOP_K
    recall_top_k: int = RECALL_TOP_K
    rerank_input_top_k: int = RERANK_INPUT_TOP_K
    fewshot_top_k: int = FEWSHOT_TOP_K
    rrf_k: int = RRF_K
    mmr_lambda: float = MMR_LAMBDA
    enable_reranker: bool = ENABLE_RERANKER
    max_context_tables: int = 8
    max_relation_hops: int = 2
    max_columns_per_table: int = 16
    enable_domain_routing: bool = True
    value_exact_match_boost: float = 2.0
    max_explain_scan_rows: int = 100_000_000

    # EXPLAIN 配置
    max_fix_retries: int = 5
    enable_explain: bool = True

    # SQL 执行配置
    enable_execute: bool = False
    execute_row_limit: int = 200
    execute_timeout: int = 30

    # 结构化结果工具（来自 Agent tool 配置 + tool 资源）
    tool_choice: str = "auto"
    tool_max_calls: int = 5
    tools: list[dict] = field(default_factory=list)

    # 查询缓存
    enable_query_cache: bool = False

    # 纠错增强配置
    max_execute_fix_retries: int = 2
    enable_enum_validate: bool = True
    enable_timeout_fallback: bool = False
    exchange_rate_injection: bool = True

    # 强制召回规则（from retrieval.pinned_rules）
    pinned_rules: list[dict] = field(default_factory=list)
    entity_resolution_rules: list[dict] = field(default_factory=list)

    # 结构化配置（from sys_config JSON，hot-reload 可更新）
    collection_search_config: dict = field(default_factory=dict)
    embedding_config: dict = field(default_factory=dict)
    index_build_config: dict = field(default_factory=dict)

    # Milvus 向量数据库（来自 agent_ref vector_db 资源绑定，fallback 到启动配置）
    milvus_uri: str = ""
    milvus_db: str = ""
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_token: str = ""

    # 安全
    token: str = ""

    # 来源标记
    config_source: str = "env"


DEFAULT_SYSTEM_PROMPT = """你是一个专业的 Apache Doris 数据分析专家。请根据下方提供的数据库 Schema、业务术语和参考示例，把用户的自然语言问题转换为可执行的 Doris SQL。

Doris 方言注意：
- DATE_TRUNC(datetime, 'unit')，不是 DATE_TRUNC('unit', datetime)
- 本月: complete_time >= DATE_TRUNC(NOW(), 'month') AND complete_time < DATE_ADD(DATE_TRUNC(NOW(), 'month'), INTERVAL 1 MONTH)
- 近30天: create_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
- 按月分组: DATE_FORMAT(create_time, '%Y-%m')"""


class AgentConfigLoader:
    """
    Agent 运行时配置加载器。

    支持两种来源（由 CONFIG_SOURCE 环境变量控制）：
      - mysql: 从 MySQL 数据库加载（默认）
      - local: 从本地 JSON 文件加载
    """

    def __init__(self, mysql_connection_string: str | None = None):
        if CONFIG_SOURCE == "local":
            self.engine = None
        else:
            if mysql_connection_string is None:
                mysql_connection_string = (
                    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
                    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
                )
            self.engine = create_engine(
                mysql_connection_string, pool_size=2, pool_recycle=3600
            )

    def load(self, agent_id: int | None = None) -> AgentRuntimeConfig:
        """
        加载 Agent 配置。

        CONFIG_SOURCE=local 时从文件加载，忽略 agent_id 参数。
        CONFIG_SOURCE=mysql 时从 MySQL 加载，优先级: agent_config > sys_config > 启动配置。
        """
        if CONFIG_SOURCE == "local":
            return self._load_from_file()

        return self._load_from_mysql(agent_id)

    def _load_from_file(self) -> AgentRuntimeConfig:
        """从本地 JSON 文件加载配置。"""
        file_path = Path(CONFIG_PROFILE)
        if not file_path.is_absolute():
            from src.retrieval.config import PROJECT_ROOT

            file_path = PROJECT_ROOT / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        logger.info(f"从本地文件加载配置: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = AgentRuntimeConfig()

        # Agent 基础信息
        agent = data.get("agent", {})
        config.agent_id = agent.get("id")
        config.agent_name = agent.get("name", "")
        config.agent_handle = agent.get("handle", "")
        config.token = agent.get("token", "")

        # sys_config 全局参数
        sys_configs = data.get("sys_config", {})
        # 将 JSON 对象转为字符串格式，与 MySQL 加载保持一致
        sys_str = {}
        for k, v in sys_configs.items():
            sys_str[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        self._apply_sys_configs(config, sys_str)

        # Agent 分段配置覆盖
        agent_configs = data.get("agent_config", {})
        if agent_configs:
            local_tools = data.get("agent_tools", [])
            self._apply_agent_configs(
                config,
                agent_configs,
                {},
                local_tools if isinstance(local_tools, list) else [],
            )
            config.config_source = "local"

        # Provider base_url 解析
        providers = data.get("providers", [])
        if providers and not config.llm_base_url:
            config.llm_base_url = providers[0].get("base_url", "")
            if not config.llm_provider:
                config.llm_provider = providers[0].get("name", "")

        # 补充默认值
        self._apply_defaults(config)

        logger.info(
            f"本地配置加载完成: agent={config.agent_name}, source={config.config_source}"
        )
        return config

    def _load_from_mysql(self, agent_id: int | None = None) -> AgentRuntimeConfig:
        """从 MySQL 数据库加载配置（原有逻辑）。"""
        config = AgentRuntimeConfig()

        # 1. 加载 sys_config（全局默认值）
        sys_configs = self._load_sys_configs()
        self._apply_sys_configs(config, sys_configs)

        # 2. 如果有 agent_id，加载 agent 配置（覆盖 sys_config）
        if agent_id:
            agent_info = self._load_agent_info(agent_id)
            if agent_info:
                config.agent_id = agent_id
                config.agent_name = agent_info.get("name", "")
                config.agent_handle = agent_info.get("handle", "")
                config.token = agent_info.get("token", "")
                agent_configs = self._load_agent_configs(agent_id)
                agent_refs = self._load_agent_refs(agent_id)
                agent_tools = self._load_agent_tools(agent_id)

                self._apply_agent_configs(
                    config,
                    agent_configs,
                    agent_refs,
                    agent_tools,
                )
                config.config_source = "agent"
            else:
                logger.warning(f"Agent {agent_id} 不存在，使用默认配置")

        # 3. 补充默认值
        self._apply_defaults(config)

        return config

    def _apply_defaults(self, config: AgentRuntimeConfig):
        """补充默认 prompt 和 token。"""
        if not config.system_prompt:
            config.system_prompt = DEFAULT_SYSTEM_PROMPT

        if not config.token:
            config.token = DEFAULT_AGENT_TOKEN

    def _load_sys_configs(self) -> dict[str, str]:
        """从 sys_config 加载全局配置。"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT config_key, config_value FROM sys_config")
                ).fetchall()
            return {row[0]: row[1] for row in rows}
        except SQLAlchemyError as exc:
            logger.warning("加载 sys_config 失败（表可能不存在）: %s", exc)
            return {}

    def _load_agent_info(self, agent_id: int) -> dict | None:
        """加载 Agent 基础信息。"""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT name, handle, status, token FROM da_agent WHERE id = :id"
                    ),
                    {"id": agent_id},
                ).fetchone()
            if row:
                return {
                    "name": row[0],
                    "handle": row[1],
                    "status": row[2],
                    "token": row[3] or "",
                }
            return None
        except SQLAlchemyError as exc:
            logger.warning("加载 Agent 信息失败: %s", exc)
            return None

    def _load_agent_configs(self, agent_id: int) -> dict[str, dict]:
        """加载 Agent 分段配置 → {section: config_dict}"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT section, config_json FROM da_agent_config WHERE agent_id = :id"
                    ),
                    {"id": agent_id},
                ).fetchall()
            result = {}
            for row in rows:
                try:
                    result[row[0]] = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    result[row[0]] = {}
            return result
        except SQLAlchemyError as exc:
            logger.warning("加载 Agent 配置失败: %s", exc)
            return {}

    def _load_agent_refs(self, agent_id: int) -> dict[str, list[str]]:
        """加载 Agent 资源引用 → {resource_type: [resource_key, ...]}"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT r.resource_type, b.binding_key "
                        "FROM da_agent_resource_binding b "
                        "JOIN sys_resource r ON r.id = b.resource_id "
                        "WHERE b.agent_id = :id AND b.enabled = 1 "
                        "AND r.resource_type != 'tool' AND r.status = 1 "
                        "ORDER BY b.sort_order, b.id"
                    ),
                    {"id": agent_id},
                ).fetchall()
            result: dict[str, list[str]] = {}
            for row in rows:
                result.setdefault(row[0], []).append(row[1])
            return result
        except SQLAlchemyError as exc:
            logger.warning("加载 Agent 资源引用失败: %s", exc)
            return {}

    def _load_resource_config(self, resource_type: str, name: str) -> dict:
        """从 sys_resource 加载资源配置。"""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT config_json FROM sys_resource "
                        "WHERE resource_type = :type AND name = :name AND status = 1"
                    ),
                    {"type": resource_type, "name": name},
                ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return {}
        except (SQLAlchemyError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("加载资源 %s/%s 失败: %s", resource_type, name, exc)
            return {}

    def _load_agent_tools(self, agent_id: int) -> list[dict]:
        """Load enabled tool contracts from the Agent binding table."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT r.name, r.display_name, r.description, "
                        "r.config_json, b.config_json "
                        "FROM da_agent_resource_binding b "
                        "JOIN sys_resource r ON r.id = b.resource_id "
                        "WHERE b.agent_id = :agent_id AND b.enabled = 1 "
                        "AND r.resource_type = 'tool' AND r.status = 1 "
                        "ORDER BY b.sort_order, b.id"
                    ),
                    {"agent_id": agent_id},
                ).fetchall()
        except SQLAlchemyError as exc:
            logger.warning("加载 Agent 工具绑定失败: %s", exc)
            return []

        tools = []
        for name, display_name, description, raw_definition, raw_binding in rows:
            try:
                definition = json.loads(raw_definition or "{}")
                binding_config = json.loads(raw_binding or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(definition, dict) or not isinstance(binding_config, dict):
                continue
            input_schema = definition.get("input_schema")
            if not isinstance(input_schema, dict):
                continue
            tools.append(
                {
                    "name": name,
                    "display_name": display_name or name,
                    "description": description or "",
                    "executor_key": str(definition.get("executor_key") or ""),
                    "execution_stage": str(
                        definition.get("execution_stage") or "channel_post_query"
                    ),
                    "state_policy": str(definition.get("state_policy") or "sticky"),
                    "intent_phrases": [
                        str(item).strip()
                        for item in (definition.get("intent_phrases") or [])
                        if str(item).strip()
                    ],
                    "input_schema": input_schema,
                    "output_schema": definition.get("output_schema") or {},
                    "requires_query_result": bool(
                        definition.get("requires_query_result")
                    ),
                    "trigger_mode": str(
                        binding_config.get("trigger_mode") or "intent_auto"
                    ),
                    "binding_config": binding_config,
                }
            )
        return tools

    def _apply_sys_configs(
        self, config: AgentRuntimeConfig, sys_configs: dict[str, str]
    ):
        """将 sys_config 值应用到 AgentRuntimeConfig。"""
        int_mappings = {
            "TABLE_SEARCH_TOP_K": "table_search_top_k",
            "COLUMN_SEARCH_TOP_K": "column_search_top_k",
            "RECALL_TOP_K": "recall_top_k",
            "RERANK_INPUT_TOP_K": "rerank_input_top_k",
            "FEWSHOT_TOP_K": "fewshot_top_k",
            "RRF_K": "rrf_k",
            "MAX_FIX_RETRIES": "max_fix_retries",
        }
        float_mappings = {
            "MMR_LAMBDA": "mmr_lambda",
        }
        bool_mappings = {
            "ENABLE_RERANKER": "enable_reranker",
        }

        for key, attr in int_mappings.items():
            if key in sys_configs:
                try:
                    setattr(config, attr, int(sys_configs[key]))
                except ValueError:
                    pass

        for key, attr in float_mappings.items():
            if key in sys_configs:
                try:
                    setattr(config, attr, float(sys_configs[key]))
                except ValueError:
                    pass

        for key, attr in bool_mappings.items():
            if key in sys_configs:
                setattr(config, attr, sys_configs[key].lower() in ("true", "1", "yes"))

        # LLM model: 格式 "provider/model" 如 "deepseek/deepseek-chat"
        if "DEFAULT_LLM_MODEL" in sys_configs:
            value = sys_configs["DEFAULT_LLM_MODEL"]
            if "/" in value:
                provider, model = value.split("/", 1)
                config.llm_provider = provider
                config.llm_model = model
            else:
                config.llm_model = value

        # 结构化 JSON 配置
        json_keys = {
            "COLLECTION_SEARCH_CONFIG": "collection_search_config",
            "EMBEDDING_CONFIG": "embedding_config",
            "INDEX_BUILD_CONFIG": "index_build_config",
        }
        for key, attr in json_keys.items():
            if key in sys_configs:
                try:
                    setattr(config, attr, json.loads(sys_configs[key]))
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"sys_config {key} JSON 解析失败，使用默认值")

        # 如果 sys_config 未配置结构化 JSON，使用环境感知默认值
        if not config.embedding_config:
            config.embedding_config = self._default_embedding_config()
        if not config.index_build_config:
            config.index_build_config = self._default_index_build_config()
        if not config.collection_search_config:
            config.collection_search_config = self._default_collection_search_config()

    def _apply_agent_configs(
        self,
        config: AgentRuntimeConfig,
        agent_configs: dict[str, dict],
        agent_refs: dict[str, list[str]],
        agent_tools: list[dict],
    ):
        """将 Agent 分段配置应用到 AgentRuntimeConfig。"""

        # model 分区
        model_cfg = agent_configs.get("model", {})
        if model_cfg:
            if "provider" in model_cfg:
                config.llm_provider = model_cfg["provider"]
            if "model" in model_cfg:
                config.llm_model = model_cfg["model"]
            if "temperature" in model_cfg:
                config.llm_temperature = float(model_cfg["temperature"])
            if "api_key" in model_cfg:
                config.llm_api_key = model_cfg["api_key"]
            if "base_url" in model_cfg:
                config.llm_base_url = model_cfg["base_url"]
            if "codex_reasoning_effort" in model_cfg:
                effort = str(model_cfg["codex_reasoning_effort"]).strip().lower()
                if effort in CODEX_REASONING_EFFORTS:
                    config.codex_reasoning_effort = effort
                else:
                    logger.warning("codex_reasoning_effort 不合法，使用默认值 low")
            if "codex_timeout_seconds" in model_cfg:
                try:
                    raw_timeout = model_cfg["codex_timeout_seconds"]
                    if isinstance(raw_timeout, bool):
                        raise TypeError
                    timeout_seconds = int(raw_timeout)
                    if not (
                        CODEX_TIMEOUT_MIN_SECONDS
                        <= timeout_seconds
                        <= CODEX_TIMEOUT_MAX_SECONDS
                    ):
                        raise ValueError
                    config.codex_timeout_seconds = timeout_seconds
                except (ValueError, TypeError):
                    logger.warning("codex_timeout_seconds 不合法，使用默认值 90")
            if "codex_max_concurrency" in model_cfg:
                try:
                    raw_concurrency = model_cfg["codex_max_concurrency"]
                    if isinstance(raw_concurrency, bool):
                        raise TypeError
                    max_concurrency = int(raw_concurrency)
                    if not 1 <= max_concurrency <= CODEX_MAX_CONCURRENCY_LIMIT:
                        raise ValueError
                    config.codex_max_concurrency = max_concurrency
                except (ValueError, TypeError):
                    logger.warning(
                        "codex_max_concurrency 不合法，使用默认值 %s",
                        CODEX_DEFAULT_MAX_CONCURRENCY,
                    )
            if "max_fix_retries" in model_cfg:
                try:
                    config.max_fix_retries = int(model_cfg["max_fix_retries"])
                except (ValueError, TypeError):
                    pass
            # Agent 级 Embedding 配置覆盖（deep merge 到 sys_config 值）
            if model_cfg.get("embedding_config"):
                config.embedding_config.update(model_cfg["embedding_config"])
                logger.info(
                    f"Agent 覆盖 embedding_config: model={model_cfg['embedding_config'].get('model')}"
                )

            # Agent 级 Reranker 配置覆盖（deep merge 到 index_build_config.reranker）
            if model_cfg.get("reranker_config"):
                if "reranker" not in config.index_build_config:
                    config.index_build_config["reranker"] = {}
                config.index_build_config["reranker"].update(
                    model_cfg["reranker_config"]
                )
                logger.info(
                    f"Agent 覆盖 reranker_config: model={model_cfg['reranker_config'].get('model')}"
                )

        # 从 agent_ref 的 provider 引用解析 base_url
        provider_refs = agent_refs.get("provider", [])
        if provider_refs and not config.llm_base_url:
            provider_name = provider_refs[0]
            res_config = self._load_resource_config("provider", provider_name)
            if res_config:
                config.llm_base_url = res_config.get("base_url", "")
                if not config.llm_provider:
                    config.llm_provider = provider_name

        # 从 agent_ref 的 vector_db 引用解析 Milvus 连接
        vector_refs = agent_refs.get("vector_db", [])
        if vector_refs:
            # resource_key 格式: "name/database"
            ref_key = vector_refs[0]
            parts = ref_key.split("/", 1)
            resource_name = parts[0]
            db_name = parts[1] if len(parts) > 1 else ""
            res_config = self._load_resource_config("vector_db", resource_name)
            if res_config:
                host = res_config.get("host", "")
                port = res_config.get("port", 19530)
                if host:
                    if ":" in host:
                        config.milvus_uri = f"http://{host}"
                    else:
                        config.milvus_uri = f"http://{host}:{port}"
                config.milvus_db = db_name or "default"
                config.milvus_user = res_config.get("user", "")
                config.milvus_password = res_config.get("password", "")
                config.milvus_token = res_config.get("token", "")
                logger.info(
                    f"Vector DB 从资源绑定加载: uri={config.milvus_uri}, db={config.milvus_db}"
                )

        # prompt 分区
        prompt_cfg = agent_configs.get("prompt", {})
        if prompt_cfg:
            if "system_prompt" in prompt_cfg:
                config.system_prompt = prompt_cfg["system_prompt"]
            if "compress_prompt" in prompt_cfg:
                config.compress_prompt = prompt_cfg["compress_prompt"]
            if "output_rules" in prompt_cfg:
                config.output_rules = prompt_cfg["output_rules"]

        # retrieval 分区
        retrieval_cfg = agent_configs.get("retrieval", {})
        if retrieval_cfg:
            retrieval_cfg = dict(retrieval_cfg)
            if "fewshot_k" in retrieval_cfg and "fewshot_top_k" not in retrieval_cfg:
                retrieval_cfg["fewshot_top_k"] = retrieval_cfg["fewshot_k"]
            int_fields = [
                "table_search_top_k",
                "column_search_top_k",
                "recall_top_k",
                "rerank_input_top_k",
                "fewshot_top_k",
                "rrf_k",
                "max_fix_retries",
                "max_context_tables",
                "max_relation_hops",
                "max_columns_per_table",
            ]
            for f in int_fields:
                if f in retrieval_cfg:
                    try:
                        setattr(config, f, int(retrieval_cfg[f]))
                    except (ValueError, TypeError):
                        pass
            if "mmr_lambda" in retrieval_cfg:
                try:
                    config.mmr_lambda = float(retrieval_cfg["mmr_lambda"])
                except (ValueError, TypeError):
                    pass
            if "enable_reranker" in retrieval_cfg:
                v = retrieval_cfg["enable_reranker"]
                config.enable_reranker = (
                    v if isinstance(v, bool) else str(v).lower() in ("true", "1")
                )
            if "enable_domain_routing" in retrieval_cfg:
                v = retrieval_cfg["enable_domain_routing"]
                config.enable_domain_routing = (
                    v if isinstance(v, bool) else str(v).lower() in ("true", "1")
                )
            if "enable_explain" in retrieval_cfg:
                v = retrieval_cfg["enable_explain"]
                config.enable_explain = (
                    v if isinstance(v, bool) else str(v).lower() in ("true", "1")
                )
            if "pinned_rules" in retrieval_cfg:
                rules = retrieval_cfg["pinned_rules"]
                if isinstance(rules, list):
                    config.pinned_rules = rules
            if "entity_resolution_rules" in retrieval_cfg:
                rules = retrieval_cfg["entity_resolution_rules"]
                if isinstance(rules, list):
                    config.entity_resolution_rules = [
                        rule for rule in rules if isinstance(rule, dict)
                    ]

        # flow 分区（SQL 执行）
        flow_cfg = agent_configs.get("flow", {})
        if flow_cfg:
            if "enable_execute" in flow_cfg:
                v = flow_cfg["enable_execute"]
                config.enable_execute = (
                    v if isinstance(v, bool) else str(v).lower() in ("true", "1")
                )
            if "execute_row_limit" in flow_cfg:
                try:
                    config.execute_row_limit = int(flow_cfg["execute_row_limit"])
                except (ValueError, TypeError):
                    pass
            if "execute_timeout" in flow_cfg:
                try:
                    config.execute_timeout = int(flow_cfg["execute_timeout"])
                except (ValueError, TypeError):
                    pass
            if "max_execute_fix_retries" in flow_cfg:
                try:
                    config.max_execute_fix_retries = int(
                        flow_cfg["max_execute_fix_retries"]
                    )
                except (ValueError, TypeError):
                    pass
            for bool_key in (
                "enable_enum_validate",
                "enable_timeout_fallback",
                "exchange_rate_injection",
            ):
                if bool_key in flow_cfg:
                    v = flow_cfg[bool_key]
                    setattr(
                        config,
                        bool_key,
                        v if isinstance(v, bool) else str(v).lower() in ("true", "1"),
                    )

        # tool 分区只保存规划策略；可用工具来自 Agent 工具绑定。
        tool_cfg = agent_configs.get("tool", {})
        choice = str(tool_cfg.get("choice") or "auto").strip().lower()
        config.tool_choice = (
            choice if choice in {"auto", "required", "none"} else "auto"
        )
        try:
            config.tool_max_calls = max(1, min(int(tool_cfg.get("max_iter", 5)), 20))
        except (TypeError, ValueError):
            config.tool_max_calls = 5
        if config.tool_choice != "none":
            config.tools = [tool for tool in agent_tools if isinstance(tool, dict)]

        # collection_overrides: Agent 级 Collection 策略覆盖
        self._merge_collection_overrides(config, agent_configs)

        # expand 分区（扩展配置，覆盖优先级低于显式字段）
        expand_cfg = agent_configs.get("expand", {})
        if expand_cfg:
            # 收集已由 model/prompt/retrieval/flow 显式设置的属性名
            explicit_keys: set[str] = set()
            # model 分区: config JSON key → config 属性名（不一致的需要映射）
            _model_key_map = {
                "provider": "llm_provider",
                "model": "llm_model",
                "temperature": "llm_temperature",
                "api_key": "llm_api_key",
                "base_url": "llm_base_url",
            }
            model_cfg_keys = agent_configs.get("model", {})
            for json_key, attr_name in _model_key_map.items():
                if json_key in model_cfg_keys:
                    explicit_keys.add(attr_name)
            if "max_fix_retries" in model_cfg_keys:
                explicit_keys.add("max_fix_retries")
            # prompt/retrieval/flow 分区: key 与属性名一致，直接收集
            for section in ("prompt", "retrieval", "flow"):
                for k in agent_configs.get(section, {}):
                    if hasattr(config, k):
                        explicit_keys.add(k)

            self._apply_expand(config, expand_cfg, explicit_keys)

    @staticmethod
    def _apply_expand(
        config: AgentRuntimeConfig,
        expand_cfg: dict,
        explicit_keys: set[str] | None = None,
    ):
        """将 expand 分区的扩展配置应用到 AgentRuntimeConfig。

        只覆盖 Agent 显式字段未设置的值（显式字段 > expand）。
        未知 key 记 warning 但不拦截。

        Args:
            config: 运行时配置对象
            expand_cfg: expand 分区 dict
            explicit_keys: 已由 model/prompt/retrieval/flow 显式设置的属性名集合
        """
        explicit_keys = explicit_keys or set()
        for key, value in expand_cfg.items():
            # 显式字段优先：已由其他分区设置的 key 不被 expand 覆盖
            if key in explicit_keys:
                logger.debug(f"expand 配置 {key}={value} 被显式字段覆盖，已跳过")
                continue
            if hasattr(config, key):
                current_value = getattr(config, key)
                value_type = type(current_value)
                cast = value_type if value_type in (int, float, bool) else None
                validated = get_expand(
                    expand_cfg,
                    key,
                    default=None,
                    cast=cast,
                )
                if validated is None:
                    continue
                setattr(config, key, validated)
            else:
                logger.warning(
                    f"expand 配置 {key}={value} 无对应 AgentRuntimeConfig 字段，已忽略"
                )

    def _merge_collection_overrides(
        self,
        config: AgentRuntimeConfig,
        agent_configs: dict[str, dict],
    ):
        """将 Agent 级 collection_overrides 逐字段 merge 到全局 collection_search_config。"""
        overrides = agent_configs.get("retrieval", {}).get("collection_overrides", {})
        if not overrides:
            return
        for col_name, col_override in overrides.items():
            if col_name in config.collection_search_config:
                config.collection_search_config[col_name].update(col_override)
            else:
                logger.warning(
                    f"collection_overrides 中的 {col_name} 不在全局配置中，已忽略"
                )

    def _default_embedding_config(self) -> dict:
        """根据 NL2SQL_ENV 生成 EMBEDDING_CONFIG 默认值。"""
        is_dev = NL2SQL_ENV == "dev"
        return {
            "model": DENSE_MODEL,
            "dim": DENSE_DIM,
            "dtype": "float16",
            "batch_size": 8 if is_dev else 64,
            "max_length": 512,
            "mrl_renormalize": is_dev,
            "instructions": {
                "table": "Given a user question, retrieve relevant database tables.",
                "column": "Given a user question, retrieve relevant database columns.",
                "enum": "Given a user question, retrieve relevant enum values.",
                "fewshot": "Given a user question, retrieve similar SQL question examples.",
                "glossary": "Given a user question, retrieve relevant business term definitions.",
            },
        }

    def _default_index_build_config(self) -> dict:
        """根据 NL2SQL_ENV 生成 INDEX_BUILD_CONFIG 默认值。"""
        is_dev = NL2SQL_ENV == "dev"
        return {
            "hnsw": {
                "M": 16 if is_dev else 32,
                "efConstruction": 200 if is_dev else 400,
                "ef_search": 64 if is_dev else 128,
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "analyzer": "chinese",
            },
            "reranker": {
                "model": RERANKER_MODEL,
                "score_threshold": 0.3,
            },
        }

    def _default_collection_search_config(self) -> dict:
        """生成 COLLECTION_SEARCH_CONFIG 默认值。"""
        is_dev = NL2SQL_ENV == "dev"
        return {
            "table": {
                "ranker_type": "weighted",
                "dense_weight": 0.4,
                "sparse_weight": 0.6,
                "recall_limit": 30 if is_dev else 100,
                "rerank": True,
                "rerank_top_n": 8,
                "final_top_n": 5,
            },
            "column": {
                "ranker_type": "weighted",
                "dense_weight": 0.4,
                "sparse_weight": 0.6,
                "recall_limit": 50 if is_dev else 200,
                "rerank": True,
                "rerank_top_n": 15,
                "final_top_n": 10,
            },
            "enum": {
                "ranker_type": "weighted",
                "dense_weight": 0.2,
                "sparse_weight": 0.8,
                "recall_limit": 20,
                "rerank": False,
                "rerank_top_n": 10,
                "final_top_n": 20,
            },
            "value": {
                "ranker_type": "bm25_only",
                "dense_weight": 0,
                "sparse_weight": 1,
                "recall_limit": 20,
                "rerank": False,
                "rerank_top_n": 10,
                "final_top_n": 20,
            },
            "fewshot": {
                "ranker_type": "weighted",
                "dense_weight": 0.8,
                "sparse_weight": 0.2,
                "recall_limit": 10,
                "rerank": True,
                "rerank_top_n": 10,
                "final_top_n": 3,
            },
            "glossary": {
                "ranker_type": "rrf",
                "dense_weight": 0.5,
                "sparse_weight": 0.5,
                "rrf_k": 20,
                "recall_limit": 10,
                "rerank": False,
                "rerank_top_n": 10,
                "final_top_n": 10,
            },
        }

    def print_config(self, config: AgentRuntimeConfig):
        """打印 Agent 运行时配置。"""
        lines = [
            "",
            "=" * 60,
            "  Agent 运行时配置",
            "=" * 60,
        ]

        if config.agent_id:
            lines += [
                "",
                "  [Agent]",
                f"    ID:       {config.agent_id}",
                f"    Name:     {config.agent_name}",
                f"    Handle:   {config.agent_handle}",
                "    Engine:   nl2sql",
            ]

        lines += [
            "",
            "  [LLM]",
            f"    Provider:    {config.llm_provider or '(默认)'}",
            f"    Model:       {config.llm_model}",
            f"    Base URL:    {config.llm_base_url or '(未配置)'}",
            f"    API Key:     {'***' + config.llm_api_key[-4:] if config.llm_api_key else '(未配置)'}",
            f"    Temperature: {config.llm_temperature}",
            f"    Codex Effort: {config.codex_reasoning_effort}",
            f"    Codex Timeout: {config.codex_timeout_seconds}s",
            f"    Codex Concurrency: {config.codex_max_concurrency}",
            "",
            "  [Prompt]",
            f"    System Prompt: {config.system_prompt[:60]}..."
            if len(config.system_prompt) > 60
            else f"    System Prompt: {config.system_prompt}",
            "",
            "  [检索参数]",
            f"    TABLE_SEARCH_TOP_K:  {config.table_search_top_k}",
            f"    COLUMN_SEARCH_TOP_K: {config.column_search_top_k}",
            f"    RECALL_TOP_K:        {config.recall_top_k}",
            f"    RERANK_INPUT_TOP_K:  {config.rerank_input_top_k}",
            f"    FEWSHOT_TOP_K:       {config.fewshot_top_k}",
            f"    RRF_K:               {config.rrf_k}",
            f"    MMR_LAMBDA:          {config.mmr_lambda}",
            f"    ENABLE_RERANKER:     {config.enable_reranker}",
            "",
            "  [校验]",
            f"    ENABLE_EXPLAIN:      {config.enable_explain}",
            f"    MAX_FIX_RETRIES:     {config.max_fix_retries}",
        ]

        # Embedding 配置
        emb = config.embedding_config
        if emb:
            lines += [
                "",
                "  [Embedding]",
                f"    Model:      {emb.get('model', '?')}",
                f"    Dim:        {emb.get('dim', '?')}",
                f"    Dtype:      {emb.get('dtype', '?')}",
                f"    Batch Size: {emb.get('batch_size', '?')}",
                f"    MRL Renorm: {emb.get('mrl_renormalize', False)}",
            ]

        # Index Build 配置
        idx = config.index_build_config
        if idx:
            hnsw = idx.get("hnsw", {})
            bm25 = idx.get("bm25", {})
            rr = idx.get("reranker", {})
            lines += [
                "",
                "  [Index Build]",
                f"    HNSW:     M={hnsw.get('M', '?')} efC={hnsw.get('efConstruction', '?')} ef={hnsw.get('ef_search', '?')}",
                f"    BM25:     k1={bm25.get('k1', '?')} b={bm25.get('b', '?')} analyzer={bm25.get('analyzer', '?')}",
                f"    Reranker: {rr.get('model', '?')} (threshold={rr.get('score_threshold', '?')})",
            ]

        # Collection 检索策略摘要
        csc = config.collection_search_config
        if csc:
            lines += ["", "  [Collection 检索策略]"]
            for col_name, col_cfg in csc.items():
                ranker = col_cfg.get("ranker_type", "?")
                rerank = "✓" if col_cfg.get("rerank") else "✗"
                final = col_cfg.get("final_top_n", "?")
                lines.append(
                    f"    {col_name:10s}  ranker={ranker:10s}  rerank={rerank}  top_n={final}"
                )

        lines += [
            "",
            f"  配置来源: {config.config_source}",
            f"  CONFIG_SOURCE: {CONFIG_SOURCE}",
            f"  CONFIG_PROFILE: {CONFIG_PROFILE or '(未设置)'}",
            "=" * 60,
        ]

        print("\n".join(lines))
