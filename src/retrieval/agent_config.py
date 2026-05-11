"""
Agent 动态配置加载 — 支持 MySQL 和本地文件两种来源。

配置来源由 CONFIG_SOURCE 环境变量控制：
  - mysql: 从 da_agent + da_agent_config + sys_config 等表加载（默认）
  - local: 从本地 JSON 文件加载（通过 config_export.py 导出）

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

from src.retrieval.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    TABLE_SEARCH_TOP_K, COLUMN_SEARCH_TOP_K, RECALL_TOP_K,
    RERANK_INPUT_TOP_K, FEWSHOT_TOP_K, RRF_K, MMR_LAMBDA,
    ENABLE_RERANKER, GLOSSARY_SCORE_THRESHOLD, DEFAULT_AGENT_TOKEN,
    NL2SQL_ENV, DENSE_MODEL, DENSE_DIM, RERANKER_MODEL,
    CONFIG_SOURCE, CONFIG_PROFILE,
)

logger = logging.getLogger(__name__)


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

    # Prompt 配置（来自 agent_config.prompt 分区）
    system_prompt: str = ""
    compress_prompt: str = ""

    # 检索参数（来自 agent_config.retrieval 分区，fallback 到 sys_config / .env）
    table_search_top_k: int = TABLE_SEARCH_TOP_K
    column_search_top_k: int = COLUMN_SEARCH_TOP_K
    recall_top_k: int = RECALL_TOP_K
    rerank_input_top_k: int = RERANK_INPUT_TOP_K
    fewshot_top_k: int = FEWSHOT_TOP_K
    rrf_k: int = RRF_K
    mmr_lambda: float = MMR_LAMBDA
    enable_reranker: bool = ENABLE_RERANKER
    glossary_score_threshold: float = GLOSSARY_SCORE_THRESHOLD

    # EXPLAIN 配置
    max_fix_retries: int = 5
    enable_explain: bool = True

    # 结构化配置（from sys_config JSON，hot-reload 可更新）
    collection_search_config: dict = field(default_factory=dict)
    embedding_config: dict = field(default_factory=dict)
    index_build_config: dict = field(default_factory=dict)

    # Milvus 向量数据库（来自 agent_ref vector_db 资源绑定，fallback 到 .env）
    milvus_uri: str = ""
    milvus_db: str = ""
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_token: str = ""

    # 安全
    token: str = ""

    # 来源标记
    config_source: str = "env"


DEFAULT_SYSTEM_PROMPT = """你是一个专业的 SQL 专家，专门将用户的自然语言问题转化为 Apache Doris SQL。

规则：
1. 必须输出一条可直接执行的 SQL，用 ```sql ``` 包裹
2. 严格使用提供的表和列，不要编造不存在的表或列
3. 列别名使用中文双引号，如 COUNT(*) AS "数量"
4. 注意参考业务上下文中的过滤条件提示
5. 参考 Few-shot 示例的 SQL 模式
6. 即使信息不完整，也要基于已有表结构尽力生成最合理的 SQL

Doris 日期函数注意：
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
                    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
                    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
                )
            self.engine = create_engine(mysql_connection_string, pool_size=2, pool_recycle=3600)

    def load(self, agent_id: int | None = None) -> AgentRuntimeConfig:
        """
        加载 Agent 配置。

        CONFIG_SOURCE=local 时从文件加载，忽略 agent_id 参数。
        CONFIG_SOURCE=mysql 时从 MySQL 加载，优先级: agent_config > sys_config > .env。
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
            raise FileNotFoundError(
                f"配置文件不存在: {file_path}\n"
                f"请先运行 python -m src.retrieval.config_export 导出配置"
            )

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
            self._apply_agent_configs(config, agent_configs, {})
            config.config_source = "local"

        # Provider base_url 解析
        providers = data.get("providers", [])
        if providers and not config.llm_base_url:
            config.llm_base_url = providers[0].get("base_url", "")
            if not config.llm_provider:
                config.llm_provider = providers[0].get("name", "")

        # 补充默认值
        self._apply_defaults(config)

        logger.info(f"本地配置加载完成: agent={config.agent_name}, source={config.config_source}")
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

                self._apply_agent_configs(config, agent_configs, agent_refs)
                config.config_source = "agent"
            else:
                logger.warning(f"Agent {agent_id} 不存在，使用默认配置")

        # 3. 补充默认值
        self._apply_defaults(config)

        return config

    def _apply_defaults(self, config: AgentRuntimeConfig):
        """补充默认 prompt、token、LLM fallback。"""
        import os

        if not config.system_prompt:
            config.system_prompt = DEFAULT_SYSTEM_PROMPT

        if not config.token:
            config.token = DEFAULT_AGENT_TOKEN

        if not config.llm_base_url or not config.llm_api_key:
            if not config.llm_base_url:
                config.llm_base_url = os.getenv("DEEPSEEK_BASE_URL", "")
            if not config.llm_api_key:
                config.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not config.llm_model:
                config.llm_model = "deepseek-chat"

    def _load_sys_configs(self) -> dict[str, str]:
        """从 sys_config 加载全局配置。"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT config_key, config_value FROM sys_config"
                )).fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.warning(f"加载 sys_config 失败（表可能不存在）: {e}")
            return {}

    def _load_agent_info(self, agent_id: int) -> dict | None:
        """加载 Agent 基础信息。"""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT name, handle, status, token FROM da_agent WHERE id = :id"
                ), {"id": agent_id}).fetchone()
            if row:
                return {"name": row[0], "handle": row[1], "status": row[2], "token": row[3] or ""}
            return None
        except Exception as e:
            logger.warning(f"加载 Agent 信息失败: {e}")
            return None

    def _load_agent_configs(self, agent_id: int) -> dict[str, dict]:
        """加载 Agent 分段配置 → {section: config_dict}"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT section, config_json FROM da_agent_config WHERE agent_id = :id"
                ), {"id": agent_id}).fetchall()
            result = {}
            for row in rows:
                try:
                    result[row[0]] = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    result[row[0]] = {}
            return result
        except Exception as e:
            logger.warning(f"加载 Agent 配置失败: {e}")
            return {}

    def _load_agent_refs(self, agent_id: int) -> dict[str, list[str]]:
        """加载 Agent 资源引用 → {resource_type: [resource_key, ...]}"""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT resource_type, resource_key FROM da_agent_ref "
                    "WHERE agent_id = :id ORDER BY sort_order"
                ), {"id": agent_id}).fetchall()
            result: dict[str, list[str]] = {}
            for row in rows:
                result.setdefault(row[0], []).append(row[1])
            return result
        except Exception as e:
            logger.warning(f"加载 Agent 资源引用失败: {e}")
            return {}

    def _load_resource_config(self, resource_type: str, name: str) -> dict:
        """从 sys_resource 加载资源配置。"""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT config_json FROM sys_resource "
                    "WHERE resource_type = :type AND name = :name AND status = 1"
                ), {"type": resource_type, "name": name}).fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return {}
        except Exception as e:
            logger.warning(f"加载资源 {resource_type}/{name} 失败: {e}")
            return {}

    def _apply_sys_configs(self, config: AgentRuntimeConfig, sys_configs: dict[str, str]):
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
            "GLOSSARY_SCORE_THRESHOLD": "glossary_score_threshold",
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
            if "max_fix_retries" in model_cfg:
                try:
                    config.max_fix_retries = int(model_cfg["max_fix_retries"])
                except (ValueError, TypeError):
                    pass
            # Agent 级 Embedding 配置覆盖（deep merge 到 sys_config 值）
            if "embedding_config" in model_cfg and model_cfg["embedding_config"]:
                config.embedding_config.update(model_cfg["embedding_config"])
                logger.info(f"Agent 覆盖 embedding_config: model={model_cfg['embedding_config'].get('model')}")

            # Agent 级 Reranker 配置覆盖（deep merge 到 index_build_config.reranker）
            if "reranker_config" in model_cfg and model_cfg["reranker_config"]:
                if "reranker" not in config.index_build_config:
                    config.index_build_config["reranker"] = {}
                config.index_build_config["reranker"].update(model_cfg["reranker_config"])
                logger.info(f"Agent 覆盖 reranker_config: model={model_cfg['reranker_config'].get('model')}")

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
                logger.info(f"Vector DB 从资源绑定加载: uri={config.milvus_uri}, db={config.milvus_db}")

        # prompt 分区
        prompt_cfg = agent_configs.get("prompt", {})
        if prompt_cfg:
            if "system_prompt" in prompt_cfg:
                config.system_prompt = prompt_cfg["system_prompt"]
            if "compress_prompt" in prompt_cfg:
                config.compress_prompt = prompt_cfg["compress_prompt"]

        # retrieval 分区
        retrieval_cfg = agent_configs.get("retrieval", {})
        if retrieval_cfg:
            int_fields = [
                "table_search_top_k", "column_search_top_k", "recall_top_k",
                "rerank_input_top_k", "fewshot_top_k", "rrf_k", "max_fix_retries",
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
            if "glossary_score_threshold" in retrieval_cfg:
                try:
                    config.glossary_score_threshold = float(retrieval_cfg["glossary_score_threshold"])
                except (ValueError, TypeError):
                    pass
            if "enable_reranker" in retrieval_cfg:
                v = retrieval_cfg["enable_reranker"]
                config.enable_reranker = v if isinstance(v, bool) else str(v).lower() in ("true", "1")
            if "enable_explain" in retrieval_cfg:
                v = retrieval_cfg["enable_explain"]
                config.enable_explain = v if isinstance(v, bool) else str(v).lower() in ("true", "1")

        # collection_overrides: Agent 级 Collection 策略覆盖
        self._merge_collection_overrides(config, agent_configs)

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
                logger.warning(f"collection_overrides 中的 {col_name} 不在全局配置中，已忽略")

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
                "rerank_top_n": 3,
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
            ]

        lines += [
            "",
            "  [LLM]",
            f"    Provider:    {config.llm_provider or '(默认)'}",
            f"    Model:       {config.llm_model}",
            f"    Base URL:    {config.llm_base_url or '(未配置)'}",
            f"    API Key:     {'***' + config.llm_api_key[-4:] if config.llm_api_key else '(未配置)'}",
            f"    Temperature: {config.llm_temperature}",
            "",
            "  [Prompt]",
            f"    System Prompt: {config.system_prompt[:60]}..." if len(config.system_prompt) > 60
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
            f"    GLOSSARY_THRESHOLD: {config.glossary_score_threshold}",
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
                lines.append(f"    {col_name:10s}  ranker={ranker:10s}  rerank={rerank}  top_n={final}")

        lines += [
            "",
            f"  配置来源: {config.config_source}",
            f"  CONFIG_SOURCE: {CONFIG_SOURCE}",
            f"  CONFIG_PROFILE: {CONFIG_PROFILE or '(未设置)'}",
            "=" * 60,
        ]

        print("\n".join(lines))
