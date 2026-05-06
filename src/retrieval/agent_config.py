"""
Agent 动态配置加载 — 从 MySQL 读取 Agent 配置和资源信息。

从 da_agent + da_agent_config + da_agent_ref + res_resource + sys_config 表
加载 Agent 的 LLM / Prompt / 检索参数等配置，替代 .env 硬编码。

使用示例::

    loader = AgentConfigLoader()
    config = loader.load(agent_id=1)
    print(config.llm_base_url)
    print(config.system_prompt)
"""

import json
import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text

from src.retrieval.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    TABLE_SEARCH_TOP_K, COLUMN_SEARCH_TOP_K, RECALL_TOP_K,
    RERANK_INPUT_TOP_K, FEWSHOT_TOP_K, RRF_K, MMR_LAMBDA,
    ENABLE_RERANKER, GLOSSARY_SCORE_THRESHOLD, DEFAULT_AGENT_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeConfig:
    """Agent 运行时配置（合并 agent_config + resource + sys_config）。"""

    # Agent 基础信息
    agent_id: int | None = None
    agent_name: str = ""
    agent_handle: str = ""

    # LLM 配置（来自 agent_config.model 分区 + res_resource provider）
    llm_provider: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.0

    # Prompt 配置（来自 agent_config.prompt 分区）
    system_prompt: str = ""

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
    """从 MySQL 加载 Agent 运行时配置。"""

    def __init__(self, mysql_connection_string: str | None = None):
        if mysql_connection_string is None:
            mysql_connection_string = (
                f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
                f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
            )
        self.engine = create_engine(mysql_connection_string, pool_size=2, pool_recycle=3600)

    def load(self, agent_id: int | None = None) -> AgentRuntimeConfig:
        """
        加载 Agent 配置。

        优先级: agent_config > sys_config > .env 默认值

        Args:
            agent_id: Agent ID，None 时只使用 sys_config + .env 默认值
        """
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

        # 3. 补充默认 system_prompt
        if not config.system_prompt:
            config.system_prompt = DEFAULT_SYSTEM_PROMPT

        # 4. Token fallback 到默认值
        if not config.token:
            config.token = DEFAULT_AGENT_TOKEN

        # 5. 补充 LLM 配置 fallback（从 .env）
        if not config.llm_base_url or not config.llm_api_key:
            import os
            if not config.llm_base_url:
                config.llm_base_url = os.getenv("DEEPSEEK_BASE_URL", "")
            if not config.llm_api_key:
                config.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not config.llm_model:
                config.llm_model = "deepseek-chat"

        return config

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
        """从 res_resource 加载资源配置。"""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT config_json FROM res_resource "
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

        # 从 agent_ref 的 provider 引用解析 base_url
        provider_refs = agent_refs.get("provider", [])
        if provider_refs and not config.llm_base_url:
            provider_name = provider_refs[0]
            res_config = self._load_resource_config("provider", provider_name)
            if res_config:
                config.llm_base_url = res_config.get("base_url", "")
                if not config.llm_provider:
                    config.llm_provider = provider_name

        # prompt 分区
        prompt_cfg = agent_configs.get("prompt", {})
        if prompt_cfg:
            if "system_prompt" in prompt_cfg:
                config.system_prompt = prompt_cfg["system_prompt"]

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
            "",
            f"  配置来源: {config.config_source}",
            "=" * 60,
        ]

        print("\n".join(lines))
