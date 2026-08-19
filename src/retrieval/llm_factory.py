"""
LLM 工厂 — 基于 LangChain init_chat_model 统一创建各厂商模型。

根据 provider 自动选择：
  - anthropic  → ChatAnthropic（原生 Messages API）
  - openai     → ChatOpenAI
  - deepseek   → ChatOpenAI + base_url
  - 其他       → ChatOpenAI + base_url（OpenAI 兼容）

使用示例::

    from src.retrieval.llm_factory import create_chat_model

    model = create_chat_model(config)
    resp = model.invoke(messages)
    print(resp.content)
"""

import logging
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.retrieval.agent_config import AgentRuntimeConfig

logger = logging.getLogger(__name__)

# provider 名称映射到 init_chat_model 识别的 model_provider
_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "deepseek": "openai",  # DeepSeek 兼容 OpenAI 协议
    "qwen": "openai",  # 通义千问（dashscope 兼容模式）
    "zhipu": "openai",  # 智谱 GLM
    "moonshot": "openai",  # Kimi
    "volcengine": "openai",  # 火山引擎（豆包）
    "baichuan": "openai",  # 百川
    "minimax": "openai",  # MiniMax
    "cohere": "openai",  # Cohere
}


def _resolve_provider(config: AgentRuntimeConfig) -> str:
    """从配置中推断 LangChain model_provider。"""
    # 优先用显式配置的 provider
    provider = (config.llm_provider or "").strip().lower()
    if provider in _PROVIDER_MAP:
        return _PROVIDER_MAP[provider]

    # 根据 base_url 域名推断
    url = (config.llm_base_url or "").lower()
    if "anthropic.com" in url:
        return "anthropic"
    if "deepseek.com" in url:
        return "openai"

    # 默认 OpenAI 兼容
    return "openai"


def create_chat_model(config: AgentRuntimeConfig) -> BaseChatModel:
    """根据 Agent 配置创建 LangChain ChatModel。"""
    if (config.llm_provider or "").strip().lower() == "codex":
        from src.retrieval.codex_chat_model import CodexChatModel

        logger.info(
            "创建 Codex LLM: model=%s, effort=%s, timeout=%ss, concurrency=%s",
            config.llm_model,
            config.codex_reasoning_effort,
            config.codex_timeout_seconds,
            config.codex_max_concurrency,
        )
        return CodexChatModel(
            model_name=config.llm_model,
            reasoning_effort=config.codex_reasoning_effort,
            timeout_seconds=config.codex_timeout_seconds,
            max_concurrency=config.codex_max_concurrency,
        )

    provider = _resolve_provider(config)

    kwargs = {
        "model": config.llm_model,
        "model_provider": provider,
        "api_key": config.llm_api_key,
    }

    # Anthropic Claude 4.x 已废弃 temperature 参数，传了会 400
    if provider == "anthropic":
        kwargs["max_tokens"] = 4096
    else:
        kwargs["temperature"] = config.llm_temperature
        if config.llm_base_url:
            kwargs["base_url"] = config.llm_base_url

    logger.info(
        f"创建 LLM: provider={provider}, model={config.llm_model}, "
        f"base_url={config.llm_base_url or '(default)'}"
    )

    return init_chat_model(**kwargs)
