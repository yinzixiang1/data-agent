"""
Reranker 精排 — 支持本地 Cross-Encoder 和在线 API 两种模式。

支持两种来源:
    - local: 通过 sentence-transformers CrossEncoder 本地推理
    - api: 通过 Reranker API（Jina/Cohere/OpenAI-compatible）在线调用

对混合检索的粗排候选做二次精排，提升 top-k 准确率。

使用示例::

    from src.retrieval.reranker import get_reranker

    reranker = get_reranker()  # ENABLE_RERANKER=false 时返回 None
    if reranker:
        ranked = reranker.rerank(
            query="活跃商户数量",
            candidates=[{"table_name": "pmt_account", "doc": {"text": "..."}, ...}],
            top_k=5,
        )
"""

import logging
from typing import Optional

from src.retrieval.config import RERANKER_MODEL, ENABLE_RERANKER

logger = logging.getLogger(__name__)

_instance: Optional["BaseReranker"] = None


def _extract_text(candidate: dict) -> str:
    """从候选项提取用于精排的文本，优先 doc.text，fallback 到 schema。"""
    text = candidate.get("doc", {}).get("text", "")
    if text:
        return text
    schema = candidate.get("schema")
    if not schema:
        return candidate.get("table_name", "")
    parts = []
    short = schema.get("table_name_short", schema.get("table_name", ""))
    if schema.get("display_name"):
        parts.append(f"{short} {schema['display_name']}")
    else:
        parts.append(short)
    if schema.get("description"):
        parts.append(schema["description"])
    if schema.get("tags"):
        parts.append("标签: " + ", ".join(str(tag) for tag in schema["tags"]))
    for col in schema.get("columns", []):
        if col.get("is_skip_index") or col.get("is_sensitive"):
            continue
        display = col.get("display_name") or ""
        description = col.get("description") or col.get("comment", "")
        business_logic = col.get("business_logic") or ""
        if any((display, description, business_logic)):
            semantic = f"字段:{col['name']}"
            if display:
                semantic += f"({display})"
            details = "；".join(
                str(value).strip()
                for value in (description, business_logic)
                if str(value).strip()
            )
            if details:
                semantic += f":{details}"
            parts.append(semantic)
    for rel in schema.get("relations", []):
        target = rel.get("target_table", "")
        if target:
            desc = rel.get("description", "")
            parts.append(f"关联:{target}" + (f"({desc})" if desc else ""))
    if schema.get("query_tips"):
        parts.append("查询注意:" + str(schema["query_tips"]))
    return "\n".join(parts)


class BaseReranker:
    """Reranker 基类，定义统一接口。"""

    def __init__(self, score_threshold: float = 0.3):
        self.score_threshold = score_threshold

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        raise NotImplementedError


class SchemaReranker(BaseReranker):
    """
    基于 Cross-Encoder 的本地 Schema 精排器。

    Attributes:
        model: sentence-transformers CrossEncoder 实例
        score_threshold: 精排分数阈值 (供调用方读取，rerank() 内不自动过滤)
    """

    def __init__(self, model_name: str = RERANKER_MODEL, score_threshold: float = 0.3):
        super().__init__(score_threshold=score_threshold)
        from sentence_transformers import CrossEncoder

        logger.info(f"加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, _extract_text(c)) for c in candidates]
        scores = self.model.predict(pairs)

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranker 精排完成: {len(candidates)} -> {min(top_k, len(ranked))}, "
            f"top={[r.get('table_name', '?') for r in ranked[:top_k]]}"
        )
        return ranked[:top_k]


class ApiReranker(BaseReranker):
    """
    在线 API Reranker，支持两种协议:

    - **OpenAI-compatible** (Jina/Cohere 风格):
      POST {base_url}/rerank  body: {"model","query","documents","top_n"}
    - **DashScope 原生** (阿里云百炼):
      POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
      body: {"model","input":{"query","documents"},"parameters":{"top_n"}}

    通过 ``api_format`` 配置项区分，默认 ``"openai"``，DashScope 设为 ``"dashscope"``。

    Attributes:
        base_url: API base URL
        api_key: 认证密钥
        api_model_name: API 请求中的 model 参数
        api_format: 协议格式 ("openai" | "dashscope")
        score_threshold: 精排分数阈值
    """

    # DashScope 原生 Reranker 端点（固定路径）
    DASHSCOPE_RERANK_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )

    def __init__(self, reranker_config: dict | None = None):
        cfg = reranker_config or {}
        super().__init__(score_threshold=cfg.get("score_threshold", 0.3))

        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.api_key = cfg.get("api_key", "")
        self.api_model_name = cfg.get("api_model_name", cfg.get("model", ""))
        self.api_format = cfg.get("api_format", "openai")

        # 自动检测: 如果 base_url 包含 dashscope.aliyuncs.com，自动切换协议
        if "dashscope.aliyuncs.com" in self.base_url:
            self.api_format = "dashscope"

        if self.api_format != "dashscope" and not self.base_url:
            raise ValueError("API Reranker 需要 base_url 配置")

        logger.info(
            f"初始化 API Reranker: format={self.api_format}, "
            f"base_url={self.base_url}, model={self.api_model_name}"
        )

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        import httpx

        if not candidates:
            return []

        documents = [_extract_text(c) for c in candidates]
        top_n = min(top_k, len(candidates))

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.api_format == "dashscope":
            url = self.DASHSCOPE_RERANK_URL
            body = {
                "model": self.api_model_name,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n},
            }
        else:
            url = f"{self.base_url}/rerank"
            body = {
                "model": self.api_model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }

        resp = httpx.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # DashScope 响应嵌套在 output 里; OpenAI-compatible 直接在顶层
        if "output" in data:
            results = data["output"].get("results", [])
        else:
            results = data.get("results", [])

        for item in results:
            idx = item["index"]
            candidates[idx]["rerank_score"] = float(item.get("relevance_score", 0))

        # 未被 API 返回的候选设为 0 分
        for c in candidates:
            if "rerank_score" not in c:
                c["rerank_score"] = 0.0

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"API Reranker 精排完成: {len(candidates)} -> {min(top_k, len(ranked))}, "
            f"top={[r.get('table_name', '?') for r in ranked[:top_k]]}"
        )
        return ranked[:top_k]


def get_reranker(index_build_config: dict | None = None) -> BaseReranker | None:
    """
    获取 Reranker 单例（懒加载）。

    根据 reranker.source 选择本地或 API 实现:
        - "local" (默认): SchemaReranker (CrossEncoder)
        - "api": ApiReranker (HTTP API)

    Args:
        index_build_config: INDEX_BUILD_CONFIG 字典，仅首次调用时生效。

    Returns:
        BaseReranker 实例，ENABLE_RERANKER=False 时返回 None
    """
    if not ENABLE_RERANKER:
        return None
    global _instance
    if _instance is None:
        cfg = (index_build_config or {}).get("reranker", {})
        source = cfg.get("source", "local")
        if source == "api":
            _instance = ApiReranker(cfg)
        else:
            model = cfg.get("model", RERANKER_MODEL)
            threshold = cfg.get("score_threshold", 0.3)
            _instance = SchemaReranker(model_name=model, score_threshold=threshold)
    return _instance
