"""
RAG 检索体系统一入口 — 串联所有模块。

SchemaRetriever 是外部调用的唯一接口，封装了完整的检索流程:
    术语解析 → 混合检索 → Reranker 精排 → 关联补回 → 枚举检索 → Few-shot → Prompt 组装

使用示例::

    retriever = SchemaRetriever()
    retriever.initialize()  # 首次启动，加载模型 + 构建索引

    result = retriever.retrieve("目前有多少活跃商户")
    print(result.relevant_tables)   # [{"table_name": "pmt_account", ...}]
    print(result.prompt_text)       # 完整 Prompt（DDL + 枚举 + 上下文 + Few-shot）
    print(result.matched_terms)     # ["活跃商户"]
"""

import logging
from dataclasses import dataclass, field

from src.retrieval.config import (
    TABLE_SEARCH_TOP_K, RERANK_INPUT_TOP_K, FEWSHOT_TOP_K,
    ENABLE_RERANKER,
)
from src.retrieval.schema_loader import SchemaLoader
from src.retrieval.embedding import get_embedding
from src.retrieval.index_manager import IndexManager
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import get_reranker
from src.retrieval.fewshot_selector import FewShotSelector
from src.retrieval.glossary_resolver import GlossaryResolver
from src.retrieval.schema_formatter import SchemaFormatter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    检索结果数据类。

    Attributes:
        relevant_tables: 检索命中的表列表，每个 dict 包含 table_name, score, schema 等
        relevant_examples: 选中的 Few-shot 示例列表
        enum_hits: 枚举值命中列表
        business_context: 术语展开的业务上下文文本
        prompt_text: 组装好的完整 Prompt 文本（直接拼入 LLM message）
        matched_terms: 命中的业务术语名列表
    """
    relevant_tables: list[dict] = field(default_factory=list)
    relevant_examples: list[dict] = field(default_factory=list)
    enum_hits: list[dict] = field(default_factory=list)
    business_context: str = ""
    prompt_text: str = ""
    matched_terms: list[str] = field(default_factory=list)


class SchemaRetriever:
    """
    RAG 检索体系统一入口。

    使用方式:
        retriever = SchemaRetriever()
        retriever.initialize()  # 启动时调用一次（强制重建索引）
        result = retriever.retrieve("目前有多少活跃商户")
    """

    def __init__(
        self,
        connection_string: str | None = None,
    ):
        """
        Args:
            connection_string: Doris SQLAlchemy 连接字符串，None 时从 config 自动拼接
        """
        self.schema_loader = SchemaLoader(
            connection_string=connection_string,
        )
        self.index_manager = IndexManager()
        self.formatter = SchemaFormatter()
        self.glossary_resolver: GlossaryResolver | None = None

        # 以下在 initialize() 中赋值
        self.searcher: HybridSearcher | None = None
        self.fewshot: FewShotSelector | None = None
        self.table_schemas: dict = {}
        self._initialized = False

    def initialize(self):
        """启动初始化：加载 Schema → 加载术语 → 强制重建索引 → 构建混合检索器。"""
        logger.info("=" * 60)
        logger.info("RAG 检索体系初始化开始")
        logger.info("=" * 60)

        embedding = get_embedding()

        # 1. 加载 Schema + 语义层 + 枚举 + Fewshot
        schemas, glossary, enums, fewshot_examples = self.schema_loader.load_all()

        # 2. 全量构建索引（含术语向量化）
        logger.info("开始全量构建索引...")
        indices = self.index_manager.build(schemas, embedding, enums, fewshot_examples, glossary)

        # 3. 初始化术语解析器（向量语义匹配）
        self.glossary_resolver = GlossaryResolver(embedding, indices["glossary_index"])

        # 4. 初始化混合检索器
        self.searcher = HybridSearcher(
            embedding=embedding,
            table_index=indices["table_index"],
            column_index=indices["column_index"],
            enum_index=indices["enum_index"],
            table_schemas=indices["table_schemas"],
        )

        self.fewshot = indices["fewshot_selector"]
        self.table_schemas = indices["table_schemas"]
        self._initialized = True

        logger.info("=" * 60)
        logger.info(
            f"RAG 初始化完成: {len(self.table_schemas)} 张表, "
            f"{len(glossary)} 条术语, "
            f"Reranker={'启用' if ENABLE_RERANKER else '关闭'}"
        )
        logger.info("=" * 60)

    def retrieve(
        self,
        user_query: str,
        top_k: int = TABLE_SEARCH_TOP_K,
        fewshot_k: int = FEWSHOT_TOP_K,
        glossary_score_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        完整 RAG 检索流程。

        流程:
            1. 业务术语解析 → enriched_query + business_context
            2. Schema 混合检索（Dense + Sparse + RRF + 列反推 + 枚举反哺 + 关联补全）
            3. Reranker 精排（如启用）
            4. Reranker 后关联表补回（被淘汰但与 top 表有关联的表）
            5. 枚举值检索
            6. Few-shot 示例检索（语义相似 + 表重叠 + MMR）
            7. Prompt 组装（DDL + 枚举映射 + 业务上下文 + Few-shot）

        Args:
            user_query: 用户原始自然语言问题，如 "目前有多少活跃商户"
            top_k: 最终返回的表数量（Reranker 后）
            fewshot_k: Few-shot 示例返回数量

        Returns:
            RetrievalResult: 包含 relevant_tables, enum_hits, prompt_text 等

        Raises:
            RuntimeError: 未调用 initialize() 时抛出
        """
        if not self._initialized:
            raise RuntimeError("SchemaRetriever 未初始化，请先调用 initialize()")

        logger.info(f"开始检索: '{user_query}'")

        # ❶ 业务术语解析
        resolve_kwargs = {}
        if glossary_score_threshold is not None:
            resolve_kwargs["score_threshold"] = glossary_score_threshold
        glossary_result = self.glossary_resolver.resolve(user_query, **resolve_kwargs)
        enriched_query = glossary_result["enriched_query"]
        business_context = glossary_result["business_context"]

        # ❷ Schema 混合检索
        rerank_k = RERANK_INPUT_TOP_K if ENABLE_RERANKER else top_k
        candidates = self.searcher.search(enriched_query, top_k=rerank_k)

        # ❸ Reranker 精排
        reranker = get_reranker()
        all_search_candidates = {c["table_name"]: c for c in candidates}
        if reranker and len(candidates) > top_k:
            candidates = reranker.rerank(user_query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # ❸.5 Reranker 后关联表补回：被 Reranker 淘汰但与 top 表有关联的表补回
        hit_names = {c["table_name"] for c in candidates}
        for c in list(candidates):
            schema = c.get("schema", {})
            for rel in schema.get("relations", []):
                related = rel.get("target_table", "")
                if related in all_search_candidates and related not in hit_names:
                    candidates.append(all_search_candidates[related])
                    hit_names.add(related)
                    logger.info(f"Reranker 后补回关联表: {related}")

        # ❹ 枚举值检索
        enum_hits = self.searcher.search_enums(user_query)

        # ❺ Few-shot 示例检索
        hit_tables = [c["table_name"] for c in candidates]
        examples = self.fewshot.select(
            query=user_query,
            tables=hit_tables,
            top_k=fewshot_k,
        )

        # ❻ Prompt 组装
        prompt_text = self.formatter.format_all(
            tables=candidates,
            examples=examples,
            business_context=business_context,
            enum_hits=enum_hits,
        )

        result = RetrievalResult(
            relevant_tables=candidates,
            relevant_examples=examples,
            enum_hits=enum_hits,
            business_context=business_context,
            prompt_text=prompt_text,
            matched_terms=glossary_result["matched_terms"],
        )

        logger.info(
            f"检索完成: tables={hit_tables}, "
            f"enums={len(enum_hits)}, examples={len(examples)}, terms={result.matched_terms}"
        )
        return result
