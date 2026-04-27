"""RAG 检索体系统一入口 — 串联所有模块"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

from src.retrieval.config import (
    INDEX_STORE_DIR, SEMANTIC_LAYER_DIR,
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
    """检索结果"""
    relevant_tables: list[dict] = field(default_factory=list)
    relevant_examples: list[dict] = field(default_factory=list)
    business_context: str = ""
    prompt_text: str = ""
    matched_terms: list[str] = field(default_factory=list)


class SchemaRetriever:
    """
    RAG 检索体系统一入口。

    使用方式:
        retriever = SchemaRetriever()
        retriever.initialize()  # 启动时调用一次
        result = retriever.retrieve("目前有多少活跃商户")
    """

    def __init__(
        self,
        connection_string: str | None = None,
        semantic_layer_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        offline: bool = False,
    ):
        self.schema_loader = SchemaLoader(
            connection_string=connection_string,
            semantic_layer_dir=semantic_layer_dir or SEMANTIC_LAYER_DIR,
            offline=offline,
        )
        self.index_manager = IndexManager(index_dir=index_dir or INDEX_STORE_DIR)
        self.formatter = SchemaFormatter()
        self.glossary_resolver = GlossaryResolver()

        # 以下在 initialize() 中赋值
        self.searcher: HybridSearcher | None = None
        self.fewshot: FewShotSelector | None = None
        self.table_schemas: dict = {}
        self._initialized = False

    def initialize(self, force_rebuild: bool = False):
        """
        启动初始化：加载 Schema → 判断是否需要重建索引 → 构建/加载索引。

        Args:
            force_rebuild: 是否强制重建（忽略 hash 对比）
        """
        logger.info("=" * 60)
        logger.info("RAG 检索体系初始化开始")
        logger.info("=" * 60)

        embedding = get_embedding()

        # 1. 加载 Schema + 语义层
        schemas, glossary = self.schema_loader.load_all()

        # 2. 加载业务术语
        self.glossary_resolver.load(glossary)

        # 3. 判断是否需要重建索引
        if force_rebuild or self.index_manager.need_rebuild(schemas):
            logger.info("开始全量构建索引...")
            indices = self.index_manager.build_and_save(schemas, embedding)
        else:
            logger.info("索引未变更，从磁盘加载...")
            indices = self.index_manager.load_all(embedding)
            # 加载时需要恢复 table_docs 中的 schema 引用
            table_schemas = indices["table_schemas"]
            for doc in indices["table_docs"]:
                doc["schema"] = table_schemas.get(doc["table_name"], {})

        # 4. 初始化混合检索器
        self.searcher = HybridSearcher(
            embedding=embedding,
            table_index=indices["table_index"],
            column_index=indices["column_index"],
            table_docs=indices["table_docs"],
            column_docs=indices["column_docs"],
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
    ) -> RetrievalResult:
        """
        完整检索流程:
        1. 业务术语解析
        2. Schema 混合检索（Dense + Sparse + RRF）
        3. 表级 + 列级合并
        4. Reranker 精排
        5. Few-shot 示例检索（语义相似 + 表重叠 + MMR）
        6. Prompt 组装（DDL 格式）

        Args:
            user_query: 用户原始提问
            top_k: 返回的表数量
            fewshot_k: 返回的 Few-shot 示例数量

        Returns:
            RetrievalResult
        """
        if not self._initialized:
            raise RuntimeError("SchemaRetriever 未初始化，请先调用 initialize()")

        logger.info(f"开始检索: '{user_query}'")

        # ❶ 业务术语解析
        glossary_result = self.glossary_resolver.resolve(user_query)
        enriched_query = glossary_result["enriched_query"]
        business_context = glossary_result["business_context"]

        # ❷ Schema 混合检索
        rerank_k = RERANK_INPUT_TOP_K if ENABLE_RERANKER else top_k
        candidates = self.searcher.search(enriched_query, top_k=rerank_k)

        # ❸ Reranker 精排
        reranker = get_reranker()
        if reranker and len(candidates) > top_k:
            candidates = reranker.rerank(user_query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # ❹ Few-shot 示例检索
        hit_tables = [c["table_name"] for c in candidates]
        examples = self.fewshot.select(
            query=user_query,
            tables=hit_tables,
            top_k=fewshot_k,
        )

        # ❺ Prompt 组装
        prompt_text = self.formatter.format_all(
            tables=candidates,
            examples=examples,
            business_context=business_context,
        )

        result = RetrievalResult(
            relevant_tables=candidates,
            relevant_examples=examples,
            business_context=business_context,
            prompt_text=prompt_text,
            matched_terms=glossary_result["matched_terms"],
        )

        logger.info(
            f"检索完成: tables={hit_tables}, "
            f"examples={len(examples)}, terms={result.matched_terms}"
        )
        return result
