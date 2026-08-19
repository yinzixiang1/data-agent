"""
RAG 检索体系统一入口 — 串联所有模块。

SchemaRetriever 是外部调用的唯一接口，封装了完整的检索流程:
    术语解析 -> Value 匹配 -> 混合检索 -> Reranker 精排 -> 关联补回
    -> 枚举检索 -> Few-shot -> Prompt 组装

使用示例::

    retriever = SchemaRetriever()
    retriever.initialize()  # 首次启动，加载模型 + 构建索引

    result = retriever.retrieve("目前有多少活跃商户")
    print(result.relevant_tables)   # [{"table_name": "pmt_account", ...}]
    print(result.prompt_text)       # 完整 Prompt
    print(result.matched_terms)     # ["活跃商户"]
    print(result.value_hits)        # [{"table_name": ..., "sql_value": ...}]
"""

import logging
from dataclasses import dataclass, field

from src.retrieval.schema_loader import SchemaLoader
from src.retrieval.embedding import get_embedding
from src.retrieval.index_manager import IndexManager
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import get_reranker
from src.retrieval.fewshot_selector import FewShotSelector
from src.retrieval.glossary_resolver import GlossaryResolver
from src.retrieval.value_indexer import ValueIndexer
from src.retrieval.ranker_strategy import get_search_params
from src.retrieval.schema_formatter import SchemaFormatter
from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.entity_resolver import EntityResolver
from src.retrieval.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)

EXCHANGE_RATE_TABLE = "warehouse_sys.sys_exchange_rate"
EXCHANGE_RATE_REQUIRED_COLUMNS = {
    f"{EXCHANGE_RATE_TABLE}.source_currency",
    f"{EXCHANGE_RATE_TABLE}.target_currency",
    f"{EXCHANGE_RATE_TABLE}.sync_time",
    f"{EXCHANGE_RATE_TABLE}.mid",
}


class RequiredSemanticTableMissingError(RuntimeError):
    """查询依赖的确定性语义表尚未绑定到当前 Agent。"""


@dataclass
class RetrievalResult:
    """
    检索结果数据类。

    Attributes:
        relevant_tables: 检索命中的表列表
        relevant_examples: 选中的 Few-shot 示例列表
        enum_hits: 枚举值命中列表
        value_hits: Schema Linking 值匹配列表
        business_context: 术语展开的业务上下文文本
        prompt_text: 组装好的完整 Prompt 文本
        matched_terms: 命中的业务术语名列表
    """

    relevant_tables: list[dict] = field(default_factory=list)
    relevant_examples: list[dict] = field(default_factory=list)
    enum_hits: list[dict] = field(default_factory=list)
    value_hits: list[dict] = field(default_factory=list)
    business_context: str = ""
    prompt_text: str = ""
    matched_terms: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    join_paths: list[list[str]] = field(default_factory=list)
    inferred_biz_line: str = ""
    context_stats: dict = field(default_factory=dict)
    query_intent: dict = field(default_factory=dict)
    requested_fields: list[dict] = field(default_factory=list)
    entity_filters: list[dict] = field(default_factory=list)
    unresolved_entities: list[dict] = field(default_factory=list)
    rejected_terms: list[str] = field(default_factory=list)


class SchemaRetriever:
    """
    RAG 检索体系统一入口。

    使用方式:
        retriever = SchemaRetriever()
        retriever.initialize()
        result = retriever.retrieve("目前有多少活跃商户")
    """

    def __init__(self, connection_string: str | None = None):
        self.schema_loader = SchemaLoader(connection_string=connection_string)
        self.index_manager = IndexManager()
        self.formatter = SchemaFormatter()
        self.glossary_resolver: GlossaryResolver | None = None
        self.value_indexer: ValueIndexer | None = None

        self.searcher: HybridSearcher | None = None
        self.fewshot: FewShotSelector | None = None
        self.table_schemas: dict = {}
        self.config: AgentRuntimeConfig | None = None
        self.context_planner: SchemaContextPlanner | None = None
        self._initialized = False

    def initialize(self, config: AgentRuntimeConfig | None = None):
        """
        启动初始化：加载配置 -> 加载 Schema -> 构建索引 -> 初始化检索器。

        Args:
            config: AgentRuntimeConfig，为 None 时通过 AgentConfigLoader 自动加载
        """
        logger.info("=" * 60)
        logger.info("RAG 检索体系初始化开始")
        logger.info("=" * 60)

        # 加载配置
        if config is None:
            loader = AgentConfigLoader()
            config = loader.load()
            loader.print_config(config)
        self.config = config
        self.index_manager = IndexManager(agent_id=config.agent_id)

        # 先确认 Agent 已绑定数据源，避免待配置状态下加载大模型。
        schemas, glossary, enums, fewshot_examples = self.schema_loader.load_all(
            agent_id=config.agent_id
        )

        # 初始化 Embedding
        embedding = get_embedding(config.embedding_config)

        # 初始化 Reranker
        get_reranker(config.index_build_config)

        from src.retrieval.config import REBUILD_INDEX_ON_STARTUP

        if REBUILD_INDEX_ON_STARTUP:
            # 全量构建索引
            logger.info("开始全量构建索引 (REBUILD_INDEX_ON_STARTUP=true)...")
            indices = self.index_manager.build(
                schemas,
                embedding,
                enums,
                fewshot_examples,
                glossary,
                index_build_config=config.index_build_config,
            )
        else:
            # 复用已有 Collection
            logger.info("连接已有索引 (REBUILD_INDEX_ON_STARTUP=false)...")
            indices = self.index_manager.connect(
                schemas,
                embedding,
                fewshot_examples,
            )

        # 初始化术语解析器
        glossary_params = get_search_params(config.collection_search_config, "glossary")
        ef_search = config.index_build_config.get("hnsw", {}).get("ef_search", 64)
        self.glossary_resolver = GlossaryResolver(
            embedding,
            indices["glossary_index"],
            search_params=glossary_params,
            ef_search=ef_search,
        )

        # 初始化 Value 索引器
        self.value_indexer = indices["value_indexer"]

        # 初始化混合检索器
        self.searcher = HybridSearcher(
            embedding=embedding,
            table_index=indices["table_index"],
            column_index=indices["column_index"],
            enum_index=indices["enum_index"],
            table_schemas=indices["table_schemas"],
            config=config,
        )

        self.fewshot = indices["fewshot_selector"]
        self.table_schemas = indices["table_schemas"]
        self.context_planner = SchemaContextPlanner(self.table_schemas)
        self._initialized = True

        logger.info("=" * 60)
        logger.info(
            f"RAG 初始化完成: {len(self.table_schemas)} 张表, "
            f"{len(glossary)} 条术语, "
            f"Reranker={'启用' if config.enable_reranker else '关闭'}"
        )
        logger.info("=" * 60)

    def rebuild_partial(self, collections: list[str]):
        """按指定 collection 类型选择性重建索引并热替换。"""
        if not self._initialized:
            raise RuntimeError("Retriever 未初始化，请先调用 initialize()")

        embedding = get_embedding()
        schemas, glossary, enums, fewshot_examples = self.schema_loader.load_all(
            agent_id=self.config.agent_id
        )

        logger.info(f"开始局部索引重建: {collections}")
        indices = self.index_manager.rebuild_partial(
            collections=collections,
            schemas=schemas,
            embedding=embedding,
            enums=enums,
            fewshot_examples=fewshot_examples,
            glossary=glossary,
            index_build_config=self.config.index_build_config,
        )

        # 热替换受影响的组件
        if "table_index" in indices:
            self.searcher.table_index = indices["table_index"]
        if "column_index" in indices:
            self.searcher.column_index = indices["column_index"]
        if "table_schemas" in indices:
            self.searcher.table_schemas = indices["table_schemas"]
            self.searcher._rebuild_short_map()
            self.table_schemas = indices["table_schemas"]
            self.context_planner = SchemaContextPlanner(self.table_schemas)
        if "enum_index" in indices:
            self.searcher.enum_index = indices["enum_index"]
        if "value_indexer" in indices:
            self.value_indexer = indices["value_indexer"]
        if "fewshot_selector" in indices:
            self.fewshot = indices["fewshot_selector"]
        if "glossary_index" in indices:
            glossary_params = get_search_params(
                self.config.collection_search_config, "glossary"
            )
            ef_search = self.config.index_build_config.get("hnsw", {}).get(
                "ef_search", 64
            )
            self.glossary_resolver = GlossaryResolver(
                embedding,
                indices["glossary_index"],
                search_params=glossary_params,
                ef_search=ef_search,
            )

        rebuilt = ", ".join(collections)
        logger.info(f"局部索引重建完成: [{rebuilt}]")
        return len(self.table_schemas)

    def retrieve(
        self,
        user_query: str,
        top_k: int | None = None,
        fewshot_k: int | None = None,
        glossary_score_threshold: float | None = None,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
        requested_field_query: str | None = None,
    ) -> RetrievalResult:
        """
        完整 RAG 检索流程。

        流程:
            1. 业务术语解析 -> enriched_query + business_context
            2. Value 匹配 (Schema Linking)
            3. Schema 混合检索
            4. Reranker 精排（如启用）
            5. Reranker 后关联表补回
            6. 枚举值检索
            7. Few-shot 示例检索
            8. Prompt 组装

        Args:
            user_query: 用户原始自然语言问题
            top_k: 最终返回的表数量，None 时用 config 值
            fewshot_k: Few-shot 示例数量，None 时用 config 值
            glossary_score_threshold: 术语匹配阈值
            biz_line: 业务线过滤（如 "banking"、"issuing"），为空则不过滤

        Returns:
            RetrievalResult
        """
        if not self._initialized:
            raise RuntimeError("SchemaRetriever 未初始化，请先调用 initialize()")

        cfg = self.config
        if top_k is None:
            top_k = cfg.table_search_top_k
        if fewshot_k is None:
            fewshot_k = cfg.fewshot_top_k

        inferred_biz_line = ""
        if not biz_line and cfg.enable_domain_routing and self.context_planner:
            inferred_biz_line = self.context_planner.infer_biz_line(user_query)
            biz_line = inferred_biz_line or None

        if biz_line:
            logger.info(f"开始检索: '{user_query}' (biz_line={biz_line})")
        else:
            logger.info(f"开始检索: '{user_query}'")

        query_analysis = QueryAnalyzer.analyze(user_query)
        requested_field_analysis = QueryAnalyzer.analyze(
            requested_field_query or user_query
        )

        entity_result = EntityResolver(
            cfg.entity_resolution_rules,
            self.table_schemas,
        ).resolve(user_query, biz_line=biz_line)
        entity_filters = entity_result["filters"]
        entity_context = EntityResolver.to_prompt_context(entity_filters)

        # 1. 业务术语解析
        resolve_kwargs = {}
        if glossary_score_threshold is not None:
            resolve_kwargs["score_threshold"] = glossary_score_threshold
        glossary_result = self.glossary_resolver.resolve(
            user_query,
            biz_line=biz_line,
            metadata_filter=metadata_filter,
            require_lexical_grounding=cfg.glossary_require_lexical_grounding,
            **resolve_kwargs,
        )
        enriched_query = glossary_result["enriched_query"]
        business_context = glossary_result["business_context"]

        # 2. Value 匹配 (Schema Linking)
        value_hits = []
        if self.value_indexer:
            value_hits = self.value_indexer.match_values(
                user_query,
                biz_line=biz_line,
                metadata_filter=metadata_filter,
                exact_match_boost=cfg.value_exact_match_boost,
            )
        required_tables = {
            hit["table_name"] for hit in value_hits if hit.get("exact_match")
        }
        required_tables.update(glossary_result.get("related_tables", []))
        required_tables.update(self.searcher._find_explicit_tables(user_query))
        required_columns = set(glossary_result.get("related_columns", []))
        required_tables.update(item["table"] for item in entity_filters)
        required_columns.update(item["qualified_column"] for item in entity_filters)

        if query_analysis.requires_exchange_rate and cfg.exchange_rate_injection:
            if EXCHANGE_RATE_TABLE not in self.table_schemas:
                raise RequiredSemanticTableMissingError(
                    "查询需要货币汇率，但当前 Agent 未绑定可用的语义表 "
                    f"{EXCHANGE_RATE_TABLE}"
                )
            required_tables.add(EXCHANGE_RATE_TABLE)
            required_columns.update(EXCHANGE_RATE_REQUIRED_COLUMNS)

        # 3. Schema 混合检索
        table_params = get_search_params(cfg.collection_search_config, "table")
        rerank_k = (
            table_params.rerank_top_n
            if cfg.enable_reranker and table_params.rerank
            else top_k
        )
        candidates = self.searcher.search(
            enriched_query,
            top_k=max(rerank_k, top_k),
            biz_line=biz_line,
            metadata_filter=metadata_filter,
            pinned_rules=cfg.pinned_rules,
            required_tables=required_tables,
        )

        # 4. Reranker 精排
        reranker = get_reranker()
        pinned_candidates = [c for c in candidates if c.get("pinned")]
        if (
            reranker
            and cfg.enable_reranker
            and table_params.rerank
            and len(candidates) > top_k
        ):
            candidates = reranker.rerank(enriched_query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # 4b. 补回被 Reranker 砍掉的 pinned 表（如汇率表）
        hit_names = {c["table_name"] for c in candidates}
        for pc in pinned_candidates:
            if pc["table_name"] not in hit_names:
                candidates.append(pc)
                hit_names.add(pc["table_name"])
                logger.info(f"Reranker 后补回 pinned 表: {pc['table_name']}")

        # 5. 在关系图中搜索最多 N 跳最短路径，补齐中间桥接表。
        join_paths: list[list[str]] = []
        if self.context_planner:
            candidates, join_paths = self.context_planner.add_join_bridges(
                candidates,
                max_hops=cfg.max_relation_hops,
                max_tables=cfg.max_context_tables,
            )

        hit_tables = [c["table_name"] for c in candidates]
        hit_table_set = set(hit_tables)

        # 6. 枚举值检索
        enum_hits = self.searcher.search_enums(
            user_query,
            biz_line=biz_line,
            metadata_filter=metadata_filter,
            table_names=hit_table_set,
        )
        value_hits = [hit for hit in value_hits if hit["table_name"] in hit_table_set]

        # 7. Few-shot 示例检索
        examples = self.fewshot.select(
            query=user_query,
            tables=hit_tables,
            top_k=fewshot_k,
            metadata_filter=metadata_filter,
            biz_line=biz_line,
            search_params=get_search_params(cfg.collection_search_config, "fewshot"),
            mmr_lambda=cfg.mmr_lambda,
        )

        # 8. 字段级上下文规划：只保留问题相关字段，主键/Join/口径字段始终保留。
        context_stats = {}
        requested_fields: list[dict] = []
        if self.context_planner:
            requested_fields = self.context_planner.resolve_requested_columns(
                candidates,
                requested_field_analysis.requested_fields,
                query=user_query,
            )
            for requirement in requested_fields:
                required_columns.update(requirement["columns"])
            candidates, context_stats = self.context_planner.prune_columns(
                candidates,
                enriched_query,
                required_columns=required_columns,
                per_table_limit=cfg.max_columns_per_table,
                preserve_time_columns=query_analysis.has_time_filter,
            )

        # 9. Prompt 组装
        intent_context = query_analysis.to_prompt_context()
        if entity_context:
            intent_context = "\n\n".join(
                part for part in (intent_context, entity_context) if part
            )
        prompt_text = self.formatter.format_all(
            tables=candidates,
            examples=examples,
            business_context=business_context,
            enum_hits=enum_hits,
            value_hits=value_hits,
            question=user_query,
            output_rules=self.config.output_rules if self.config else "",
            intent_context=intent_context,
        )
        context_stats["prompt_chars"] = len(prompt_text)
        context_stats["prompt_tokens_estimate"] = max(1, len(prompt_text) // 3)

        result = RetrievalResult(
            relevant_tables=candidates,
            relevant_examples=examples,
            enum_hits=enum_hits,
            value_hits=value_hits,
            business_context=business_context,
            prompt_text=prompt_text,
            matched_terms=glossary_result["matched_terms"],
            required_columns=sorted(required_columns),
            join_paths=join_paths,
            inferred_biz_line=inferred_biz_line,
            context_stats=context_stats,
            query_intent=query_analysis.to_dict(),
            requested_fields=requested_fields,
            entity_filters=entity_filters,
            unresolved_entities=entity_result["unresolved"],
            rejected_terms=glossary_result.get("rejected_terms", []),
        )

        logger.info(
            f"检索完成: tables={hit_tables}, values={len(value_hits)}, "
            f"enums={len(enum_hits)}, examples={len(examples)}, terms={result.matched_terms}, "
            f"entity_filters={len(entity_filters)}"
        )
        return result
