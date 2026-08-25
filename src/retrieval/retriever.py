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

import json
import logging
from dataclasses import dataclass, field

from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.embedding import get_embedding
from src.retrieval.entity_resolver import EntityResolver
from src.retrieval.fewshot_selector import FewShotSelector
from src.retrieval.glossary_resolver import GlossaryResolver
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.index_manager import IndexManager
from src.retrieval.ranker_strategy import get_search_params
from src.retrieval.reranker import get_reranker
from src.retrieval.schema_formatter import SchemaFormatter
from src.retrieval.schema_loader import SchemaLoader
from src.retrieval.value_indexer import ValueIndexer

logger = logging.getLogger(__name__)


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
    semantic_table_evidence: list[dict] = field(default_factory=list)


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

    def rebuild_all(self) -> int:
        """Rebuild every persisted collection from the current source of truth."""
        return self.rebuild_partial(["table", "enum", "value", "fewshot", "glossary"])

    @staticmethod
    def _drop_weak_table_candidates(
        candidates: list[dict],
        score_threshold: float | None,
    ) -> tuple[list[dict], list[str]]:
        """Drop reranked tables that have neither score nor column evidence."""
        if score_threshold is None:
            return candidates, []
        kept: list[dict] = []
        dropped: list[str] = []
        for candidate in candidates:
            rerank_score = candidate.get("rerank_score")
            is_weak = (
                rerank_score is not None
                and float(rerank_score) < score_threshold
                and not candidate.get("hit_by_column")
                and not candidate.get("pinned")
            )
            if is_weak:
                dropped.append(str(candidate.get("table_name") or ""))
            else:
                kept.append(candidate)
        return kept, dropped

    @staticmethod
    def _filter_enums_by_selected_columns(
        enum_hits: list[dict],
        candidates: list[dict],
    ) -> tuple[list[dict], int]:
        """Keep enum evidence only when its owning column remains in context."""
        selected_by_table = {
            str(candidate.get("table_name") or ""): {
                str(column).casefold()
                for column in candidate.get("selected_columns", [])
                if column
            }
            for candidate in candidates
        }
        filtered = [
            hit
            for hit in enum_hits
            if str(hit.get("column_name") or "").casefold()
            in selected_by_table.get(str(hit.get("table_name") or ""), set())
        ]
        return filtered, len(enum_hits) - len(filtered)

    def retrieve(
        self,
        user_query: str,
        top_k: int | None = None,
        fewshot_k: int | None = None,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
        query_state: dict | None = None,
        original_query: str | None = None,
        inherited_tables: set[str] | None = None,
        inherited_columns: set[str] | None = None,
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
            biz_line: 业务线过滤（如 "banking"、"issuing"），为空则不过滤
            original_query: 本轮用户原话，用于防止改写丢失受控术语
            inherited_tables: 上一轮已验证 SQL 中仍适用的表
            inherited_columns: 上一轮已验证 SQL 中仍适用的列

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

        state = query_state if isinstance(query_state, dict) else {}

        entity_result = EntityResolver(
            cfg.entity_resolution_rules,
            self.table_schemas,
        ).resolve(user_query, biz_line=biz_line)
        entity_filters = entity_result["filters"]
        entity_context = EntityResolver.to_prompt_context(entity_filters)

        # 1. 业务术语解析。多轮改写可能把本轮原话中的受控术语、枚举
        # 或 ID 换成自然语言表达，因此同时保留改写后的完整问题与原话。
        grounding_query = user_query
        if original_query and original_query.strip() != user_query.strip():
            grounding_query = f"{user_query}\n本轮用户原话：{original_query.strip()}"
        glossary_result = self.glossary_resolver.resolve(
            grounding_query,
            biz_line=biz_line,
            metadata_filter=metadata_filter,
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
        required_tables: set[str] = set()
        required_tables.update(glossary_result.get("related_tables", []))
        required_tables.update(self.searcher._find_explicit_tables(user_query))
        required_tables.update(inherited_tables or set())
        required_columns = set(glossary_result.get("related_columns", []))
        required_columns.update(inherited_columns or set())
        required_tables.update(item["table"] for item in entity_filters)
        required_columns.update(item["qualified_column"] for item in entity_filters)

        # An exact value such as LOCAL/SWIFT can exist on several unrelated
        # tables.  Treat it as column evidence only after stronger signals
        # (glossary, explicit table, prior SQL, entity link)
        # have established the owning table; never let a shared enum value pin
        # every table that happens to contain it.
        grounded_required_tables = {
            variant
            for table in required_tables
            for variant in {str(table), str(table).rsplit(".", 1)[-1]}
        }
        required_columns.update(
            f"{hit['table_name']}.{hit['column_name']}"
            for hit in value_hits
            if hit.get("exact_match")
            and {
                str(hit.get("table_name") or ""),
                str(hit.get("table_name") or "").rsplit(".", 1)[-1],
            }
            & grounded_required_tables
            and hit.get("column_name")
        )

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

        candidates, dropped_tables = self._drop_weak_table_candidates(
            candidates,
            reranker.score_threshold if reranker and cfg.enable_reranker else None,
        )
        if dropped_tables:
            logger.info(
                "Low-evidence table candidates removed",
                extra={
                    "dropped_count": len(dropped_tables),
                    "dropped_tables": dropped_tables,
                },
            )

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

        # QueryState 中的展示维度是用户明确要求输出的字段。将它们
        # 映射到已召回 Schema，同时用于字段裁剪保护和最终 SELECT 校验，
        # 避免“召回了表却裁掉用户要的列”。
        requested_fields: list[dict] = []
        if self.context_planner:
            requested_fields = self.context_planner.resolve_requested_columns(
                candidates,
                [str(value) for value in state.get("dimensions", []) if value],
                query=user_query,
            )
            required_columns.update(
                column
                for requirement in requested_fields
                for column in requirement.get("columns", [])
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
        if self.context_planner:
            candidates, context_stats = self.context_planner.prune_columns(
                candidates,
                enriched_query,
                required_columns=required_columns,
                per_table_limit=cfg.max_columns_per_table,
                preserve_time_columns=bool(state.get("time_range")),
            )
            enum_hits, dropped_enum_count = self._filter_enums_by_selected_columns(
                enum_hits,
                candidates,
            )
            context_stats["weak_tables_dropped"] = len(dropped_tables)
            context_stats["unrelated_enums_dropped"] = dropped_enum_count

        # 9. Prompt 组装
        intent_context = ""
        if state:
            intent_context = (
                "以下是模型合并得到的粗粒度查询摘要，仅用于辅助理解；"
                "完整需求以【用户问题】为准：\n"
                + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            )
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
            query_intent={"state": state},
            requested_fields=requested_fields,
            entity_filters=entity_filters,
            unresolved_entities=entity_result["unresolved"],
            rejected_terms=glossary_result.get("rejected_terms", []),
            semantic_table_evidence=glossary_result.get("table_evidence", []),
        )

        logger.info(
            f"检索完成: tables={hit_tables}, values={len(value_hits)}, "
            f"enums={len(enum_hits)}, examples={len(examples)}, terms={result.matched_terms}, "
            f"entity_filters={len(entity_filters)}"
        )
        return result
