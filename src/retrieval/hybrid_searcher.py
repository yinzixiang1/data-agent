"""
混合检索 — Dense + BM25 hybrid search，表级 + 列级 + 枚举联合。

检索流程:
    1. 表级混合检索（Dense + BM25，per-collection ranker）
    2. 列级混合检索 -> 命中的列反推其所属表，给表加分
    3. 枚举值检索 -> 命中的枚举反哺其关联表分数
    4. 关联表补全 -> top 表的 relations 中的关联表获得 bonus 分

使用示例::

    searcher = HybridSearcher(embedding, table_index, column_index,
                              enum_index, table_schemas, config)

    results = searcher.search("活跃商户数量", top_k=5)
    enums = searcher.search_enums("持牌商户")
"""

import logging

from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.milvus_filter import build_metadata_filter
from src.retrieval.ranker_strategy import get_search_params
from src.retrieval.agent_config import AgentRuntimeConfig

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器：表级 + 列级 + 枚举联合检索，多信号融合排序。

    Attributes:
        embedding: Qwen3Embedding 实例
        table_index: 表级 MilvusIndex
        column_index: 列级 MilvusIndex
        enum_index: 枚举值 MilvusIndex
        table_schemas: {table_name: schema_dict} 映射
        config: AgentRuntimeConfig（提供 collection_search_config 和 index_build_config）
    """

    def __init__(
        self,
        embedding: Qwen3Embedding,
        table_index: MilvusIndex,
        column_index: MilvusIndex,
        enum_index: MilvusIndex,
        table_schemas: dict,
        config: AgentRuntimeConfig,
    ):
        self.embedding = embedding
        self.table_index = table_index
        self.column_index = column_index
        self.enum_index = enum_index
        self.table_schemas = table_schemas
        self.config = config
        self._rebuild_short_map()

    def _rebuild_short_map(self):
        """构建 short_name -> full_name 反查映射（用于关联表解析）。"""
        self._short_to_full: dict[str, str] = {}
        for full_name in self.table_schemas:
            short = full_name.split(".", 1)[1] if "." in full_name else full_name
            self._short_to_full[short] = full_name

    def _resolve_full_name(self, target: str) -> str | None:
        """将关联表名（可能是短名或全名）解析为 table_schemas 中的全限定名。"""
        if target in self.table_schemas:
            return target
        return self._short_to_full.get(target)

    @property
    def _ef_search(self) -> int:
        return self.config.index_build_config.get("hnsw", {}).get("ef_search", 64)

    @staticmethod
    def _build_filter(
        biz_line: str = "", metadata_filter: dict | None = None
    ) -> str | None:
        """构建 Milvus 过滤表达式，支持 biz_line + 任意 metadata KV 组合过滤。"""
        return build_metadata_filter(biz_line, metadata_filter)

    def search(
        self,
        query: str,
        top_k: int = 5,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
        pinned_rules: list[dict] | None = None,
    ) -> list[dict]:
        """
        表级 + 列级混合检索，融合枚举反哺、关联表补全和强制召回规则。

        Args:
            query: 用户查询（可能已经过术语增强）
            top_k: 最终返回的表数量
            biz_line: 业务线过滤，为空则不过滤
            metadata_filter: 任意 KV 过滤
            pinned_rules: 强制召回规则列表（从 Agent 配置读取）

        Returns:
            list[dict]: 按 score 降序，每个元素含 table_name, score, source,
                hit_by_column, pinned, schema
        """
        csc = self.config.collection_search_config
        filter_expr = self._build_filter(biz_line or "", metadata_filter)

        # ── 表级混合检索 ──
        table_params = get_search_params(csc, "table")
        q_dense_table = self.embedding.encode_query(query, "table")
        table_results = self.table_index.hybrid_search(
            q_dense_table,
            query,
            ranker=table_params.ranker,
            recall_k=table_params.recall_limit,
            output_fields=["table_name"],
            filter_expr=filter_expr,
            ef_search=self._ef_search,
        )

        # ── 列级混合检索 -> 反推表 ──
        column_hit_tables: set[str] = set()
        if self.column_index.count > 0:
            col_params = get_search_params(csc, "column")
            q_dense_col = self.embedding.encode_query(query, "column")
            col_results = self.column_index.hybrid_search(
                q_dense_col,
                query,
                ranker=col_params.ranker,
                recall_k=col_params.recall_limit,
                output_fields=["table_name"],
                filter_expr=filter_expr,
                ef_search=self._ef_search,
            )
            for doc_id, score, entity in col_results:
                column_hit_tables.add(entity["table_name"])

        # ── 合并：表级 + 列级反推 bonus ──
        table_scores: dict[str, float] = {}
        for doc_id, score, entity in table_results:
            table_name = entity["table_name"]
            table_scores[table_name] = max(table_scores.get(table_name, 0), score)

        for table_name in column_hit_tables:
            if table_name in table_scores:
                table_scores[table_name] += 0.01
            else:
                table_scores[table_name] = 0.01

        # ── 枚举命中反哺表分数 ──
        enum_boost_tables: set[str] = set()
        if self.enum_index.count > 0:
            enum_params = get_search_params(csc, "enum")
            q_dense_enum = self.embedding.encode_query(query, "enum")
            enum_results = self.enum_index.hybrid_search(
                q_dense_enum,
                query,
                ranker=enum_params.ranker,
                recall_k=enum_params.recall_limit,
                output_fields=["table_name"],
                filter_expr=filter_expr,
                ef_search=self._ef_search,
            )
            for doc_id, score, entity in enum_results:
                enum_boost_tables.add(entity["table_name"])
            for table_name in enum_boost_tables:
                if table_name in self.table_schemas:
                    table_scores.setdefault(table_name, 0)
                    table_scores[table_name] += 0.02
            if enum_boost_tables:
                logger.info(f"枚举反哺: {enum_boost_tables}")

        # ── 关联表补全 ──
        current_top = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        relation_boosted: set[str] = set()
        for table_name, score in current_top:
            schema = self.table_schemas.get(table_name, {})
            for rel in schema.get("relations", []):
                target_short = rel.get("target_table", "")
                if not target_short:
                    continue
                related = self._resolve_full_name(target_short)
                if not related or related == table_name:
                    continue
                bonus = score * 0.1
                table_scores.setdefault(related, 0)
                table_scores[related] += bonus
                relation_boosted.add(related)
        if relation_boosted:
            logger.info(f"关联补全: {relation_boosted}")

        # ── 强制召回规则 ──
        pinned_tables = self._apply_pinned_rules(
            query, table_scores, pinned_rules or []
        )

        # 按分数排序，pinned 表不受 top_k 截断
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        top_names = {t[0] for t in sorted_tables}
        for pt in pinned_tables:
            if pt not in top_names:
                sorted_tables.append((pt, table_scores[pt]))

        # 组装结果
        pinned_set = set(pinned_tables)
        results = []
        for table_name, score in sorted_tables:
            schema = self.table_schemas.get(table_name, {})
            results.append(
                {
                    "table_name": table_name,
                    "score": score,
                    "source": "hybrid",
                    "hit_by_column": table_name in column_hit_tables,
                    "pinned": table_name in pinned_set,
                    "schema": schema,
                }
            )

        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"表级候选={len(table_results)}, 列级反推={len(column_hit_tables)}, "
            f"枚举反哺={len(enum_boost_tables)}, 关联补全={len(relation_boosted)}, "
            f"最终={len(results)} 张表: {[r['table_name'] for r in results]}"
        )
        return results

    def _apply_pinned_rules(
        self,
        query: str,
        table_scores: dict[str, float],
        rules: list[dict],
    ) -> list[str]:
        """根据可配置规则检测意图，命中时将指定表强制注入候选。

        规则触发逻辑（OR 关系）:
            路径1: keywords 中任一词出现在 query 中 → 直接触发
            路径2: entities 中任一词出现在 query 中 且 entity_keywords 中任一词也出现 → 组合触发

        Args:
            query: 用户查询文本
            table_scores: 当前表分数字典（就地修改）
            rules: 规则列表，每条含 name, table, keywords, entities, entity_keywords, min_score_ratio

        Returns:
            命中的 pinned 表名列表
        """
        pinned: list[str] = []
        query_upper = query.upper()

        for rule in rules:
            table = rule.get("table", "")
            if not table or table not in self.table_schemas:
                continue

            # 路径1: keywords 直接命中
            keywords = rule.get("keywords", [])
            direct_hit = (
                any(kw.upper() in query_upper for kw in keywords) if keywords else False
            )

            # 路径2: entities + entity_keywords 组合命中
            combo_hit = False
            if not direct_hit:
                entities = rule.get("entities", [])
                entity_keywords = rule.get("entity_keywords", [])
                has_entity = (
                    any(e.upper() in query_upper for e in entities)
                    if entities
                    else False
                )
                has_keyword = (
                    any(kw in query for kw in entity_keywords)
                    if entity_keywords
                    else False
                )
                combo_hit = has_entity and has_keyword

            if not direct_hit and not combo_hit:
                continue

            # 命中：保底分注入
            ratio = rule.get("min_score_ratio", 0.8)
            max_score = max(table_scores.values()) if table_scores else 0.5
            min_score = max_score * ratio
            old_score = table_scores.get(table, 0)
            table_scores[table] = max(old_score, min_score)
            pinned.append(table)

            trigger = "关键词" if direct_hit else "实体+动词"
            logger.info(
                f"强制召回[{rule.get('name', table)}]({trigger}): "
                f"{table} score={old_score:.4f}->{table_scores[table]:.4f}"
            )

        return pinned

    def search_enums(
        self,
        query: str,
        top_k: int = 8,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        枚举值检索 — 将用户自然语言映射到实际枚举值。

        Args:
            query: 用户原始查询
            top_k: 返回的最大枚举命中数
            biz_line: 业务线过滤，为空则不过滤
            metadata_filter: 任意 KV 过滤

        Returns:
            list[dict]: table_name, column_name, enum_label_cn, sql_value, score
        """
        if self.enum_index.count == 0:
            return []

        enum_params = get_search_params(self.config.collection_search_config, "enum")
        q_dense = self.embedding.encode_query(query, "enum")
        filter_expr = self._build_filter(biz_line or "", metadata_filter)

        results = self.enum_index.hybrid_search(
            q_dense,
            query,
            ranker=enum_params.ranker,
            recall_k=max(top_k, enum_params.recall_limit),
            output_fields=[
                "table_name",
                "column_name",
                "enum_label_cn",
                "sql_value",
                "enum_code",
            ],
            filter_expr=filter_expr,
            ef_search=self._ef_search,
        )

        enum_hits = []
        for doc_id, score, entity in results[:top_k]:
            enum_hits.append(
                {
                    "table_name": entity["table_name"],
                    "column_name": entity["column_name"],
                    "enum_label_cn": entity["enum_label_cn"],
                    "sql_value": entity["sql_value"],
                    "score": score,
                }
            )

        if enum_hits:
            top_hits = [
                f"{item['enum_label_cn']}->{item['column_name']}={item['sql_value']}"
                for item in enum_hits[:3]
            ]
            logger.info(f"枚举检索: {len(enum_hits)} 条命中, top={top_hits}")
        return enum_hits
