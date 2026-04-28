"""混合检索 — Dense + Sparse hybrid search，表级 + 列级 + 枚举联合"""

import logging

from src.retrieval.config import RECALL_TOP_K
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BGEEmbedding

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器：
    1. 表级 + 列级 hybrid search → RRF 融合
    2. 枚举值检索 → 将业务术语映射到 SQL 条件
    """

    def __init__(
        self,
        embedding: BGEEmbedding,
        table_index: MilvusIndex,
        column_index: MilvusIndex,
        enum_index: MilvusIndex,
        table_schemas: dict,
    ):
        self.embedding = embedding
        self.table_index = table_index
        self.column_index = column_index
        self.enum_index = enum_index
        self.table_schemas = table_schemas

    def search(self, query: str, top_k: int = 5, recall_k: int = RECALL_TOP_K) -> list[dict]:
        """
        表级 + 列级混合检索。

        Returns:
            [{"table_name", "score", "source", "hit_by_column", "schema"}, ...]
        """
        q_output = self.embedding.encode_query(query)
        q_dense = q_output["dense_vecs"]
        q_sparse = q_output["lexical_weights"][0]

        # ── 表级混合检索 ──
        table_results = self.table_index.hybrid_search(
            q_dense, q_sparse, top_k=recall_k, recall_k=recall_k,
            output_fields=["table_name"],
        )

        # ── 列级混合检索 → 反推表 ──
        column_hit_tables: set[str] = set()
        if self.column_index.count > 0:
            col_results = self.column_index.hybrid_search(
                q_dense, q_sparse, top_k=recall_k, recall_k=recall_k,
                output_fields=["table_name"],
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
            enum_results = self.enum_index.hybrid_search(
                q_dense, q_sparse, top_k=8, recall_k=8,
                output_fields=["table_name"],
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
        current_top = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        relation_boosted: set[str] = set()
        for table_name, score in current_top:
            schema = self.table_schemas.get(table_name, {})
            for rel in schema.get("relations", []):
                related = rel.get("target_table", "")
                if not related or related not in self.table_schemas or related == table_name:
                    continue
                bonus = score * 0.1
                table_scores.setdefault(related, 0)
                table_scores[related] += bonus
                relation_boosted.add(related)
        if relation_boosted:
            logger.info(f"关联补全: {relation_boosted}")

        # 按分数排序
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 组装结果
        results = []
        for table_name, score in sorted_tables:
            schema = self.table_schemas.get(table_name, {})
            results.append({
                "table_name": table_name,
                "score": score,
                "source": "hybrid",
                "hit_by_column": table_name in column_hit_tables,
                "schema": schema,
            })

        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"表级候选={len(table_results)}, 列级反推={len(column_hit_tables)}, "
            f"枚举反哺={len(enum_boost_tables)}, 关联补全={len(relation_boosted)}, "
            f"最终={len(results)} 张表: {[r['table_name'] for r in results]}"
        )
        return results

    def search_enums(self, query: str, top_k: int = 8) -> list[dict]:
        """
        枚举值检索 — 将用户自然语言映射到实际枚举值。

        Returns:
            [{"table_name", "column_name", "enum_label_cn", "sql_value", "score"}, ...]
        """
        if self.enum_index.count == 0:
            return []

        q_output = self.embedding.encode_query(query)
        q_dense = q_output["dense_vecs"]
        q_sparse = q_output["lexical_weights"][0]

        results = self.enum_index.hybrid_search(
            q_dense, q_sparse, top_k=top_k, recall_k=top_k,
            output_fields=["table_name", "column_name", "enum_label_cn", "sql_value", "enum_code"],
        )

        enum_hits = []
        for doc_id, score, entity in results:
            enum_hits.append({
                "table_name": entity["table_name"],
                "column_name": entity["column_name"],
                "enum_label_cn": entity["enum_label_cn"],
                "sql_value": entity["sql_value"],
                "score": score,
            })

        if enum_hits:
            logger.info(
                f"枚举检索: {len(enum_hits)} 条命中, "
                f"top={[f'{e['enum_label_cn']}→{e['column_name']}={e['sql_value']}' for e in enum_hits[:3]]}"
            )
        return enum_hits
