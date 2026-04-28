"""
混合检索 — Dense + Sparse hybrid search，表级 + 列级 + 枚举联合。

检索流程:
    1. 表级混合检索（Dense + Sparse → RRF）
    2. 列级混合检索 → 命中的列反推其所属表，给表加分
    3. 枚举值检索 → 命中的枚举反哺其关联表分数
    4. 关联表补全 → top 表的 relations 中的关联表获得 bonus 分

使用示例::

    searcher = HybridSearcher(embedding, table_index, column_index, enum_index, table_schemas)

    # 表级检索
    results = searcher.search("活跃商户数量", top_k=5)
    # results: [{"table_name": "pmt_account", "score": 0.85, "schema": {...}}, ...]

    # 枚举值检索
    enums = searcher.search_enums("持牌商户")
    # enums: [{"table_name": "pmt_account", "column_name": "account_type",
    #          "enum_label_cn": "LPSP", "sql_value": "2000", "score": 0.9}, ...]
"""

import logging

from src.retrieval.config import RECALL_TOP_K
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BGEEmbedding

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器：表级 + 列级 + 枚举联合检索，多信号融合排序。

    Attributes:
        embedding: BGEEmbedding 实例
        table_index: 表级 MilvusIndex
        column_index: 列级 MilvusIndex
        enum_index: 枚举值 MilvusIndex
        table_schemas: {table_name: schema_dict} 映射
    """

    def __init__(
        self,
        embedding: BGEEmbedding,
        table_index: MilvusIndex,
        column_index: MilvusIndex,
        enum_index: MilvusIndex,
        table_schemas: dict,
    ):
        """
        Args:
            embedding: BGEEmbedding 实例，用于编码查询
            table_index: 表级 Collection 的 MilvusIndex
            column_index: 列级 Collection 的 MilvusIndex
            enum_index: 枚举值 Collection 的 MilvusIndex
            table_schemas: {table_name: schema_dict} 映射，用于读取关联表信息
        """
        self.embedding = embedding
        self.table_index = table_index
        self.column_index = column_index
        self.enum_index = enum_index
        self.table_schemas = table_schemas

    def search(self, query: str, top_k: int = 5, recall_k: int = RECALL_TOP_K) -> list[dict]:
        """
        表级 + 列级混合检索，融合枚举反哺和关联表补全。

        评分规则:
            - 表级检索: RRF 基础分
            - 列级命中: +0.01（列所属的表）
            - 枚举反哺: +0.02（枚举值关联的表）
            - 关联补全: +parent_score × 0.1（top 表的关联表）

        Args:
            query: 用户查询（可能已经过术语增强），如 "活跃商户 account_status is_delete"
            top_k: 最终返回的表数量
            recall_k: Dense/Sparse 各自的召回数量

        Returns:
            list[dict]: 按 score 降序排列，每个元素包含:
                - "table_name" (str): 表名
                - "score" (float): 综合得分
                - "source" (str): 固定为 "hybrid"
                - "hit_by_column" (bool): 是否被列级检索命中
                - "schema" (dict): 完整表 Schema
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

        用途: 当用户说 "持牌商户" 时，检索到 account_type=2000，
        注入 Prompt 帮助 LLM 生成正确的 WHERE 条件。

        Args:
            query: 用户原始查询
            top_k: 返回的最大枚举命中数

        Returns:
            list[dict]: 每条包含:
                - "table_name" (str): 关联表名
                - "column_name" (str): 字段名
                - "enum_label_cn" (str): 枚举中文标签
                - "sql_value" (str): SQL 中使用的实际值
                - "score" (float): 检索相关度分数
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
