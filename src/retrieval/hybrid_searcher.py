"""混合检索 — Milvus Dense + Sparse hybrid search，表级 + 列级联合"""

import json
import logging

from src.retrieval.config import RECALL_TOP_K
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BGEEmbedding

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器（Milvus 版）：
    1. Milvus hybrid search: Dense + Sparse → RRF 融合（Milvus 内置）
    2. 表级 + 列级联合，列级命中反推表
    """

    def __init__(
        self,
        embedding: BGEEmbedding,
        table_index: MilvusIndex,
        column_index: MilvusIndex,
        table_docs: list[dict],
        column_docs: list[dict],
    ):
        self.embedding = embedding
        self.table_index = table_index
        self.column_index = column_index
        self.table_docs = table_docs
        self.column_docs = column_docs

    def search(self, query: str, top_k: int = 5, recall_k: int = RECALL_TOP_K) -> list[dict]:
        """
        完整混合检索流程。

        Returns:
            [{"table_name", "score", "source", "doc", "schema"}, ...]
        """
        # 编码 query
        q_output = self.embedding.encode_query(query)
        q_dense = q_output["dense_vecs"]  # (1, 1024)
        q_sparse = q_output["lexical_weights"][0]  # dict

        # ── 表级混合检索（Milvus RRF） ──
        table_results = self.table_index.hybrid_search(
            q_dense, q_sparse, top_k=recall_k, recall_k=recall_k,
        )

        # ── 列级混合检索（Milvus RRF）→ 反推表 ──
        column_hit_tables: set[str] = set()
        if self.column_index.count > 0:
            col_results = self.column_index.hybrid_search(
                q_dense, q_sparse, top_k=recall_k, recall_k=recall_k,
            )
            for doc_id, score, doc_json in col_results:
                doc = json.loads(doc_json)
                column_hit_tables.add(doc["table_name"])

        # ── 合并：表级 + 列级反推 bonus ──
        table_scores: dict[str, float] = {}
        table_doc_map: dict[str, dict] = {}

        for doc_id, score, doc_json in table_results:
            doc = json.loads(doc_json)
            table_name = doc["table_name"]
            table_scores[table_name] = max(table_scores.get(table_name, 0), score)
            if table_name not in table_doc_map:
                table_doc_map[table_name] = doc

        # 列级反推表 bonus
        for table_name in column_hit_tables:
            if table_name in table_scores:
                table_scores[table_name] += 0.01
            else:
                table_scores[table_name] = 0.01

        # 按分数排序
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 组装结果（从 table_docs 获取完整 schema）
        full_doc_map = {doc["table_name"]: doc for doc in self.table_docs}

        results = []
        for table_name, score in sorted_tables:
            doc = full_doc_map.get(table_name, table_doc_map.get(table_name, {}))
            results.append({
                "table_name": table_name,
                "score": score,
                "source": "hybrid",
                "hit_by_column": table_name in column_hit_tables,
                "doc": doc,
                "schema": doc.get("schema", {}),
            })

        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"表级候选={len(table_results)}, 列级反推={len(column_hit_tables)}, "
            f"最终={len(results)} 张表: {[r['table_name'] for r in results]}"
        )
        return results
