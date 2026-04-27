"""Milvus 向量存储 — 替换 FAISS DenseIndex + 自建 SparseIndex"""

import logging
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from scipy.sparse import csr_array
import numpy as np

from src.retrieval.config import MILVUS_URI, MILVUS_DB, RRF_K

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI, db_name="default")
        # 确保数据库存在
        dbs = _client.list_databases()
        if MILVUS_DB not in dbs:
            _client.create_database(MILVUS_DB)
            logger.info(f"创建 Milvus 数据库: {MILVUS_DB}")
        _client.using_database(MILVUS_DB)
        logger.info(f"Milvus 连接成功: {MILVUS_URI}, db={MILVUS_DB}")
    return _client


def _to_sparse_vectors(lexical_weights_list: list[dict]) -> list:
    """将 BGE-M3 lexical_weights 转为 Milvus sparse vector (csr_array)"""
    sparse_vecs = []
    for weights in lexical_weights_list:
        if not weights:
            sparse_vecs.append(csr_array((1, 1)))
            continue
        indices = [int(k) for k in weights.keys()]
        values = [float(v) for v in weights.values()]
        max_idx = max(indices) + 1
        row = csr_array(
            (values, ([0] * len(indices), indices)),
            shape=(1, max_idx),
        )
        sparse_vecs.append(row)
    return sparse_vecs


class MilvusIndex:
    """
    Milvus 混合索引：Dense + Sparse 存储在同一个 Collection 中。
    支持 hybrid search (Dense + Sparse → RRF 融合)。
    """

    def __init__(self, collection_name: str, dim: int = 1024):
        self.collection_name = collection_name
        self.dim = dim
        self.client = get_milvus_client()

    def exists(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def drop(self):
        if self.exists():
            self.client.drop_collection(self.collection_name)
            logger.info(f"已删除 Collection: {self.collection_name}")

    def create(self):
        """创建 Collection: id + dense_vec + sparse_vec + doc_json"""
        if self.exists():
            self.drop()

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense_vec", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vec", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("doc_json", DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vec", index_type="FLAT", metric_type="IP")
        index_params.add_index(field_name="sparse_vec", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"创建 Collection: {self.collection_name}")

    def insert(
        self,
        dense_vecs: np.ndarray,
        lexical_weights_list: list[dict],
        doc_jsons: list[str],
    ):
        """批量插入 Dense + Sparse 向量"""
        n = len(doc_jsons)
        # 归一化 dense
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        dense_normed = (dense_vecs / norms).astype(np.float32)

        sparse_vecs = _to_sparse_vectors(lexical_weights_list)

        data = [
            {"id": i, "dense_vec": dense_normed[i].tolist(), "sparse_vec": sparse_vecs[i], "doc_json": doc_jsons[i]}
            for i in range(n)
        ]
        self.client.insert(self.collection_name, data)
        logger.info(f"插入 {n} 条到 {self.collection_name}")

    def hybrid_search(
        self,
        query_dense: np.ndarray,
        query_sparse_weights: dict,
        top_k: int = 20,
        recall_k: int = 20,
    ) -> list[tuple[int, float, str]]:
        """
        混合检索: Dense + Sparse → RRF 融合。

        Returns:
            [(doc_id, score, doc_json), ...]
        """
        # 归一化 query dense
        q = query_dense.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # 构建 sparse query
        q_sparse = _to_sparse_vectors([query_sparse_weights])

        # 两路 ANN 请求
        dense_req = AnnSearchRequest(
            data=q.tolist(),
            anns_field="dense_vec",
            param={"metric_type": "IP"},
            limit=recall_k,
        )
        sparse_req = AnnSearchRequest(
            data=q_sparse,
            anns_field="sparse_vec",
            param={"metric_type": "IP"},
            limit=recall_k,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=RRF_K),
            limit=top_k,
            output_fields=["doc_json"],
        )

        output = []
        for hit in results[0]:
            output.append((
                hit["id"],
                hit["distance"],
                hit["entity"]["doc_json"],
            ))
        return output

    def dense_search(
        self,
        query_dense: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float, str]]:
        """纯 Dense 检索（用于 Few-shot 选择）"""
        q = query_dense.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        results = self.client.search(
            collection_name=self.collection_name,
            data=q.tolist(),
            anns_field="dense_vec",
            limit=top_k,
            output_fields=["doc_json"],
            search_params={"metric_type": "IP"},
        )

        output = []
        for hit in results[0]:
            output.append((
                hit["id"],
                hit["distance"],
                hit["entity"]["doc_json"],
            ))
        return output

    def query_all(self) -> list[dict]:
        """查询所有文档（按 id 排序），返回 [{"id": int, "doc_json": str}, ...]"""
        if not self.exists():
            return []
        results = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["doc_json"],
            limit=10000,
        )
        results.sort(key=lambda x: x["id"])
        return results

    @property
    def count(self) -> int:
        if not self.exists():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)


class MilvusMetaStore:
    """Milvus 元数据存储 — 用于存储 schema_hash 等非向量键值数据"""

    COLLECTION = "nl2sql_metadata"

    def __init__(self):
        self.client = get_milvus_client()

    def _ensure_collection(self):
        if not self.client.has_collection(self.COLLECTION):
            schema = self.client.create_schema(auto_id=False)
            schema.add_field("key", DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field("value", DataType.VARCHAR, max_length=65535)
            # Milvus 要求至少一个向量字段
            schema.add_field("_dummy_vec", DataType.FLOAT_VECTOR, dim=2)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="_dummy_vec", index_type="FLAT", metric_type="L2")

            self.client.create_collection(
                collection_name=self.COLLECTION,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"创建元数据 Collection: {self.COLLECTION}")

    def set(self, key: str, value: str):
        """设置元数据（upsert 语义）"""
        self._ensure_collection()
        self.client.upsert(self.COLLECTION, [{"key": key, "value": value, "_dummy_vec": [0.0, 0.0]}])

    def get(self, key: str) -> str | None:
        """获取元数据"""
        if not self.client.has_collection(self.COLLECTION):
            return None
        results = self.client.query(
            self.COLLECTION,
            filter=f'key == "{key}"',
            output_fields=["value"],
        )
        if results:
            return results[0]["value"]
        return None

    def drop(self):
        if self.client.has_collection(self.COLLECTION):
            self.client.drop_collection(self.COLLECTION)
            logger.info(f"已删除元数据 Collection: {self.COLLECTION}")
