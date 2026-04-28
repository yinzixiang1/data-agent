"""Milvus 向量存储"""

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
        dbs = _client.list_databases()
        if MILVUS_DB not in dbs:
            _client.create_database(MILVUS_DB)
            logger.info(f"创建数据库: {MILVUS_DB}")
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
    Milvus Collection 封装。
    支持自定义标量字段 + Dense/Sparse 混合检索。
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

    def create(self, scalar_fields: list[dict] | None = None):
        """
        创建 Collection。

        基础字段自动包含: id(INT64 PK), dense_vec(FLOAT_VECTOR), sparse_vec(SPARSE_FLOAT_VECTOR)
        scalar_fields 示例:
            [{"name": "table_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True}, ...]
        """
        if self.exists():
            self.drop()

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense_vec", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vec", DataType.SPARSE_FLOAT_VECTOR)

        if scalar_fields:
            for f in scalar_fields:
                kwargs = {k: v for k, v in f.items() if k not in ("name", "dtype", "inverted")}
                schema.add_field(f["name"], f["dtype"], **kwargs)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vec", index_type="FLAT", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vec", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        # 标量倒排索引
        if scalar_fields:
            for f in scalar_fields:
                if f.get("inverted"):
                    index_params.add_index(field_name=f["name"], index_type="INVERTED")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"创建 Collection: {self.collection_name}")

    def insert(self, dense_vecs: np.ndarray, lexical_weights_list: list[dict], rows: list[dict]):
        """
        批量插入。

        dense_vecs: (n, dim) numpy array
        lexical_weights_list: BGE-M3 稀疏权重
        rows: 标量字段数据 [{"table_name": "xxx", ...}, ...]
        """
        n = len(rows)
        sparse_vecs = _to_sparse_vectors(lexical_weights_list)

        data = []
        for i in range(n):
            row = {
                "id": i,
                "dense_vec": dense_vecs[i].astype(np.float32).tolist(),
                "sparse_vec": sparse_vecs[i],
            }
            row.update(rows[i])
            data.append(row)

        self.client.insert(self.collection_name, data)
        logger.info(f"插入 {n} 条到 {self.collection_name}")

    def hybrid_search(
        self,
        query_dense: np.ndarray,
        query_sparse_weights: dict,
        top_k: int = 20,
        recall_k: int = 20,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float, dict]]:
        """
        混合检索: Dense + Sparse → RRF 融合。

        Returns: [(doc_id, score, entity_dict), ...]
        """
        q = query_dense.astype(np.float32).reshape(1, -1)
        q_sparse = _to_sparse_vectors([query_sparse_weights])

        dense_req = AnnSearchRequest(
            data=q.tolist(),
            anns_field="dense_vec",
            param={"metric_type": "COSINE"},
            limit=recall_k,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=q_sparse,
            anns_field="sparse_vec",
            param={"metric_type": "IP"},
            limit=recall_k,
            expr=filter_expr,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=RRF_K),
            limit=top_k,
            output_fields=output_fields or ["*"],
        )

        return [(hit["id"], hit["distance"], hit["entity"]) for hit in results[0]]

    def dense_search(
        self,
        query_dense: np.ndarray,
        top_k: int = 20,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float, dict]]:
        """纯 Dense 检索"""
        q = query_dense.astype(np.float32).reshape(1, -1)

        results = self.client.search(
            collection_name=self.collection_name,
            data=q.tolist(),
            anns_field="dense_vec",
            limit=top_k,
            output_fields=output_fields or ["*"],
            search_params={"metric_type": "COSINE"},
            filter=filter_expr,
        )

        return [(hit["id"], hit["distance"], hit["entity"]) for hit in results[0]]

    def query_all(self, output_fields: list[str] | None = None) -> list[dict]:
        """查询所有文档（按 id 排序）"""
        if not self.exists():
            return []
        results = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=output_fields or ["*"],
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
    """元数据存储 — 用于存储 schema_hash 等键值数据"""

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
