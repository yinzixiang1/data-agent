"""
Milvus 向量存储 — Collection 管理、混合检索。

封装了 MilvusClient，提供 Dense + Sparse 混合检索（RRF 融合）和标量过滤能力。

使用示例::

    from src.retrieval.milvus_store import MilvusIndex

    # 创建表级 Collection 并插入数据
    idx = MilvusIndex("nl2sql_table", dim=1024)
    idx.create(scalar_fields=[
        {"name": "table_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    ])
    idx.insert(dense_vecs, lexical_weights, [{"table_name": "pmt_account"}])

    # 混合检索
    hits = idx.hybrid_search(q_dense, q_sparse, top_k=5)
    # hits: [(doc_id, score, {"table_name": "pmt_account", ...}), ...]
"""

import logging
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from scipy.sparse import csr_array
import numpy as np

from src.retrieval.config import MILVUS_URI, MILVUS_DB, RRF_K

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """
    获取 Milvus 客户端单例（懒加载）。

    首次调用时连接 Milvus 服务，若目标数据库不存在则自动创建。

    Returns:
        MilvusClient 实例，已切换到 config.MILVUS_DB 数据库
    """
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
    """
    将 BGE-M3 的 lexical_weights 转为 Milvus 所需的 csr_array 稀疏向量。

    Args:
        lexical_weights_list: BGE-M3 encode 返回的稀疏权重列表，
            每个元素为 {token_id(int): weight(float)} 字典

    Returns:
        list[csr_array]: 与输入等长的稀疏向量列表，每个为 shape (1, max_idx) 的 csr_array
    """
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
    Milvus Collection 封装，支持自定义标量字段 + Dense/Sparse 混合检索。

    每个 Collection 自动包含三个基础字段:
        - id (INT64): 主键
        - dense_vec (FLOAT_VECTOR): 密集向量，用于语义检索
        - sparse_vec (SPARSE_FLOAT_VECTOR): 稀疏向量，用于关键词检索

    Attributes:
        collection_name: Milvus Collection 名称
        dim: Dense 向量维度（默认 1024，对应 BGE-M3）
        client: MilvusClient 实例
    """

    def __init__(self, collection_name: str, dim: int = 1024):
        """
        Args:
            collection_name: Collection 名称，如 "nl2sql_table"
            dim: Dense 向量维度，BGE-M3 为 1024
        """
        self.collection_name = collection_name
        self.dim = dim
        self.client = get_milvus_client()

    def exists(self) -> bool:
        """检查 Collection 是否存在。"""
        return self.client.has_collection(self.collection_name)

    def drop(self):
        """删除 Collection（如果存在）。"""
        if self.exists():
            self.client.drop_collection(self.collection_name)
            logger.info(f"已删除 Collection: {self.collection_name}")

    def create(self, scalar_fields: list[dict] | None = None):
        """
        创建 Collection（如果已存在则先删除再重建）。

        基础字段(id, dense_vec, sparse_vec)自动创建，只需传入额外的标量字段。

        Args:
            scalar_fields: 自定义标量字段列表，每个字段为 dict，支持的键:
                - "name" (str): 字段名，如 "table_name"
                - "dtype" (DataType): 数据类型，如 DataType.VARCHAR
                - "max_length" (int): VARCHAR 最大长度
                - "inverted" (bool): 是否创建倒排索引（加速标量过滤）
                示例: [{"name": "table_name", "dtype": DataType.VARCHAR,
                         "max_length": 128, "inverted": True}]
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
        批量插入文档到 Collection。

        Args:
            dense_vecs: Dense 向量矩阵，shape (n, dim)，float32 numpy array
            lexical_weights_list: BGE-M3 稀疏权重列表，长度 n，
                每个元素为 {token_id: weight} 字典
            rows: 标量字段数据列表，长度 n，每个元素为 dict，
                键对应 create() 时定义的 scalar_fields，如:
                [{"table_name": "pmt_account", "table_cn_name": "商户账户表"}, ...]
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
        混合检索: Dense(COSINE) + Sparse(IP) → RRF 融合排序。

        Args:
            query_dense: 查询的 Dense 向量，shape (1, dim) 或 (dim,)
            query_sparse_weights: 查询的 Sparse 权重，{token_id: weight} 字典
            top_k: 最终返回的文档数量
            recall_k: Dense/Sparse 各自召回的候选数量（融合前的池子大小）
            output_fields: 需要返回的标量字段名列表，None 表示返回所有字段
            filter_expr: Milvus 标量过滤表达式，如 'table_name == "pmt_account"'

        Returns:
            list[tuple]: [(doc_id, rrf_score, entity_dict), ...]
                - doc_id: 文档 ID (int)
                - rrf_score: RRF 融合后的分数 (float)
                - entity_dict: 标量字段数据 (dict)
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
        """
        纯 Dense 向量检索（COSINE 相似度）。

        Args:
            query_dense: 查询向量，shape (1, dim) 或 (dim,)
            top_k: 返回数量
            output_fields: 需要返回的标量字段名列表，None 表示返回所有
            filter_expr: 标量过滤表达式

        Returns:
            list[tuple]: [(doc_id, cosine_score, entity_dict), ...]
        """
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

    @property
    def count(self) -> int:
        """Collection 中的文档总数，不存在时返回 0。"""
        if not self.exists():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)
