"""
Milvus 向量存储 — BM25 Function + HNSW Dense 混合检索。

每个 Collection 自动包含:
    - id (INT64 PK)
    - text (VARCHAR): BM25 输入原文, jieba 分词
    - bm25_sparse (SPARSE_FLOAT_VECTOR): BM25 Function 自动生成
    - dense_vec (FLOAT_VECTOR): Dense 向量 (has_dense=True 时)

Dense 和 Sparse 完全解耦:
    - Dense 由 Qwen3Embedding 外部编码后写入
    - Sparse 由 Milvus BM25 Function 在 insert 时自动生成

使用示例::

    from src.retrieval.milvus_store import MilvusIndex

    # 含 Dense 的 Collection
    idx = MilvusIndex("nl2sql_table", dim=1024)
    idx.create(scalar_fields=[...], index_config={"hnsw": {"M": 16}})
    idx.insert(dense_vecs, texts, rows)
    hits = idx.hybrid_search(q_dense, "查询文本", ranker, recall_k=30)

    # 纯 BM25 的 Collection (nl2sql_value)
    val_idx = MilvusIndex("nl2sql_value", has_dense=False)
    val_idx.create(scalar_fields=[...])
    val_idx.insert(None, texts, rows)
    hits = val_idx.bm25_search("持牌商户", top_k=20)
"""

import logging

import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
)

from src.retrieval.config import (
    MILVUS_URI,
    MILVUS_DB,
    MILVUS_USER,
    MILVUS_PASSWORD,
    MILVUS_TOKEN,
)

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None

# 运行时覆盖（由 app.py 启动时通过 configure() 注入 Agent 绑定的资源配置）
_runtime_uri: str = ""
_runtime_db: str = ""
_runtime_user: str = ""
_runtime_password: str = ""
_runtime_token: str = ""


def configure(
    uri: str = "", db: str = "", user: str = "", password: str = "", token: str = ""
):
    """注入运行时 Milvus 连接参数（覆盖 .env 默认值）。必须在 get_milvus_client() 首次调用前执行。"""
    global _runtime_uri, _runtime_db, _runtime_user, _runtime_password, _runtime_token
    _runtime_uri = uri
    _runtime_db = db
    _runtime_user = user
    _runtime_password = password
    _runtime_token = token


def get_milvus_client() -> MilvusClient:
    """
    获取 Milvus 客户端单例（懒加载）。

    优先使用 configure() 注入的参数，fallback 到 config.py 环境变量。
    首次调用时连接 Milvus 服务，若目标数据库不存在则自动创建。

    Returns:
        MilvusClient 实例，已切换到目标数据库
    """
    global _client
    if _client is None:
        uri = _runtime_uri or MILVUS_URI
        db = _runtime_db or MILVUS_DB
        user = _runtime_user or MILVUS_USER
        password = _runtime_password or MILVUS_PASSWORD
        token = _runtime_token or MILVUS_TOKEN

        conn_kwargs = {"uri": uri, "db_name": "default", "timeout": 10}
        if token:
            conn_kwargs["token"] = token
        elif user and password:
            conn_kwargs["token"] = f"{user}:{password}"
        _client = MilvusClient(**conn_kwargs)
        dbs = _client.list_databases()
        if db not in dbs:
            _client.create_database(db)
            logger.info(f"创建数据库: {db}")
        _client.using_database(db)
        logger.info(f"Milvus 连接成功: {uri}, db={db}, user={user or '(anonymous)'}")
    return _client


class MilvusIndex:
    """
    Milvus Collection 封装，支持 BM25 Function + HNSW Dense 混合检索。

    Attributes:
        collection_name: Collection 名称
        dim: Dense 向量维度
        has_dense: 是否包含 Dense 向量 (False 时为纯 BM25 Collection)
        client: MilvusClient 实例
    """

    def __init__(self, collection_name: str, dim: int = 1024, has_dense: bool = True):
        """
        Args:
            collection_name: Collection 名称，如 "nl2sql_table"
            dim: Dense 向量维度
            has_dense: 是否包含 Dense 向量，False 时为纯 BM25 (nl2sql_value)
        """
        self.collection_name = collection_name
        self.dim = dim
        self.has_dense = has_dense
        self._cached_count: int | None = None
        self.client = get_milvus_client()

    def exists(self) -> bool:
        """检查 Collection 是否存在。"""
        return self.client.has_collection(self.collection_name)

    def drop(self):
        """删除 Collection（如果存在）。"""
        if self.exists():
            self.client.drop_collection(self.collection_name)
            self._cached_count = None
            logger.info(f"已删除 Collection: {self.collection_name}")

    def create(
        self,
        scalar_fields: list[dict] | None = None,
        index_config: dict | None = None,
        bm25_config: dict | None = None,
    ):
        """
        创建 Collection（如已存在则先删除）。

        自动创建基础字段和索引:
            - text (VARCHAR) + BM25 Function -> bm25_sparse
            - dense_vec (FLOAT_VECTOR) + HNSW 索引 [仅 has_dense=True]
            - 用户自定义 scalar_fields

        Args:
            scalar_fields: 自定义标量字段列表，每个 dict 支持:
                - "name": 字段名
                - "dtype": DataType 枚举
                - "max_length": VARCHAR 最大长度
                - "inverted": 是否创建倒排索引
            index_config: INDEX_BUILD_CONFIG 字典，包含 hnsw / bm25 子配置
            bm25_config: BM25 参数 (k1, b, analyzer)，为 None 时从 index_config 读取
        """
        if self.exists():
            self.drop()

        idx_cfg = index_config or {}
        hnsw_cfg = idx_cfg.get("hnsw", {})
        bm25_cfg = bm25_config or idx_cfg.get("bm25", {})
        analyzer = bm25_cfg.get("analyzer", "chinese")

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)

        # text 字段: BM25 Function 的输入
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": analyzer},
        )

        # BM25 sparse 输出字段
        schema.add_field("bm25_sparse", DataType.SPARSE_FLOAT_VECTOR)

        # BM25 Function: text -> bm25_sparse
        bm25_function = Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["bm25_sparse"],
        )
        schema.add_function(bm25_function)

        # Dense 向量字段 (可选)
        if self.has_dense:
            schema.add_field("dense_vec", DataType.FLOAT_VECTOR, dim=self.dim)

        # 用户自定义标量字段
        if scalar_fields:
            for f in scalar_fields:
                kwargs = {
                    k: v for k, v in f.items() if k not in ("name", "dtype", "inverted")
                }
                schema.add_field(f["name"], f["dtype"], **kwargs)

        # 索引
        index_params = self.client.prepare_index_params()

        if self.has_dense:
            index_params.add_index(
                field_name="dense_vec",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": hnsw_cfg.get("M", 16),
                    "efConstruction": hnsw_cfg.get("efConstruction", 200),
                },
            )

        index_params.add_index(
            field_name="bm25_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

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
        logger.info(
            f"创建 Collection: {self.collection_name} "
            f"(has_dense={self.has_dense}, analyzer={analyzer})"
        )

    def insert(
        self,
        dense_vecs: np.ndarray | None,
        texts: list[str],
        rows: list[dict],
        bm25_texts: list[str] | None = None,
    ):
        """
        批量插入文档到 Collection。

        Args:
            dense_vecs: Dense 向量, shape (N, dim), has_dense=False 时传 None
            texts: 文本列表, 默认写入 ``text`` 字段供 BM25 Function 使用
            rows: 标量字段数据列表
            bm25_texts: 可选, 当 BM25 需要不同于 Dense 编码的文本时使用
                (如 Glossary: Dense 编码 "术语: 定义", BM25 用 "术语 同义词")
        """
        n = len(rows)
        actual_texts = bm25_texts if bm25_texts is not None else texts

        data = []
        for i in range(n):
            row = {"id": i, "text": actual_texts[i]}
            if self.has_dense and dense_vecs is not None:
                row["dense_vec"] = dense_vecs[i].astype(np.float32).tolist()
            row.update(rows[i])
            data.append(row)

        self.client.insert(self.collection_name, data)
        self._cached_count = n
        logger.info(f"插入 {n} 条到 {self.collection_name}")

    def hybrid_search(
        self,
        query_dense: np.ndarray,
        query_text: str,
        ranker,
        recall_k: int = 20,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
        ef_search: int = 64,
    ) -> list[tuple[int, float, dict]]:
        """
        Dense + BM25 混合检索。

        Args:
            query_dense: 查询 Dense 向量, shape (1, dim) 或 (dim,)
            query_text: 查询文本 (Milvus BM25 自动分词)
            ranker: Milvus Ranker 实例 (WeightedRanker / RRFRanker)
            recall_k: 每路召回数量
            output_fields: 返回的标量字段
            filter_expr: 标量过滤表达式
            ef_search: HNSW 搜索参数 ef

        Returns:
            list[tuple]: [(doc_id, score, entity_dict), ...]
        """
        q = query_dense.astype(np.float32).reshape(1, -1)

        dense_req = AnnSearchRequest(
            data=q.tolist(),
            anns_field="dense_vec",
            param={"metric_type": "COSINE", "params": {"ef": ef_search}},
            limit=recall_k,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="bm25_sparse",
            param={"metric_type": "BM25"},
            limit=recall_k,
            expr=filter_expr,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=recall_k,
            output_fields=output_fields or ["*"],
        )

        return [(hit["id"], hit["distance"], hit["entity"]) for hit in results[0]]

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 20,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float, dict]]:
        """
        纯 BM25 文本检索 (nl2sql_value 等纯 Sparse Collection 专用)。

        Args:
            query_text: 查询文本
            top_k: 返回数量
            output_fields: 返回的标量字段
            filter_expr: 标量过滤表达式

        Returns:
            list[tuple]: [(doc_id, bm25_score, entity_dict), ...]
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field="bm25_sparse",
            limit=top_k,
            output_fields=output_fields or ["*"],
            search_params={"metric_type": "BM25"},
            filter=filter_expr,
        )

        return [(hit["id"], hit["distance"], hit["entity"]) for hit in results[0]]

    def dense_search(
        self,
        query_dense: np.ndarray,
        top_k: int = 20,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
        ef_search: int = 64,
    ) -> list[tuple[int, float, dict]]:
        """
        纯 Dense 向量检索 (COSINE)。

        Args:
            query_dense: 查询向量, shape (1, dim) 或 (dim,)
            top_k: 返回数量
            output_fields: 返回的标量字段
            filter_expr: 标量过滤表达式
            ef_search: HNSW 搜索参数 ef

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
            search_params={"metric_type": "COSINE", "params": {"ef": ef_search}},
            filter=filter_expr,
        )

        return [(hit["id"], hit["distance"], hit["entity"]) for hit in results[0]]

    @property
    def count(self) -> int:
        """Collection 中的文档总数，不存在时返回 0。"""
        if self._cached_count is not None:
            return self._cached_count
        if not self.exists():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)
