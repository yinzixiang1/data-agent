"""
索引构建管理 — 将 Schema 文档编码为向量并写入 6 个 Milvus Collection。

每次启动时强制全量重建，确保索引与 Schema 一致。

6 个 Collection:
    - nl2sql_table:   表级索引 (Dense + BM25)
    - nl2sql_column:  列级索引 (Dense + BM25)
    - nl2sql_enum:    枚举值索引 (Dense + BM25)
    - nl2sql_value:   值级索引 (BM25-only, Schema Linking)
    - nl2sql_fewshot: Few-shot 示例索引 (Dense + BM25)
    - nl2sql_glossary: 术语索引 (Dense + BM25, 分离编码)

使用示例::

    from src.retrieval.index_manager import IndexManager
    from src.retrieval.embedding import get_embedding

    mgr = IndexManager()
    embedding = get_embedding(config.embedding_config)
    indices = mgr.build(schemas, embedding, enums, index_build_config=config.index_build_config)
"""

import json
import logging

from pymilvus import DataType

from src.retrieval.document_builder import DocumentBuilder
from src.retrieval.collection_names import agent_collection_name
from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.fewshot_selector import FewShotSelector
from src.retrieval.value_indexer import ValueIndexer

logger = logging.getLogger(__name__)

# ── Collection 名称 ──
TABLE_COLLECTION = "nl2sql_table"
COLUMN_COLLECTION = "nl2sql_column"
ENUM_COLLECTION = "nl2sql_enum"
VALUE_COLLECTION = "nl2sql_value"
FEWSHOT_COLLECTION = "nl2sql_fewshot"
GLOSSARY_COLLECTION = "nl2sql_glossary"

# ── Collection Schema 定义 ──

TABLE_FIELDS = [
    {"name": "db_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {
        "name": "table_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {"name": "biz_line", "dtype": DataType.VARCHAR, "max_length": 64, "inverted": True},
    {"name": "metadata", "dtype": DataType.JSON},
    {"name": "table_cn_name", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "table_comment", "dtype": DataType.VARCHAR, "max_length": 4096},
    {"name": "business_domain", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "schema_json", "dtype": DataType.VARCHAR, "max_length": 65535},
]

COLUMN_FIELDS = [
    {"name": "db_name", "dtype": DataType.VARCHAR, "max_length": 128},
    {
        "name": "table_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {"name": "biz_line", "dtype": DataType.VARCHAR, "max_length": 64, "inverted": True},
    {"name": "metadata", "dtype": DataType.JSON},
    {"name": "column_name", "dtype": DataType.VARCHAR, "max_length": 128},
    {"name": "column_cn_name", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "column_type", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "column_comment", "dtype": DataType.VARCHAR, "max_length": 1024},
    {"name": "enum_values", "dtype": DataType.VARCHAR, "max_length": 4096},
    {"name": "is_enum", "dtype": DataType.BOOL},
]

ENUM_FIELDS = [
    {
        "name": "table_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {
        "name": "column_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {"name": "biz_line", "dtype": DataType.VARCHAR, "max_length": 64, "inverted": True},
    {"name": "metadata", "dtype": DataType.JSON},
    {"name": "enum_code", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "enum_label_cn", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "description", "dtype": DataType.VARCHAR, "max_length": 1024},
    {"name": "synonyms", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "sql_value", "dtype": DataType.VARCHAR, "max_length": 64},
]

VALUE_FIELDS = [
    {
        "name": "table_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {
        "name": "column_name",
        "dtype": DataType.VARCHAR,
        "max_length": 128,
        "inverted": True,
    },
    {"name": "biz_line", "dtype": DataType.VARCHAR, "max_length": 64, "inverted": True},
    {"name": "metadata", "dtype": DataType.JSON},
    {"name": "enum_code", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "enum_label_cn", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "sql_value", "dtype": DataType.VARCHAR, "max_length": 64},
]

FEWSHOT_FIELDS = [
    {"name": "question", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql", "dtype": DataType.VARCHAR, "max_length": 8192},
    {"name": "involved_tables", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "difficulty", "dtype": DataType.VARCHAR, "max_length": 32},
    {"name": "metadata", "dtype": DataType.JSON},
]

GLOSSARY_FIELDS = [
    {"name": "term", "dtype": DataType.VARCHAR, "max_length": 256, "inverted": True},
    {"name": "definition", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql_hint", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "related_tables", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "related_columns", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "synonyms", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "metadata", "dtype": DataType.JSON},
]


class IndexManager:
    """
    索引构建管理器 — 全量构建 6 个 Milvus Collection。
    """

    def __init__(self, agent_id: int | None = None):
        self.agent_id = agent_id

    def _name(self, base_name: str) -> str:
        return agent_collection_name(base_name, self.agent_id)

    def build(
        self,
        schemas: list[dict],
        embedding: Qwen3Embedding,
        enums: list[dict] | None = None,
        fewshot_examples: list[dict] | None = None,
        glossary: dict[str, dict] | None = None,
        index_build_config: dict | None = None,
    ) -> dict:
        """
        全量构建索引: 构建文档 -> 编码 -> 写入 Milvus。

        Args:
            schemas: 表 Schema 列表
            embedding: Qwen3Embedding 实例
            enums: 枚举条目列表
            fewshot_examples: Few-shot 示例列表
            glossary: 业务术语字典
            index_build_config: INDEX_BUILD_CONFIG (HNSW/BM25 参数)

        Returns:
            dict: table_index, column_index, enum_index, value_indexer,
                  fewshot_selector, glossary_index, table_schemas
        """
        idx_cfg = index_build_config or {}
        dim = embedding.dim
        builder = DocumentBuilder()

        # 1. 构建文档
        table_docs, column_docs, enum_docs = builder.build_all(schemas, enums)
        value_docs = builder.build_value_documents(enums or [])
        logger.info(
            f"文档构建: {len(table_docs)} 表级, {len(column_docs)} 列级, "
            f"{len(enum_docs)} 枚举值, {len(value_docs)} 值级"
        )

        # 2. 表级索引
        table_index = MilvusIndex(self._name(TABLE_COLLECTION), dim=dim)
        table_index.create(TABLE_FIELDS, index_config=idx_cfg)
        if table_docs:
            table_texts = [d["text"] for d in table_docs]
            dense_vecs = embedding.encode(
                table_texts, instruction=embedding.instructions.get("table", "")
            )
            table_rows = [
                {
                    "db_name": d["schema"].get("database", ""),
                    "table_name": d["table_name"],
                    "biz_line": d["schema"].get("biz_line", ""),
                    "metadata": d["schema"].get("metadata", {}),
                    "table_cn_name": d["schema"].get("display_name", ""),
                    "table_comment": (d["schema"].get("description") or "")[:4096],
                    "business_domain": ", ".join(d["schema"].get("tags", [])),
                    "schema_json": json.dumps(d["schema"], ensure_ascii=False)[:65535],
                }
                for d in table_docs
            ]
            table_index.insert(dense_vecs, table_texts, table_rows)

        # 3. 列级索引
        # 构建 table_name -> biz_line / database / metadata 映射
        table_biz_map = {s["table_name"]: s.get("biz_line", "") for s in schemas}
        table_db_map = {s["table_name"]: s.get("database", "") for s in schemas}
        table_meta_map = {s["table_name"]: s.get("metadata", {}) for s in schemas}
        column_index = MilvusIndex(self._name(COLUMN_COLLECTION), dim=dim)
        column_index.create(COLUMN_FIELDS, index_config=idx_cfg)
        if column_docs:
            column_texts = [d["text"] for d in column_docs]
            dense_vecs = embedding.encode(
                column_texts, instruction=embedding.instructions.get("column", "")
            )
            column_rows = [
                {
                    "db_name": table_db_map.get(d["table_name"], ""),
                    "table_name": d["table_name"],
                    "biz_line": table_biz_map.get(d["table_name"], ""),
                    "metadata": table_meta_map.get(d["table_name"], {}),
                    "column_name": d["column_name"],
                    "column_cn_name": d.get("column_cn_name", ""),
                    "column_type": d.get("column_type", ""),
                    "column_comment": d.get("column_comment", ""),
                    "enum_values": d.get("enum_values_summary", "")[:4096],
                    "is_enum": d.get("is_enum", False),
                }
                for d in column_docs
            ]
            column_index.insert(dense_vecs, column_texts, column_rows)

        # 4. 枚举值索引
        enum_index = MilvusIndex(self._name(ENUM_COLLECTION), dim=dim)
        enum_index.create(ENUM_FIELDS, index_config=idx_cfg)
        if enum_docs:
            enum_texts = [d["text"] for d in enum_docs]
            dense_vecs = embedding.encode(
                enum_texts, instruction=embedding.instructions.get("enum", "")
            )
            enum_rows = [
                {
                    "table_name": d["table_name"],
                    "column_name": d["column_name"],
                    "biz_line": d.get("biz_line", ""),
                    "metadata": d.get("metadata", {}),
                    "enum_code": d["enum_code"],
                    "enum_label_cn": d["enum_label_cn"],
                    "description": d.get("description", ""),
                    "synonyms": d.get("synonyms", ""),
                    "sql_value": d["sql_value"],
                }
                for d in enum_docs
            ]
            enum_index.insert(dense_vecs, enum_texts, enum_rows)

        # 5. 值级索引 (BM25-only)
        value_index = MilvusIndex(self._name(VALUE_COLLECTION), has_dense=False)
        value_index.create(VALUE_FIELDS, index_config=idx_cfg)
        value_idx = ValueIndexer(value_index)
        value_idx.build_index(value_docs)

        # 6. Few-shot 索引
        fewshot_index = MilvusIndex(self._name(FEWSHOT_COLLECTION), dim=dim)
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        fewshot.build_index(fewshot_examples or [], index_config=idx_cfg)

        # 7. 术语索引 (Dense + BM25 分离编码)
        glossary_index = MilvusIndex(self._name(GLOSSARY_COLLECTION), dim=dim)
        glossary_index.create(GLOSSARY_FIELDS, index_config=idx_cfg)
        if glossary:
            dense_texts = []
            bm25_texts = []
            glossary_rows = []
            for term, info in glossary.items():
                definition = info.get("definition", "")
                synonyms = info.get("synonyms", [])
                dense_texts.append(f"{term}: {definition}" if definition else term)
                bm25_texts.append(" ".join([term] + synonyms))
                glossary_rows.append(
                    {
                        "term": term,
                        "definition": (definition or "")[:2048],
                        "sql_hint": (info.get("sql_hint", "") or "")[:2048],
                        "related_tables": json.dumps(
                            info.get("related_tables", []), ensure_ascii=False
                        )[:2048],
                        "related_columns": json.dumps(
                            info.get("related_columns", []), ensure_ascii=False
                        )[:2048],
                        "synonyms": json.dumps(synonyms, ensure_ascii=False)[:2048],
                        "metadata": info.get("metadata", {}),
                    }
                )
            dense_vecs = embedding.encode(
                dense_texts, instruction=embedding.instructions.get("glossary", "")
            )
            glossary_index.insert(
                dense_vecs, dense_texts, glossary_rows, bm25_texts=bm25_texts
            )
            logger.info(
                f"术语索引构建: {len(glossary_rows)} 条 (Dense=完整文本, BM25=术语名)"
            )

        # 8. table_name -> schema 映射
        table_schemas = {s["table_name"]: s for s in schemas}

        logger.info("索引构建完成")

        return {
            "table_index": table_index,
            "column_index": column_index,
            "enum_index": enum_index,
            "value_indexer": value_idx,
            "fewshot_selector": fewshot,
            "glossary_index": glossary_index,
            "table_schemas": table_schemas,
        }

    def connect(
        self,
        schemas: list[dict],
        embedding: Qwen3Embedding,
        fewshot_examples: list[dict] | None = None,
    ) -> dict:
        """
        连接已有 Collection（不重建索引），仅初始化内存对象。

        比 build() 快得多：跳过 drop/create/insert，只构造 MilvusIndex 引用
        和 FewShotSelector 内存状态（用于 MMR 多样性选择）。

        Args:
            schemas: 表 Schema 列表（用于 table_schemas 映射）
            embedding: Qwen3Embedding 实例（FewShotSelector 需要）
            fewshot_examples: Few-shot 示例列表（FewShotSelector 需要）

        Returns:
            dict: 与 build() 返回格式一致
        """
        dim = embedding.dim

        table_index = MilvusIndex(self._name(TABLE_COLLECTION), dim=dim)
        column_index = MilvusIndex(self._name(COLUMN_COLLECTION), dim=dim)
        enum_index = MilvusIndex(self._name(ENUM_COLLECTION), dim=dim)
        value_index = MilvusIndex(self._name(VALUE_COLLECTION), has_dense=False)
        value_idx = ValueIndexer(value_index)
        glossary_index = MilvusIndex(self._name(GLOSSARY_COLLECTION), dim=dim)

        # FewShotSelector 需要内存中的 examples + embeddings 用于 MMR
        fewshot_index = MilvusIndex(self._name(FEWSHOT_COLLECTION), dim=dim)
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        examples = fewshot_examples or []
        if examples:
            fewshot.examples = examples
            texts = [ex["question"] for ex in examples]
            fewshot.embeddings = embedding.encode(texts)
            fewshot.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
            logger.info(f"Few-shot 内存状态加载: {len(examples)} 条示例")

        table_schemas = {s["table_name"]: s for s in schemas}

        collection_names = [
            self._name(name)
            for name in [
                TABLE_COLLECTION,
                COLUMN_COLLECTION,
                ENUM_COLLECTION,
                VALUE_COLLECTION,
                FEWSHOT_COLLECTION,
                GLOSSARY_COLLECTION,
            ]
        ]
        counts = {name: MilvusIndex(name, dim=dim).count for name in collection_names}
        logger.info(f"已连接现有 Collection: {counts}")

        return {
            "table_index": table_index,
            "column_index": column_index,
            "enum_index": enum_index,
            "value_indexer": value_idx,
            "fewshot_selector": fewshot,
            "glossary_index": glossary_index,
            "table_schemas": table_schemas,
        }

    def rebuild_partial(
        self,
        collections: list[str],
        schemas: list[dict],
        embedding: Qwen3Embedding,
        enums: list[dict] | None = None,
        fewshot_examples: list[dict] | None = None,
        glossary: dict[str, dict] | None = None,
        index_build_config: dict | None = None,
    ) -> dict:
        """
        按指定 collection 类型选择性重建索引。

        Args:
            collections: 要重建的类型列表，可选值: table, enum, value, fewshot, glossary

        Returns:
            dict: 仅包含被重建的索引对象
        """
        idx_cfg = index_build_config or {}
        dim = embedding.dim
        result = {}
        builder = DocumentBuilder()

        if "table" in collections:
            logger.info("重建 table + column 索引...")
            table_docs, column_docs, _ = builder.build_all(schemas, None)
            table_biz_map = {s["table_name"]: s.get("biz_line", "") for s in schemas}
            table_db_map = {s["table_name"]: s.get("database", "") for s in schemas}

            table_index = MilvusIndex(self._name(TABLE_COLLECTION), dim=dim)
            table_index.create(TABLE_FIELDS, index_config=idx_cfg)
            if table_docs:
                table_texts = [d["text"] for d in table_docs]
                dense_vecs = embedding.encode(
                    table_texts, instruction=embedding.instructions.get("table", "")
                )
                table_rows = [
                    {
                        "db_name": d["schema"].get("database", ""),
                        "table_name": d["table_name"],
                        "biz_line": d["schema"].get("biz_line", ""),
                        "metadata": d["schema"].get("metadata", {}),
                        "table_cn_name": d["schema"].get("display_name", ""),
                        "table_comment": (d["schema"].get("description") or "")[:4096],
                        "business_domain": ", ".join(d["schema"].get("tags", [])),
                        "schema_json": json.dumps(d["schema"], ensure_ascii=False)[
                            :65535
                        ],
                    }
                    for d in table_docs
                ]
                table_index.insert(dense_vecs, table_texts, table_rows)
            result["table_index"] = table_index

            column_index = MilvusIndex(self._name(COLUMN_COLLECTION), dim=dim)
            column_index.create(COLUMN_FIELDS, index_config=idx_cfg)
            table_meta_map = {s["table_name"]: s.get("metadata", {}) for s in schemas}
            if column_docs:
                column_texts = [d["text"] for d in column_docs]
                dense_vecs = embedding.encode(
                    column_texts, instruction=embedding.instructions.get("column", "")
                )
                column_rows = [
                    {
                        "db_name": table_db_map.get(d["table_name"], ""),
                        "table_name": d["table_name"],
                        "biz_line": table_biz_map.get(d["table_name"], ""),
                        "metadata": table_meta_map.get(d["table_name"], {}),
                        "column_name": d["column_name"],
                        "column_cn_name": d.get("column_cn_name", ""),
                        "column_type": d.get("column_type", ""),
                        "column_comment": d.get("column_comment", ""),
                        "enum_values": d.get("enum_values_summary", "")[:4096],
                        "is_enum": d.get("is_enum", False),
                    }
                    for d in column_docs
                ]
                column_index.insert(dense_vecs, column_texts, column_rows)
            result["column_index"] = column_index
            result["table_schemas"] = {s["table_name"]: s for s in schemas}
            logger.info(
                f"table + column 索引重建完成: {len(table_docs)} 表, {len(column_docs)} 列"
            )

        if "enum" in collections:
            logger.info("重建 enum 索引...")
            _, _, enum_docs = builder.build_all(schemas, enums)
            enum_index = MilvusIndex(self._name(ENUM_COLLECTION), dim=dim)
            enum_index.create(ENUM_FIELDS, index_config=idx_cfg)
            if enum_docs:
                enum_texts = [d["text"] for d in enum_docs]
                dense_vecs = embedding.encode(
                    enum_texts, instruction=embedding.instructions.get("enum", "")
                )
                enum_rows = [
                    {
                        "table_name": d["table_name"],
                        "column_name": d["column_name"],
                        "biz_line": d.get("biz_line", ""),
                        "metadata": d.get("metadata", {}),
                        "enum_code": d["enum_code"],
                        "enum_label_cn": d["enum_label_cn"],
                        "description": d.get("description", ""),
                        "synonyms": d.get("synonyms", ""),
                        "sql_value": d["sql_value"],
                    }
                    for d in enum_docs
                ]
                enum_index.insert(dense_vecs, enum_texts, enum_rows)
            result["enum_index"] = enum_index
            logger.info(f"enum 索引重建完成: {len(enum_docs)} 条")

        if "value" in collections:
            logger.info("重建 value 索引...")
            value_docs = builder.build_value_documents(enums or [])
            value_index = MilvusIndex(self._name(VALUE_COLLECTION), has_dense=False)
            value_index.create(VALUE_FIELDS, index_config=idx_cfg)
            value_idx = ValueIndexer(value_index)
            value_idx.build_index(value_docs)
            result["value_indexer"] = value_idx
            logger.info(f"value 索引重建完成: {len(value_docs)} 条")

        if "fewshot" in collections:
            logger.info("重建 fewshot 索引...")
            fewshot_index = MilvusIndex(self._name(FEWSHOT_COLLECTION), dim=dim)
            fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
            fewshot.build_index(fewshot_examples or [], index_config=idx_cfg)
            result["fewshot_selector"] = fewshot
            logger.info(f"fewshot 索引重建完成: {len(fewshot_examples or [])} 条")

        if "glossary" in collections:
            logger.info("重建 glossary 索引...")
            glossary_index = MilvusIndex(self._name(GLOSSARY_COLLECTION), dim=dim)
            glossary_index.create(GLOSSARY_FIELDS, index_config=idx_cfg)
            if glossary:
                dense_texts = []
                bm25_texts = []
                glossary_rows = []
                for term, info in glossary.items():
                    definition = info.get("definition", "")
                    synonyms = info.get("synonyms", [])
                    dense_texts.append(f"{term}: {definition}" if definition else term)
                    bm25_texts.append(" ".join([term] + synonyms))
                    glossary_rows.append(
                        {
                            "term": term,
                            "definition": (definition or "")[:2048],
                            "sql_hint": (info.get("sql_hint", "") or "")[:2048],
                            "related_tables": json.dumps(
                                info.get("related_tables", []), ensure_ascii=False
                            )[:2048],
                            "related_columns": json.dumps(
                                info.get("related_columns", []), ensure_ascii=False
                            )[:2048],
                            "synonyms": json.dumps(synonyms, ensure_ascii=False)[:2048],
                            "metadata": info.get("metadata", {}),
                        }
                    )
                dense_vecs = embedding.encode(
                    dense_texts, instruction=embedding.instructions.get("glossary", "")
                )
                glossary_index.insert(
                    dense_vecs, dense_texts, glossary_rows, bm25_texts=bm25_texts
                )
            result["glossary_index"] = glossary_index
            logger.info(f"glossary 索引重建完成: {len(glossary or {})} 条")

        return result
