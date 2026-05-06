"""
索引构建管理 — 将 Schema 文档编码为向量并写入 Milvus Collection。

每次启动时强制全量重建，确保索引与 Schema 一致。

使用示例::

    from src.retrieval.index_manager import IndexManager
    from src.retrieval.embedding import get_embedding

    mgr = IndexManager()
    embedding = get_embedding()
    indices = mgr.build(schemas, embedding, enums)
    # indices 包含: table_index, column_index, enum_index, fewshot_selector, table_schemas
"""

import json
import logging

from pymilvus import DataType

from src.retrieval.document_builder import DocumentBuilder
from src.retrieval.embedding import BGEEmbedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.fewshot_selector import FewShotSelector

logger = logging.getLogger(__name__)

# ── Collection 名称 ──
TABLE_COLLECTION = "nl2sql_table"
COLUMN_COLLECTION = "nl2sql_column"
ENUM_COLLECTION = "nl2sql_enum"
FEWSHOT_COLLECTION = "nl2sql_fewshot"

# ── Collection Schema 定义 ──

TABLE_FIELDS = [
    {"name": "db_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {"name": "table_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {"name": "table_cn_name", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "table_comment", "dtype": DataType.VARCHAR, "max_length": 4096},
    {"name": "business_domain", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "schema_json", "dtype": DataType.VARCHAR, "max_length": 65535},
]

COLUMN_FIELDS = [
    {"name": "db_name", "dtype": DataType.VARCHAR, "max_length": 128},
    {"name": "table_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {"name": "column_name", "dtype": DataType.VARCHAR, "max_length": 128},
    {"name": "column_cn_name", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "column_type", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "column_comment", "dtype": DataType.VARCHAR, "max_length": 1024},
    {"name": "enum_values", "dtype": DataType.VARCHAR, "max_length": 4096},
    {"name": "is_enum", "dtype": DataType.BOOL},
]

ENUM_FIELDS = [
    {"name": "table_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {"name": "column_name", "dtype": DataType.VARCHAR, "max_length": 128, "inverted": True},
    {"name": "enum_code", "dtype": DataType.VARCHAR, "max_length": 64},
    {"name": "enum_label_cn", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "description", "dtype": DataType.VARCHAR, "max_length": 1024},
    {"name": "synonyms", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "sql_value", "dtype": DataType.VARCHAR, "max_length": 64},
]

FEWSHOT_FIELDS = [
    {"name": "question", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql", "dtype": DataType.VARCHAR, "max_length": 8192},
    {"name": "involved_tables", "dtype": DataType.VARCHAR, "max_length": 512},
    {"name": "difficulty", "dtype": DataType.VARCHAR, "max_length": 32},
]

GLOSSARY_COLLECTION = "nl2sql_glossary"

GLOSSARY_FIELDS = [
    {"name": "term", "dtype": DataType.VARCHAR, "max_length": 256, "inverted": True},
    {"name": "definition", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "sql_hint", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "related_tables", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "related_columns", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "synonyms", "dtype": DataType.VARCHAR, "max_length": 2048},
]


class IndexManager:
    """
    索引构建管理器 — 全量构建所有 Milvus Collection。

    构建的 Collection:
        1. nl2sql_table — 表级索引（每张表一条）
        2. nl2sql_column — 列级索引（每个有效列一条）
        3. nl2sql_enum — 枚举值索引（每个枚举值一条）
        4. nl2sql_fewshot — Few-shot 示例索引（来自 MySQL da_fewshot + da_table_query）
    """

    def build(
        self,
        schemas: list[dict],
        embedding: BGEEmbedding,
        enums: list[dict] | None = None,
        fewshot_examples: list[dict] | None = None,
        glossary: dict[str, dict] | None = None,
    ) -> dict:
        """
        全量构建索引: 构建文档 → BGE-M3 编码 → 写入 Milvus。

        Args:
            schemas: 表 Schema 列表（SchemaLoader.load_all() 返回）
            embedding: BGEEmbedding 实例，用于向量编码
            enums: 枚举条目列表，为 None 时跳过枚举索引
            fewshot_examples: Few-shot 示例列表，每条含 question, sql, tables, difficulty
            glossary: 业务术语字典，{term: {definition, sql_hint, related_tables, related_columns}}

        Returns:
            dict，包含:
                - "table_index" (MilvusIndex): 表级索引
                - "column_index" (MilvusIndex): 列级索引
                - "enum_index" (MilvusIndex): 枚举值索引
                - "fewshot_selector" (FewShotSelector): Few-shot 选择器（已加载数据）
                - "glossary_index" (MilvusIndex): 术语索引
                - "table_schemas" (dict): {table_name: schema_dict} 映射
        """
        builder = DocumentBuilder()

        # 1. 构建文档
        table_docs, column_docs, enum_docs = builder.build_all(schemas, enums)
        logger.info(f"文档构建: {len(table_docs)} 表级, {len(column_docs)} 列级, {len(enum_docs)} 枚举值")

        # 2. 表级索引
        table_index = MilvusIndex(TABLE_COLLECTION)
        table_index.create(TABLE_FIELDS)

        table_texts = [d["text"] for d in table_docs]
        table_output = embedding.encode(table_texts, return_dense=True, return_sparse=True)

        table_rows = [
            {
                "db_name": d["schema"].get("database", ""),
                "table_name": d["table_name"],
                "table_cn_name": d["schema"].get("display_name", ""),
                "table_comment": (d["schema"].get("description") or "")[:4096],
                "business_domain": ", ".join(d["schema"].get("tags", [])),
                "schema_json": json.dumps(d["schema"], ensure_ascii=False),
            }
            for d in table_docs
        ]
        table_index.insert(table_output["dense_vecs"], table_output["lexical_weights"], table_rows)

        # 3. 列级索引
        column_index = MilvusIndex(COLUMN_COLLECTION)
        column_index.create(COLUMN_FIELDS)

        if column_docs:
            column_texts = [d["text"] for d in column_docs]
            column_output = embedding.encode(column_texts, return_dense=True, return_sparse=True)

            column_rows = [
                {
                    "db_name": schemas[0].get("database", "") if schemas else "",
                    "table_name": d["table_name"],
                    "column_name": d["column_name"],
                    "column_cn_name": d.get("column_cn_name", ""),
                    "column_type": d.get("column_type", ""),
                    "column_comment": d.get("column_comment", ""),
                    "enum_values": d.get("enum_values_summary", "")[:4096],
                    "is_enum": d.get("is_enum", False),
                }
                for d in column_docs
            ]
            column_index.insert(column_output["dense_vecs"], column_output["lexical_weights"], column_rows)

        # 4. 枚举值索引
        enum_index = MilvusIndex(ENUM_COLLECTION)
        enum_index.create(ENUM_FIELDS)

        if enum_docs:
            enum_texts = [d["text"] for d in enum_docs]
            enum_output = embedding.encode(enum_texts, return_dense=True, return_sparse=True)

            enum_rows = [
                {
                    "table_name": d["table_name"],
                    "column_name": d["column_name"],
                    "enum_code": d["enum_code"],
                    "enum_label_cn": d["enum_label_cn"],
                    "description": d.get("description", ""),
                    "synonyms": d.get("synonyms", ""),
                    "sql_value": d["sql_value"],
                }
                for d in enum_docs
            ]
            enum_index.insert(enum_output["dense_vecs"], enum_output["lexical_weights"], enum_rows)

        # 5. Few-shot 索引
        fewshot_index = MilvusIndex(FEWSHOT_COLLECTION)
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        fewshot.build_index(fewshot_examples or [])

        # 6. 术语索引
        glossary_index = MilvusIndex(GLOSSARY_COLLECTION)
        glossary_index.create(GLOSSARY_FIELDS)

        if glossary:
            dense_texts = []   # Dense: term + definition（语义完整）
            sparse_texts = []  # Sparse: term + 同义词（词汇精确匹配）
            glossary_rows = []
            for term, info in glossary.items():
                definition = info.get("definition", "")
                synonyms = info.get("synonyms", [])
                dense_texts.append(f"{term}: {definition}" if definition else term)
                sparse_parts = [term] + synonyms
                sparse_texts.append(" ".join(sparse_parts))
                glossary_rows.append({
                    "term": term,
                    "definition": (definition or "")[:2048],
                    "sql_hint": (info.get("sql_hint", "") or "")[:2048],
                    "related_tables": json.dumps(info.get("related_tables", []), ensure_ascii=False)[:2048],
                    "related_columns": json.dumps(info.get("related_columns", []), ensure_ascii=False)[:2048],
                    "synonyms": json.dumps(synonyms, ensure_ascii=False)[:2048],
                })

            # Dense 用完整文本编码（语义匹配）
            dense_output = embedding.encode(dense_texts, return_dense=True, return_sparse=False)
            # Sparse 只用术语名编码（词汇精确匹配）
            sparse_output = embedding.encode(sparse_texts, return_dense=False, return_sparse=True)
            glossary_index.insert(dense_output["dense_vecs"], sparse_output["lexical_weights"], glossary_rows)
            logger.info(f"术语索引构建: {len(glossary_rows)} 条 (Dense=完整文本, Sparse=术语名)")

        # 7. table_name → schema 映射
        table_schemas = {s["table_name"]: s for s in schemas}

        logger.info("索引构建完成")

        return {
            "table_index": table_index,
            "column_index": column_index,
            "enum_index": enum_index,
            "fewshot_selector": fewshot,
            "glossary_index": glossary_index,
            "table_schemas": table_schemas,
        }
