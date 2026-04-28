"""索引生命周期管理"""

import json
import hashlib
import logging

from pymilvus import DataType

from src.retrieval.document_builder import DocumentBuilder
from src.retrieval.embedding import BGEEmbedding
from src.retrieval.milvus_store import MilvusIndex, MilvusMetaStore
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


class IndexManager:
    """索引生命周期管理"""

    def __init__(self):
        self.meta_store = MilvusMetaStore()

    def compute_schema_hash(
        self, schemas: list[dict], enums: list[dict] | None = None,
    ) -> str:
        """计算 Schema + 枚举 内容 hash"""
        hashable = []
        for s in schemas:
            hashable.append({
                "table_name": s.get("table_name"),
                "display_name": s.get("display_name"),
                "description": s.get("description"),
                "tags": s.get("tags"),
                "columns": [
                    {"name": c["name"], "type": c.get("type", ""), "comment": c.get("comment", ""),
                     "display_name": c.get("display_name", "")}
                    for c in s.get("columns", [])
                ],
                "relations": s.get("relations"),
                "common_queries": [q.get("question", "") + q.get("sql", "") for q in s.get("common_queries", [])],
            })
        if enums:
            hashable.append({"enums": enums})
        content = json.dumps(hashable, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def need_rebuild(
        self, schemas: list[dict], enums: list[dict] | None = None,
    ) -> bool:
        """检查是否需要重建索引"""
        current_hash = self.compute_schema_hash(schemas, enums)

        stored_hash = self.meta_store.get("schema_hash")
        if not stored_hash:
            logger.info("无 schema_hash，需要构建索引")
            return True

        if stored_hash != current_hash:
            logger.info("Schema 内容已变更，需要重建索引")
            return True

        table_idx = MilvusIndex(TABLE_COLLECTION)
        fewshot_idx = MilvusIndex(FEWSHOT_COLLECTION)

        if not table_idx.exists() or not fewshot_idx.exists():
            logger.info("Collection 不存在，需要重建")
            return True

        return False

    def build_and_save(
        self,
        schemas: list[dict],
        embedding: BGEEmbedding,
        enums: list[dict] | None = None,
    ) -> dict:
        """
        全量构建索引: 编码文档 → 写入向量存储。

        Returns:
            {
                "table_index": MilvusIndex,
                "column_index": MilvusIndex,
                "enum_index": MilvusIndex,
                "fewshot_selector": FewShotSelector,
                "table_schemas": dict,
            }
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
        all_examples = []
        for schema in schemas:
            for q in schema.get("common_queries", []):
                all_examples.append({
                    "question": q["question"],
                    "sql": q["sql"],
                    "tables": q.get("tables", [schema["table_name"]]),
                    "difficulty": q.get("difficulty", ""),
                })
        fewshot.build_index(all_examples)

        # 6. table_name → schema 映射
        table_schemas = {s["table_name"]: s for s in schemas}

        # 7. 保存 schema_hash
        schema_hash = self.compute_schema_hash(schemas, enums)
        self.meta_store.set("schema_hash", schema_hash)
        logger.info("索引构建完成")

        return {
            "table_index": table_index,
            "column_index": column_index,
            "enum_index": enum_index,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }

    def load_all(self, embedding: BGEEmbedding) -> dict:
        """加载已有索引和元数据"""
        logger.info("加载索引和元数据")

        table_index = MilvusIndex(TABLE_COLLECTION)
        column_index = MilvusIndex(COLUMN_COLLECTION)
        enum_index = MilvusIndex(ENUM_COLLECTION)
        fewshot_index = MilvusIndex(FEWSHOT_COLLECTION)

        # 加载表级文档 → 重建 table_schemas
        table_rows = table_index.query_all(output_fields=["table_name", "schema_json"])
        table_schemas = {}
        for row in table_rows:
            schema_json = row.get("schema_json", "{}")
            if schema_json:
                table_schemas[row["table_name"]] = json.loads(schema_json)

        # 加载 Few-shot 示例
        fewshot_rows = fewshot_index.query_all(output_fields=["question", "sql", "involved_tables", "difficulty"])
        examples = [
            {
                "question": row["question"],
                "sql": row["sql"],
                "tables": row.get("involved_tables", "").split(",") if row.get("involved_tables") else [],
                "difficulty": row.get("difficulty", ""),
            }
            for row in fewshot_rows
        ]

        # 恢复 FewShotSelector（需要 embeddings 做 MMR）
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        if examples:
            fewshot.examples = examples
            fewshot.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
            texts = [ex["question"] for ex in examples]
            output = embedding.encode(texts, return_dense=True, return_sparse=False)
            fewshot.embeddings = output["dense_vecs"]

        logger.info(
            f"索引加载完成: table={table_index.count}, column={column_index.count}, "
            f"enum={enum_index.count}, fewshot={fewshot_index.count}, schemas={len(table_schemas)}"
        )

        return {
            "table_index": table_index,
            "column_index": column_index,
            "enum_index": enum_index,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }
