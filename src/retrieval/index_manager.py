"""索引生命周期管理"""

import json
import hashlib
import logging

from src.retrieval.document_builder import DocumentBuilder
from src.retrieval.embedding import BGEEmbedding
from src.retrieval.milvus_store import MilvusIndex, MilvusMetaStore
from src.retrieval.fewshot_selector import FewShotSelector

logger = logging.getLogger(__name__)

# Milvus Collection 名称
TABLE_COLLECTION = "nl2sql_table"
COLUMN_COLLECTION = "nl2sql_column"
FEWSHOT_COLLECTION = "nl2sql_fewshot"


class IndexManager:
    """索引生命周期管理"""

    def __init__(self):
        self.meta_store = MilvusMetaStore()

    def compute_schema_hash(self, schemas: list[dict]) -> str:
        """计算 Schema 内容 hash"""
        hashable = []
        for s in schemas:
            hashable.append({
                "table_name": s.get("table_name"),
                "display_name": s.get("display_name"),
                "description": s.get("description"),
                "tags": s.get("tags"),
                "columns": [
                    {"name": c["name"], "type": c.get("type", ""), "comment": c.get("comment", ""),
                     "display_name": c.get("display_name", ""), "enum_values": c.get("enum_values")}
                    for c in s.get("columns", [])
                ],
                "relations": s.get("relations"),
                "common_queries": [q.get("question", "") + q.get("sql", "") for q in s.get("common_queries", [])],
            })
        content = json.dumps(hashable, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def need_rebuild(self, schemas: list[dict]) -> bool:
        """检查是否需要重建索引"""
        current_hash = self.compute_schema_hash(schemas)

        stored_hash = self.meta_store.get("schema_hash")
        if not stored_hash:
            logger.info("无 schema_hash，需要构建索引")
            return True

        if stored_hash != current_hash:
            logger.info("Schema 内容已变更，需要重建索引")
            return True

        # 检查 Milvus Collection 是否存在
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
    ) -> dict:
        """
        全量构建索引: 编码文档 → 写入向量存储。

        Returns:
            {
                "table_index": MilvusIndex,
                "column_index": MilvusIndex,
                "fewshot_selector": FewShotSelector,
                "table_schemas": dict,
            }
        """
        builder = DocumentBuilder()

        # 1. 构建文档
        table_docs, column_docs = builder.build_all(schemas)
        logger.info(f"文档构建: {len(table_docs)} 表级, {len(column_docs)} 列级")

        # 2. 表级索引 → Milvus（doc_json 含完整 schema）
        table_index = MilvusIndex(TABLE_COLLECTION)
        table_index.create()

        table_texts = [d["text"] for d in table_docs]
        table_output = embedding.encode(table_texts, return_dense=True, return_sparse=True)

        table_doc_jsons = [
            json.dumps({
                "table_name": d["table_name"],
                "doc_type": d.get("doc_type", "table"),
                "text": d["text"],
                "schema": d["schema"],
            }, ensure_ascii=False)
            for d in table_docs
        ]
        table_index.insert(
            table_output["dense_vecs"].copy(),
            table_output["lexical_weights"],
            table_doc_jsons,
        )

        # 3. 列级索引 → Milvus
        column_index = MilvusIndex(COLUMN_COLLECTION)
        column_index.create()

        if column_docs:
            column_texts = [d["text"] for d in column_docs]
            column_output = embedding.encode(column_texts, return_dense=True, return_sparse=True)

            col_doc_jsons = [
                json.dumps({
                    "table_name": d["table_name"],
                    "column_name": d["column_name"],
                    "doc_type": d.get("doc_type", "column"),
                    "text": d["text"],
                }, ensure_ascii=False)
                for d in column_docs
            ]
            column_index.insert(
                column_output["dense_vecs"].copy(),
                column_output["lexical_weights"],
                col_doc_jsons,
            )

        # 4. Few-shot 索引 → Milvus
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

        # 5. table_name → schema 映射
        table_schemas = {s["table_name"]: s for s in schemas}

        # 6. 保存 schema_hash 到 Milvus 元数据
        schema_hash = self.compute_schema_hash(schemas)
        self.meta_store.set("schema_hash", schema_hash)
        logger.info("索引构建完成")

        return {
            "table_index": table_index,
            "column_index": column_index,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }

    def load_all(self, embedding: BGEEmbedding) -> dict:
        """加载已有索引和元数据"""
        logger.info("加载索引和元数据")

        table_index = MilvusIndex(TABLE_COLLECTION)
        column_index = MilvusIndex(COLUMN_COLLECTION)
        fewshot_index = MilvusIndex(FEWSHOT_COLLECTION)

        # 加载表级文档 → 重建 table_schemas
        table_rows = table_index.query_all()
        table_schemas = {}
        for row in table_rows:
            doc = json.loads(row["doc_json"])
            if "schema" in doc:
                table_schemas[doc["table_name"]] = doc["schema"]

        # 加载 Few-shot 示例
        fewshot_rows = fewshot_index.query_all()
        examples = [json.loads(row["doc_json"]) for row in fewshot_rows]

        # 恢复 FewShotSelector（需要 embeddings 做 MMR）
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        if examples:
            fewshot.examples = examples
            fewshot.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
            texts = [ex["question"] for ex in examples]
            output = embedding.encode(texts, return_dense=True, return_sparse=False)
            fewshot.embeddings = output["dense_vecs"]

        logger.info(
            f"索引加载完成: table={table_index.count}, "
            f"column={column_index.count}, fewshot={fewshot_index.count}, "
            f"schemas={len(table_schemas)}"
        )

        return {
            "table_index": table_index,
            "column_index": column_index,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }
