"""索引生命周期管理 — Milvus 版"""

import json
import hashlib
import logging
from pathlib import Path

from src.retrieval.config import INDEX_STORE_DIR
from src.retrieval.document_builder import DocumentBuilder
from src.retrieval.embedding import BGEEmbedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.fewshot_selector import FewShotSelector

logger = logging.getLogger(__name__)

# Milvus Collection 名称
TABLE_COLLECTION = "nl2sql_table"
COLUMN_COLLECTION = "nl2sql_column"
FEWSHOT_COLLECTION = "nl2sql_fewshot"


class IndexManager:
    """索引生命周期管理（Milvus 版）"""

    def __init__(self, index_dir: str | Path = INDEX_STORE_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

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
        hash_file = self.index_dir / "schema_hash.txt"
        current_hash = self.compute_schema_hash(schemas)

        if not hash_file.exists():
            return True

        stored_hash = hash_file.read_text().strip()
        if stored_hash != current_hash:
            logger.info("Schema 内容已变更，需要重建索引")
            return True

        # 检查 Milvus Collection 是否存在
        table_idx = MilvusIndex(TABLE_COLLECTION)
        column_idx = MilvusIndex(COLUMN_COLLECTION)
        fewshot_idx = MilvusIndex(FEWSHOT_COLLECTION)

        if not table_idx.exists() or not fewshot_idx.exists():
            logger.info("Milvus Collection 不存在，需要重建")
            return True

        # 检查元数据文件
        for f in ["table_docs.json", "column_docs.json", "fewshot_examples.json", "table_schemas.json"]:
            if not (self.index_dir / f).exists():
                logger.info(f"元数据文件缺失: {f}，需要重建")
                return True

        return False

    def build_and_save(
        self,
        schemas: list[dict],
        embedding: BGEEmbedding,
    ) -> dict:
        """
        全量构建索引: 编码文档 → 写入 Milvus → 保存元数据。

        Returns:
            {
                "table_index": MilvusIndex,
                "column_index": MilvusIndex,
                "table_docs": list[dict],
                "column_docs": list[dict],
                "fewshot_selector": FewShotSelector,
                "table_schemas": dict,
            }
        """
        builder = DocumentBuilder()

        # 1. 构建文档
        table_docs, column_docs = builder.build_all(schemas)
        logger.info(f"文档构建: {len(table_docs)} 表级, {len(column_docs)} 列级")

        # 2. 表级索引 → Milvus
        table_index = MilvusIndex(TABLE_COLLECTION)
        table_index.create()

        table_texts = [d["text"] for d in table_docs]
        table_output = embedding.encode(table_texts, return_dense=True, return_sparse=True)

        table_doc_jsons = [
            json.dumps({"table_name": d["table_name"], "text": d["text"]}, ensure_ascii=False)
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
                json.dumps({"table_name": d["table_name"], "column_name": d["column_name"], "text": d["text"]}, ensure_ascii=False)
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

        # 6. 保存元数据到本地（Milvus 存向量，本地存文档元数据）
        self._save_metadata(table_docs, column_docs, all_examples, table_schemas, schemas)

        return {
            "table_index": table_index,
            "column_index": column_index,
            "table_docs": table_docs,
            "column_docs": column_docs,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }

    def load_all(self, embedding: BGEEmbedding) -> dict:
        """从 Milvus + 本地元数据加载"""
        logger.info("从 Milvus + 本地元数据加载索引")

        table_index = MilvusIndex(TABLE_COLLECTION)
        column_index = MilvusIndex(COLUMN_COLLECTION)
        fewshot_index = MilvusIndex(FEWSHOT_COLLECTION)

        # 本地元数据
        table_docs = json.loads((self.index_dir / "table_docs.json").read_text(encoding="utf-8"))
        column_docs = json.loads((self.index_dir / "column_docs.json").read_text(encoding="utf-8"))
        examples = json.loads((self.index_dir / "fewshot_examples.json").read_text(encoding="utf-8"))
        table_schemas = json.loads((self.index_dir / "table_schemas.json").read_text(encoding="utf-8"))

        # 恢复 FewShotSelector（需要 embeddings 做 MMR）
        fewshot = FewShotSelector(embedding, milvus_index=fewshot_index)
        if examples:
            fewshot.examples = examples
            fewshot.example_table_sets = [set(ex.get("tables", [])) for ex in examples]
            # 重新编码以获取 embeddings（MMR 需要）
            texts = [ex["question"] for ex in examples]
            output = embedding.encode(texts, return_dense=True, return_sparse=False)
            fewshot.embeddings = output["dense_vecs"]

        logger.info(
            f"索引加载完成: table={table_index.count}, "
            f"column={column_index.count}, fewshot={fewshot_index.count}"
        )

        return {
            "table_index": table_index,
            "column_index": column_index,
            "table_docs": table_docs,
            "column_docs": column_docs,
            "fewshot_selector": fewshot,
            "table_schemas": table_schemas,
        }

    def _save_metadata(self, table_docs, column_docs, examples, table_schemas, schemas):
        """保存文档元数据到本地（向量已在 Milvus 中）"""
        table_docs_save = [{k: v for k, v in d.items() if k != "schema"} for d in table_docs]
        (self.index_dir / "table_docs.json").write_text(
            json.dumps(table_docs_save, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.index_dir / "column_docs.json").write_text(
            json.dumps(column_docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.index_dir / "fewshot_examples.json").write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.index_dir / "table_schemas.json").write_text(
            json.dumps(table_schemas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        schema_hash = self.compute_schema_hash(schemas)
        (self.index_dir / "schema_hash.txt").write_text(schema_hash)
        logger.info(f"元数据保存完成: {self.index_dir}")
