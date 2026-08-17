"""
知识库检索器 — 从 Milvus 检索知识文档并组装 Prompt。

支持:
    1. 从 MySQL (da_knowledge_doc) 同步文档到 Milvus
    2. 文本分块 (固定窗口 + 重叠)
    3. Dense + BM25 混合检索
    4. 知识问答 Prompt 组装

使用示例::

    from src.retrieval.knowledge_retriever import KnowledgeRetriever
    from src.retrieval.embedding import get_embedding

    kr = KnowledgeRetriever(get_embedding())
    kr.sync_from_db(agent_id=1)

    chunks = kr.retrieve("什么是清算周期", top_k=5)
    prompt = kr.format_prompt("什么是清算周期", chunks)
"""

import logging
import re
from dataclasses import dataclass, field

from pymilvus import DataType, RRFRanker

from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.embedding import BaseEmbedding
from src.retrieval.collection_names import agent_collection_name

logger = logging.getLogger(__name__)

COLLECTION_NAME = "nl2sql_knowledge"

# 分块参数
CHUNK_SIZE = 512  # 每个 chunk 的目标字符数
CHUNK_OVERLAP = 64  # 相邻 chunk 重叠字符数
MIN_CHUNK_SIZE = 50  # 低于此长度的 chunk 丢弃


@dataclass
class KnowledgeResult:
    """知识检索结果。"""

    chunks: list[dict] = field(default_factory=list)
    prompt_text: str = ""
    source_docs: list[str] = field(default_factory=list)


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    将文本按段落和固定窗口分块。

    策略:
        1. 先按双换行拆段落
        2. 短段落合并，长段落按 chunk_size 滑窗切割
    """
    if not text or not text.strip():
        return []

    # 按双换行或多空行分段
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}" if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            # 长段落按窗口切割
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
            else:
                buffer = para
                continue
            buffer = ""

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


class KnowledgeRetriever:
    """
    知识库检索器。

    管理 Milvus 中的知识文档向量集合，
    提供文档同步、检索和 Prompt 组装功能。
    """

    def __init__(self, embedding: BaseEmbedding, agent_id: int | None = None):
        self.embedding = embedding
        self.agent_id = agent_id
        self.collection_name = agent_collection_name(COLLECTION_NAME, agent_id)
        self.index: MilvusIndex | None = None
        self._initialized = False

    def _ensure_index(self):
        """确保 Milvus index 已初始化。"""
        if self.index is None:
            self.index = MilvusIndex(self.collection_name, dim=self.embedding.dim)
        return self.index

    def connect(self):
        """连接到已有的知识库 Collection（不重建）。"""
        idx = self._ensure_index()
        if idx.exists():
            self._initialized = True
            logger.info(
                f"知识库 Collection 已连接: {self.collection_name} ({idx.count} 条)"
            )
        else:
            logger.info(f"知识库 Collection 不存在: {self.collection_name}，等待同步")

    def sync_from_db(self, agent_id: int | None = None):
        """
        从 MySQL 加载知识文档，分块 + 向量化后写入 Milvus。

        Args:
            agent_id: Agent ID，用于过滤绑定的知识库 (is_public=1 + Agent 绑定的)
        """
        if self.agent_id is not None:
            if agent_id is not None and agent_id != self.agent_id:
                raise ValueError(
                    f"知识检索器绑定 Agent {self.agent_id}，不能同步 Agent {agent_id}"
                )
            agent_id = self.agent_id

        from sqlalchemy import create_engine, text as sql_text
        from src.retrieval.config import (
            MYSQL_HOST,
            MYSQL_PORT,
            MYSQL_USER,
            MYSQL_PASSWORD_URL,
            MYSQL_DATABASE,
        )

        url = (
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
            f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
        )
        engine = create_engine(url, pool_size=2, pool_recycle=3600)

        try:
            with engine.connect() as conn:
                # 加载启用的知识库 ID（公共 + Agent 绑定）
                if agent_id:
                    kb_rows = conn.execute(
                        sql_text(
                            "SELECT kb.id, kb.name FROM da_knowledge_base kb "
                            "WHERE kb.status = 1 AND ("
                            "  kb.is_public = 1 "
                            "  OR kb.id IN ("
                            "    SELECT CAST(r.resource_key AS UNSIGNED) FROM da_agent_ref r "
                            "    WHERE r.agent_id = :agent_id AND r.resource_type = 'knowledge'"
                            "  )"
                            ")"
                        ),
                        {"agent_id": agent_id},
                    ).fetchall()
                else:
                    kb_rows = conn.execute(
                        sql_text(
                            "SELECT id, name FROM da_knowledge_base WHERE status = 1"
                        )
                    ).fetchall()

                if not kb_rows:
                    logger.info("无启用的知识库，跳过同步")
                    return

                kb_ids = [r[0] for r in kb_rows]
                kb_names = {r[0]: r[1] for r in kb_rows}
                logger.info(
                    f"知识库同步: {len(kb_ids)} 个知识库 — {list(kb_names.values())}"
                )

                # 加载文档
                placeholders = ",".join(str(i) for i in kb_ids)
                docs = conn.execute(
                    sql_text(
                        f"SELECT id, kb_id, title, content, doc_type "
                        f"FROM da_knowledge_doc WHERE kb_id IN ({placeholders}) AND content != ''"
                    )
                ).fetchall()

            if not docs:
                logger.info("无知识文档，跳过同步")
                return

            logger.info(f"加载 {len(docs)} 篇文档，开始分块...")

            # 分块
            all_chunks = []
            all_texts = []
            for doc in docs:
                doc_id, kb_id, title, content, doc_type = doc
                chunks = _chunk_text(content)
                for i, chunk in enumerate(chunks):
                    # BM25 文本 = 标题 + 内容
                    bm25_text = f"{title} {chunk}" if title else chunk
                    all_texts.append(bm25_text)
                    all_chunks.append(
                        {
                            "doc_id": doc_id,
                            "kb_id": kb_id,
                            "kb_name": kb_names.get(kb_id, ""),
                            "title": title,
                            "chunk_idx": i,
                            "chunk_text": chunk,
                        }
                    )

            if not all_chunks:
                logger.info("分块结果为空，跳过")
                return

            logger.info(f"分块完成: {len(docs)} 篇文档 → {len(all_chunks)} 个 chunk")

            # Dense 编码
            logger.info("开始 Dense 编码...")
            dense_vecs = self.embedding.encode(all_texts)
            logger.info(f"Dense 编码完成: shape={dense_vecs.shape}")

            # 创建 Collection 并写入
            idx = self._ensure_index()
            idx.create(
                scalar_fields=[
                    {"name": "doc_id", "dtype": DataType.INT64},
                    {"name": "kb_id", "dtype": DataType.INT64, "inverted": True},
                    {"name": "kb_name", "dtype": DataType.VARCHAR, "max_length": 256},
                    {"name": "title", "dtype": DataType.VARCHAR, "max_length": 512},
                    {"name": "chunk_idx", "dtype": DataType.INT64},
                    {
                        "name": "chunk_text",
                        "dtype": DataType.VARCHAR,
                        "max_length": 65535,
                    },
                ],
            )

            rows = [
                {
                    "doc_id": c["doc_id"],
                    "kb_id": c["kb_id"],
                    "kb_name": c["kb_name"],
                    "title": c["title"],
                    "chunk_idx": c["chunk_idx"],
                    "chunk_text": c["chunk_text"],
                }
                for c in all_chunks
            ]

            idx.insert(dense_vecs, all_texts, rows)
            self._initialized = True
            logger.info(
                f"知识库同步完成: {len(all_chunks)} 个 chunk 已写入 {self.collection_name}"
            )

            # 更新 MySQL 文档状态
            with engine.connect() as conn:
                doc_ids = list({c["doc_id"] for c in all_chunks})
                placeholders = ",".join(str(d) for d in doc_ids)
                conn.execute(
                    sql_text(
                        f"UPDATE da_knowledge_doc SET status = 1 WHERE id IN ({placeholders})"
                    )
                )
                conn.commit()

        finally:
            engine.dispose()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        kb_ids: list[int] | None = None,
    ) -> list[dict]:
        """
        检索与 query 相关的知识文档 chunk。

        Args:
            query: 用户问题
            top_k: 返回数量
            kb_ids: 限定的知识库 ID 列表（为空则不过滤）

        Returns:
            [{"title", "chunk_text", "kb_name", "score", ...}, ...]
        """
        if not self._initialized:
            logger.warning("知识库未初始化，返回空结果")
            return []

        idx = self._ensure_index()
        if not idx.exists() or idx.count == 0:
            return []

        # 编码查询
        q_vec = self.embedding.encode_query(query, collection_type="glossary")

        # 构建过滤表达式
        filter_expr = None
        if kb_ids:
            id_list = ", ".join(str(i) for i in kb_ids)
            filter_expr = f"kb_id in [{id_list}]"

        # 混合检索
        ranker = RRFRanker(k=60)
        hits = idx.hybrid_search(
            query_dense=q_vec,
            query_text=query,
            ranker=ranker,
            recall_k=top_k * 3,
            output_fields=[
                "doc_id",
                "kb_id",
                "kb_name",
                "title",
                "chunk_idx",
                "chunk_text",
            ],
            filter_expr=filter_expr,
        )

        results = []
        seen_chunks = set()
        for doc_id, score, entity in hits[:top_k]:
            chunk_key = (entity.get("doc_id"), entity.get("chunk_idx"))
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            results.append(
                {
                    "doc_id": entity.get("doc_id"),
                    "kb_id": entity.get("kb_id"),
                    "kb_name": entity.get("kb_name", ""),
                    "title": entity.get("title", ""),
                    "chunk_text": entity.get("chunk_text", ""),
                    "chunk_idx": entity.get("chunk_idx"),
                    "score": round(score, 4),
                }
            )

        logger.info(f"知识检索完成: query='{query}', hits={len(results)}")
        return results

    def format_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        将检索到的知识 chunk 组装为 LLM Prompt。

        Args:
            question: 用户问题
            chunks: retrieve() 返回的 chunk 列表

        Returns:
            完整的 User Message 文本
        """
        if not chunks:
            return (
                f"用户问题：{question}\n\n未找到相关知识文档，请根据你的通用知识回答。"
            )

        context_parts = []
        for i, c in enumerate(chunks, 1):
            source = f"[{c.get('kb_name', '')}] {c.get('title', '')}"
            context_parts.append(f"【文档 {i}】{source}\n{c['chunk_text']}")

        context = "\n\n".join(context_parts)

        return (
            f"【参考文档】\n{context}\n\n"
            f"【用户问题】\n{question}\n\n"
            f"【回答要求】\n"
            f"请根据上述参考文档回答用户问题。如果文档中没有直接答案，"
            f"请说明并基于文档信息给出合理推断。引用来源文档编号。"
        )
