"""
Schema Linking 值索引 — 将用户 query 中的实体值链接到 Schema 字段。

纯 BM25 检索，不使用 Dense 向量。

检索流程:
    1. jieba 分词 + 引号内容 + 长 token 抽取 -> 候选实体列表
    2. 每个候选对 nl2sql_value 做 BM25 检索
    3. 完全匹配加权 (value == candidate 时 score x 2.0)
    4. 结果注入 Prompt 的 "已识别实体值" 区域

使用示例::

    from src.retrieval.value_indexer import ValueIndexer
    from src.retrieval.milvus_store import MilvusIndex

    value_index = MilvusIndex("nl2sql_value", has_dense=False)
    indexer = ValueIndexer(value_index)
    indexer.build_index(value_docs)

    matches = indexer.match_values("查询持牌商户的交易量")
    # [{"table_name": "pmt_account", "column_name": "account_type",
    #   "enum_label_cn": "持牌商户", "sql_value": "2000", ...}]
"""

import logging
import re

from src.retrieval.milvus_filter import build_metadata_filter
from src.retrieval.milvus_store import MilvusIndex

logger = logging.getLogger(__name__)


class ValueIndexer:
    """
    Schema Linking 值索引器 (纯 BM25)。

    Attributes:
        milvus_index: MilvusIndex 实例 (has_dense=False)
    """

    def __init__(self, milvus_index: MilvusIndex):
        """
        Args:
            milvus_index: 纯 BM25 的 MilvusIndex (has_dense=False)
        """
        self.milvus_index = milvus_index

    def build_index(self, value_docs: list[dict]):
        """
        从预构建的值级文档写入 Milvus。

        Args:
            value_docs: DocumentBuilder.build_value_documents() 返回的文档列表
        """
        if not value_docs:
            logger.info("无值级文档，跳过 value 索引构建")
            return

        texts = [d["text"] for d in value_docs]
        rows = [
            {
                "table_name": d["table_name"],
                "column_name": d["column_name"],
                "biz_line": d.get("biz_line", ""),
                "metadata": d.get("metadata", {}),
                "enum_code": d["enum_code"],
                "enum_label_cn": d["enum_label_cn"],
                "sql_value": d["sql_value"],
            }
            for d in value_docs
        ]
        self.milvus_index.insert(None, texts, rows)
        logger.info(f"Value 索引构建完成: {len(value_docs)} 条")

    def extract_entities(self, query: str) -> list[str]:
        """
        从用户 query 中抽取候选实体值。

        策略:
            1. 中英文引号包裹的内容 (最高优先)
            2. 大写英文词 (如 SGD, USD, SWIFT) — 通常是代码/缩写
            3. jieba 分词中的名词 / 未登录词 / 英文词 (>= 2 字符)

        Args:
            query: 用户原始查询

        Returns:
            去重后的候选实体列表
        """
        import jieba.posseg as pseg

        candidates: set[str] = set()

        # 引号内容
        for m in re.finditer(r'["""\u201c]([^"""\u201d]+)["""\u201d]', query):
            candidates.add(m.group(1).strip())
        for m in re.finditer(r"['''\u2018]([^'''\u2019]+)['''\u2019]", query):
            candidates.add(m.group(1).strip())

        # 大写英文词（>=2字符）：币种代码、协议名等
        for m in re.finditer(r"[A-Z]{2,}", query):
            candidates.add(m.group())

        # jieba 词性标注: 保留名词类 + 未登录词 + 英文词
        keep_flags = {"n", "nr", "ns", "nt", "nz", "ng", "vn", "an", "x", "eng"}
        for word, flag in pseg.cut(query):
            if len(word) >= 2 and flag in keep_flags:
                candidates.add(word)

        candidates.discard("")
        return sorted(candidates, key=lambda value: (-len(value), value.casefold()))

    def match_values(
        self,
        query: str,
        top_k_per_entity: int = 3,
        exact_match_boost: float = 2.0,
        biz_line: str | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        对 query 中的候选实体做 BM25 匹配。

        Args:
            query: 用户原始查询
            top_k_per_entity: 每个候选实体最多返回的匹配数
            biz_line: 业务线过滤，为空则不过滤
            metadata_filter: 任意 KV 过滤

        Returns:
            list[dict]: 按 score 降序，每条包含:
                - table_name, column_name, enum_label_cn, sql_value
                - matched_entity: 匹配到的原始候选文本
                - score: BM25 分数 (完全匹配 x2.0)
        """
        if self.milvus_index.count == 0:
            return []

        candidates = self.extract_entities(query)
        if not candidates:
            return []

        filter_expr = build_metadata_filter(biz_line or "", metadata_filter)
        best_by_key: dict[tuple[str, str, str], dict] = {}

        for entity in candidates:
            hits = self.milvus_index.bm25_search(
                entity,
                top_k=top_k_per_entity,
                output_fields=[
                    "table_name",
                    "column_name",
                    "enum_label_cn",
                    "sql_value",
                ],
                filter_expr=filter_expr,
            )
            for doc_id, score, ent in hits:
                exact_match = any(
                    str(value).casefold() == entity.casefold()
                    for value in (
                        ent.get("enum_label_cn", ""),
                        ent.get("sql_value", ""),
                    )
                    if value is not None
                )
                if exact_match:
                    score *= exact_match_boost

                key = (
                    ent["table_name"],
                    ent["column_name"],
                    str(ent["sql_value"]),
                )
                candidate = {
                    "table_name": ent["table_name"],
                    "column_name": ent["column_name"],
                    "enum_label_cn": ent.get("enum_label_cn", ""),
                    "sql_value": ent["sql_value"],
                    "matched_entity": entity,
                    "exact_match": exact_match,
                    "score": score,
                }
                current = best_by_key.get(key)
                if current is None or (candidate["exact_match"], candidate["score"]) > (
                    current["exact_match"],
                    current["score"],
                ):
                    best_by_key[key] = candidate

        results = list(best_by_key.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        if results:
            logger.info(
                f"Value 匹配: {len(candidates)} 候选 -> {len(results)} 命中, "
                f"top={[(r['enum_label_cn'], r['column_name']) for r in results[:3]]}"
            )
        return results
