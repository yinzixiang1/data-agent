"""
混合检索 — Dense + BM25 hybrid search，表级 + 列级 + 枚举联合。

检索流程:
    1. 表级混合检索（Dense + BM25，per-collection ranker）
    2. 列级混合检索 -> 命中的列反推其所属表，给表加分
    3. 枚举值检索 -> 命中的枚举反哺其关联表分数
    4. 关联表补全 -> top 表的 relations 中的关联表获得 bonus 分

使用示例::

    searcher = HybridSearcher(embedding, table_index, column_index,
                              enum_index, table_schemas, config)

    results = searcher.search("活跃商户数量", top_k=5)
    enums = searcher.search_enums("持牌商户")
"""

import logging
import re

from src.retrieval.embedding import Qwen3Embedding
from src.retrieval.milvus_store import MilvusIndex
from src.retrieval.ranker_strategy import get_search_params
from src.retrieval.agent_config import AgentRuntimeConfig

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器：表级 + 列级 + 枚举联合检索，多信号融合排序。

    Attributes:
        embedding: Qwen3Embedding 实例
        table_index: 表级 MilvusIndex
        column_index: 列级 MilvusIndex
        enum_index: 枚举值 MilvusIndex
        table_schemas: {table_name: schema_dict} 映射
        config: AgentRuntimeConfig（提供 collection_search_config 和 index_build_config）
    """

    def __init__(
        self,
        embedding: Qwen3Embedding,
        table_index: MilvusIndex,
        column_index: MilvusIndex,
        enum_index: MilvusIndex,
        table_schemas: dict,
        config: AgentRuntimeConfig,
    ):
        self.embedding = embedding
        self.table_index = table_index
        self.column_index = column_index
        self.enum_index = enum_index
        self.table_schemas = table_schemas
        self.config = config
        self._rebuild_short_map()

    def _rebuild_short_map(self):
        """构建 short_name -> full_name 反查映射（用于关联表解析）。"""
        self._short_to_full: dict[str, str] = {}
        for full_name in self.table_schemas:
            short = full_name.split(".", 1)[1] if "." in full_name else full_name
            self._short_to_full[short] = full_name

    def _resolve_full_name(self, target: str) -> str | None:
        """将关联表名（可能是短名或全名）解析为 table_schemas 中的全限定名。"""
        if target in self.table_schemas:
            return target
        return self._short_to_full.get(target)

    @property
    def _ef_search(self) -> int:
        return self.config.index_build_config.get("hnsw", {}).get("ef_search", 64)

    def _biz_filter(self, biz_line: str | None) -> str | None:
        """生成 biz_line 过滤表达式，sys 业务线的数据不被过滤。"""
        if not biz_line:
            return None
        return f'biz_line == "{biz_line}" or biz_line == "sys"'

    def search(self, query: str, top_k: int = 5, biz_line: str | None = None) -> list[dict]:
        """
        表级 + 列级混合检索，融合枚举反哺和关联表补全。

        评分规则:
            - 表级检索: ranker 融合基础分
            - 列级命中: +0.01（列所属的表）
            - 枚举反哺: +0.02（枚举值关联的表）
            - 关联补全: +parent_score x 0.1（top 表的关联表）

        Args:
            query: 用户查询（可能已经过术语增强）
            top_k: 最终返回的表数量
            biz_line: 业务线过滤，为空则不过滤

        Returns:
            list[dict]: 按 score 降序，每个元素含 table_name, score, source,
                hit_by_column, schema
        """
        csc = self.config.collection_search_config
        filter_expr = self._biz_filter(biz_line)

        # ── 表级混合检索 ──
        table_params = get_search_params(csc, "table")
        q_dense_table = self.embedding.encode_query(query, "table")
        table_results = self.table_index.hybrid_search(
            q_dense_table,
            query,
            ranker=table_params.ranker,
            recall_k=table_params.recall_limit,
            output_fields=["table_name"],
            filter_expr=filter_expr,
            ef_search=self._ef_search,
        )

        # ── 列级混合检索 -> 反推表 ──
        column_hit_tables: set[str] = set()
        if self.column_index.count > 0:
            col_params = get_search_params(csc, "column")
            q_dense_col = self.embedding.encode_query(query, "column")
            col_results = self.column_index.hybrid_search(
                q_dense_col,
                query,
                ranker=col_params.ranker,
                recall_k=col_params.recall_limit,
                output_fields=["table_name"],
                filter_expr=filter_expr,
                ef_search=self._ef_search,
            )
            for doc_id, score, entity in col_results:
                column_hit_tables.add(entity["table_name"])

        # ── 合并：表级 + 列级反推 bonus ──
        table_scores: dict[str, float] = {}
        for doc_id, score, entity in table_results:
            table_name = entity["table_name"]
            table_scores[table_name] = max(table_scores.get(table_name, 0), score)

        for table_name in column_hit_tables:
            if table_name in table_scores:
                table_scores[table_name] += 0.01
            else:
                table_scores[table_name] = 0.01

        # ── 枚举命中反哺表分数 ──
        enum_boost_tables: set[str] = set()
        if self.enum_index.count > 0:
            enum_params = get_search_params(csc, "enum")
            q_dense_enum = self.embedding.encode_query(query, "enum")
            enum_results = self.enum_index.hybrid_search(
                q_dense_enum,
                query,
                ranker=enum_params.ranker,
                recall_k=enum_params.recall_limit,
                output_fields=["table_name"],
                filter_expr=filter_expr,
                ef_search=self._ef_search,
            )
            for doc_id, score, entity in enum_results:
                enum_boost_tables.add(entity["table_name"])
            for table_name in enum_boost_tables:
                if table_name in self.table_schemas:
                    table_scores.setdefault(table_name, 0)
                    table_scores[table_name] += 0.02
            if enum_boost_tables:
                logger.info(f"枚举反哺: {enum_boost_tables}")

        # ── 关联表补全 ──
        current_top = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        relation_boosted: set[str] = set()
        for table_name, score in current_top:
            schema = self.table_schemas.get(table_name, {})
            for rel in schema.get("relations", []):
                target_short = rel.get("target_table", "")
                if not target_short:
                    continue
                related = self._resolve_full_name(target_short)
                if not related or related == table_name:
                    continue
                bonus = score * 0.1
                table_scores.setdefault(related, 0)
                table_scores[related] += bonus
                relation_boosted.add(related)
        if relation_boosted:
            logger.info(f"关联补全: {relation_boosted}")

        # ── 币种转换意图检测 → 强制注入汇率表 ──
        self._inject_exchange_rate_table(query, table_scores)

        # 按分数排序
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 组装结果
        results = []
        for table_name, score in sorted_tables:
            schema = self.table_schemas.get(table_name, {})
            results.append({
                "table_name": table_name,
                "score": score,
                "source": "hybrid",
                "hit_by_column": table_name in column_hit_tables,
                "schema": schema,
            })

        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"表级候选={len(table_results)}, 列级反推={len(column_hit_tables)}, "
            f"枚举反哺={len(enum_boost_tables)}, 关联补全={len(relation_boosted)}, "
            f"最终={len(results)} 张表: {[r['table_name'] for r in results]}"
        )
        return results

    # 常见法币/加密货币代码（3 位大写字母）
    _CURRENCY_CODES = {
        "USD", "EUR", "GBP", "JPY", "CNY", "CNH", "HKD", "SGD", "AUD", "CAD",
        "CHF", "NZD", "KRW", "TWD", "THB", "MYR", "IDR", "PHP", "VND", "INR",
        "AED", "SAR", "BRL", "MXN", "ZAR", "TRY", "RUB", "PLN", "SEK", "NOK",
        "DKK", "CZK", "HUF", "ILS", "CLP", "ARS", "PEN", "COP", "NGN", "KES",
        "BTC", "ETH", "USDT", "USDC",
    }
    _CONVERT_KEYWORDS = re.compile(r"转|换|兑|换算|转换|折算|convert", re.IGNORECASE)
    _EXCHANGE_RATE_KEYWORDS = re.compile(r"汇率|exchange\s*rate|FX", re.IGNORECASE)

    def _inject_exchange_rate_table(self, query: str, table_scores: dict[str, float]):
        """检测币种/汇率意图，命中时将汇率表注入候选。

        触发路径（OR 关系）:
            1. 直接提及: query 含 "汇率" / "exchange rate" / "FX"
            2. 币种 + 转换: query 含币种代码(SGD/USD...) + 转换关键词(转/换/兑...)
        """
        # 路径1: 直接提及汇率
        direct_hit = bool(self._EXCHANGE_RATE_KEYWORDS.search(query))

        # 路径2: 币种代码 + 转换关键词
        found_currencies: set[str] = set()
        if not direct_hit:
            found_currencies = {m.group().upper() for m in re.finditer(r"[A-Za-z]{3,5}", query)} & self._CURRENCY_CODES
            if not found_currencies or not self._CONVERT_KEYWORDS.search(query):
                return

        # 从 table_schemas 中查找含 exchange_rate 的表
        exchange_table = None
        for full_name in self.table_schemas:
            short = full_name.split(".", 1)[1] if "." in full_name else full_name
            if "exchange_rate" in short and "source" not in short:
                exchange_table = full_name
                break

        if not exchange_table:
            return

        trigger = "汇率关键词" if direct_hit else f"币种{found_currencies}"
        max_score = max(table_scores.values()) if table_scores else 0.5
        if exchange_table in table_scores:
            table_scores[exchange_table] += 0.02
            logger.info(f"汇率表意图检测({trigger}): 加分 {exchange_table}")
        else:
            table_scores[exchange_table] = max_score * 0.8
            logger.info(f"汇率表意图检测({trigger}): 注入 {exchange_table} (score={max_score * 0.8:.4f})")

    def search_enums(self, query: str, top_k: int = 8, biz_line: str | None = None) -> list[dict]:
        """
        枚举值检索 — 将用户自然语言映射到实际枚举值。

        Args:
            query: 用户原始查询
            top_k: 返回的最大枚举命中数
            biz_line: 业务线过滤，为空则不过滤

        Returns:
            list[dict]: table_name, column_name, enum_label_cn, sql_value, score
        """
        if self.enum_index.count == 0:
            return []

        enum_params = get_search_params(self.config.collection_search_config, "enum")
        q_dense = self.embedding.encode_query(query, "enum")
        filter_expr = self._biz_filter(biz_line)

        results = self.enum_index.hybrid_search(
            q_dense,
            query,
            ranker=enum_params.ranker,
            recall_k=max(top_k, enum_params.recall_limit),
            output_fields=["table_name", "column_name", "enum_label_cn", "sql_value", "enum_code"],
            filter_expr=filter_expr,
            ef_search=self._ef_search,
        )

        enum_hits = []
        for doc_id, score, entity in results[:top_k]:
            enum_hits.append({
                "table_name": entity["table_name"],
                "column_name": entity["column_name"],
                "enum_label_cn": entity["enum_label_cn"],
                "sql_value": entity["sql_value"],
                "score": score,
            })

        if enum_hits:
            logger.info(
                f"枚举检索: {len(enum_hits)} 条命中, "
                f"top={[f'{e['enum_label_cn']}->{e['column_name']}={e['sql_value']}' for e in enum_hits[:3]]}"
            )
        return enum_hits
