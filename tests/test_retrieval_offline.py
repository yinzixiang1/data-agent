"""
离线测试脚本 — 不依赖 Doris/MySQL，用 mock Schema 验证 RAG 全链路。

运行: python -m tests.test_retrieval_offline
"""

import sys
import logging
from pathlib import Path

# 项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_mock_schemas() -> list[dict]:
    """构造 mock Schema 数据（模拟 Doris DDL + MySQL 语义层合并后的结果）"""
    return [
        {
            "database": "dwd_banking",
            "table_name": "pmt_account",
            "table_comment": "普通账户表",
            "display_name": "商户账户表",
            "description": "支付系统的核心账户表，记录所有商户/个人的账户信息",
            "tags": ["账户", "商户", "KYC", "风险", "支付", "开户"],
            "query_tips": "查询时需带 is_delete=0 过滤已删除记录",
            "columns": [
                {"name": "paid", "type": "INT", "key": "UNI", "comment": "", "display_name": "账户主键ID"},
                {"name": "customer_id", "type": "VARCHAR(108)", "key": "", "comment": "客户ID", "display_name": "客户ID"},
                {"name": "account_type", "type": "INT", "key": "", "comment": "账户类型",
                 "display_name": "账户类型",
                 "enum_values": [{"value": 1000, "label": "普通账户"}, {"value": 2000, "label": "LPSP"}, {"value": 2001, "label": "TPSP"}, {"value": 3000, "label": "RPSP"}]},
                {"name": "account_name", "type": "VARCHAR(765)", "key": "", "comment": "商户短名", "display_name": "商户名称"},
                {"name": "legal_entity_type", "type": "INT", "key": "", "comment": "法人实体类型",
                 "display_name": "法人实体类型",
                 "enum_values": [{"value": 1000, "label": "个人"}, {"value": 2000, "label": "企业"}]},
                {"name": "risk_rating_level", "type": "VARCHAR(72)", "key": "", "comment": "风险等级",
                 "display_name": "风险等级",
                 "enum_values": ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]},
                {"name": "risk_score", "type": "INT", "key": "", "comment": "风险评分", "display_name": "风险评分"},
                {"name": "country", "type": "VARCHAR(9)", "key": "", "comment": "", "display_name": "国家"},
                {"name": "monthly_revenue", "type": "VARCHAR(288)", "key": "", "comment": "月收入/营业额", "display_name": "月营业额"},
                {"name": "verification_status", "type": "TINYINT", "key": "", "comment": "KYC/KYB认证状态",
                 "display_name": "KYC/KYB认证状态",
                 "enum_values": [{"value": -1, "label": "拒绝"}, {"value": 1, "label": "已通过"}, {"value": 2, "label": "审核中"}]},
                {"name": "account_status", "type": "TINYINT", "key": "", "comment": "账户状态",
                 "display_name": "账户状态",
                 "enum_values": [{"value": 0, "label": "默认"}, {"value": 1, "label": "活跃"}, {"value": 2, "label": "停用"}]},
                {"name": "white_label_status", "type": "TINYINT", "key": "", "comment": "白标状态",
                 "display_name": "白标状态",
                 "enum_values": [{"value": 0, "label": "未开通"}, {"value": 1, "label": "已开通"}]},
                {"name": "create_time", "type": "DATETIME", "key": "", "comment": "创建时间", "display_name": "创建时间"},
                {"name": "main_email", "type": "VARCHAR(360)", "key": "", "comment": "开户邮箱", "display_name": "开户邮箱"},
                {"name": "is_delete", "type": "INT", "key": "", "comment": "是否删除", "display_name": "是否删除",
                 "enum_values": [{"value": 0, "label": "未删除"}, {"value": 1, "label": "已删除"}]},
                {"name": "metadata", "type": "STRING", "key": "", "comment": "元数据", "skip_index": True},
                {"name": "identification_value", "type": "VARCHAR(144)", "key": "", "comment": "", "sensitive": True},
            ],
            "relations": [
                {"column": "customer_id", "target_table": "dim_customer", "target_column": "customer_id", "join_type": "LEFT JOIN"},
            ],
        },
        {
            "database": "dwd_banking",
            "table_name": "pmt_transaction",
            "table_comment": "交易记录表",
            "display_name": "交易流水表",
            "description": "记录所有支付交易的明细流水，包含交易金额、币种、状态、渠道等信息",
            "tags": ["交易", "支付", "流水", "收款"],
            "columns": [
                {"name": "txn_id", "type": "BIGINT", "key": "UNI", "comment": "交易ID", "display_name": "交易ID"},
                {"name": "paid", "type": "INT", "key": "", "comment": "账户ID", "display_name": "账户ID"},
                {"name": "amount", "type": "DECIMAL(18,2)", "key": "", "comment": "交易金额", "display_name": "交易金额"},
                {"name": "currency", "type": "VARCHAR(10)", "key": "", "comment": "币种", "display_name": "币种"},
                {"name": "txn_status", "type": "VARCHAR(20)", "key": "", "comment": "交易状态",
                 "display_name": "交易状态",
                 "enum_values": ["SUCCESS", "FAILED", "PENDING", "CANCELLED"]},
                {"name": "channel", "type": "VARCHAR(50)", "key": "", "comment": "交易渠道", "display_name": "交易渠道"},
                {"name": "create_time", "type": "DATETIME", "key": "", "comment": "交易时间", "display_name": "交易时间"},
            ],
            "relations": [
                {"column": "paid", "target_table": "pmt_account", "target_column": "paid", "join_type": "INNER JOIN"},
            ],
        },
    ]


def build_mock_glossary() -> dict:
    """构造 mock 术语表"""
    return {
        "活跃商户": {
            "term": "活跃商户",
            "definition": "account_status = 1 且 is_delete = 0 的账户",
            "sql_hint": "account_status = 1 AND is_delete = 0",
            "related_tables": ["pmt_account"],
            "related_columns": ["pmt_account.account_status", "pmt_account.is_delete"],
        },
        "KYC通过率": {
            "term": "KYC通过率",
            "definition": "KYC认证通过的账户数占总账户数的比例",
            "sql_hint": "COUNT(CASE WHEN verification_status = 1 THEN 1 END) / COUNT(*)",
            "related_tables": ["pmt_account"],
            "related_columns": ["pmt_account.verification_status"],
        },
        "高风险商户": {
            "term": "高风险商户",
            "definition": "风险等级为 HIGH 或 VERY_HIGH 的商户",
            "sql_hint": "risk_rating_level IN ('HIGH', 'VERY_HIGH')",
            "related_tables": ["pmt_account"],
            "related_columns": ["pmt_account.risk_rating_level"],
        },
        "LPSP": {
            "term": "LPSP",
            "definition": "License Payment Service Provider，持牌支付服务商",
            "sql_hint": "account_type = 2000",
            "related_tables": ["pmt_account"],
            "related_columns": ["pmt_account.account_type"],
        },
    }


def build_mock_fewshot() -> list[dict]:
    """构造 mock Few-shot 示例"""
    return [
        {
            "question": "目前有多少活跃商户",
            "sql": "SELECT COUNT(*) AS \"活跃商户数\" FROM dwd_banking.pmt_account WHERE account_status = 1 AND is_delete = 0",
            "tables": ["pmt_account"],
            "difficulty": "easy",
        },
        {
            "question": "各国家的商户数量分布",
            "sql": "SELECT country AS \"国家\", COUNT(*) AS \"商户数\" FROM dwd_banking.pmt_account WHERE is_delete = 0 GROUP BY country ORDER BY COUNT(*) DESC",
            "tables": ["pmt_account"],
            "difficulty": "easy",
        },
    ]


def main():
    from src.retrieval.embedding import get_embedding
    from src.retrieval.index_manager import IndexManager
    from src.retrieval.hybrid_searcher import HybridSearcher
    from src.retrieval.reranker import get_reranker
    from src.retrieval.glossary_resolver import GlossaryResolver
    from src.retrieval.schema_formatter import SchemaFormatter
    from src.retrieval.config import RERANK_INPUT_TOP_K, TABLE_SEARCH_TOP_K, ENABLE_RERANKER

    # 1. 准备数据
    schemas = build_mock_schemas()
    glossary = build_mock_glossary()
    fewshot_examples = build_mock_fewshot()

    # 2. 初始化
    embedding = get_embedding()
    index_manager = IndexManager()
    indices = index_manager.build(schemas, embedding, fewshot_examples=fewshot_examples)

    searcher = HybridSearcher(
        embedding=embedding,
        table_index=indices["table_index"],
        column_index=indices["column_index"],
        enum_index=indices["enum_index"],
        table_schemas=indices["table_schemas"],
    )

    fewshot = indices["fewshot_selector"]
    glossary_resolver = GlossaryResolver()
    glossary_resolver.load(glossary)
    formatter = SchemaFormatter()
    reranker = get_reranker()

    # 3. 测试查询
    test_queries = [
        "目前有多少活跃商户",
        "KYC通过率是多少",
        "高风险商户有哪些",
        "各国家的商户分布",
        "今天的交易总额",
        "LPSP和TPSP分别有多少",
        "最近一个月新开户数",
        "交易成功率",
        "pmt_account",
        "白标商户数量",
    ]

    print("\n" + "=" * 80)
    print("RAG 检索测试")
    print("=" * 80)

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"问题: {query}")
        print(f"{'─' * 60}")

        # 术语解析
        g_result = glossary_resolver.resolve(query)
        enriched = g_result["enriched_query"]
        if g_result["matched_terms"]:
            print(f"  术语匹配: {g_result['matched_terms']}")
            print(f"  增强查询: {enriched}")

        # 混合检索
        rerank_k = RERANK_INPUT_TOP_K if ENABLE_RERANKER else TABLE_SEARCH_TOP_K
        candidates = searcher.search(enriched, top_k=rerank_k)

        # Reranker
        if reranker and len(candidates) > TABLE_SEARCH_TOP_K:
            candidates = reranker.rerank(query, candidates, top_k=TABLE_SEARCH_TOP_K)

        print(f"  命中表: {[c['table_name'] for c in candidates]}")
        for c in candidates:
            score_key = "rerank_score" if "rerank_score" in c else "score"
            print(f"    - {c['table_name']}: {score_key}={c.get(score_key, 0):.4f}")

        # Few-shot
        examples = fewshot.select(query, tables=[c["table_name"] for c in candidates], top_k=3)
        if examples:
            print(f"  Few-shot: {[ex['question'][:30] for ex in examples]}")

        # Prompt 预览
        prompt = formatter.format_all(candidates, examples, g_result["business_context"])
        print(f"  Prompt 长度: {len(prompt)} 字符")

    print(f"\n{'=' * 80}")
    print("全部测试通过")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
