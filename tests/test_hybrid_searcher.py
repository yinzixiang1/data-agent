from src.retrieval.agent_config import AgentRuntimeConfig
from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.hybrid_searcher import HybridSearcher


class _Embedding:
    def encode_query(self, _query: str, _collection_type: str) -> list[float]:
        return [0.1]


class _Index:
    def __init__(self, results: list[tuple[int, float, dict]]) -> None:
        self.results = results
        self.count = len(results)

    def hybrid_search(self, *_args, **_kwargs) -> list[tuple[int, float, dict]]:
        return self.results


def _searcher(
    table_results: list[tuple[int, float, dict]],
    schemas: dict[str, dict],
) -> HybridSearcher:
    return HybridSearcher(
        _Embedding(),
        _Index(table_results),
        _Index([]),
        _Index([]),
        schemas,
        AgentRuntimeConfig(enable_reranker=False),
    )


def test_relation_neighbor_does_not_become_recall_candidate() -> None:
    schemas = {
        "analytics.transactions": {
            "table_name": "analytics.transactions",
            "relations": [
                {
                    "column_name": "balance_id",
                    "target_table": "balances",
                    "target_column": "balance_id",
                }
            ],
        },
        "analytics.balances": {
            "table_name": "analytics.balances",
            "relations": [],
        },
    }
    searcher = _searcher([(1, 0.9, {"table_name": "analytics.transactions"})], schemas)

    result = searcher.search("统计本月交易额", top_k=5)

    assert [candidate["table_name"] for candidate in result] == [
        "analytics.transactions"
    ]


def test_equal_scores_use_table_name_as_stable_tie_breaker() -> None:
    schemas = {
        "analytics.z_table": {"table_name": "analytics.z_table"},
        "analytics.a_table": {"table_name": "analytics.a_table"},
    }
    searcher = _searcher(
        [
            (1, 0.8, {"table_name": "analytics.z_table"}),
            (2, 0.8, {"table_name": "analytics.a_table"}),
        ],
        schemas,
    )

    result = searcher.search("统计交易额", top_k=5)

    assert [candidate["table_name"] for candidate in result] == [
        "analytics.a_table",
        "analytics.z_table",
    ]


def test_semantically_required_table_remains_pinned() -> None:
    schemas = {
        "analytics.orders": {"table_name": "analytics.orders"},
        "analytics.ledger": {"table_name": "analytics.ledger"},
    }
    searcher = _searcher([(1, 0.9, {"table_name": "analytics.ledger"})], schemas)

    result = searcher.search(
        "统计本月业务金额",
        top_k=1,
        required_tables={"analytics.orders"},
    )

    assert [candidate["table_name"] for candidate in result] == [
        "analytics.ledger",
        "analytics.orders",
    ]
    assert result[1]["pinned"] is True


def test_relation_graph_only_adds_join_bridge_after_direct_recall() -> None:
    schemas = {
        "analytics.orders": {
            "table_name": "analytics.orders",
            "relations": [
                {
                    "column": "order_id",
                    "target_table": "order_items",
                    "target_column": "order_id",
                    "cardinality": "one_to_many",
                }
            ],
        },
        "analytics.order_items": {
            "table_name": "analytics.order_items",
            "relations": [
                {
                    "column": "user_id",
                    "target_table": "users",
                    "target_column": "user_id",
                    "cardinality": "many_to_one",
                }
            ],
        },
        "analytics.users": {
            "table_name": "analytics.users",
            "relations": [],
        },
    }
    planner = SchemaContextPlanner(schemas)
    direct_candidates = [
        {
            "table_name": "analytics.orders",
            "score": 0.9,
            "schema": schemas["analytics.orders"],
        },
        {
            "table_name": "analytics.users",
            "score": 0.8,
            "schema": schemas["analytics.users"],
        },
    ]

    candidates, paths = planner.add_join_bridges(direct_candidates)

    assert paths == [["analytics.orders", "analytics.order_items", "analytics.users"]]
    assert [candidate["table_name"] for candidate in candidates] == [
        "analytics.orders",
        "analytics.users",
        "analytics.order_items",
    ]
    assert candidates[-1]["relation_bridge"] is True
