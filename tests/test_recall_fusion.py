import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import pymilvus  # noqa: F401
except ModuleNotFoundError:
    pymilvus = types.ModuleType("pymilvus")

    class FakeRanker:
        def __init__(self, *args, **kwargs) -> None:
            pass

    for name in ("MilvusClient", "Function", "FunctionType", "AnnSearchRequest"):
        setattr(pymilvus, name, object)
    pymilvus.DataType = SimpleNamespace(VARCHAR="VARCHAR", JSON="JSON")
    pymilvus.RRFRanker = FakeRanker
    pymilvus.WeightedRanker = FakeRanker
    sys.modules["pymilvus"] = pymilvus

from src.retrieval.glossary_resolver import GlossaryResolver
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.milvus_filter import add_table_name_filter
from src.retrieval.ranker_strategy import CollectionSearchParams
from src.retrieval.value_indexer import ValueIndexer
from src.retrieval.fewshot_selector import FewShotSelector


class FakeEmbedding:
    def encode_query(self, query: str, collection_type: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeIndex:
    def __init__(self, results: list[tuple[int, float, dict]]) -> None:
        self.results = results
        self.count = len(results)

    def hybrid_search(self, *args, **kwargs) -> list[tuple[int, float, dict]]:
        return self.results


def build_searcher() -> HybridSearcher:
    schemas = {
        name: {"table_name": name, "relations": []}
        for name in (
            "sales.orders",
            "sales.decoy",
            "sales.column_hit",
            "sales.enum_hit",
            "sales.value_hit",
            "archive.orders",
        )
    }
    config = SimpleNamespace(
        collection_search_config={},
        index_build_config={"hnsw": {"ef_search": 32}},
    )
    return HybridSearcher(
        embedding=FakeEmbedding(),
        table_index=FakeIndex(
            [
                (0, 0.8, {"table_name": "sales.orders"}),
                (1, 0.7, {"table_name": "sales.decoy"}),
            ]
        ),
        column_index=FakeIndex([(0, 100.0, {"table_name": "sales.column_hit"})]),
        enum_index=FakeIndex([(0, 50.0, {"table_name": "sales.enum_hit"})]),
        table_schemas=schemas,
        config=config,
    )


def test_cross_collection_scores_are_normalized_and_required_tables_are_pinned():
    searcher = build_searcher()

    results = searcher.search(
        "订单中的持牌商户",
        top_k=5,
        required_tables={"sales.value_hit"},
    )
    by_name = {item["table_name"]: item for item in results}

    assert by_name["sales.column_hit"]["score"] == pytest.approx(0.2)
    assert by_name["sales.enum_hit"]["score"] == pytest.approx(0.16)
    assert by_name["sales.value_hit"]["pinned"] is True
    assert by_name["sales.value_hit"]["score"] == pytest.approx(0.72)


def test_explicit_table_detection_and_relation_resolution_are_schema_aware():
    searcher = build_searcher()

    assert searcher._find_explicit_tables("查询 `sales`.`orders` 的订单数") == {
        "sales.orders"
    }
    assert searcher._resolve_full_name("orders") is None
    assert (
        searcher._resolve_full_name("orders", source_table="sales.decoy")
        == "sales.orders"
    )


def test_enum_results_are_filtered_before_top_k_is_applied():
    searcher = build_searcher()
    searcher.enum_index = FakeIndex(
        [
            (
                0,
                0.9,
                {
                    "table_name": "sales.decoy",
                    "column_name": "status",
                    "enum_label_cn": "无关状态",
                    "sql_value": "0",
                },
            ),
            (
                1,
                0.8,
                {
                    "table_name": "sales.orders",
                    "column_name": "status",
                    "enum_label_cn": "已完成",
                    "sql_value": "1",
                },
            ),
        ]
    )

    results = searcher.search_enums(
        "已完成订单",
        top_k=1,
        table_names={"sales.orders"},
    )

    assert [item["table_name"] for item in results] == ["sales.orders"]


def test_table_filter_is_escaped_and_combined_with_metadata_filter():
    result = add_table_name_filter(
        '(metadata["tenant"] == "uat")',
        {"sales.orders", 'sales.order"archive'},
    )

    assert result == (
        '((metadata["tenant"] == "uat")) and '
        '(table_name in ["sales.order\\"archive", "sales.orders"])'
    )


def test_value_linking_marks_canonical_exact_matches():
    index = SimpleNamespace(
        count=1,
        bm25_search=lambda *args, **kwargs: [
            (
                0,
                1.0,
                {
                    "table_name": "sales.value_hit",
                    "column_name": "account_type",
                    "enum_label_cn": "持牌商户",
                    "sql_value": "2000",
                },
            )
        ],
    )
    value_indexer = ValueIndexer(index)
    value_indexer.extract_entities = lambda query: ["持牌商户"]

    results = value_indexer.match_values("查询持牌商户")

    assert results[0]["exact_match"] is True
    assert results[0]["score"] == 2.0


def test_value_linking_keeps_exact_match_when_fuzzy_hit_arrives_first():
    def bm25_search(entity: str, **kwargs) -> list[tuple[int, float, dict]]:
        return [
            (
                0,
                1.0 if entity == "商户" else 0.7,
                {
                    "table_name": "sales.value_hit",
                    "column_name": "account_type",
                    "enum_label_cn": "持牌商户",
                    "sql_value": "2000",
                },
            )
        ]

    value_indexer = ValueIndexer(SimpleNamespace(count=1, bm25_search=bm25_search))
    value_indexer.extract_entities = lambda query: ["商户", "持牌商户"]

    results = value_indexer.match_values("查询持牌商户")

    assert len(results) == 1
    assert results[0]["matched_entity"] == "持牌商户"
    assert results[0]["exact_match"] is True


def test_glossary_top_k_caps_query_expansion():
    index = FakeIndex(
        [
            (0, 1.0, {"term": "术语一", "definition": "定义一"}),
            (1, 0.9, {"term": "术语二", "definition": "定义二"}),
        ]
    )
    params = CollectionSearchParams("weighted", object(), 10, False, 10, 10)
    resolver = GlossaryResolver(FakeEmbedding(), index, params)

    result = resolver.resolve("业务问题", top_k=1, score_threshold=0.1)

    assert result["matched_terms"] == ["术语一"]
    assert "定义二" not in result["business_context"]


def test_fewshot_uses_table_structure_and_quality_after_hybrid_recall():
    selector = FewShotSelector(FakeEmbedding())
    selector.examples = [
        {
            "id": 1,
            "question": "语义接近但查明细",
            "sql": "SELECT * FROM archive.logs",
            "tables": ["archive.logs"],
            "metadata": {"quality_score": 0.5},
        },
        {
            "id": 2,
            "question": "订单数量",
            "sql": "SELECT COUNT(*) FROM sales.orders",
            "tables": ["sales.orders"],
            "metadata": {"quality_score": 1.0, "dialect": "doris"},
        },
    ]
    selector.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    selector.example_table_sets = [{"archive.logs"}, {"sales.orders"}]
    selector.milvus_index = FakeIndex([(0, 0.9, {}), (1, 0.8, {})])
    params = CollectionSearchParams("weighted", object(), 10, True, 10, 10)

    result = selector.select(
        "订单有多少笔",
        tables=["sales.orders"],
        top_k=1,
        search_params=params,
    )

    assert result[0]["id"] == 2
