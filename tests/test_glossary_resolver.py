from src.retrieval.glossary_resolver import GlossaryResolver
from src.retrieval.ranker_strategy import CollectionSearchParams


class _Embedding:
    def encode_query(self, query: str, collection_type: str):
        return [0.1]


class _Index:
    count = 2

    def __init__(self, results):
        self.results = results

    def hybrid_search(self, *_args, **_kwargs):
        return self.results


def _entity(term: str, synonyms: list[str], table: str) -> dict:
    return {
        "term": term,
        "definition": f"{term} definition",
        "sql_hint": f"use {table}",
        "related_tables": f'["{table}"]',
        "related_columns": "[]",
        "synonyms": str(synonyms).replace("'", '"'),
    }


def _resolver(results) -> GlossaryResolver:
    return GlossaryResolver(
        _Embedding(),
        _Index(results),
        CollectionSearchParams(
            ranker_type="rrf",
            ranker=object(),
            recall_limit=10,
            rerank=False,
            rerank_top_n=3,
            final_top_n=3,
            rrf_k=60,
        ),
    )


def test_grounded_term_excludes_unrelated_semantic_candidates() -> None:
    resolver = _resolver(
        [
            (1, 0.032, _entity("transfer", ["内部转账"], "orders")),
            (2, 0.031, _entity("USD折算", ["转美元"], "exchange_rate")),
        ]
    )

    result = resolver.resolve("按内部转账方向和原币种统计")

    assert result["matched_terms"] == ["transfer"]
    assert result["related_tables"] == ["orders"]


def test_multiple_explicit_terms_are_all_kept() -> None:
    resolver = _resolver(
        [
            (1, 0.032, _entity("payout", ["出金"], "orders")),
            (2, 0.031, _entity("USD折算", ["转美元"], "exchange_rate")),
        ]
    )

    result = resolver.resolve("统计出金金额并转美元")

    assert result["matched_terms"] == ["payout", "USD折算"]
    assert result["related_tables"] == ["exchange_rate", "orders"]


def test_semantic_only_candidates_do_not_pin_physical_tables() -> None:
    resolver = _resolver(
        [
            (1, 0.032, _entity("渠道资金", ["渠道余额"], "channel_funds")),
            (2, 0.031, _entity("USD折算", ["转美元"], "exchange_rate")),
        ]
    )

    result = resolver.resolve("按天查询账务流水")

    assert result["matched_terms"] == []
    assert result["related_tables"] == []
    assert result["related_columns"] == []
    assert result["rejected_terms"] == ["渠道资金", "USD折算"]


def test_short_chinese_term_uses_word_boundary_instead_of_substring() -> None:
    resolver = _resolver([(1, 0.032, _entity("payin", ["入金"], "orders"))])

    false_positive = resolver.resolve("按币种统计买入金额")
    actual_term = resolver.resolve("统计最近一个月的入金金额")

    assert false_positive["matched_terms"] == []
    assert actual_term["matched_terms"] == ["payin"]


def test_glossary_recall_covers_small_controlled_dictionary() -> None:
    index = _Index([])
    index.count = 29
    resolver = GlossaryResolver(
        _Embedding(),
        index,
        CollectionSearchParams(
            ranker_type="rrf",
            ranker=object(),
            recall_limit=10,
            rerank=False,
            rerank_top_n=3,
            final_top_n=3,
            rrf_k=60,
        ),
    )

    resolver.resolve("汇款人国家")

    # The fake index records no arguments, so exercise the formula directly
    # through a recording replacement.
    class _RecordingIndex(_Index):
        def __init__(self):
            super().__init__([])
            self.count = 29
            self.recall_k = 0

        def hybrid_search(self, *_args, **kwargs):
            self.recall_k = kwargs["recall_k"]
            return []

    recording = _RecordingIndex()
    resolver.glossary_index = recording
    resolver.resolve("汇款人国家")

    assert recording.recall_k == 29


def test_all_explicit_terms_are_kept_even_above_default_top_k() -> None:
    resolver = _resolver(
        [
            (1, 0.034, _entity("payin", [], "orders")),
            (2, 0.033, _entity("LOCAL", [], "orders")),
            (3, 0.032, _entity("SWIFT", [], "orders")),
            (4, 0.031, _entity("渠道", [], "orders")),
        ]
    )

    result = resolver.resolve("payin 按 LOCAL/SWIFT 和渠道统计")

    assert result["matched_terms"] == ["payin", "LOCAL", "SWIFT", "渠道"]
