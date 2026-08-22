from __future__ import annotations

import numpy as np

from src.retrieval.milvus_store import MilvusIndex


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def insert(self, collection_name: str, data: list[dict]) -> None:
        self.calls.append(("insert", (collection_name, data)))

    def flush(self, collection_name: str) -> None:
        self.calls.append(("flush", collection_name))


def _index(client: _FakeMilvusClient, *, has_dense: bool = True) -> MilvusIndex:
    index = MilvusIndex.__new__(MilvusIndex)
    index.collection_name = "test_collection"
    index.dim = 2
    index.has_dense = has_dense
    index._cached_count = None
    index.client = client
    return index


def test_insert_flushes_before_updating_cached_count() -> None:
    client = _FakeMilvusClient()
    index = _index(client)

    index.insert(
        np.array([[0.1, 0.2]], dtype=np.float32),
        ["账户余额"],
        [{"table_name": "banking_account_balance"}],
    )

    assert [call[0] for call in client.calls] == ["insert", "flush"]
    assert client.calls[1] == ("flush", "test_collection")
    assert index.count == 1


def test_flush_failure_does_not_report_unpersisted_count() -> None:
    class _FailingFlushClient(_FakeMilvusClient):
        def flush(self, collection_name: str) -> None:
            super().flush(collection_name)
            raise RuntimeError("flush failed")

    index = _index(_FailingFlushClient())

    try:
        index.insert(None, ["账户余额"], [{"name": "balance"}])
    except RuntimeError as exc:
        assert str(exc) == "flush failed"
    else:
        raise AssertionError("flush failure must propagate")

    assert index._cached_count is None
