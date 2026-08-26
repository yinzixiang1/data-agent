from src.tools.result_snapshot import ResultSnapshotStore


def _query_result() -> dict:
    return {
        "columns": ["month", "amount"],
        "rows": [["2026-07", 10], ["2026-08", 12]],
        "row_count": 2,
        "truncated": False,
    }


def test_snapshot_returns_an_independent_copy_for_the_same_query_context() -> None:
    store = ResultSnapshotStore()
    source = _query_result()
    snapshot_id = store.put(
        source,
        session_id="session-1",
        agent_id=7,
        user_id="user-1",
        context_summary="query-context",
    )
    source["rows"][0][1] = 999

    restored = store.get(
        snapshot_id,
        session_id="session-1",
        agent_id=7,
        user_id="user-1",
        context_summary="query-context",
    )

    assert restored == _query_result()
    assert snapshot_id.startswith("nl2sql:result_snapshot:")


def test_snapshot_rejects_a_different_session_user_or_query_context() -> None:
    store = ResultSnapshotStore()
    snapshot_id = store.put(
        _query_result(),
        session_id="session-1",
        agent_id=7,
        user_id="user-1",
        context_summary="query-context",
    )

    assert (
        store.get(
            snapshot_id,
            session_id="session-2",
            agent_id=7,
            user_id="user-1",
            context_summary="query-context",
        )
        is None
    )
    assert (
        store.get(
            snapshot_id,
            session_id="session-1",
            agent_id=7,
            user_id="user-2",
            context_summary="query-context",
        )
        is None
    )
    assert (
        store.get(
            snapshot_id,
            session_id="session-1",
            agent_id=7,
            user_id="user-1",
            context_summary="another-query",
        )
        is None
    )
