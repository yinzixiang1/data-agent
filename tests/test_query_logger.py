import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import query_logger


class _Result:
    lastrowid = 42


class _Connection:
    def __init__(self) -> None:
        self.sql = ""
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, statement, parameters):
        self.sql = str(statement)
        return _Result()

    def commit(self) -> None:
        self.committed = True


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


class _ReplayResult:
    def __init__(self, row: dict | None) -> None:
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class _ReplayConnection(_Connection):
    def __init__(self, row: dict | None) -> None:
        super().__init__()
        self.row = row
        self.parameters = {}

    def execute(self, statement, parameters):
        self.sql = str(statement)
        self.parameters = parameters
        return _ReplayResult(self.row)


def test_query_log_insert_sets_required_feedback_reason(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(
        query_logger,
        "create_engine",
        lambda *_args, **_kwargs: _Engine(connection),
    )

    log_id = query_logger.QueryLogger("mysql+pymysql://unused").log(
        session_id="session-1",
        user_query="活跃商户有多少？",
    )

    assert log_id == 42
    assert "feedback_reason" in connection.sql
    assert connection.committed is True


def test_completed_lark_response_can_be_replayed(monkeypatch) -> None:
    response = {"session_id": "session-1", "question": "活跃商户有多少？"}
    connection = _ReplayConnection(
        {
            "id": 42,
            "execution_result": json.dumps(
                query_logger.QueryLogger.lark_response_snapshot(response)
            ),
        }
    )
    monkeypatch.setattr(
        query_logger,
        "create_engine",
        lambda *_args, **_kwargs: _Engine(connection),
    )

    replay = query_logger.QueryLogger("mysql+pymysql://unused").get_lark_response(
        trace_id="evt-1",
        user_id="ou-1",
        agent_id=1,
        user_query="活跃商户有多少？",
    )

    assert replay == (42, response)
    assert "caller = 'lark'" in connection.sql
    assert connection.parameters == {
        "trace_id": "evt-1",
        "user_id": "ou-1",
        "agent_id": 1,
        "user_query": "活跃商户有多少？",
    }


def test_lark_response_snapshot_keeps_only_card_sized_result() -> None:
    wrapped = query_logger.QueryLogger.lark_response_snapshot(
        {
            "session_id": "session-1",
            "question": "查询明细",
            "sql": "SELECT * FROM a_large_table",
            "trace": {"very": "large"},
            "query_result": {
                "columns": [f"c{index}" for index in range(10)],
                "rows": [
                    [f"r{row}c{column}" for column in range(10)] for row in range(20)
                ],
                "row_count": 20,
                "truncated": True,
            },
        }
    )
    snapshot = wrapped[query_logger._LARK_RESPONSE_SNAPSHOT_KEY]

    assert "sql" not in snapshot
    assert "trace" not in snapshot
    assert len(snapshot["query_result"]["columns"]) == 6
    assert len(snapshot["query_result"]["rows"]) == 8
    assert len(snapshot["query_result"]["rows"][0]) == 6
    assert snapshot["query_result"]["row_count"] == 20
