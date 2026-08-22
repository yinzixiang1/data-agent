from contextlib import nullcontext

from src.retrieval.schema_loader import SchemaLoader


class _Rows:
    def fetchall(self):
        return [
            (
                7,
                "按渠道统计订单",
                "SELECT channel_code, COUNT(*) FROM db.orders GROUP BY channel_code",
                '["db.orders"]',
                "L1",
                '{"source":"verified"}',
                "banking",
            )
        ]


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement, _params):
        self.statements.append(str(statement))
        return _Rows()


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self):
        return nullcontext(self.connection)


def test_loader_only_uses_reviewed_fewshot_table() -> None:
    connection = _Connection()
    loader = SchemaLoader.__new__(SchemaLoader)
    loader.mysql_engine = _Engine(connection)

    examples = loader.load_fewshot(exec_db_ids=[3], agent_id=1)

    assert len(examples) == 1
    assert examples[0]["question"] == "按渠道统计订单"
    assert examples[0]["tables"] == ["db.orders"]
    assert examples[0]["metadata"]["business"] == "banking"
    assert len(connection.statements) == 1
    assert "da_semantic_fewshot" in connection.statements[0]
    assert "da_semantic_query" not in connection.statements[0]
