from src.retrieval.context_planner import SchemaContextPlanner
from src.retrieval.document_builder import DocumentBuilder


def _schemas():
    return {
        "sales.customers": {
            "table_name": "sales.customers",
            "table_name_short": "customers",
            "biz_line": "banking",
            "tags": ["客户"],
            "columns": [
                {"name": "customer_id", "key": "PRI"},
                {"name": "customer_name", "display_name": "客户名称"},
            ],
            "relations": [],
        },
        "sales.orders": {
            "table_name": "sales.orders",
            "table_name_short": "orders",
            "biz_line": "banking",
            "tags": ["订单"],
            "columns": [
                {"name": "order_id", "key": "PRI"},
                {"name": "customer_id", "is_skip_index": True},
                {"name": "amount", "display_name": "订单金额"},
                {"name": "internal_note", "is_skip_index": True},
                {"name": "created_at", "display_name": "创建时间"},
            ],
            "relations": [
                {
                    "column": "customer_id",
                    "target_table": "customers",
                    "target_column": "customer_id",
                },
                {
                    "column": "order_id",
                    "target_table": "order_items",
                    "target_column": "order_id",
                },
            ],
        },
        "sales.order_items": {
            "table_name": "sales.order_items",
            "table_name_short": "order_items",
            "biz_line": "banking",
            "tags": ["订单明细"],
            "columns": [
                {"name": "order_id"},
                {"name": "product_id"},
            ],
            "relations": [
                {
                    "column": "product_id",
                    "target_table": "products",
                    "target_column": "product_id",
                }
            ],
        },
        "sales.products": {
            "table_name": "sales.products",
            "table_name_short": "products",
            "biz_line": "banking",
            "tags": ["产品"],
            "columns": [{"name": "product_id", "key": "PRI"}],
            "relations": [],
        },
        "card.cards": {
            "table_name": "card.cards",
            "table_name_short": "cards",
            "biz_line": "issuing",
            "tags": ["发卡"],
            "columns": [{"name": "card_id", "key": "PRI"}],
            "relations": [],
        },
    }


def test_join_path_adds_missing_bridge_table():
    planner = SchemaContextPlanner(_schemas())
    candidates = [
        {"table_name": "sales.orders", "schema": _schemas()["sales.orders"]},
        {"table_name": "sales.products", "schema": _schemas()["sales.products"]},
    ]

    planned, paths = planner.add_join_bridges(candidates, max_hops=2, max_tables=8)

    assert [item["table_name"] for item in planned] == [
        "sales.orders",
        "sales.products",
        "sales.order_items",
    ]
    assert paths == [["sales.orders", "sales.order_items", "sales.products"]]
    assert planned[-1]["relation_bridge"] is True


def test_column_pruning_keeps_keys_join_and_question_fields():
    schemas = _schemas()
    planner = SchemaContextPlanner(schemas)
    candidates = [
        {
            "table_name": "sales.orders",
            "schema": schemas["sales.orders"],
            "matched_columns": [{"column_name": "amount", "score": 1.0}],
        }
    ]

    planned, stats = planner.prune_columns(
        candidates, "统计订单金额", per_table_limit=3
    )

    selected = planned[0]["selected_columns"]
    assert selected == ["order_id", "customer_id", "amount"]
    assert "internal_note" not in selected
    assert stats["columns_pruned"] == 2


def test_time_query_keeps_likely_time_columns_beyond_regular_limit():
    schemas = _schemas()
    schemas["sales.orders"]["columns"].extend(
        [
            {"name": "completed_at", "type": "datetime"},
            {"name": "updated_at", "type": "datetime"},
            {"name": "deleted_at", "type": "datetime"},
        ]
    )
    planner = SchemaContextPlanner(schemas)

    planned, _ = planner.prune_columns(
        [{"table_name": "sales.orders", "schema": schemas["sales.orders"]}],
        "统计最近7天每天的订单数量",
        per_table_limit=3,
        preserve_time_columns=True,
    )

    selected = planned[0]["selected_columns"]
    assert "created_at" in selected
    assert "completed_at" in selected
    assert "updated_at" in selected
    assert "deleted_at" not in selected


def test_domain_routing_requires_clear_evidence():
    planner = SchemaContextPlanner(_schemas())

    assert planner.infer_biz_line("issuing 发卡数量") == "issuing"
    assert planner.infer_biz_line("查询最近数据") == ""


def test_inline_enum_values_are_added_to_entity_value_index():
    schemas = [
        {
            "table_name": "sales.orders",
            "biz_line": "banking",
            "columns": [
                {
                    "name": "status",
                    "display_name": "订单状态",
                    "enum_values": [
                        {"value": "SUCCESS", "label": "成功", "synonyms": ["已完成"]}
                    ],
                }
            ],
        }
    ]

    documents = DocumentBuilder().build_value_documents([], schemas)

    assert documents == [
        {
            "table_name": "sales.orders",
            "column_name": "status",
            "biz_line": "banking",
            "metadata": {},
            "enum_code": "SUCCESS",
            "enum_label_cn": "成功",
            "sql_value": "SUCCESS",
            "text": "成功 已完成",
        }
    ]
