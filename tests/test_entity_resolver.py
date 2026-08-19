from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.entity_resolver import EntityResolver


TABLE = "dwd_bi_banking.banking_account"
SCHEMAS = {
    TABLE: {
        "table_name": TABLE,
        "columns": [
            {"name": "account_id"},
            {"name": "short_account_id"},
            {"name": "account_name"},
        ],
    }
}
RULE = {
    "name": "账户标识解析",
    "business": "banking",
    "table": TABLE,
    "generic_terms": ["账户"],
    "name_terms": ["账户名称", "账户名"],
    "long_id_terms": ["账户ID", "account_id"],
    "short_id_terms": ["短账户ID", "short_account_id"],
    "name_column": "account_name",
    "long_id_column": "account_id",
    "short_id_column": "short_account_id",
    "long_id_length": 32,
}


def resolve(query: str) -> dict:
    return EntityResolver([RULE], SCHEMAS).resolve(query, biz_line="banking")


def test_generic_short_value_uses_short_id_column() -> None:
    result = resolve("查询账户为 12345 的最近一个月交易")

    assert result["unresolved"] == []
    assert result["filters"][0]["qualified_column"] == (f"{TABLE}.short_account_id")
    assert result["filters"][0]["value"] == "12345"


def test_generic_32_character_value_uses_long_id_column() -> None:
    account_id = "a" * 32

    result = resolve(f"查询账户={account_id}的交易")

    assert result["filters"][0]["column"] == "account_id"


def test_explicit_name_and_id_terms_override_length_routing() -> None:
    name_result = resolve("查询账户名称为 yzx 的交易")
    id_result = resolve("查询账户ID为 12345 的交易")

    assert name_result["filters"][0]["column"] == "account_name"
    assert id_result["filters"][0]["column"] == "account_id"


def test_value_longer_than_configured_id_length_requires_clarification() -> None:
    result = resolve(f"查询账户为 {'a' * 33} 的交易")

    assert result["filters"] == []
    assert "超过规则允许" in result["unresolved"][0]["issue"]


def test_missing_length_configuration_is_not_silently_defaulted() -> None:
    invalid_rule = {**RULE, "long_id_length": None}

    result = EntityResolver([invalid_rule], SCHEMAS).resolve("账户为 12345")

    assert result["filters"] == []
    assert "未配置有效" in result["unresolved"][0]["issue"]


def test_unrelated_query_does_not_create_a_filter() -> None:
    assert resolve("展示最近一个月交易")["filters"] == []


def test_entity_rules_are_loaded_from_agent_retrieval_config() -> None:
    loader = AgentConfigLoader.__new__(AgentConfigLoader)
    config = AgentRuntimeConfig()

    loader._apply_agent_configs(
        config,
        {
            "retrieval": {
                "entity_resolution_rules": [RULE],
                "glossary_require_lexical_grounding": False,
            }
        },
        {},
    )

    assert config.entity_resolution_rules == [RULE]
    assert config.glossary_require_lexical_grounding is False
