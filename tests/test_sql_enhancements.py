"""
SQL 纠错增强 — 单元测试。

覆盖:
  - extract_where_values: WHERE 条件提取
  - validate_enum_values: 枚举值交叉校验
  - check_result_anomalies: 结果合理性规则预检
  - simplify_sql_for_timeout: 超时降级 SQL 简化
  - get_expand / _apply_expand: 配置读取与优先级

运行:
  cd data-agen && python -m pytest tests/test_sql_enhancements.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.retrieval.sql_validator import SQLValidator
from src.retrieval.agent_config import get_expand, AgentRuntimeConfig, AgentConfigLoader
from src.retrieval.collection_names import agent_collection_name
from src.retrieval.milvus_filter import build_metadata_filter
from src.retrieval.query_cache import QueryCache
from src.retrieval.schema_loader import AgentDatasourceNotConfiguredError, SchemaLoader


# ════════════════════════════════════════════
# security guards
# ════════════════════════════════════════════

class _NoConnectionEngine:
    def connect(self):
        raise AssertionError("unsafe SQL must be rejected before opening a database connection")


class TestReadOnlySQLGuard:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM analytics.orders",
            "WITH recent AS (SELECT * FROM analytics.orders) SELECT * FROM recent",
            "SELECT 'DELETE FROM users' AS example",
            "SELECT 1 /* UPDATE users SET role = 'admin' */",
        ],
    )
    def test_allows_read_only_queries(self, sql):
        assert SQLValidator.validate_read_only(sql) == (True, "")

    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE analytics.orders SET status = 1",
            "DELETE FROM analytics.orders",
            "SELECT 1; DROP TABLE analytics.orders",
            "SELECT * FROM analytics.orders INTO OUTFILE '/tmp/orders.csv'",
            "SELECT status INTO @last_status FROM analytics.orders",
            "SELECT * FROM analytics.orders FOR UPDATE",
            "WITH changed AS (DELETE FROM analytics.orders RETURNING *) SELECT * FROM changed",
        ],
    )
    def test_rejects_unsafe_queries(self, sql):
        allowed, reason = SQLValidator.validate_read_only(sql)
        assert allowed is False
        assert reason

    def test_execute_and_explain_fail_before_connecting(self):
        validator = SQLValidator(_NoConnectionEngine())
        assert validator.execute("DELETE FROM analytics.orders")["success"] is False
        assert validator.explain("UPDATE analytics.orders SET status = 1")["valid"] is False


class TestDatabaseAccessGuard:
    AUTHORIZED = {"analytics", "shared"}

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM analytics.orders",
            "SELECT * FROM `analytics`.`orders` a JOIN shared.users u ON a.user_id = u.id",
            "WITH recent AS (SELECT * FROM analytics.orders) SELECT * FROM recent",
            "WITH recent(id) AS (SELECT id FROM analytics.orders) SELECT * FROM recent",
            "WITH RECURSIVE tree AS (SELECT * FROM analytics.nodes) SELECT * FROM tree",
            "SELECT * FROM /* generated */ analytics.orders",
        ],
    )
    def test_allows_only_authorized_qualified_tables(self, sql):
        allowed, reason, databases = SQLValidator.validate_database_access(
            sql,
            self.AUTHORIZED,
        )
        assert allowed is True, reason
        assert databases <= self.AUTHORIZED

    @pytest.mark.parametrize(
        ("sql", "expected_message"),
        [
            ("SELECT * FROM secret.orders", "未在当前 Agent"),
            (
                "SELECT * FROM analytics.orders a JOIN unqualified_users u ON a.user_id = u.id",
                "数据库限定名",
            ),
            ("SELECT * FROM analytics.orders, secret.users", "显式 JOIN"),
            (
                "WITH recent AS (SELECT * FROM analytics.orders) SELECT * FROM local_orders",
                "数据库限定名",
            ),
            ("SELECT * FROM information_schema.tables", "未在当前 Agent"),
        ],
    )
    def test_rejects_unverifiable_or_unauthorized_tables(self, sql, expected_message):
        allowed, reason, _ = SQLValidator.validate_database_access(sql, self.AUTHORIZED)
        assert allowed is False
        assert expected_message in reason


class TestMilvusFilterGuard:
    def test_escapes_values_and_supports_scalars(self):
        expression = build_metadata_filter(
            'retail" or true or "',
            {"scenario": 'x" or true or "', "enabled": True, "level": 2},
        )
        assert 'retail\\" or true or \\"' in expression
        assert 'x\\" or true or \\"' in expression
        assert 'metadata["enabled"] == true' in expression
        assert 'metadata["level"] == 2' in expression

    @pytest.mark.parametrize(
        ("metadata", "expected_message"),
        [
            ({'bad"] or true or metadata["x': "value"}, "非法 metadata filter key"),
            ({"valid_key": {"nested": "value"}}, "不支持"),
            ({"valid_key": float("nan")}, "NaN"),
        ],
    )
    def test_rejects_unsafe_filter_inputs(self, metadata, expected_message):
        with pytest.raises(ValueError, match=expected_message):
            build_metadata_filter(metadata_filter=metadata)


def test_schema_loader_fails_closed_without_agent_datasources(monkeypatch):
    loader = SchemaLoader.__new__(SchemaLoader)
    monkeypatch.setattr(loader, "_load_agent_exec_db_ids", lambda agent_id: [])

    with pytest.raises(AgentDatasourceNotConfiguredError, match="拒绝加载全量语义层"):
        loader.load_all(agent_id=42)


class _StaticEmbedding:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_query_cache_isolates_contexts():
    cache = QueryCache(_StaticEmbedding())
    cache.put("订单数量", {"sql": "SELECT 1"}, context_key='{"agent_id":1}')

    assert cache.get("订单数量", context_key='{"agent_id":1}') == {"sql": "SELECT 1"}
    assert cache.get("订单数量", context_key='{"agent_id":2}') is None


def test_milvus_collection_names_are_agent_scoped():
    assert agent_collection_name("nl2sql_table", None) == "nl2sql_table_agent_local"
    assert agent_collection_name("nl2sql_table", 7) == "nl2sql_table_agent_7"
    assert agent_collection_name("nl2sql_table", 8) != agent_collection_name(
        "nl2sql_table",
        7,
    )


# ════════════════════════════════════════════
# extract_where_values
# ════════════════════════════════════════════

class TestExtractWhereValues:
    def test_string_equal(self):
        sql = "SELECT * FROM t WHERE status = 'success' AND name = \"test\""
        vals = SQLValidator.extract_where_values(sql)
        assert any(v["column"] == "status" and v["value"] == "success" for v in vals)
        assert any(v["column"] == "name" and v["value"] == "test" for v in vals)

    def test_numeric_equal(self):
        sql = "SELECT * FROM t WHERE status = 1 AND type = 2"
        vals = SQLValidator.extract_where_values(sql)
        assert any(v["column"] == "status" and v["value"] == "1" for v in vals)
        assert any(v["column"] == "type" and v["value"] == "2" for v in vals)

    def test_excludes_limit_offset(self):
        sql = "SELECT * FROM t WHERE id = 1 LIMIT 10"
        vals = SQLValidator.extract_where_values(sql)
        cols = [v["column"] for v in vals]
        assert "LIMIT" not in cols
        assert "id" in cols

    def test_backtick_column(self):
        sql = "SELECT * FROM t WHERE `status` = 'active'"
        vals = SQLValidator.extract_where_values(sql)
        assert any(v["column"] == "status" and v["value"] == "active" for v in vals)

    def test_no_where(self):
        sql = "SELECT COUNT(*) FROM t"
        vals = SQLValidator.extract_where_values(sql)
        assert vals == []

    def test_complex_sql(self):
        sql = """
        SELECT a.id, b.name
        FROM dwd_banking.pmt_account a
        JOIN dwd_banking.pmt_order b ON a.id = b.account_id
        WHERE a.status = 'ACTIVE'
          AND b.currency = 'SGD'
          AND b.type = 3
        """
        vals = SQLValidator.extract_where_values(sql)
        assert any(v["column"] == "status" and v["value"] == "ACTIVE" for v in vals)
        assert any(v["column"] == "currency" and v["value"] == "SGD" for v in vals)
        assert any(v["column"] == "type" and v["value"] == "3" for v in vals)


# ════════════════════════════════════════════
# validate_enum_values
# ════════════════════════════════════════════

class TestValidateEnumValues:
    """enum_hits 是扁平结构，每条一个枚举值。"""

    ENUM_HITS = [
        {"table_name": "t1", "column_name": "status", "enum_label_cn": "成功", "sql_value": "1", "score": 0.9},
        {"table_name": "t1", "column_name": "status", "enum_label_cn": "失败", "sql_value": "2", "score": 0.8},
        {"table_name": "t1", "column_name": "status", "enum_label_cn": "处理中", "sql_value": "3", "score": 0.7},
        {"table_name": "t1", "column_name": "currency", "enum_label_cn": "人民币", "sql_value": "CNY", "score": 0.9},
        {"table_name": "t1", "column_name": "currency", "enum_label_cn": "美元", "sql_value": "USD", "score": 0.8},
        {"table_name": "t1", "column_name": "currency", "enum_label_cn": "新加坡元", "sql_value": "SGD", "score": 0.7},
    ]

    def test_match_ok(self):
        """枚举值匹配，无 mismatch。"""
        where = [{"column": "status", "operator": "=", "value": "1"}]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert result == []

    def test_mismatch_detected(self):
        """用文本值 'success' 但枚举定义是 1/2/3。"""
        where = [{"column": "status", "operator": "=", "value": "success"}]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert len(result) == 1
        assert result[0]["column"] == "status"
        assert result[0]["sql_value"] == "success"

    def test_chinese_label_suggestion(self):
        """用中文标签 '成功' 应生成替换建议。"""
        where = [{"column": "status", "operator": "=", "value": "成功"}]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert len(result) == 1
        assert "应使用" in result[0]["suggestion"]
        assert "1" in result[0]["suggestion"]

    def test_no_enum_for_column(self):
        """WHERE 列不在 enum_hits 中，不报 mismatch。"""
        where = [{"column": "unknown_col", "operator": "=", "value": "abc"}]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert result == []

    def test_multiple_mismatches(self):
        """多列同时不匹配。"""
        where = [
            {"column": "status", "operator": "=", "value": "active"},
            {"column": "currency", "operator": "=", "value": "RMB"},
        ]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert len(result) == 2

    def test_case_insensitive_column(self):
        """列名大小写不敏感。"""
        where = [{"column": "STATUS", "operator": "=", "value": "1"}]
        result = SQLValidator.validate_enum_values(where, self.ENUM_HITS)
        assert result == []

    def test_empty_enum_hits(self):
        where = [{"column": "status", "operator": "=", "value": "1"}]
        result = SQLValidator.validate_enum_values(where, [])
        assert result == []


# ════════════════════════════════════════════
# check_result_anomalies
# ════════════════════════════════════════════

class TestCheckResultAnomalies:
    def test_negative_count(self):
        warnings = SQLValidator.check_result_anomalies(
            "查询交易数量", "SELECT count FROM t", ["count"], [["-5"]]
        )
        assert any("负数" in w for w in warnings)

    def test_no_negative_normal(self):
        warnings = SQLValidator.check_result_anomalies(
            "查询交易数量", "SELECT count FROM t", ["count"], [["100"]]
        )
        assert not any("负数" in w for w in warnings)

    def test_time_keyword_no_time_condition(self):
        warnings = SQLValidator.check_result_anomalies(
            "查询本月交易", "SELECT * FROM t", ["id"], [["1"]]
        )
        assert any("时间" in w for w in warnings)

    def test_time_keyword_with_time_condition(self):
        warnings = SQLValidator.check_result_anomalies(
            "查询本月交易",
            "SELECT * FROM t WHERE create_time >= '2026-05-01'",
            ["id"], [["1"]]
        )
        assert not any("时间" in w for w in warnings)

    def test_single_row_but_expect_multi(self):
        warnings = SQLValidator.check_result_anomalies(
            "按月统计交易量",
            "SELECT month, count FROM t GROUP BY month",
            ["month", "count"], [["2026-05", "100"]]
        )
        assert any("仅返回 1 行" in w for w in warnings)

    def test_multi_row_ok(self):
        warnings = SQLValidator.check_result_anomalies(
            "按月统计交易量",
            "SELECT month, count FROM t GROUP BY month",
            ["month", "count"],
            [["2026-04", "80"], ["2026-05", "100"]]
        )
        assert not any("仅返回 1 行" in w for w in warnings)

    def test_no_anomaly(self):
        warnings = SQLValidator.check_result_anomalies(
            "查询账户",
            "SELECT * FROM t WHERE create_time > '2026-01-01'",
            ["id", "name"], [["1", "a"], ["2", "b"]]
        )
        assert warnings == []

    def test_negative_amount_column(self):
        """amount 列也触发负数检查。"""
        warnings = SQLValidator.check_result_anomalies(
            "查询金额", "SELECT total_amount FROM t", ["total_amount"], [["-999"]]
        )
        assert any("负数" in w for w in warnings)


# ════════════════════════════════════════════
# simplify_sql_for_timeout
# ════════════════════════════════════════════

class TestSimplifySqlForTimeout:
    def test_level1_shrink_limit(self):
        sql = "SELECT * FROM t ORDER BY id LIMIT 200"
        result = SQLValidator.simplify_sql_for_timeout(sql, 1)
        assert "LIMIT 50" in result
        assert "LIMIT 200" not in result

    def test_level1_add_limit(self):
        sql = "SELECT * FROM t ORDER BY id"
        result = SQLValidator.simplify_sql_for_timeout(sql, 1)
        assert "LIMIT 50" in result

    def test_level2_remove_order_by(self):
        sql = "SELECT * FROM t ORDER BY create_time DESC LIMIT 50"
        result = SQLValidator.simplify_sql_for_timeout(sql, 2)
        assert result is not None
        assert "ORDER BY" not in result
        assert "LIMIT" in result

    def test_level2_no_order_by_returns_none(self):
        sql = "SELECT * FROM t WHERE id = 1 LIMIT 50"
        result = SQLValidator.simplify_sql_for_timeout(sql, 2)
        assert result is None

    def test_level3_returns_none(self):
        sql = "SELECT * FROM t"
        result = SQLValidator.simplify_sql_for_timeout(sql, 3)
        assert result is None

    def test_level1_case_insensitive(self):
        sql = "SELECT * FROM t limit 500"
        result = SQLValidator.simplify_sql_for_timeout(sql, 1)
        assert "50" in result


# ════════════════════════════════════════════
# get_expand
# ════════════════════════════════════════════

class TestGetExpand:
    def test_basic_read(self):
        cfg = {"max_tokens": 4096}
        assert get_expand(cfg, "max_tokens", default=2048, cast=int) == 4096

    def test_default_when_missing(self):
        assert get_expand({}, "max_tokens", default=2048, cast=int) == 2048

    def test_bool_cast_true(self):
        assert get_expand({"flag": "true"}, "flag", default=False, cast=bool) is True
        assert get_expand({"flag": True}, "flag", default=False, cast=bool) is True
        assert get_expand({"flag": "1"}, "flag", default=False, cast=bool) is True

    def test_bool_cast_false(self):
        assert get_expand({"flag": "false"}, "flag", default=True, cast=bool) is False
        assert get_expand({"flag": False}, "flag", default=True, cast=bool) is False

    def test_validation_pass(self):
        cfg = {"max_execute_fix_retries": 3}
        assert get_expand(cfg, "max_execute_fix_retries", default=2, cast=int) == 3

    def test_validation_fail_fallback(self):
        """超出范围 → 降级到默认值。"""
        cfg = {"max_execute_fix_retries": 99}
        assert get_expand(cfg, "max_execute_fix_retries", default=2, cast=int) == 2

    def test_validation_negative(self):
        cfg = {"execute_timeout": -1}
        assert get_expand(cfg, "execute_timeout", default=30, cast=int) == 30

    def test_cast_failure_fallback(self):
        cfg = {"max_tokens": "not_a_number"}
        assert get_expand(cfg, "max_tokens", default=4096, cast=int) == 4096

    def test_none_value(self):
        cfg = {"key": None}
        assert get_expand(cfg, "key", default=42) == 42

    def test_unknown_key_no_validator(self):
        """未注册 EXPAND_VALIDATORS 的 key，通过不拦截。"""
        cfg = {"custom_flag": True}
        assert get_expand(cfg, "custom_flag", default=False, cast=bool) is True

    def test_float_cast(self):
        cfg = {"intent_threshold": "0.8"}
        assert get_expand(cfg, "intent_threshold", default=0.7, cast=float) == 0.8


# ════════════════════════════════════════════
# _apply_expand 优先级
# ════════════════════════════════════════════

class TestApplyExpandPriority:
    def test_expand_skips_explicit_keys(self):
        """显式字段已设置 → expand 不覆盖。"""
        config = AgentRuntimeConfig()
        config.execute_timeout = 60  # 模拟 flow 分区已设置

        expand_cfg = {"execute_timeout": 10, "max_execute_fix_retries": 5}
        explicit_keys = {"execute_timeout"}  # flow 分区设置了

        AgentConfigLoader._apply_expand(config, expand_cfg, explicit_keys)

        assert config.execute_timeout == 60  # 保持 flow 设置的值
        assert config.max_execute_fix_retries == 5  # expand 覆盖成功

    def test_expand_applies_when_no_explicit(self):
        """无显式设置 → expand 正常覆盖。"""
        config = AgentRuntimeConfig()

        expand_cfg = {"execute_timeout": 45}
        AgentConfigLoader._apply_expand(config, expand_cfg, set())

        assert config.execute_timeout == 45

    def test_expand_invalid_value_ignored(self):
        """expand 值不合法 → 保持默认。"""
        config = AgentRuntimeConfig()
        assert config.max_execute_fix_retries == 2  # 默认值

        expand_cfg = {"max_execute_fix_retries": 99}  # 超出 0-5 范围
        AgentConfigLoader._apply_expand(config, expand_cfg, set())

        assert config.max_execute_fix_retries == 2  # 保持默认

    def test_expand_unknown_key_ignored(self):
        """expand 含未知 key → 忽略不报错。"""
        config = AgentRuntimeConfig()
        expand_cfg = {"totally_unknown_key": "value"}
        AgentConfigLoader._apply_expand(config, expand_cfg, set())
        assert not hasattr(config, "totally_unknown_key")


# ════════════════════════════════════════════
# AgentRuntimeConfig 新字段默认值
# ════════════════════════════════════════════

class TestAgentRuntimeConfigDefaults:
    def test_enhancement_defaults(self):
        config = AgentRuntimeConfig()
        assert config.max_execute_fix_retries == 2
        assert config.enable_empty_analysis is True
        assert config.enable_enum_validate is True
        assert config.enable_result_check is True
        assert config.enable_timeout_fallback is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
