"""
查询日志记录 — 将每次 NL2SQL 查询结果写入 sys_query_log 表。

对接 dataAgent-admin-api 定义的 sys_query_log 表结构，
支持后续在控制台查看查询历史、统计分析、用户反馈闭环。

使用示例::

    logger = QueryLogger()
    log_id = logger.log(
        session_id="abc123",
        user_query="有多少活跃商户",
        matched_tables=["pmt_account"],
        matched_terms=["活跃商户"],
        generated_sql="SELECT COUNT(*) FROM pmt_account WHERE ...",
        is_success=True,
        execution_time_ms=1200,
        retry_count=0,
    )
"""

import json
import logging

from sqlalchemy import create_engine, text

from src.retrieval.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_PASSWORD_URL, MYSQL_DATABASE,
)

logger = logging.getLogger(__name__)


class QueryLogger:
    """将查询结果写入 MySQL sys_query_log 表。"""

    def __init__(self, mysql_connection_string: str | None = None):
        if mysql_connection_string is None:
            mysql_connection_string = (
                f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD_URL}"
                f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
            )
        self.engine = create_engine(mysql_connection_string, pool_size=2, pool_recycle=3600)

    def log(
        self,
        session_id: str = "",
        user_query: str = "",
        intent: str = "",
        matched_tables: list[str] | None = None,
        matched_terms: list[str] | None = None,
        generated_sql: str = "",
        execution_result: dict | str = "",
        execution_time_ms: int = 0,
        retry_count: int = 0,
        is_success: bool = True,
        agent_id: int | None = None,
        scenario: str = "",
        business: str = "",
        caller: str = "",
        user_id: str = "",
        user_name: str = "",
        trace_id: str = "",
        matched_fewshot: list[dict] | None = None,
    ) -> int | None:
        """
        写入一条查询日志。

        Returns:
            插入的 log ID，失败时返回 None
        """
        try:
            matched_tables_json = json.dumps(matched_tables or [], ensure_ascii=False)
            matched_terms_json = json.dumps(matched_terms or [], ensure_ascii=False)
            matched_fewshot_json = json.dumps(matched_fewshot or [], ensure_ascii=False)
            if isinstance(execution_result, dict):
                execution_result = json.dumps(execution_result, ensure_ascii=False)

            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "INSERT INTO sys_query_log "
                    "(session_id, user_query, intent, matched_tables, matched_terms, "
                    "generated_sql, execution_result, execution_time_ms, retry_count, is_success, "
                    "user_feedback, feedback_score, feedback_comment, "
                    "agent_id, scenario, business, caller, user_id, user_name, trace_id, matched_fewshot) "
                    "VALUES (:session_id, :user_query, :intent, :matched_tables, :matched_terms, "
                    ":generated_sql, :execution_result, :execution_time_ms, :retry_count, :is_success, "
                    "'', 0, '', "
                    ":agent_id, :scenario, :business, :caller, :user_id, :user_name, :trace_id, :matched_fewshot)"
                ), {
                    "session_id": session_id,
                    "user_query": user_query,
                    "intent": intent,
                    "matched_tables": matched_tables_json,
                    "matched_terms": matched_terms_json,
                    "generated_sql": generated_sql,
                    "execution_result": execution_result or "",
                    "execution_time_ms": execution_time_ms,
                    "retry_count": retry_count,
                    "is_success": 1 if is_success else 0,
                    "agent_id": agent_id,
                    "scenario": scenario,
                    "business": business,
                    "caller": caller,
                    "user_id": user_id,
                    "user_name": user_name,
                    "trace_id": trace_id,
                    "matched_fewshot": matched_fewshot_json,
                })
                conn.commit()
                log_id = result.lastrowid
                logger.info(f"查询日志已记录: id={log_id}, success={is_success}")
                return log_id

        except Exception as e:
            logger.warning(f"查询日志记录失败: {e}")
            return None
