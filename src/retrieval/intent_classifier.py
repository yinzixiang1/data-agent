"""
意图分类器 — 根据 engine_type（从 da_agent_source 推导）和问题内容路由到 SQL / 知识问答 / 闲聊。

engine_type 由 AgentConfigLoader 从 da_agent_source 推导：
  - 仅有 source_type=1 (业务库)  → nl2sql
  - 仅有 source_type=2 (知识库)  → knowledge
  - 两者都有                     → hybrid

hybrid 模式下使用关键词规则 + LLM 兜底判断用户意图。

使用示例::

    classifier = IntentClassifier()
    intent = classifier.classify("什么是清算周期", engine_type="hybrid")
    # "knowledge"

    intent = classifier.classify("本月交易总额多少", engine_type="hybrid")
    # "sql"
"""

import logging
import re

logger = logging.getLogger(__name__)


# 数据查询意图关键词（倾向 SQL）
_SQL_PATTERNS = [
    r"多少", r"几个", r"几条", r"数量", r"总[额量数]", r"合计", r"汇总",
    r"统计", r"平均", r"最[大小高低]", r"排[名行序]", r"top\s?\d",
    r"占比", r"比例", r"环比", r"同比", r"趋势", r"增长",
    r"查[询一下]", r"列[出举]", r"展示", r"看[看一下]",
    r"本[月周日年]", r"上[月周日年]", r"近\d+[天日月周年]",
    r"哪[些个].*最", r"按.*分组", r"group\s*by",
]

# 知识问答意图关键词（倾向文档检索）
_KNOWLEDGE_PATTERNS = [
    r"什么是", r"是什么", r"解释.*一下", r"介绍.*一下",
    r"如何", r"怎么[办做样]", r"为什么", r"为啥",
    r"流程", r"步骤", r"规则", r"规范", r"标准",
    r"文档", r"说明", r"指南", r"手册",
    r"区别", r"差异", r"对比",
    r"含义", r"意思", r"定义",
]


def _score_patterns(query: str, patterns: list[str]) -> int:
    """统计 query 命中的模式数量。"""
    count = 0
    q = query.lower()
    for p in patterns:
        if re.search(p, q, re.IGNORECASE):
            count += 1
    return count


class IntentClassifier:
    """
    用户意图分类器。

    策略:
        - engine_type="nl2sql":  一律返回 "sql"
        - engine_type="knowledge": 一律返回 "knowledge"
        - engine_type="hybrid":  关键词评分 → 分数差明显时直接判定，否则默认 "sql"
    """

    def classify(
        self,
        query: str,
        engine_type: str = "nl2sql",
        llm_client=None,
    ) -> str:
        """
        判断用户意图。

        Args:
            query: 用户问题
            engine_type: Agent 引擎类型 (nl2sql / knowledge / hybrid)
            llm_client: LangChain ChatModel (保留扩展，当前未使用)

        Returns:
            "sql" | "knowledge" | "chitchat"
        """
        if engine_type == "nl2sql":
            return "sql"
        if engine_type == "knowledge":
            return "knowledge"

        # hybrid 模式：关键词评分
        sql_score = _score_patterns(query, _SQL_PATTERNS)
        kb_score = _score_patterns(query, _KNOWLEDGE_PATTERNS)

        logger.info(f"意图分类: sql_score={sql_score}, kb_score={kb_score}, query='{query}'")

        if kb_score > sql_score and kb_score >= 2:
            return "knowledge"
        if sql_score > kb_score:
            return "sql"

        # 平局或都未命中：默认 sql（NL2SQL 是核心能力）
        return "sql"
