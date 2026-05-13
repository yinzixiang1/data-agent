"""
多轮对话上下文压缩 — 将历史问答 + 当前补充合并为一个完整的自然语言问题。

解决的问题:
    用户第1轮: "张3今天有多少笔交易"
    用户第2轮: "不包含手续费，一次兑换属于一笔交易"
    → 压缩为: "张3今天有多少笔交易，不包含手续费，一次兑换属于一笔交易"

这样 RAG 检索时用完整问题去检索，命中率更高。

使用示例::

    compressor = ContextCompressor(client, model="deepseek-chat")
    merged = compressor.compress(
        history_summary="张3今天有多少笔交易|||pmt_finance_transactions,pmt_account",
        current_question="不包含手续费，一次兑换属于一笔交易",
    )
    # "张3今天有多少笔交易，不包含手续费，一次兑换属于一笔交易"
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

COMPRESS_PROMPT = """你是一个查询合并助手。用户在多轮对话中逐步补充查询条件。

请将下面的历史查询和用户新补充内容合并成一个完整、独立的自然语言问题。

要求:
1. 保留历史中的所有查询条件（人名、时间、筛选条件等）
2. 把新补充的内容融合进去
3. 输出一个完整的问题，不需要解释
4. 如果新补充明显是一个全新的问题（与历史无关），直接返回新问题原文

## 历史查询
{history}

## 用户新补充
{current}

## 合并后的完整问题"""


class ContextCompressor:
    """多轮对话上下文压缩器。"""

    def __init__(self, model: BaseChatModel, custom_prompt: str = ""):
        self.model = model
        self.prompt_template = custom_prompt.strip() if custom_prompt else COMPRESS_PROMPT

    def compress(self, history_summary: str, current_question: str) -> str:
        """
        将历史摘要 + 当前问题压缩为一个完整问题。

        Args:
            history_summary: 上一轮的摘要，格式 "question|||table1,table2,..."
            current_question: 当前用户输入

        Returns:
            合并后的完整自然语言问题
        """
        parts = history_summary.split("|||", 1)
        prev_question = parts[0] if parts else ""
        prev_tables = parts[1] if len(parts) > 1 else ""

        history_text = f"问题: {prev_question}"
        if prev_tables:
            history_text += f"\n涉及表: {prev_tables}"

        prompt = self.prompt_template.format(history=history_text, current=current_question)

        try:
            resp = self.model.invoke([HumanMessage(content=prompt)])
            merged = resp.content.strip()
            logger.info(f"上下文压缩: '{current_question}' → '{merged}'")
            return merged
        except Exception as e:
            logger.warning(f"上下文压缩失败，使用原始问题: {e}")
            return current_question

    @staticmethod
    def build_summary(question: str, tables: list[str]) -> str:
        """
        构建本轮摘要，供下一轮使用。

        Args:
            question: 本轮的完整问题（可能已经过压缩）
            tables: 本轮命中的表名列表

        Returns:
            摘要字符串，格式 "question|||table1,table2,..."
        """
        tables_str = ",".join(tables) if tables else ""
        return f"{question}|||{tables_str}"
