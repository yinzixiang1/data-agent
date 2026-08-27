"""Dispatch controlled Agent-side result tools after SQL execution."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.tools.analysis import AnalysisInvoker, AnalysisSkipped, execute_analysis

logger = logging.getLogger(__name__)


def execute_agent_result_tools(
    tool_calls: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    *,
    query_result: dict[str, Any] | None,
    analysis_context: dict[str, Any] | None = None,
    missing_result_error: str = "查询没有产生可供工具处理的结果",
    invoke: AnalysisInvoker,
) -> list[dict[str, Any]]:
    """Execute only registered tools assigned to the Agent post-query stage."""
    definitions = {
        str(tool.get("name") or ""): tool
        for tool in tool_definitions
        if isinstance(tool, dict) and tool.get("name")
    }
    results = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        definition = definitions.get(name)
        if not definition or definition.get("execution_stage") != "agent_post_query":
            continue
        started = time.monotonic()
        if query_result is None:
            results.append(
                _tool_result(
                    name,
                    "skipped",
                    started,
                    error=missing_result_error,
                )
            )
            continue
        executor_key = str(definition.get("executor_key") or "")
        arguments = call.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        binding_config = definition.get("binding_config")
        binding_config = binding_config if isinstance(binding_config, dict) else {}
        runtime_config = definition.get("runtime_config")
        runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
        try:
            if executor_key != "analyze_result":
                raise RuntimeError(f"未注册的 Agent 结果工具执行器: {executor_key}")
            output, usage = execute_analysis(
                query_result=query_result,
                arguments=arguments,
                binding_config=binding_config,
                runtime_config=runtime_config,
                analysis_context=analysis_context,
                invoke=invoke,
            )
            result = _tool_result(name, "success", started, output=output)
            if usage:
                result["usage"] = {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                }
            results.append(result)
        except AnalysisSkipped as exc:
            results.append(_tool_result(name, "skipped", started, error=str(exc)))
        except Exception as exc:
            logger.exception(
                "Agent result tool execution failed",
                extra={"tool_name": name, "executor_key": executor_key},
            )
            results.append(
                _tool_result(
                    name,
                    "failed",
                    started,
                    error=f"智能分析执行异常: {type(exc).__name__}",
                )
            )
    return results


def _tool_result(
    name: str,
    status: str,
    started: float,
    *,
    output: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "output": output or {},
        "error": error[:1000],
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
