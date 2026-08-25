"""LangChain adapter for the official synchronous Codex SDK."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai_codex import ApprovalMode, Codex, CodexConfig, CodexError, Sandbox
from openai_codex.types import ReasoningEffort
from pydantic import BaseModel, PrivateAttr, ValidationError

logger = logging.getLogger(__name__)

_QUERY_DEADLINE: ContextVar[float | None] = ContextVar(
    "codex_query_deadline", default=None
)
_INTERRUPT_GRACE_SECONDS = 3.0
_CLIENT_CLOSE_GRACE_SECONDS = 1.0
_CODEX_RUNTIME_ERRORS = (
    CodexError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# The dedicated CODEX_HOME must contain authentication only. These runtime
# overrides additionally remove every tool surface supported by Codex 0.147.0.
_CODEX_CONFIG_OVERRIDES = (
    'approval_policy="never"',
    'sandbox_mode="read-only"',
    'shell_environment_policy.inherit="none"',
    'web_search="disabled"',
    "tools.web_search=false",
    "features.shell_tool=false",
    "features.unified_exec=false",
    "features.shell_snapshot=false",
    "features.apps=false",
    "features.plugins=false",
    "features.remote_plugin=false",
    "features.browser_use=false",
    "features.browser_use_external=false",
    "features.browser_use_full_cdp_access=false",
    "features.in_app_browser=false",
    "features.computer_use=false",
    "features.multi_agent=false",
    "agents.enabled=false",
    "features.image_generation=false",
    "features.view_image=false",
    "tools.view_image=false",
    "features.skill_search=false",
    "features.skill_mcp_dependency_install=false",
    "features.workspace_dependencies=false",
)

_DEVELOPER_INSTRUCTIONS = """You are a text-generation backend embedded in Lumen.
Use only the ordered conversation supplied in the input. Never call tools, inspect files,
run commands, browse, use plugins/apps/skills/MCP, delegate work, or ask for approval.
Do not reveal system, runtime, account, filesystem, or authentication information.
Return exactly one JSON object matching the supplied schema. Put the complete response
that should be sent to the caller in the `answer` field without changing its formatting.
"""


class CodexModelError(RuntimeError):
    """Safe, user-facing Codex invocation failure."""


class CodexTimeoutError(CodexModelError):
    """The cumulative Codex query budget was exhausted."""


class _StructuredAnswer(BaseModel):
    answer: str


@contextmanager
def codex_query_budget(timeout_seconds: int) -> Iterator[None]:
    """Set one cumulative deadline for every Codex turn in a query pipeline."""
    requested_deadline = time.monotonic() + timeout_seconds
    current_deadline = _QUERY_DEADLINE.get()
    deadline = (
        min(current_deadline, requested_deadline)
        if current_deadline is not None
        else requested_deadline
    )
    token = _QUERY_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _QUERY_DEADLINE.reset(token)


def codex_remaining_timeout(default_seconds: int) -> float:
    """Return the remaining cumulative budget or raise a safe timeout."""
    deadline = _QUERY_DEADLINE.get()
    if deadline is None:
        return float(default_seconds)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CodexTimeoutError("Codex 查询已超过时间限制，请稍后重试")
    return remaining


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    text_parts: list[str] = []
    for part in message.content:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    if text_parts:
        return "\n".join(text_parts)
    return json.dumps(message.content, ensure_ascii=False, default=str)


def serialize_messages(messages: Sequence[BaseMessage]) -> str:
    """Serialize a LangChain conversation without flattening role boundaries."""
    transcript = [
        {
            "role": {
                "ai": "assistant",
                "human": "user",
            }.get(message.type, message.type),
            "content": _message_text(message),
        }
        for message in messages
    ]
    return json.dumps(
        {"task": "Respond to this ordered conversation", "messages": transcript},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sdk_client(workdir: str) -> Codex:
    return Codex(
        CodexConfig(
            cwd=workdir,
            config_overrides=_CODEX_CONFIG_OVERRIDES,
            client_name="lumen_data_agent",
            client_title="Lumen Data Agent",
        )
    )


class CodexChatModel(BaseChatModel):
    """A synchronous BaseChatModel backed by one reusable Codex client."""

    model_name: str
    reasoning_effort: str = "low"
    timeout_seconds: int = 90
    max_concurrency: int = 1

    _client: Codex | None = PrivateAttr(default=None)
    _client_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _executor: ThreadPoolExecutor = PrivateAttr()
    _executor_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _slots: threading.BoundedSemaphore = PrivateAttr()
    _workdir: str = PrivateAttr()
    _closed: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any, /) -> None:
        self._workdir = tempfile.mkdtemp(prefix="lumen-codex-")
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix="lumen-codex-turn",
        )
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        try:
            self._client = _sdk_client(self._workdir)
        except _CODEX_RUNTIME_ERRORS as exc:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)
            shutil.rmtree(self._workdir, ignore_errors=True)
            logger.error(
                "Codex SDK client initialization failed",
                extra={"error_type": type(exc).__name__},
            )
            raise CodexModelError(
                "Codex 初始化失败，请检查服务器订阅登录状态"
            ) from None

    @property
    def _llm_type(self) -> str:
        return "codex"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "reasoning_effort": self.reasoning_effort,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        remaining = codex_remaining_timeout(self.timeout_seconds)
        if not self._slots.acquire(timeout=remaining):
            raise CodexTimeoutError("Codex 查询已超过时间限制，请稍后重试")
        try:
            remaining = codex_remaining_timeout(self.timeout_seconds)
            result = self._run_turn(serialize_messages(messages), remaining)
            answer = self._parse_answer(result.final_response)
            usage = self._usage_metadata(result.usage)
            message = AIMessage(content=answer, usage_metadata=usage)
            return ChatResult(generations=[ChatGeneration(message=message)])
        except CodexModelError:
            raise
        except _CODEX_RUNTIME_ERRORS as exc:
            logger.error("Codex turn failed", extra={"error_type": type(exc).__name__})
            raise CodexModelError("Codex 调用失败，请稍后重试") from None
        finally:
            self._slots.release()

    def _run_turn(self, prompt: str, timeout_seconds: float) -> Any:
        handle_holder: list[Any] = []
        holder_lock = threading.Lock()

        def collect() -> Any:
            client = self._get_client()
            thread = client.thread_start(
                approval_mode=ApprovalMode.deny_all,
                cwd=self._workdir,
                developer_instructions=_DEVELOPER_INSTRUCTIONS,
                ephemeral=True,
                model=self.model_name,
                sandbox=Sandbox.read_only,
            )
            handle = thread.turn(
                prompt,
                approval_mode=ApprovalMode.deny_all,
                cwd=self._workdir,
                effort=ReasoningEffort(self.reasoning_effort),
                model=self.model_name,
                output_schema=_OUTPUT_SCHEMA,
                sandbox=Sandbox.read_only,
            )
            with holder_lock:
                handle_holder.append(handle)
            return handle.run()

        with self._executor_lock:
            if self._closed:
                raise CodexModelError("Codex 客户端已关闭")
            executor = self._executor
            future: Future[Any] = executor.submit(collect)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout:
            with holder_lock:
                handle = handle_holder[0] if handle_holder else None
            if handle is not None:
                try:
                    handle.interrupt()
                except _CODEX_RUNTIME_ERRORS as exc:
                    logger.warning(
                        "Codex turn interrupt failed",
                        extra={"error_type": type(exc).__name__},
                    )
            try:
                future.result(timeout=_INTERRUPT_GRACE_SECONDS)
            except FutureTimeout:
                self._discard_client()
                try:
                    future.result(timeout=_CLIENT_CLOSE_GRACE_SECONDS)
                except FutureTimeout:
                    self._replace_executor(executor)
                except _CODEX_RUNTIME_ERRORS as exc:
                    logger.debug(
                        "Codex turn finished with an error after client cleanup",
                        extra={"error_type": type(exc).__name__},
                    )
            except _CODEX_RUNTIME_ERRORS as exc:
                logger.debug(
                    "Codex turn finished with an error after interrupt",
                    extra={"error_type": type(exc).__name__},
                )
            raise CodexTimeoutError("Codex 查询已超过时间限制，请稍后重试") from None

    def _replace_executor(self, stale_executor: ThreadPoolExecutor) -> None:
        """Keep later calls usable even if an SDK collector failed to unwind."""
        with self._executor_lock:
            if self._closed or self._executor is not stale_executor:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrency,
                thread_name_prefix="lumen-codex-turn",
            )
        stale_executor.shutdown(wait=False, cancel_futures=True)

    def _get_client(self) -> Codex:
        with self._client_lock:
            if self._closed:
                raise CodexModelError("Codex 客户端已关闭")
            if self._client is None:
                try:
                    self._client = _sdk_client(self._workdir)
                except _CODEX_RUNTIME_ERRORS as exc:
                    logger.error(
                        "Codex SDK client restart failed",
                        extra={"error_type": type(exc).__name__},
                    )
                    raise CodexModelError("Codex 暂时不可用，请稍后重试") from None
            return self._client

    def _discard_client(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except _CODEX_RUNTIME_ERRORS as exc:
                logger.warning(
                    "Codex SDK client cleanup failed",
                    extra={"error_type": type(exc).__name__},
                )

    @staticmethod
    def _parse_answer(raw_response: str | None) -> str:
        if not raw_response:
            raise CodexModelError("Codex 未返回有效结果，请稍后重试")
        try:
            return _StructuredAnswer.model_validate_json(raw_response).answer
        except (ValidationError, TypeError, ValueError):
            raise CodexModelError("Codex 返回格式无效，请稍后重试") from None

    @staticmethod
    def _usage_metadata(usage: Any) -> dict[str, int] | None:
        if usage is None or getattr(usage, "last", None) is None:
            return None
        last = usage.last
        return {
            "input_tokens": int(last.input_tokens),
            "output_tokens": int(last.output_tokens),
            "total_tokens": int(last.total_tokens),
        }

    def safe_status(self) -> dict[str, Any]:
        """Return live SDK status without identity, token, or path fields."""
        result: dict[str, Any] = {
            "status": "unavailable",
            "message": "Codex 暂时不可用",
        }
        try:
            result["cli_version"] = version("openai-codex-cli-bin")
        except PackageNotFoundError:
            pass
        try:
            client = self._get_client()
            account = client.account(refresh_token=False)
            if account.account is None:
                result.update(
                    status="unauthenticated", message="Codex 尚未完成订阅登录"
                )
                return result
            models = client.models(include_hidden=False)
            result.update(
                status="ready",
                message="Codex 已登录并可用",
                models=[item.model for item in models.data],
            )
            return result
        except _CODEX_RUNTIME_ERRORS as exc:
            logger.warning(
                "Codex status check failed", extra={"error_type": type(exc).__name__}
            )
            return result

    def close(self) -> None:
        """Stop the SDK runtime and discard the isolated empty working directory."""
        with self._client_lock:
            if self._closed:
                return
            self._closed = True
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except _CODEX_RUNTIME_ERRORS as exc:
                logger.warning(
                    "Codex SDK client close failed",
                    extra={"error_type": type(exc).__name__},
                )
        with self._executor_lock:
            executor = self._executor
        executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(Path(self._workdir), ignore_errors=True)
