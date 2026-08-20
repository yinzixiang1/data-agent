import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval import codex_chat_model as codex_module
from src.retrieval.agent_config import AgentConfigLoader, AgentRuntimeConfig
from src.retrieval.codex_chat_model import (
    CodexChatModel,
    CodexModelError,
    CodexTimeoutError,
    codex_query_budget,
    serialize_messages,
)


class FakeHandle:
    def __init__(self, response: str = '{"answer":"ok"}'):
        self.response = response
        self.interrupted = False

    def run(self):
        usage = SimpleNamespace(
            last=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5)
        )
        return SimpleNamespace(final_response=self.response, usage=usage)

    def interrupt(self):
        self.interrupted = True


class FakeThread:
    def __init__(self, handle: FakeHandle):
        self.handle = handle
        self.turn_kwargs = None

    def turn(self, prompt, **kwargs):
        self.prompt = prompt
        self.turn_kwargs = kwargs
        return self.handle


class FakeCodex:
    def __init__(self, handle: FakeHandle | None = None):
        self.handle = handle or FakeHandle()
        self.thread_calls = []
        self.threads = []
        self.closed = False

    def thread_start(self, **kwargs):
        self.thread_calls.append(kwargs)
        thread = FakeThread(self.handle)
        self.threads.append(thread)
        return thread

    def account(self, **kwargs):
        return SimpleNamespace(
            account=SimpleNamespace(email="must-not-leak@example.com"),
            requires_openai_auth=True,
        )

    def models(self, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(model="model-a"), SimpleNamespace(model="model-b")]
        )

    def close(self):
        self.closed = True


def test_message_serialization_preserves_supported_roles():
    payload = json.loads(
        serialize_messages([SystemMessage("rules"), HumanMessage("question")])
    )

    assert payload["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]


def test_codex_tool_surfaces_and_shell_environment_are_disabled():
    overrides = set(codex_module._CODEX_CONFIG_OVERRIDES)
    assert "features.view_image=false" in overrides
    assert "tools.view_image=false" in overrides
    assert 'shell_environment_policy.inherit="none"' in overrides
    assert "features.shell_tool=false" in overrides
    assert "features.unified_exec=false" in overrides


def test_provider_codex_is_explicit_and_has_no_fallback(monkeypatch):
    from src.retrieval import llm_factory

    sentinel = object()
    monkeypatch.setattr(codex_module, "CodexChatModel", lambda **kwargs: sentinel)
    monkeypatch.setattr(
        llm_factory,
        "init_chat_model",
        lambda **kwargs: pytest.fail("Codex must not use the OpenAI fallback"),
    )

    config = AgentRuntimeConfig(llm_provider="codex", llm_model="model-a")
    assert llm_factory.create_chat_model(config) is sentinel

    def fail_codex(**kwargs):
        raise CodexModelError("safe failure")

    monkeypatch.setattr(codex_module, "CodexChatModel", fail_codex)
    with pytest.raises(CodexModelError, match="safe failure"):
        llm_factory.create_chat_model(config)


def test_codex_model_config_defaults_and_validation():
    loader = AgentConfigLoader.__new__(AgentConfigLoader)
    config = AgentRuntimeConfig()
    loader._apply_agent_configs(
        config,
        {
            "model": {
                "codex_reasoning_effort": "xhigh",
                "codex_timeout_seconds": 100,
                "codex_max_concurrency": 2,
            }
        },
        {},
    )
    assert config.codex_reasoning_effort == "xhigh"
    assert config.codex_timeout_seconds == 100
    assert config.codex_max_concurrency == 2

    invalid = AgentRuntimeConfig()
    loader._apply_agent_configs(
        invalid,
        {
            "model": {
                "codex_reasoning_effort": "extreme",
                "codex_timeout_seconds": 120,
                "codex_max_concurrency": 0,
            }
        },
        {},
    )
    assert invalid.codex_reasoning_effort == "low"
    assert invalid.codex_timeout_seconds == 90
    assert invalid.codex_max_concurrency == 4


def test_codex_structured_response_and_isolated_turns(monkeypatch):
    fake = FakeCodex(FakeHandle('{"answer":"```sql\\nSELECT 1\\n```"}'))
    monkeypatch.setattr(codex_module, "_sdk_client", lambda workdir: fake)
    model = CodexChatModel(model_name="model-a")
    try:
        first = model.invoke([HumanMessage("one")])
        second = model.invoke([HumanMessage("two")])
        assert first.content == "```sql\nSELECT 1\n```"
        assert second.content == first.content
        assert len(fake.thread_calls) == 2
        assert all(call["ephemeral"] is True for call in fake.thread_calls)
        assert all(call["cwd"] == model._workdir for call in fake.thread_calls)
        assert all(thread.turn_kwargs["output_schema"] for thread in fake.threads)
    finally:
        model.close()


def test_invalid_structured_response_is_controlled(monkeypatch):
    fake = FakeCodex(FakeHandle("not-json"))
    monkeypatch.setattr(codex_module, "_sdk_client", lambda workdir: fake)
    model = CodexChatModel(model_name="model-a")
    try:
        with pytest.raises(CodexModelError, match="返回格式无效"):
            model.invoke([HumanMessage("one")])
    finally:
        model.close()


def test_timeout_interrupts_and_replaces_stuck_executor(monkeypatch):
    release = threading.Event()

    class StuckHandle(FakeHandle):
        def run(self):
            release.wait()
            return super().run()

    stuck_handle = StuckHandle()
    clients = [FakeCodex(stuck_handle), FakeCodex()]
    monkeypatch.setattr(codex_module, "_sdk_client", lambda workdir: clients.pop(0))
    monkeypatch.setattr(codex_module, "_INTERRUPT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(codex_module, "_CLIENT_CLOSE_GRACE_SECONDS", 0.01)
    model = CodexChatModel(model_name="model-a", timeout_seconds=10)
    old_executor = model._executor
    try:
        with codex_query_budget(0.01), pytest.raises(CodexTimeoutError):
            model.invoke([HumanMessage("slow")])
        assert stuck_handle.interrupted is True
        assert model._executor is not old_executor
        release.set()
        assert model.invoke([HumanMessage("next")]).content == "ok"
    finally:
        release.set()
        model.close()


def test_status_never_returns_identity_or_token(monkeypatch):
    fake = FakeCodex()
    monkeypatch.setattr(codex_module, "_sdk_client", lambda workdir: fake)
    model = CodexChatModel(model_name="model-a")
    try:
        status = model.safe_status()
        assert status["status"] == "ready"
        assert status["models"] == ["model-a", "model-b"]
        assert set(status) <= {"status", "message", "cli_version", "models"}
        assert "must-not-leak" not in json.dumps(status)
    finally:
        model.close()


def test_status_endpoint_filters_unknown_sensitive_fields(monkeypatch):
    import app as service

    class StatusClient:
        def safe_status(self):
            return {
                "status": "ready",
                "message": "ok",
                "models": ["model-a"],
                "email": "must-not-leak@example.com",
                "token": "must-not-leak",
                "path": "/private/account/path",
            }

    monkeypatch.setattr(service, "_verify_admin_token", lambda request: None)
    monkeypatch.setattr(
        service, "agent_config", AgentRuntimeConfig(llm_provider="codex")
    )
    monkeypatch.setattr(service, "llm_client", StatusClient())

    response = asyncio.run(service.codex_status(object()))
    payload = response.model_dump(exclude_none=True)
    assert payload == {"status": "ready", "message": "ok", "models": ["model-a"]}


def test_codex_test_runs_selected_model_without_leaking_response(monkeypatch):
    import app as service

    tested_models = []

    monkeypatch.setattr(service, "_verify_admin_token", lambda request: None)
    monkeypatch.setattr(
        service,
        "_run_codex_smoke_test",
        lambda model_name: tested_models.append(model_name),
    )

    response = asyncio.run(
        service.codex_test(service.CodexTestRequest(model=" model-a "), object())
    )

    assert tested_models == ["model-a"]
    assert response.status == "success"
    assert response.message == "model-a 调用成功"
    assert response.latency_ms is not None


def test_full_query_runs_in_worker_and_respects_concurrency(monkeypatch):
    import app as service

    active = 0
    max_active = 0
    worker_threads = set()
    main_thread = threading.get_ident()
    state_lock = threading.Lock()

    def fake_run_query(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        worker_threads.add(threading.get_ident())
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return {"sql": "SELECT 1"}

    monkeypatch.setattr(service, "run_query", fake_run_query)
    config = AgentRuntimeConfig(
        llm_provider="codex", codex_timeout_seconds=1, codex_max_concurrency=1
    )
    semaphore = asyncio.Semaphore(1)

    async def run_scenario():
        await asyncio.gather(
            service._run_query_in_worker("one", config, object(), semaphore=semaphore),
            service._run_query_in_worker("two", config, object(), semaphore=semaphore),
        )

    asyncio.run(run_scenario())
    assert max_active == 1
    assert worker_threads and main_thread not in worker_threads


def test_queue_time_consumes_cumulative_budget(monkeypatch):
    import app as service

    monkeypatch.setattr(
        service,
        "run_query",
        lambda *args, **kwargs: (time.sleep(0.08), {"sql": "SELECT 1"})[1],
    )
    config = AgentRuntimeConfig(
        llm_provider="codex", codex_timeout_seconds=0.03, codex_max_concurrency=1
    )
    semaphore = asyncio.Semaphore(1)

    async def run_scenario():
        first = asyncio.create_task(
            service._run_query_in_worker("one", config, object(), semaphore=semaphore)
        )
        await asyncio.sleep(0.005)
        with pytest.raises(CodexTimeoutError):
            await service._run_query_in_worker(
                "two", config, object(), semaphore=semaphore
            )
        await first

    asyncio.run(run_scenario())


def test_stale_runtime_is_rejected_before_using_closed_client(monkeypatch):
    import app as service

    called = False

    def fake_run_query(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(service, "run_query", fake_run_query)
    old_client = object()
    old_semaphore = asyncio.Semaphore(1)
    config = AgentRuntimeConfig(llm_provider="codex", codex_timeout_seconds=1)
    lock = asyncio.Lock()
    monkeypatch.setattr(service, "llm_client", object())
    monkeypatch.setattr(service, "codex_query_semaphore", asyncio.Semaphore(1))

    async def run_scenario():
        with pytest.raises(CodexModelError, match="配置刚刚更新"):
            await service._run_query_in_worker(
                "question",
                config,
                old_client,
                semaphore=old_semaphore,
                runtime_swap_lock=lock,
            )

    asyncio.run(run_scenario())
    assert called is False
