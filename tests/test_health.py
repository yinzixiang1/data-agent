import asyncio

from starlette.responses import Response

import app as service


def test_health_returns_service_unavailable_until_agent_is_ready(monkeypatch):
    monkeypatch.setattr(service, "retriever", None)
    monkeypatch.setattr(service, "validator", None)
    response = Response()

    payload = asyncio.run(service.health(response))

    assert response.status_code == 503
    assert payload["ready"] is False
    assert payload["initialized"] is False


def test_health_returns_ok_when_agent_is_ready(monkeypatch):
    retriever = type("ReadyRetriever", (), {"_initialized": True})()
    monkeypatch.setattr(service, "retriever", retriever)
    monkeypatch.setattr(service, "validator", object())
    response = Response()

    payload = asyncio.run(service.health(response))

    assert response.status_code == 200
    assert payload["ready"] is True
    assert payload["initialized"] is True
