import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from app.integrations.errors import IntegrationRequestError
from app.integrations.llama_client import LlamaClient


def test_llama_client_disables_opaque_sdk_retries_and_reuses_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    close_count = 0

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok": true}')
                    )
                ]
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructor_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setattr("app.integrations.llama_client.AsyncOpenAI", FakeAsyncOpenAI)
    settings = SimpleNamespace(
        local_llama_enabled=True,
        local_solver_llama_base_url="http://localhost:18080",
        local_solver_llama_model="local-test-model",
        local_solver_llama_timeout_seconds=2.0,
    )
    client = LlamaClient(settings)

    async def run_requests() -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            first = await client.generate_json(
                prompt="Return JSON.",
                model="tiny-routing-model",
                thinking_budget_tokens=0,
                timeout_seconds=1.0,
                operation="local_geometry_extraction",
                trace_id="retry-test",
            )
            second = await client.generate_json(
                prompt="Return JSON again.",
                model="tiny-routing-model",
                timeout_seconds=1.0,
            )
            return first, second
        finally:
            await client.aclose()

    first_result, second_result = asyncio.run(run_requests())

    assert constructor_kwargs["max_retries"] == 0
    assert constructor_kwargs["base_url"] == "http://localhost:18080/v1"
    assert len(requests) == 2
    assert requests[0]["model"] == "tiny-routing-model"
    assert requests[0]["extra_body"] == {"thinking_budget_tokens": 0}
    assert first_result == {"ok": True}
    assert second_result == {"ok": True}
    assert close_count == 1


def test_llama_connection_refusal_logs_root_cause_and_opens_shared_circuit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_count = 0

    class RefusingCompletions:
        async def create(self, **kwargs: Any) -> Any:
            nonlocal request_count
            request_count += 1
            request = httpx.Request("POST", "http://localhost:28080/v1/chat/completions")
            connection_error = APIConnectionError(request=request)
            raise connection_error from httpx.ConnectError(
                "[Errno 111] Connection refused", request=request
            )

    class RefusingAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=RefusingCompletions())

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "app.integrations.llama_client.AsyncOpenAI", RefusingAsyncOpenAI
    )
    settings = SimpleNamespace(
        local_llama_enabled=True,
        local_solver_llama_base_url="http://localhost:28080",
        local_solver_llama_model="local:connection-refusal-test",
        local_solver_llama_timeout_seconds=2.0,
        local_llama_unavailable_cooldown_seconds=60.0,
    )
    first_client = LlamaClient(settings)

    async def run_requests() -> None:
        with caplog.at_level(logging.WARNING), pytest.raises(IntegrationRequestError):
            await first_client.generate_json(
                prompt="Return JSON.",
                operation="local_route_classification",
                trace_id="connection-refusal-test",
            )

        assert request_count == 1
        assert first_client.available is False
        second_client = LlamaClient(settings)
        with pytest.raises(IntegrationRequestError, match="temporarily unavailable"):
            await second_client.generate_json(
                prompt="Return JSON.",
                operation="local_geometry_extraction",
                trace_id="same-request-or-next-request",
            )
        await first_client.aclose()
        await second_client.aclose()

    asyncio.run(run_requests())

    failure_log = next(
        record.getMessage()
        for record in caplog.records
        if "Llama-server request failed" in record.getMessage()
    )
    assert "operation=local_route_classification" in failure_log
    assert "Connection error" in failure_log
    assert "ConnectError: [Errno 111] Connection refused" in failure_log
    assert request_count == 1


def test_llama_client_supports_mock_transport_and_closes_pool() -> None:
    requests: list[httpx.Request] = []
    settings = SimpleNamespace(
        local_llama_enabled=True,
        local_solver_llama_base_url="http://llama.test",
        local_solver_llama_model="local-test-model",
        local_solver_llama_timeout_seconds=2.0,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "local-test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"ok":true}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = LlamaClient(settings, transport=httpx.MockTransport(handler))
    underlying_client = client.http_client

    async def run_requests() -> None:
        for _ in range(2):
            assert await client.generate_json(prompt="Return JSON.") == {"ok": True}
            assert client.http_client is underlying_client
            assert underlying_client.is_closed is False
        await client.aclose()

    asyncio.run(run_requests())

    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert underlying_client.is_closed is True
