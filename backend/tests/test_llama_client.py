import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError

from app.integrations.errors import IntegrationRequestError
from app.integrations.llama_client import LlamaClient


def test_llama_client_disables_opaque_sdk_retries_and_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs: dict[str, Any] = {}
    request_kwargs: dict[str, Any] = {}

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            request_kwargs.update(kwargs)
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

    monkeypatch.setattr("app.integrations.llama_client.AsyncOpenAI", FakeAsyncOpenAI)
    client = LlamaClient()
    client.settings = cast(
        Any,
        SimpleNamespace(
            local_llama_enabled=True,
            local_solver_llama_base_url="http://localhost:18080",
            local_solver_llama_model="local-test-model",
            local_solver_llama_timeout_seconds=2.0,
        ),
    )

    result = asyncio.run(
        client.generate_json(
            prompt="Return JSON.",
            model="tiny-routing-model",
            thinking_budget_tokens=0,
            timeout_seconds=1.0,
            operation="local_geometry_extraction",
            trace_id="retry-test",
        )
    )

    assert constructor_kwargs["max_retries"] == 0
    assert constructor_kwargs["base_url"] == "http://localhost:18080/v1"
    assert request_kwargs["model"] == "tiny-routing-model"
    assert request_kwargs["extra_body"] == {"thinking_budget_tokens": 0}
    assert result == {"ok": True}


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
    first_client = LlamaClient()
    first_client.settings = cast(Any, settings)

    with caplog.at_level(logging.WARNING), pytest.raises(IntegrationRequestError):
        asyncio.run(
            first_client.generate_json(
                prompt="Return JSON.",
                operation="local_route_classification",
                trace_id="connection-refusal-test",
            )
        )

    assert request_count == 1
    assert first_client.available is False
    failure_log = next(
        record.getMessage()
        for record in caplog.records
        if "Llama-server request failed" in record.getMessage()
    )
    assert "operation=local_route_classification" in failure_log
    assert "Connection error" in failure_log
    assert "ConnectError: [Errno 111] Connection refused" in failure_log

    second_client = LlamaClient()
    second_client.settings = cast(Any, settings)
    with pytest.raises(IntegrationRequestError, match="temporarily unavailable"):
        asyncio.run(
            second_client.generate_json(
                prompt="Return JSON.",
                operation="local_geometry_extraction",
                trace_id="same-request-or-next-request",
            )
        )
    assert request_count == 1
