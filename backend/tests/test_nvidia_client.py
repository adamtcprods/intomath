import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.integrations.errors import IntegrationRequestError
from app.integrations.nvidia_client import NvidiaClient
from app.services.model_router import (
    NVIDIA_GPT_OSS_20B_MODEL,
    NVIDIA_GPT_OSS_120B_MODEL,
    NVIDIA_NEMOTRON_NANO_MODEL,
    NVIDIA_NEMOTRON_SUPER_MODEL,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        nvidia_direct_enabled=True,
        nvidia_api_key="test-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        remote_model_attempt_timeout_seconds=25.0,
        nvidia_large_model_attempt_timeout_seconds=50.0,
    )


@pytest.mark.parametrize(
    ("model", "family"),
    [
        (NVIDIA_NEMOTRON_SUPER_MODEL, "nemotron"),
        (NVIDIA_NEMOTRON_NANO_MODEL, "nemotron"),
        (NVIDIA_GPT_OSS_120B_MODEL, "gpt_oss"),
        (NVIDIA_GPT_OSS_20B_MODEL, "gpt_oss"),
    ],
)
def test_nvidia_payload_uses_family_controls_without_unsupported_response_format(
    model: str,
    family: str,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "nim-test",
                "model": model,
                "choices": [{"message": {"content": '{"ok":true}'}}],
            },
        )

    client = NvidiaClient(_settings(), transport=httpx.MockTransport(handler))
    result = asyncio.run(
        client.complete_json(
            model=model,
            system_prompt="Return JSON.",
            user_prompt="Test",
            json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            schema_name="test_schema",
            operation="structured_math_solution",
            trace_id="nvidia-family-test",
        )
    )

    assert result == {"ok": True}
    assert len(requests) == 1
    payload = requests[0]
    assert payload["model"] == model
    assert payload["stream"] is False
    assert "response_format" not in payload
    assert "Required JSON schema" in payload["messages"][0]["content"]
    if family == "gpt_oss":
        assert payload["reasoning_effort"] == "low"
        assert "chat_template_kwargs" not in payload
        assert "reasoning_budget" not in payload
    else:
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["reasoning_budget"] == 64
        assert "reasoning_effort" not in payload


def test_nvidia_http_error_exposes_bounded_status_and_body() -> None:
    error_body = '{"error":{"message":"model is loading"}}'
    client = NvidiaClient(
        _settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text=error_body, request=request)
        ),
    )

    with pytest.raises(IntegrationRequestError) as exc_info:
        asyncio.run(
            client.complete_json(
                model=NVIDIA_NEMOTRON_NANO_MODEL,
                system_prompt="Return JSON.",
                user_prompt="Test",
                operation="geometry_extraction",
                trace_id="nvidia-http-error",
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.response_body == error_body
