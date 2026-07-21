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
    "model",
    [
        NVIDIA_GPT_OSS_120B_MODEL,
        NVIDIA_GPT_OSS_20B_MODEL,
    ],
)
def test_nvidia_payload_uses_family_controls_without_unsupported_response_format(
    model: str,
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
    assert "JSON-escape every backslash" in payload["messages"][0]["content"]
    assert payload["reasoning_effort"] == "medium"
    assert "chat_template_kwargs" not in payload
    assert "reasoning_budget" not in payload


@pytest.mark.parametrize(
    ("operation", "expected_effort"),
    [
        ("structured_math_solution", "medium"),
        ("structured_math_steps_repair", "medium"),
        ("structured_math_solution_repair", "medium"),
        ("geometry_extraction", "low"),
        ("local_route_classification", "low"),
    ],
)
def test_nvidia_reasoning_effort_matches_operation(
    operation: str, expected_effort: str
) -> None:
    assert NvidiaClient(_settings())._reasoning_effort(operation) == expected_effort


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
                model=NVIDIA_GPT_OSS_20B_MODEL,
                system_prompt="Return JSON.",
                user_prompt="Test",
                operation="geometry_extraction",
                trace_id="nvidia-http-error",
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.response_body == error_body


def test_nvidia_parser_prefers_complete_schema_object_over_earlier_fragment() -> None:
    complete_payload = {
        "answer": {"text": "The result is 180 degrees.", "latex": "180^\\circ"},
        "steps": [],
        "parts": [],
        "confidence": 0.8,
        "warnings": [],
    }
    content = (
        'Draft fragment: {"text":"premature answer","latex":"180"}\n'
        + json.dumps(complete_payload)
    )
    client = NvidiaClient(
        _settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                json={
                    "model": NVIDIA_GPT_OSS_20B_MODEL,
                    "choices": [{"message": {"content": content}}],
                },
            )
        ),
    )

    result = asyncio.run(
        client.complete_json(
            model=NVIDIA_GPT_OSS_20B_MODEL,
            system_prompt="Return JSON.",
            user_prompt="Test",
            json_schema={
                "type": "object",
                "required": ["answer", "steps", "parts", "confidence", "warnings"],
            },
        )
    )

    assert result == complete_payload


def test_nvidia_parser_reconstructs_schema_object_from_separate_fragments() -> None:
    schema = {
        "required": ["answer", "steps", "parts", "confidence", "warnings"],
    }
    content = "\n".join(
        [
            '{"answer":{"text":"Recovered proof","latex":"180^\\\\circ"}}',
            '{"steps":[]}',
            '{"parts":[]}',
            '{"confidence":0.7}',
            '{"warnings":[]}',
        ]
    )

    result = NvidiaClient(_settings())._loads_json_response(
        content,
        model=NVIDIA_GPT_OSS_20B_MODEL,
        json_schema=schema,
    )

    assert result == {
        "answer": {"text": "Recovered proof", "latex": "180^\\circ"},
        "steps": [],
        "parts": [],
        "confidence": 0.7,
        "warnings": [],
    }
