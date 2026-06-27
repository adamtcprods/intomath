import pytest

from app.integrations.openrouter_client import OpenRouterClient


def test_extracts_json_from_responses_output_text() -> None:
    client = OpenRouterClient()
    data = {
        "status": "completed",
        "output_text": '{"answer": 42}',
    }

    text = client._extract_response_text(data, model="test-model")

    assert client._loads_json_response(text, model="test-model") == {"answer": 42}


def test_extracts_json_from_responses_output_items() -> None:
    client = OpenRouterClient()
    data = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '```json\n{"ok": true}\n```',
                    }
                ],
            }
        ],
    }

    text = client._extract_response_text(data, model="test-model")

    assert client._loads_json_response(text, model="test-model") == {"ok": True}


def test_extracts_json_after_model_preamble() -> None:
    client = OpenRouterClient()
    text = 'We need to solve this carefully.\n{"answer": 42}\nDone.'

    assert client._loads_json_response(text, model="test-model") == {"answer": 42}


def test_extracts_first_complete_json_object_with_trailing_text() -> None:
    client = OpenRouterClient()
    text = '{"answer": {"text": "x=2"}}\nExtra explanation that should be ignored.'

    assert client._loads_json_response(text, model="test-model") == {
        "answer": {"text": "x=2"}
    }


def test_chat_payload_uses_json_schema_response_format() -> None:
    client = OpenRouterClient()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    payload = client._chat_json_payload(
        model="test-model",
        system_prompt="Return JSON.",
        user_prompt="Solve 1+1",
        temperature=0.2,
        json_schema=schema,
        schema_name="math_answer",
        max_tokens=100,
    )

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "math_answer",
            "strict": True,
            "schema": schema,
        },
    }
    assert payload["messages"][0]["role"] == "system"
    assert "Do not echo the schema" in payload["messages"][0]["content"]
    assert payload["plugins"] == [{"id": "response-healing"}]
    assert payload["reasoning"] == {"exclude": True}


def test_openrouter_api_error_is_reported_without_choices_key() -> None:
    client = OpenRouterClient()
    data = {
        "error": {"message": "Rate limit exceeded. Please try again later."},
        "error_type": "rate_limit_exceeded",
    }

    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        client._raise_response_error(data, model="test-model")


def test_missing_responses_output_reports_shape_instead_of_choices_key() -> None:
    client = OpenRouterClient()
    data = {"status": "failed", "incomplete_details": {"reason": "max_output_tokens"}}

    with pytest.raises(RuntimeError, match="returned no output text"):
        client._extract_response_text(data, model="test-model")
