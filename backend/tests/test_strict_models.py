import asyncio

import pytest

from app.core.model_policy import StructuredModelEndpoint
from app.services.solver_pipeline.content_repair import (
    repair_missing_structured_steps,
)
from app.services.solver_pipeline.strict_models import (
    MISSING_STEPS_RECOVERY_WARNING,
    StructuredPayloadValidationError,
    validate_structured_payload,
)


def test_structured_payload_normalizes_safe_shape_deviations() -> None:
    payload = {
        "solution": {
            "answer": {
                "text": "The angles are supplementary.",
                "latex": "180^\\circ",
                "unused_note": "drop me",
            },
            "steps": [
                {
                    "index": "1",
                    "title": "Establish cyclicity",
                    "explanation": "Show the relevant quadrilateral is cyclic.",
                    "latex": "\\angle KIL + \\angle YPX = 180^\\circ",
                    "unused_note": "drop me",
                }
            ],
            "confidence": 1,
            "metadata": {"provider": "test"},
        }
    }

    result = validate_structured_payload(
        payload,
        request_id="test-request",
        provider="test",
        model="test-model",
    )

    assert result["answer"]["text"] == "The angles are supplementary."
    assert result["steps"][0]["index"] == 1
    assert result["steps"][0]["latex"] == [
        "\\angle KIL + \\angle YPX = 180^\\circ"
    ]
    assert result["parts"] == []
    assert result["warnings"] == []
    assert result["confidence"] == 1.0


def test_structured_payload_still_rejects_bare_answer_fragment() -> None:
    with pytest.raises(StructuredPayloadValidationError):
        validate_structured_payload(
            {"text": "Only a fragment", "latex": "180"},
            request_id="test-request",
            provider="test",
            model="test-model",
        )


def test_structured_payload_recovers_missing_steps_from_answer() -> None:
    result = validate_structured_payload(
        {
            "answer": {
                "text": "The proof concludes that the angle sum is 180 degrees.",
                "latex": "180^\\circ",
            },
            "parts": [],
            "confidence": 0.7,
            "warnings": [],
        },
        request_id="test-request",
        provider="test",
        model="test-model",
    )

    assert result["steps"][0]["explanation"].startswith("The proof concludes")
    assert "conclusion-only step" in result["warnings"][0]


def test_structured_payload_promotes_single_part_steps_to_top_level() -> None:
    step = {
        "index": 1,
        "title": "Cyclicity",
        "explanation": "Establish the cyclic quadrilateral.",
    }
    result = validate_structured_payload(
        {
            "answer": {"text": "Proved.", "latex": None},
            "parts": [
                {
                    "answer": {"text": "Proved.", "latex": None},
                    "steps": [step],
                }
            ],
            "confidence": 0.8,
            "warnings": [],
        },
        request_id="test-request",
        provider="test",
        model="test-model",
    )

    assert result["steps"][0]["title"] == "Cyclicity"


def test_missing_step_repair_replaces_conclusion_only_recovery() -> None:
    class CompletionClient:
        async def complete_json(self, **_: object) -> dict:
            return {
                "steps": [
                    {
                        "index": index,
                        "title": f"Proof step {index}",
                        "explanation": f"Justification {index}.",
                    }
                    for index in range(1, 4)
                ]
            }

    payload = validate_structured_payload(
        {
            "answer": {"text": "The angle sum is 180 degrees.", "latex": "180"},
            "confidence": 0.7,
        },
        request_id="test-request",
        provider="test",
        model="test-model",
    )
    result = asyncio.run(
        repair_missing_structured_steps(
            payload,
            problem_text="Prove the angle sum is 180 degrees.",
            completion_client=CompletionClient(),
            candidate=StructuredModelEndpoint(
                provider="test",
                model="test-model",
                routing_model="test:test-model",
            ),
            timeout_seconds=1.0,
            request_id="test-request",
        )
    )

    assert len(result["steps"]) == 3
    assert MISSING_STEPS_RECOVERY_WARNING not in result["warnings"]
