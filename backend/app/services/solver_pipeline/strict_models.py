"""Strict Pydantic models for validating structured solver responses."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


logger = logging.getLogger(__name__)
MISSING_STEPS_RECOVERY_WARNING = (
    "The solver omitted structured proof steps; a conclusion-only step was "
    "recovered from its answer."
)


class StrictStructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    latex: str | None


class StrictStructuredStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    title: str
    explanation: str
    why_it_happens: str | None
    common_mistakes: list[str]
    alternative_approaches: list[str]
    hints: list[str]
    exam_tip: str | None
    latex: list[str]


class StrictStructuredPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    question: str
    answer: StrictStructuredAnswer
    steps: list[StrictStructuredStep]


class StrictStructuredSolvePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: StrictStructuredAnswer
    steps: list[StrictStructuredStep]
    parts: list[StrictStructuredPart]
    confidence: float
    warnings: list[str]


class StructuredPayloadValidationError(ValueError):
    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


def _string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def _normalize_answer(value: Any) -> Any:
    if isinstance(value, str):
        return {"text": value, "latex": None}
    if not isinstance(value, dict):
        return value
    return {
        "text": value.get("text"),
        "latex": value.get("latex"),
    }


def _normalize_step(value: Any, *, fallback_index: int) -> Any:
    if isinstance(value, str):
        return {
            "index": fallback_index,
            "title": f"Step {fallback_index}",
            "explanation": value,
            "why_it_happens": None,
            "common_mistakes": [],
            "alternative_approaches": [],
            "hints": [],
            "exam_tip": None,
            "latex": [],
        }
    if not isinstance(value, dict):
        return value
    raw_index = value.get("index", fallback_index)
    if isinstance(raw_index, str) and raw_index.strip().isdigit():
        raw_index = int(raw_index.strip())
    title = value.get("title", f"Step {fallback_index}")
    return {
        "index": raw_index,
        "title": title,
        "explanation": value.get("explanation"),
        "why_it_happens": value.get("why_it_happens"),
        "common_mistakes": _string_list(value.get("common_mistakes")),
        "alternative_approaches": _string_list(
            value.get("alternative_approaches")
        ),
        "hints": _string_list(value.get("hints")),
        "exam_tip": value.get("exam_tip"),
        "latex": _string_list(value.get("latex")),
    }


def _normalize_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    root = payload
    if "answer" not in root:
        nested_candidates = [
            value
            for value in root.values()
            if isinstance(value, dict) and "answer" in value
        ]
        if len(nested_candidates) == 1:
            root = nested_candidates[0]
    if "answer" not in root:
        return payload

    raw_parts = root.get("parts", [])
    parts: Any = raw_parts
    if isinstance(raw_parts, list):
        parts = []
        for part_index, part in enumerate(raw_parts):
            if not isinstance(part, dict):
                parts.append(part)
                continue
            part_steps = part.get("steps")
            normalized_part_steps = (
                [
                    _normalize_step(step, fallback_index=step_index)
                    for step_index, step in enumerate(part_steps, start=1)
                ]
                if isinstance(part_steps, list)
                else part_steps
            )
            parts.append(
                {
                    "label": part.get("label", chr(ord("a") + part_index)),
                    "question": part.get("question", ""),
                    "answer": _normalize_answer(part.get("answer")),
                    "steps": normalized_part_steps,
                }
            )

    raw_steps = root.get("steps")
    if raw_steps is None:
        for alias in (
            "proof_steps",
            "solution_steps",
            "reasoning_steps",
            "proof",
            "solution",
            "reasoning",
        ):
            alias_value = root.get(alias)
            if isinstance(alias_value, dict) and "steps" in alias_value:
                alias_value = alias_value.get("steps")
            if isinstance(alias_value, (list, str)):
                raw_steps = alias_value if isinstance(alias_value, list) else [alias_value]
                break
    if (
        (raw_steps is None or raw_steps == [])
        and isinstance(raw_parts, list)
        and len(raw_parts) == 1
        and isinstance(raw_parts[0], dict)
        and isinstance(raw_parts[0].get("steps"), list)
        and raw_parts[0]["steps"]
    ):
        raw_steps = raw_parts[0]["steps"]

    recovery_warning: str | None = None
    if (raw_steps is None or raw_steps == []) and not raw_parts:
        normalized_answer = _normalize_answer(root.get("answer"))
        answer_text = (
            normalized_answer.get("text")
            if isinstance(normalized_answer, dict)
            else None
        )
        if isinstance(answer_text, str) and answer_text.strip():
            raw_steps = [answer_text]
            recovery_warning = MISSING_STEPS_RECOVERY_WARNING

    steps = (
        [
            _normalize_step(step, fallback_index=index)
            for index, step in enumerate(raw_steps, start=1)
        ]
        if isinstance(raw_steps, list)
        else raw_steps
    )

    confidence = root.get("confidence", 0.5)
    if type(confidence) is int:
        confidence = float(confidence)
    warnings = _string_list(root.get("warnings"))
    if recovery_warning and isinstance(warnings, list):
        warnings.append(recovery_warning)
    return {
        "answer": _normalize_answer(root.get("answer")),
        "steps": steps,
        "parts": parts,
        "confidence": confidence,
        "warnings": warnings,
    }


def validate_structured_payload(
    payload: dict[str, Any],
    *,
    request_id: str | None,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """Validate a structured response and log a bounded failure summary."""
    normalized_payload = _normalize_structured_payload(payload)
    try:
        return StrictStructuredSolvePayload.model_validate(
            normalized_payload
        ).model_dump(mode="python")
    except ValidationError as exc:
        issue_codes = Counter(
            str(item.get("type", "invalid")).replace(".", "_")
            for item in exc.errors(include_url=False)
        )
        bare_step = (
            {"index", "title", "latex"}.issubset(normalized_payload)
            and "answer" not in normalized_payload
            and "steps" not in normalized_payload
        )
        failure_shape = "bare_step_object" if bare_step else "schema_validation_failure"
        logger.warning(
            "Structured solve schema validation failed request_id=%s provider=%s "
            "model=%s operation=structured_math_solution failure_shape=%s "
            "issue_count=%s issue_codes=%s",
            request_id,
            provider,
            model,
            failure_shape,
            len(exc.errors(include_url=False)),
            json.dumps(dict(sorted(issue_codes.items())), sort_keys=True),
        )
        raise StructuredPayloadValidationError(
            "Structured solve response was a bare step object instead of the required solution object."
            if bare_step
            else "Structured solve response did not match the required schema.",
            failure_code=failure_shape,
        ) from exc


__all__ = [
    "MISSING_STEPS_RECOVERY_WARNING",
    "StrictStructuredAnswer",
    "StrictStructuredPart",
    "StrictStructuredSolvePayload",
    "StrictStructuredStep",
    "StructuredPayloadValidationError",
    "validate_structured_payload",
]
