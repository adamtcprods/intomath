"""Coerce structured model payloads into solver response drafts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolvePart, SolveStep
from app.services.fallback_solver import FallbackSolver

from .subquestion import DetectedSubquestion, alpha_label, subquestion_with_context


@dataclass
class StructuredSolveDraft:
    answer: SolveAnswer
    steps: list[SolveStep]
    confidence: float
    warnings: list[str]
    parts: list[SolvePart] = field(default_factory=list)
    solver_model: str | None = None


def draft_from_payload(
    payload: dict[str, Any],
    *,
    expected_subquestions: list[DetectedSubquestion] | None = None,
) -> StructuredSolveDraft:
    expected_subquestions = expected_subquestions or []
    answer = answer_from_raw(payload.get("answer", {}))
    steps = steps_from_raw(payload.get("steps", []))
    parts = parts_from_payload(
        payload.get("parts", []),
        expected_subquestions=expected_subquestions,
        top_level_answer=answer,
        top_level_steps=steps,
    )

    if expected_subquestions and len(expected_subquestions) > 1 and not parts:
        raise ValueError("Model returned no per-question parts")
    if expected_subquestions and len(expected_subquestions) > 1:
        validate_subquestion_step_depth(parts, expected_subquestions)
    if not steps and parts:
        steps = reindex_steps(parts[0].steps)
    if not steps:
        raise ValueError("Model returned no steps")

    confidence = float(payload.get("confidence", 0.0) or 0.0)
    warnings = coerce_string_list(payload.get("warnings", []))
    return StructuredSolveDraft(
        answer=answer,
        steps=steps,
        confidence=confidence,
        warnings=warnings,
        parts=parts,
    )


def validate_subquestion_step_depth(
    parts: list[SolvePart], expected_subquestions: list[DetectedSubquestion]
) -> None:
    for index, subquestion in enumerate(expected_subquestions):
        if index >= len(parts):
            raise ValueError(f"Model returned no part for question {subquestion.label}")
        part = parts[index]
        if is_proof_like_question(subquestion.question) and len(part.steps) < 3:
            raise ValueError(
                f"Model returned too few steps for proof question {subquestion.label}"
            )


def is_proof_like_question(question: str) -> bool:
    normalized = FallbackSolver().normalize_text(question)
    proof_markers = {
        "prove",
        "proof",
        "show that",
        "justify",
        "chung minh",
        "cyclic",
        "noi tiep",
        "angle bisector",
        "bisects",
        "phan giac",
    }
    return any(marker in normalized for marker in proof_markers)


def answer_from_raw(raw_answer: Any) -> SolveAnswer:
    if isinstance(raw_answer, str):
        raw_answer = {"text": raw_answer}
    if not isinstance(raw_answer, dict):
        raw_answer = {"text": str(raw_answer) if raw_answer is not None else ""}
    if "text" not in raw_answer or raw_answer.get("text") is None:
        raw_answer = {**raw_answer, "text": ""}
    return SolveAnswer.model_validate(raw_answer)


def steps_from_raw(raw_steps: Any) -> list[SolveStep]:
    if raw_steps is None:
        return []
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        return []
    return [SolveStep.model_validate(step) for step in raw_steps]


def parts_from_payload(
    raw_parts: Any,
    *,
    expected_subquestions: list[DetectedSubquestion],
    top_level_answer: SolveAnswer,
    top_level_steps: list[SolveStep],
) -> list[SolvePart]:
    if raw_parts is None:
        raw_parts = []
    if isinstance(raw_parts, dict):
        raw_parts = [raw_parts]
    if not isinstance(raw_parts, list):
        raw_parts = []

    if expected_subquestions and len(expected_subquestions) > 1:
        return [
            part_from_raw(
                raw_parts[index] if index < len(raw_parts) else {},
                index=index,
                expected_subquestion=subquestion,
                top_level_answer=top_level_answer,
                top_level_steps=top_level_steps,
                require_steps=True,
            )
            for index, subquestion in enumerate(expected_subquestions)
        ]

    parts: list[SolvePart] = []
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, dict):
            continue
        part = part_from_raw(
            raw_part,
            index=index,
            expected_subquestion=None,
            top_level_answer=top_level_answer,
            top_level_steps=top_level_steps,
            require_steps=False,
        )
        if part.steps:
            parts.append(part)
    return parts


def part_from_raw(
    raw_part: Any,
    *,
    index: int,
    expected_subquestion: DetectedSubquestion | None,
    top_level_answer: SolveAnswer,
    top_level_steps: list[SolveStep],
    require_steps: bool,
) -> SolvePart:
    raw_part = raw_part if isinstance(raw_part, dict) else {}
    label = alpha_label(index)
    question = expected_subquestion.question if expected_subquestion else ""
    if not question:
        question = str(raw_part.get("question") or raw_part.get("prompt") or "")

    answer = answer_from_raw(raw_part.get("answer", {}))
    if not answer.text:
        answer = SolveAnswer(
            text=top_level_answer.text
            or f"See the step-by-step guide for question {label}.",
            latex=top_level_answer.latex,
        )

    steps = steps_from_raw(raw_part.get("steps", []))
    if not steps:
        steps = fallback_steps_for_part(top_level_steps, label, index)
    if require_steps and not steps:
        raise ValueError(f"Model returned no steps for question {label}")

    return SolvePart(
        label=label,
        question=question,
        answer=answer,
        steps=reindex_steps(steps),
    )


def fallback_steps_for_part(
    steps: list[SolveStep], label: str, index: int
) -> list[SolveStep]:
    if not steps:
        return []

    label_pattern = re.compile(
        rf"(?:\b(?:part|question)\s*\(?{re.escape(label)}\)?\b|\({re.escape(label)}\)|\b{re.escape(label)}[\).:])",
        flags=re.IGNORECASE,
    )
    matching_steps = [
        step
        for step in steps
        if label_pattern.search(f"{step.title} {step.explanation}")
    ]
    if matching_steps:
        return reindex_steps(matching_steps)
    if index < len(steps):
        return reindex_steps([steps[index]])
    return []


def reindex_steps(steps: list[SolveStep]) -> list[SolveStep]:
    return [
        step.model_copy(update={"index": index + 1})
        for index, step in enumerate(steps)
    ]


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def fallback_draft_for_subquestions(
    *,
    fallback_solver: FallbackSolver,
    text: str,
    problem_type: ProblemType,
    difficulty: Difficulty,
    subquestions: list[DetectedSubquestion],
    warning: str | None = None,
    suppress_config_warnings: bool = False,
    warning_filter: Callable[[list[str]], list[str]] | None = None,
) -> StructuredSolveDraft:
    answer, steps, confidence, warnings = fallback_solver.solve(
        text, problem_type, difficulty
    )
    warnings = list(warnings)
    if suppress_config_warnings and warning_filter is not None:
        warnings = warning_filter(warnings)
    if warning:
        warnings.append(warning)

    if len(subquestions) <= 1:
        return StructuredSolveDraft(answer, steps, confidence, warnings, parts=[])

    parts: list[SolvePart] = []
    confidences = [confidence]
    for subquestion in subquestions:
        subquestion_text = subquestion_with_context(text, subquestion.question)
        part_answer, part_steps, part_confidence, part_warnings = fallback_solver.solve(
            subquestion_text, problem_type, difficulty
        )
        confidences.append(part_confidence)
        if suppress_config_warnings and warning_filter is not None:
            part_warnings = warning_filter(part_warnings)
        warnings.extend(
            f"Question {subquestion.label}: {part_warning}"
            for part_warning in part_warnings
        )
        if not part_steps:
            part_steps = [
                SolveStep(
                    index=1,
                    title=f"Review question {subquestion.label}",
                    explanation=(
                        "This subquestion was detected, but the local fallback solver "
                        "could not produce detailed steps for it."
                    ),
                    why_it_happens=(
                        "Some proof and construction prompts need more structure before "
                        "a reliable step-by-step solution can be produced."
                    ),
                    hints=[
                        "Try again with a clearer problem statement or type this subquestion separately."
                    ],
                )
            ]
        parts.append(
            SolvePart(
                label=subquestion.label,
                question=subquestion.question,
                answer=part_answer,
                steps=reindex_steps(part_steps),
            )
        )

    labels = ", ".join(part.label for part in parts)
    return StructuredSolveDraft(
        answer=SolveAnswer(
            text=(
                f"This problem has {len(parts)} subquestions: {labels}. "
                "Each subquestion has its own answer and step-by-step guide."
            ),
            latex=None,
        ),
        steps=[
            SolveStep(
                index=1,
                title="Use the per-question guides",
                explanation=(
                    f"The prompt was split into {labels}. Select a question "
                    "to view its dedicated steps."
                ),
                why_it_happens=(
                    "Multi-part prompts need separate step sequences so each "
                    "requested proof or computation is traceable."
                ),
            )
        ],
        confidence=min(confidences),
        warnings=warnings,
        parts=parts,
    )


__all__ = [
    "StructuredSolveDraft",
    "answer_from_raw",
    "coerce_string_list",
    "draft_from_payload",
    "fallback_draft_for_subquestions",
    "fallback_steps_for_part",
    "is_proof_like_question",
    "part_from_raw",
    "parts_from_payload",
    "reindex_steps",
    "steps_from_raw",
    "validate_subquestion_step_depth",
]
