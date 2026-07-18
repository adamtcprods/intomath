"""Content-quality checks for mathematical notation and scratch-work prose."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredContentIssue:
    code: str
    path: str
    step_index: int
    part_index: int | None
    fields: tuple[str, ...]
    message: str


MATH_NOTATION_PATTERN = re.compile(
    r"\^|_[A-Za-z0-9{]|\\(?:int|sqrt|frac|sum|prod|gcd|le|ge|neq)\b|"
    r"\b(?:gcd|lcm|sqrt)\s*\(|(?:<=|>=)|[≤≥≠≈∞∫√]|"
    r"(?:[A-Za-z0-9})\]])\s*(?:=|<|>|\+|\*|/)\s*(?:[A-Za-z0-9({\[])"
)
STRONG_SCRATCH_PATTERN = re.compile(
    r"\b(?:wait|hold on|scratch that|never mind|i was wrong|not sure|"
    r"let me (?:try|check|rethink|reconsider)|is that right|does that work|hmm+)\b",
    flags=re.IGNORECASE,
)
HEDGE_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|probably|i think|i guess|seems? like|might be|could be)\b",
    flags=re.IGNORECASE,
)


def contains_clear_math_notation(value: str | None) -> bool:
    return bool(value and MATH_NOTATION_PATTERN.search(value))


def looks_like_scratch_work(value: str | None) -> bool:
    if not value:
        return False
    if STRONG_SCRATCH_PATTERN.search(value):
        return True
    if len(HEDGE_PATTERN.findall(value)) >= 2:
        return True
    if "?" in value and re.search(
        r"\b(?:i|we|let me|should i|what if)\b", value, flags=re.IGNORECASE
    ):
        return True

    assignments: dict[str, set[str]] = {}
    for variable, conclusion in re.findall(
        r"\b([A-Za-z])\s*=\s*([^,;.?!]+)", value
    ):
        assignments.setdefault(variable.casefold(), set()).add(conclusion.strip())
    has_contradictory_transition = bool(
        re.search(r"\b(?:but|actually|however|no[,;:]?)\b", value, re.IGNORECASE)
    )
    return has_contradictory_transition and any(
        len(conclusions) > 1 for conclusions in assignments.values()
    )


def structured_content_issues(payload: dict[str, Any]) -> list[StructuredContentIssue]:
    locations: list[tuple[str, int | None, int, dict[str, Any]]] = []
    for step_index, step in enumerate(payload.get("steps", [])):
        if isinstance(step, dict):
            locations.append((f"steps[{step_index}]", None, step_index, step))
    for part_index, part in enumerate(payload.get("parts", [])):
        if not isinstance(part, dict):
            continue
        for step_index, step in enumerate(part.get("steps", [])):
            if isinstance(step, dict):
                locations.append(
                    (
                        f"parts[{part_index}].steps[{step_index}]",
                        part_index,
                        step_index,
                        step,
                    )
                )

    issues: list[StructuredContentIssue] = []
    for path, part_index, step_index, step in locations:
        explanation = str(step.get("explanation") or "")
        why_it_happens = str(step.get("why_it_happens") or "")
        latex = step.get("latex")
        latex_empty = not (
            isinstance(latex, list)
            and any(isinstance(item, str) and item.strip() for item in latex)
        )
        if latex_empty and (
            contains_clear_math_notation(explanation)
            or contains_clear_math_notation(why_it_happens)
        ):
            issues.append(
                StructuredContentIssue(
                    code="missing_latex_for_math",
                    path=path,
                    step_index=step_index,
                    part_index=part_index,
                    fields=("latex",),
                    message=(
                        f"{path} contains mathematical notation but its latex array is empty."
                    ),
                )
            )

        scratch_fields = tuple(
            field_name
            for field_name, value in (
                ("explanation", explanation),
                ("why_it_happens", why_it_happens),
            )
            if looks_like_scratch_work(value)
        )
        if scratch_fields:
            issues.append(
                StructuredContentIssue(
                    code="scratch_work_style",
                    path=path,
                    step_index=step_index,
                    part_index=part_index,
                    fields=scratch_fields,
                    message=(
                        f"{path} contains scratch-work markers and may not read as a finished explanation."
                    ),
                )
            )
    return issues


__all__ = [
    "HEDGE_PATTERN",
    "MATH_NOTATION_PATTERN",
    "STRONG_SCRATCH_PATTERN",
    "StructuredContentIssue",
    "contains_clear_math_notation",
    "looks_like_scratch_work",
    "structured_content_issues",
]
