"""Bounded repair and logging helpers for structured solution content."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from app.integrations.errors import exception_diagnostics
from app.integrations.protocols import StructuredCompletionClient
from app.services.model_router import StructuredModelEndpoint

from .content_quality import StructuredContentIssue, structured_content_issues
from .prompts import (
    SOLVE_RESPONSE_JSON_SCHEMA,
    SOLVE_STEPS_REPAIR_JSON_SCHEMA,
    STRUCTURED_CONTENT_REPAIR_SYSTEM_PROMPT,
)
from .strict_models import (
    MISSING_STEPS_RECOVERY_WARNING,
    StrictStructuredSolvePayload,
    validate_structured_payload,
)


logger = logging.getLogger(__name__)


async def repair_missing_structured_steps(
    payload: dict[str, Any],
    *,
    problem_text: str,
    completion_client: StructuredCompletionClient,
    candidate: StructuredModelEndpoint,
    timeout_seconds: float,
    max_tokens: int = 2_000,
    request_id: str | None,
) -> dict[str, Any]:
    warnings = payload.get("warnings", [])
    if MISSING_STEPS_RECOVERY_WARNING not in warnings:
        return payload

    logger.warning(
        "Structured solve missing-step repair started request_id=%s provider=%s "
        "model=%s operation=structured_math_steps_repair",
        request_id,
        candidate.provider,
        candidate.model,
    )
    try:
        repair_payload = await completion_client.complete_json(
            model=candidate.model,
            system_prompt=(
                "Return exactly one JSON object containing only a steps array. "
                "Write 4 to 7 concise, logically complete proof steps for the problem. "
                "Preserve the supplied conclusion and include the concrete geometry "
                "relations that justify each step."
            ),
            user_prompt=(
                f"Problem:\n{problem_text}\n\n"
                "Conclusion already produced:\n"
                + json.dumps(payload.get("answer"), ensure_ascii=False)
            ),
            temperature=0.1,
            json_schema=SOLVE_STEPS_REPAIR_JSON_SCHEMA,
            schema_name="structured_math_steps_repair",
            max_tokens=max_tokens,
            require_parameters=False,
            allow_schema_downgrade=False,
            repair_invalid_json=False,
            timeout_seconds=timeout_seconds,
            operation="structured_math_steps_repair",
            trace_id=request_id,
        )
        repaired_steps = repair_payload.get("steps")
        if not isinstance(repaired_steps, list) or not repaired_steps:
            raise ValueError("The missing-step repair returned no proof steps.")
        merged = deepcopy(payload)
        merged["steps"] = repaired_steps
        merged["warnings"] = [
            warning
            for warning in warnings
            if warning != MISSING_STEPS_RECOVERY_WARNING
        ]
        validated = validate_structured_payload(
            merged,
            request_id=request_id,
            provider=candidate.provider,
            model=candidate.model,
        )
        logger.info(
            "Structured solve missing-step repair completed request_id=%s provider=%s "
            "model=%s operation=structured_math_steps_repair steps=%s",
            request_id,
            candidate.provider,
            candidate.model,
            len(validated["steps"]),
        )
        return validated
    except Exception as exc:
        diagnostics = exception_diagnostics(exc)
        logger.warning(
            "Structured solve missing-step repair failed request_id=%s provider=%s "
            "model=%s operation=structured_math_steps_repair error_type=%s "
            "error_message=%s status_code=%s response_body=%s",
            request_id,
            candidate.provider,
            candidate.model,
            diagnostics.error_type,
            diagnostics.error_message,
            diagnostics.status_code,
            diagnostics.response_body,
        )
        return payload


def log_structured_step_quality(
    payload: dict[str, Any],
    *,
    request_id: str | None,
    provider: str,
    model: str,
    stage: str,
) -> list[StructuredContentIssue]:
    issues = structured_content_issues(payload)
    issue_codes_by_path: dict[str, list[str]] = {}
    for issue in issues:
        issue_codes_by_path.setdefault(issue.path, []).append(issue.code)

    step_locations: list[tuple[str, dict[str, Any]]] = []
    for step_index, step in enumerate(payload.get("steps", [])):
        if isinstance(step, dict):
            step_locations.append((f"steps[{step_index}]", step))
    for part_index, part in enumerate(payload.get("parts", [])):
        if not isinstance(part, dict):
            continue
        for step_index, step in enumerate(part.get("steps", [])):
            if isinstance(step, dict):
                step_locations.append(
                    (f"parts[{part_index}].steps[{step_index}]", step)
                )

    for path, step in step_locations:
        latex = step.get("latex")
        latex_empty = not (
            isinstance(latex, list)
            and any(isinstance(item, str) and item.strip() for item in latex)
        )
        logger.info(
            "Structured solve step content quality request_id=%s provider=%s "
            "model=%s stage=%s step_path=%s latex_empty=%s issue_codes=%s",
            request_id,
            provider,
            model,
            stage,
            path,
            latex_empty,
            json.dumps(sorted(issue_codes_by_path.get(path, []))),
        )
    return issues


async def repair_structured_content(
    payload: dict[str, Any],
    issues: list[StructuredContentIssue],
    *,
    completion_client: StructuredCompletionClient,
    candidate: StructuredModelEndpoint,
    timeout_seconds: float,
    max_tokens: int = 2_500,
    request_id: str | None,
) -> dict[str, Any]:
    scratch_issues = [issue for issue in issues if issue.code == "scratch_work_style"]
    if not scratch_issues:
        return payload

    issue_summary = [
        {"path": issue.path, "fields": list(issue.fields)}
        for issue in scratch_issues
    ]
    logger.warning(
        "Structured solve content repair started request_id=%s provider=%s "
        "model=%s operation=structured_math_solution_repair flagged_steps=%s",
        request_id,
        candidate.provider,
        candidate.model,
        len(scratch_issues),
    )
    try:
        repair_payload = await completion_client.complete_json(
            model=candidate.model,
            system_prompt=STRUCTURED_CONTENT_REPAIR_SYSTEM_PROMPT,
            user_prompt=(
                "Flagged fields:\n"
                + json.dumps(issue_summary, ensure_ascii=False, separators=(",", ":"))
                + "\n\nOriginal solution JSON:\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
            temperature=0.1,
            json_schema=SOLVE_RESPONSE_JSON_SCHEMA,
            schema_name="structured_math_solution_repair",
            max_tokens=max_tokens,
            require_parameters=False,
            allow_schema_downgrade=False,
            repair_invalid_json=False,
            timeout_seconds=timeout_seconds,
            operation="structured_math_solution_repair",
            trace_id=request_id,
        )
        repaired = StrictStructuredSolvePayload.model_validate(
            repair_payload
        ).model_dump(mode="python")
        merged = merge_structured_content_repair(payload, repaired, scratch_issues)
        remaining = sum(
            issue.code == "scratch_work_style"
            for issue in structured_content_issues(merged)
        )
        logger.info(
            "Structured solve content repair completed request_id=%s provider=%s "
            "model=%s operation=structured_math_solution_repair remaining_scratch_issues=%s",
            request_id,
            candidate.provider,
            candidate.model,
            remaining,
        )
        return merged
    except Exception as exc:
        diagnostics = exception_diagnostics(exc)
        logger.warning(
            "Structured solve content repair failed request_id=%s provider=%s "
            "model=%s operation=structured_math_solution_repair error_type=%s "
            "error_message=%s status_code=%s response_body=%s",
            request_id,
            candidate.provider,
            candidate.model,
            diagnostics.error_type,
            diagnostics.error_message,
            diagnostics.status_code,
            diagnostics.response_body,
        )
        return payload


def merge_structured_content_repair(
    original: dict[str, Any],
    repaired: dict[str, Any],
    issues: list[StructuredContentIssue],
) -> dict[str, Any]:
    merged = deepcopy(original)
    for issue in issues:
        original_step = structured_step_at(merged, issue)
        repaired_step = structured_step_at(repaired, issue)
        if original_step is None or repaired_step is None:
            raise ValueError(f"Structured content repair omitted {issue.path}")
        for field_name in issue.fields:
            original_step[field_name] = repaired_step[field_name]
        repaired_latex = repaired_step.get("latex")
        if isinstance(repaired_latex, list) and any(
            isinstance(item, str) and item.strip() for item in repaired_latex
        ):
            original_step["latex"] = repaired_latex
    return StrictStructuredSolvePayload.model_validate(merged).model_dump(mode="python")


def structured_step_at(
    payload: dict[str, Any], issue: StructuredContentIssue
) -> dict[str, Any] | None:
    steps: Any
    if issue.part_index is None:
        steps = payload.get("steps", [])
    else:
        parts = payload.get("parts", [])
        if not isinstance(parts, list) or issue.part_index >= len(parts):
            return None
        part = parts[issue.part_index]
        if not isinstance(part, dict):
            return None
        steps = part.get("steps", [])
    if not isinstance(steps, list) or issue.step_index >= len(steps):
        return None
    step = steps[issue.step_index]
    return step if isinstance(step, dict) else None


def append_content_quality_warnings(
    payload: dict[str, Any], issues: list[StructuredContentIssue]
) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        payload["warnings"] = warnings
    for issue in issues:
        if issue.code == "missing_latex_for_math":
            warning = (
                f"{issue.path} contains math notation but has no LaTeX formula "
                "for math rendering."
            )
        elif issue.code == "scratch_work_style":
            warning = (
                f"{issue.path} may contain unedited scratch work after one bounded "
                "cleanup attempt."
            )
        else:
            continue
        if warning not in warnings:
            warnings.append(warning)


__all__ = [
    "append_content_quality_warnings",
    "log_structured_step_quality",
    "merge_structured_content_repair",
    "repair_missing_structured_steps",
    "repair_structured_content",
    "structured_step_at",
]
