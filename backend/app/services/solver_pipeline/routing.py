"""Routing-decision transformations used by the solver orchestrator."""

from __future__ import annotations

from app.services.local_solver_selector import (
    LOCAL_SOLVER_MIN_CONFIDENCE,
    UNSUPPORTED_LOCAL_SOLVER_MARKER,
)
from app.services.local_solver_types import LocalSolveResult
from app.services.model_router import (
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    LOCAL_LLAMA_TRIVIA_MODEL,
    RoutingDecision,
)

from .response_builder import StructuredSolveDraft


def with_local_solver_routing(
    routing: RoutingDecision,
    local_result: LocalSolveResult,
    *,
    original_text: str,
) -> RoutingDecision:
    is_trivia = local_result.reason == "answered a math trivia or concept question"
    solver_model = (
        LOCAL_LLAMA_TRIVIA_MODEL if is_trivia else LOCAL_DETERMINISTIC_SOLVER_MODEL
    )
    reason_parts = [
        routing.reason,
        f"local llama.cpp trivia solver used because it {local_result.reason}"
        if is_trivia
        else f"deterministic local solver used because it {local_result.reason}",
    ]
    if local_result.detector_model:
        reason_parts.append(
            f"local llama.cpp model {local_result.detector_model} produced the answer"
            if is_trivia
            else f"local llama.cpp detector {local_result.detector_model} selected a supported canonical form"
        )
    if not is_trivia and local_result.normalized_text.strip() != original_text.strip():
        reason_parts.append("prompt was normalized before deterministic solving")
    return RoutingDecision(
        problem_type=local_result.problem_type or routing.problem_type,
        difficulty=routing.difficulty,
        parser_model=LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
        solver_model=solver_model,
        vision_model=routing.vision_model,
        reason="; ".join(reason_parts),
    )


def with_structured_solver_routing(
    routing: RoutingDecision, *, solver_model: str
) -> RoutingDecision:
    if solver_model == routing.solver_model:
        return routing
    return RoutingDecision(
        problem_type=routing.problem_type,
        difficulty=routing.difficulty,
        parser_model=routing.parser_model,
        solver_model=solver_model,
        vision_model=routing.vision_model,
        reason=f"{routing.reason}; result produced by {solver_model}",
    )


def with_local_subquestion_routing(
    routing: RoutingDecision, *, reason: str
) -> RoutingDecision:
    return RoutingDecision(
        problem_type=routing.problem_type,
        difficulty=routing.difficulty,
        parser_model=LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
        solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
        vision_model=routing.vision_model,
        reason=f"{routing.reason}; deterministic local solver used because it {reason}",
    )


def is_supported_local_draft(draft: StructuredSolveDraft) -> bool:
    if draft.confidence < LOCAL_SOLVER_MIN_CONFIDENCE:
        return False
    return not any(
        UNSUPPORTED_LOCAL_SOLVER_MARKER in warning for warning in draft.warnings
    )


def without_backend_config_warnings(warnings: list[str]) -> list[str]:
    backend_only_markers = (
        "Configure NVIDIA_API_KEY",
        "NVIDIA_API_KEY is not configured",
        "Model-backed solving failed",
        "model backend is not configured",
        "model client is disabled",
        "JSON repair attempt failed",
        "returned invalid JSON",
    )
    return [
        warning
        for warning in warnings
        if not any(marker in warning for marker in backend_only_markers)
    ]


__all__ = [
    "is_supported_local_draft",
    "with_local_solver_routing",
    "with_local_subquestion_routing",
    "with_structured_solver_routing",
    "without_backend_config_warnings",
]
