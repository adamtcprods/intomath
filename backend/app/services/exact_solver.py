"""Strict, model-free entry point for IntoMath's exact solver grammars."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.fallback_solver import FallbackSolver
from app.services.local_solver_types import LocalSolveResult

_MAX_EXACT_PROMPT_LENGTH = 1_000
_ARITHMETIC_BODY = r"[0-9+\-*/^().\s]+"
_LINEAR_SIDE = r"[0-9xX+\-*/^().\s]+"
_QUADRATIC_BODY = r"[0-9xX+\-*/^().\s]+"


@dataclass(frozen=True)
class ExactSolveResult:
    """A solved prompt plus all routing facts that do not require a model."""

    local_result: LocalSolveResult
    problem_type: ProblemType
    difficulty: Difficulty
    normalized_prompt: str
    visualization_environment: VisualizationEnvironment | None
    visualization_search_terms: tuple[str, ...]
    reason: str
    solve_route: Literal["deterministic"] = "deterministic"


def try_solve_exact(
    text: str,
    fallback_solver: FallbackSolver | None = None,
) -> ExactSolveResult | None:
    """Solve only a complete, unambiguous member of an exact grammar.

    This strict boundary intentionally does not use the fallback solver's permissive
    text extractors as its acceptance rule. Once a whole prompt matches, the existing
    recursive AST parsers and response builders remain the execution authority.
    """

    solver = fallback_solver or FallbackSolver()
    prompt = _normalize_prompt(text)
    if not prompt or len(prompt) > _MAX_EXACT_PROMPT_LENGTH:
        return None

    arithmetic = _match_arithmetic(prompt)
    if arithmetic is not None and solver._try_arithmetic(arithmetic) is not None:
        return _solve_accepted(
            solver=solver,
            normalized_prompt=arithmetic,
            problem_type=ProblemType.arithmetic,
            difficulty=Difficulty.easy,
            visualization_environment=None,
            visualization_search_terms=(),
            reason="matched the exact numeric arithmetic grammar",
        )

    linear = _match_linear_relation(prompt)
    if linear is not None and solver._try_linear_equation(linear) is not None:
        return _solve_accepted(
            solver=solver,
            normalized_prompt=linear,
            problem_type=ProblemType.algebra,
            difficulty=Difficulty.easy,
            visualization_environment=None,
            visualization_search_terms=(),
            reason="matched the exact one-variable linear relation grammar",
        )

    quadratic = _match_quadratic_graph(prompt, solver)
    if quadratic is not None:
        return _solve_accepted(
            solver=solver,
            normalized_prompt=f"Graph y = {quadratic}",
            problem_type=ProblemType.algebra,
            difficulty=Difficulty.medium,
            visualization_environment=VisualizationEnvironment.graphing,
            visualization_search_terms=("Function", "Vertex", "Root"),
            reason="matched the exact quadratic graph-analysis grammar",
        )

    return None


def _solve_accepted(
    *,
    solver: FallbackSolver,
    normalized_prompt: str,
    problem_type: ProblemType,
    difficulty: Difficulty,
    visualization_environment: VisualizationEnvironment | None,
    visualization_search_terms: tuple[str, ...],
    reason: str,
) -> ExactSolveResult | None:
    answer, steps, confidence, warnings = solver.solve(
        normalized_prompt,
        problem_type,
        difficulty,
    )
    if confidence < 0.70 or any(
        "outside the local deterministic solver" in warning for warning in warnings
    ):
        return None
    local_result = LocalSolveResult(
        answer=answer,
        steps=steps,
        confidence=confidence,
        warnings=warnings,
        normalized_text=normalized_prompt,
        problem_type=problem_type,
        reason=reason,
    )
    return ExactSolveResult(
        local_result=local_result,
        problem_type=problem_type,
        difficulty=difficulty,
        normalized_prompt=normalized_prompt,
        visualization_environment=visualization_environment,
        visualization_search_terms=visualization_search_terms,
        reason=reason,
    )


def _normalize_prompt(text: str) -> str:
    normalized = (
        text.strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("²", "^2")
    )
    return re.sub(r"[ \t\r\f\v]+", " ", normalized)


def _strip_exact_prefix(
    prompt: str,
    prefixes: tuple[str, ...],
) -> str:
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    match = re.fullmatch(
        rf"(?:(?:{prefix_pattern})\s*:?\s*)?(.+?)\s*\??",
        prompt,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _match_arithmetic(prompt: str) -> str | None:
    if "\n" in prompt:
        return None
    expression = _strip_exact_prefix(
        prompt,
        ("calculate", "compute", "evaluate", "simplify", "what is"),
    )
    if not re.fullmatch(_ARITHMETIC_BODY, expression):
        return None
    if not re.search(r"\d", expression) or not re.search(r"[-+*/^]", expression):
        return None
    return expression


def _match_linear_relation(prompt: str) -> str | None:
    if "\n" in prompt:
        return None
    relation = _strip_exact_prefix(
        prompt,
        ("solve", "solve for x", "find x"),
    )
    if not re.fullmatch(
        rf"{_LINEAR_SIDE}(?:<=|>=|=|<|>){_LINEAR_SIDE}",
        relation,
    ):
        return None
    if relation.lower().count("x") == 0:
        return None
    if len(re.findall(r"<=|>=|=|<|>", relation)) != 1:
        return None
    return relation.strip()


def _match_quadratic_graph(
    prompt: str,
    solver: FallbackSolver,
) -> str | None:
    if "\n" in prompt:
        return None
    match = re.fullmatch(
        rf"(?:(?:graph|plot|analyze|analyse)\s+)?"
        rf"(?:y|f\s*\(\s*x\s*\))\s*=\s*({_QUADRATIC_BODY})\s*\??",
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    expression = match.group(1).strip()
    coefficients = solver._parse_quadratic_coefficients(expression)
    if coefficients is None or math.isclose(coefficients[0], 0.0):
        return None
    return expression


__all__ = ["ExactSolveResult", "try_solve_exact"]
