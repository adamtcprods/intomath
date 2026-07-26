"""Execution of routes selected by the unified local router."""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.model_policy import SolveRoute
from app.integrations.llama_client import LlamaClient
from app.schemas.common import Difficulty, ProblemType
from app.services.exact_solver import ExactSolveResult, try_solve_exact
from app.services.fallback_solver import FallbackSolver
from app.services.llama_trivia_solver import (
    LlamaTriviaSolver,
    trivia_result_to_local_solve_result,
)
from app.services.local_solver_types import LocalSolveResult

LOCAL_SOLVER_MIN_CONFIDENCE = 0.70
UNSUPPORTED_LOCAL_SOLVER_MARKER = "outside the local deterministic solver"


class LocalSolverSelector:
    """Runs an already selected deterministic or local-trivia route.

    Routing and deterministic selection deliberately do not make separate model
    calls. The deterministic route always rechecks the original prompt through
    ``try_solve_exact``; router-provided normalization is never executable input.
    """

    def __init__(
        self,
        fallback_solver: FallbackSolver | None = None,
        llama_client: LlamaClient | None = None,
        settings: Any | None = None,
        trivia_solver: LlamaTriviaSolver | None = None,
    ) -> None:
        self.fallback_solver = fallback_solver or FallbackSolver()
        self.llama_client = llama_client or LlamaClient()
        self.settings = settings or get_settings()
        self.trivia_solver = trivia_solver or LlamaTriviaSolver(self.llama_client)

    def try_solve_exact(self, text: str) -> ExactSolveResult | None:
        return try_solve_exact(text, self.fallback_solver)

    async def solve_selected_route(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
        solve_route: SolveRoute,
    ) -> LocalSolveResult | None:
        if solve_route is SolveRoute.deterministic:
            exact = self.try_solve_exact(text)
            return exact.local_result if exact is not None else None

        if solve_route is not SolveRoute.local_trivia:
            return None
        if not getattr(self.settings, "local_solver_llama_trivia_enabled", False):
            return None
        if problem_type is ProblemType.geometry:
            return None

        trivia_result = await self.trivia_solver.solve(
            text,
            problem_type,
            difficulty,
        )
        if trivia_result is None:
            return None
        return trivia_result_to_local_solve_result(
            trivia_result,
            problem_type=problem_type,
            original_text=text,
        )

    async def solve_if_supported(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
    ) -> LocalSolveResult | None:
        """Compatibility wrapper for callers outside the unified pipeline."""

        exact = self.try_solve_exact(text)
        if exact is not None:
            return exact.local_result
        return await self.solve_selected_route(
            text,
            problem_type,
            difficulty,
            SolveRoute.local_trivia,
        )


__all__ = [
    "LOCAL_SOLVER_MIN_CONFIDENCE",
    "LocalSolverSelector",
    "UNSUPPORTED_LOCAL_SOLVER_MARKER",
]
