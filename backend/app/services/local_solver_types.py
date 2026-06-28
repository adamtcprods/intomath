"""Shared dataclasses for the local solver pipeline.

Kept in a separate module to avoid circular imports between
local_solver_selector and llama_trivia_solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.common import ProblemType
from app.schemas.solve import SolveAnswer, SolveStep


@dataclass(frozen=True)
class LocalSolveResult:
    answer: SolveAnswer
    steps: list[SolveStep]
    confidence: float
    warnings: list[str]
    normalized_text: str
    problem_type: ProblemType | None
    reason: str
    detector_model: str | None = None


@dataclass(frozen=True)
class LocalSolveDetection:
    use_local_solver: bool
    normalized_prompt: str | None
    reason: str | None
