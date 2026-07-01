from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.integrations.llama_client import LlamaClient
from app.schemas.common import Difficulty, ProblemType
from app.services.fallback_solver import FallbackSolver
from app.services.llama_trivia_solver import (
    LlamaTriviaSolver,
    trivia_result_to_local_solve_result,
)
from app.services.local_solver_types import LocalSolveDetection, LocalSolveResult

LOCAL_SOLVER_MIN_CONFIDENCE = 0.70
UNSUPPORTED_LOCAL_SOLVER_MARKER = "outside the local deterministic solver"

LOCAL_SOLVER_DETECTION_PROMPT = """
You are a strict routing classifier for IntoMath's deterministic local math solver.
Return exactly one JSON object and nothing else.

The local solver can ONLY solve these prompt shapes:
1. Arithmetic expressions written with digits and operators, e.g. "12*(3+4)-5".
2. One-variable linear equations or inequalities in x with numeric coefficients, parentheses, multiplication, or division, e.g. "2(x + 3) = 14" or "x/2 + 3 < 7".
3. Quadratic graph analysis for y=... or f(x)=..., e.g. "y = x^2 - 4x + 3".
4. Simple constructions: perpendicular bisector of AB, circle with center O and radius r, or construct/draw triangle ABC.

If the problem can be normalized into one of those exact shapes, set use_local_solver=true and put that canonical text in normalized_prompt.
If it needs proof, calculus, trigonometry, statistics, word-problem reasoning, systems, factoring beyond graph analysis, geometry theorem reasoning, or you are unsure, set use_local_solver=false.
Do not solve the problem. Do not include the answer.

JSON schema:
{"use_local_solver": true, "normalized_prompt": "canonical supported prompt", "reason": "short reason"}
""".strip()


class LocalSolverSelector:
    """Selects deterministic solving when it can be trusted.

    The deterministic fallback solver is always the final gate. llama.cpp may suggest
    that a prompt can be normalized into a supported shape, but we only use that
    route if the deterministic solver then returns a high-confidence result.

    If both deterministic solving and llama.cpp normalization fail, and the question
    looks like math trivia or a concept question, the llama.cpp trivia solver is
    attempted as a zero-cost local alternative before falling through to the cloud.
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

    async def solve_if_supported(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
    ) -> LocalSolveResult | None:
        if not self.settings.local_solver_first:
            return None

        # 1. Try the fast, fully deterministic solver first.
        direct_result = self._solve_with_deterministic_solver(
            text=text,
            problem_type=problem_type,
            difficulty=difficulty,
            reason="matched a deterministic local solver pattern",
        )
        if direct_result is not None:
            return direct_result

        # 2. Ask Llama-server to normalise the prompt into a canonical supported form,
        #    then re-run the deterministic solver on the normalised text.
        if self._should_use_llama_detector(text):
            detection = await self._detect_with_llama(text)
            if detection is not None and detection.use_local_solver:
                normalized_prompt = (detection.normalized_prompt or "").strip()
                if normalized_prompt:
                    reason = detection.reason or "Llama detector normalized the prompt"
                    normalised_result = self._solve_with_deterministic_solver(
                        text=normalized_prompt,
                        problem_type=problem_type,
                        difficulty=difficulty,
                        reason=reason,
                        detector_model=getattr(self.llama_client, "model", None),
                    )
                    if normalised_result is not None:
                        return normalised_result

        # 3. Fall back to the llama.cpp trivia solver for concept / trivia questions.
        if getattr(self.settings, "local_solver_llama_trivia_enabled", False):
            trivia_result = await self.trivia_solver.solve(
                text, problem_type, difficulty
            )
            if trivia_result is not None:
                return trivia_result_to_local_solve_result(
                    trivia_result,
                    problem_type=problem_type,
                    original_text=text,
                )

        return None

    def _solve_with_deterministic_solver(
        self,
        *,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
        reason: str,
        detector_model: str | None = None,
    ) -> LocalSolveResult | None:
        answer, steps, confidence, warnings = self.fallback_solver.solve(
            text, problem_type, difficulty
        )
        if not self._is_supported_local_result(confidence, warnings):
            return None

        return LocalSolveResult(
            answer=answer,
            steps=steps,
            confidence=confidence,
            warnings=warnings,
            normalized_text=text,
            problem_type=self._infer_supported_problem_type(text),
            reason=reason,
            detector_model=detector_model,
        )

    def _is_supported_local_result(
        self, confidence: float, warnings: list[str]
    ) -> bool:
        if confidence < LOCAL_SOLVER_MIN_CONFIDENCE:
            return False
        return not any(
            UNSUPPORTED_LOCAL_SOLVER_MARKER in warning for warning in warnings
        )

    def _should_use_llama_detector(self, text: str) -> bool:
        if not self.settings.local_solver_llama_detection_enabled:
            return False
        if not getattr(self.llama_client, "enabled", False):
            return False

        stripped = text.strip()
        return 3 <= len(stripped) <= 1_000

    async def _detect_with_llama(self, text: str) -> LocalSolveDetection | None:
        prompt = f"{LOCAL_SOLVER_DETECTION_PROMPT}\n\nProblem:\n{text}"
        try:
            payload = await self.llama_client.generate_json(prompt=prompt)
        except Exception:
            return None

        return LocalSolveDetection(
            use_local_solver=self._coerce_bool(payload.get("use_local_solver")),
            normalized_prompt=self._coerce_optional_string(
                payload.get("normalized_prompt")
            ),
            reason=self._coerce_optional_string(payload.get("reason")),
        )

    def _infer_supported_problem_type(self, text: str) -> ProblemType | None:
        if self.fallback_solver._try_linear_equation(text) is not None:
            return ProblemType.algebra
        if self.fallback_solver._try_linear_graph(text) is not None:
            return ProblemType.algebra
        if self.fallback_solver._try_quadratic_graph(text) is not None:
            return ProblemType.algebra
        if self.fallback_solver._try_geometry_construction(text) is not None:
            return ProblemType.geometry
        if self.fallback_solver._try_arithmetic(text) is not None:
            return ProblemType.arithmetic
        return None

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    def _coerce_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
