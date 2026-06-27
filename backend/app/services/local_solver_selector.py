from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.integrations.ollama_client import OllamaClient
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolveStep
from app.services.fallback_solver import FallbackSolver

LOCAL_SOLVER_MIN_CONFIDENCE = 0.70
UNSUPPORTED_LOCAL_SOLVER_MARKER = "outside the local deterministic solver"

LOCAL_SOLVER_DETECTION_PROMPT = """
You are a strict routing classifier for IntoMath's deterministic local math solver.
Return exactly one JSON object and nothing else.

The local solver can ONLY solve these prompt shapes:
1. Arithmetic expressions written with digits and operators, e.g. "12*(3+4)-5".
2. One-variable linear equations in x with numeric coefficients, parentheses, multiplication, or division, e.g. "2(x + 3) = 14" or "x/2 + 3 = 7".
3. Quadratic graph analysis for y=... or f(x)=..., e.g. "y = x^2 - 4x + 3".
4. Simple constructions: perpendicular bisector of AB, circle with center O and radius r, or construct/draw triangle ABC.

If the problem can be normalized into one of those exact shapes, set use_local_solver=true and put that canonical text in normalized_prompt.
If it needs proof, calculus, trigonometry, statistics, word-problem reasoning, systems, inequalities, factoring beyond graph analysis, geometry theorem reasoning, or you are unsure, set use_local_solver=false.
Do not solve the problem. Do not include the answer.

JSON schema:
{"use_local_solver": true, "normalized_prompt": "canonical supported prompt", "reason": "short reason"}
""".strip()


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


class LocalSolverSelector:
    """Selects deterministic solving when it can be trusted.

    The deterministic fallback solver is always the final gate. Ollama may suggest
    that a prompt can be normalized into a supported shape, but we only use that
    route if the deterministic solver then returns a high-confidence result.
    """

    def __init__(
        self,
        fallback_solver: FallbackSolver | None = None,
        ollama_client: OllamaClient | None = None,
        settings: Any | None = None,
    ) -> None:
        self.fallback_solver = fallback_solver or FallbackSolver()
        self.ollama_client = ollama_client or OllamaClient()
        self.settings = settings or get_settings()

    async def solve_if_supported(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
    ) -> LocalSolveResult | None:
        if not self.settings.local_solver_first:
            return None

        direct_result = self._solve_with_deterministic_solver(
            text=text,
            problem_type=problem_type,
            difficulty=difficulty,
            reason="matched a deterministic local solver pattern",
        )
        if direct_result is not None:
            return direct_result

        if not self._should_use_ollama_detector(text):
            return None

        detection = await self._detect_with_ollama(text)
        if detection is None or not detection.use_local_solver:
            return None

        normalized_prompt = (detection.normalized_prompt or "").strip()
        if not normalized_prompt:
            return None

        reason = detection.reason or "Ollama detector normalized the prompt"
        return self._solve_with_deterministic_solver(
            text=normalized_prompt,
            problem_type=problem_type,
            difficulty=difficulty,
            reason=reason,
            detector_model=getattr(self.ollama_client, "model", None),
        )

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

    def _should_use_ollama_detector(self, text: str) -> bool:
        if not self.settings.local_solver_ollama_detection_enabled:
            return False
        if not getattr(self.ollama_client, "enabled", False):
            return False

        lowered = text.lower().strip()
        if len(lowered) < 3:
            return False
        if any(
            proof_marker in lowered
            for proof_marker in {
                "prove",
                "proof",
                "show that",
                "justify",
                "chứng minh",
                "chung minh",
            }
        ):
            return False
        if any(
            advanced_marker in lowered
            for advanced_marker in {
                "derivative",
                "differentiate",
                "integral",
                "integrate",
                "limit",
                "sin",
                "cos",
                "tan",
                "mean",
                "median",
                "variance",
                "probability",
            }
        ):
            return False

        operator_markers = {
            "=",
            "+",
            "-",
            "*",
            "/",
            "^",
            "²",
            "plus",
            "minus",
            "times",
            "divided",
            "equals",
        }
        solver_intent_markers = {
            "solve",
            "evaluate",
            "simplify",
            "calculate",
            "what is",
        }
        graph_markers = {"graph", "function", "parabola", "vertex"}
        construction_markers = {
            "circle",
            "triangle",
            "perpendicular",
            "bisector",
            "midpoint",
            "construct",
            "draw",
        }
        has_operator = any(marker in lowered for marker in operator_markers)
        has_solver_intent = any(marker in lowered for marker in solver_intent_markers)
        has_digit_or_variable = bool(re.search(r"\d|\bx\b", lowered))

        return (
            (has_operator and (has_solver_intent or has_digit_or_variable))
            or any(marker in lowered for marker in graph_markers)
            or any(marker in lowered for marker in construction_markers)
        )

    async def _detect_with_ollama(self, text: str) -> LocalSolveDetection | None:
        prompt = f"{LOCAL_SOLVER_DETECTION_PROMPT}\n\nProblem:\n{text}"
        try:
            payload = await self.ollama_client.generate_json(prompt=prompt)
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
        if self.fallback_solver._try_quadratic_graph(text) is not None:
            return ProblemType.functions
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
