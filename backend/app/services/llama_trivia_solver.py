"""llama.cpp-backed zero-cost trivia and concept fallback solver.

Handles math trivia and conceptual questions that the deterministic solver
cannot answer but that are too simple to warrant a cloud LLM call.

This solver is always a best-effort layer. Any timeout, malformed response,
or low confidence causes it to return None, letting the caller fall through
to the cloud model pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.integrations.llama_client import LlamaClient
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolveStep
from app.services.local_solver_types import LocalSolveResult

LLAMA_TRIVIA_MIN_CONFIDENCE = 0.60

LLAMA_TRIVIA_PROMPT = """You are a concise math tutor. Answer the math question below.
Return exactly one JSON object and nothing else.

JSON schema (all fields required):
{
  "answer_text": "short plain-English answer in one or two sentences",
  "answer_latex": "LaTeX expression for the answer, or null if not applicable",
  "steps": [
    {
      "title": "step title",
      "explanation": "step explanation",
      "why_it_happens": "optional – why this step matters, or null",
      "hints": ["optional hint"],
      "exam_tip": "optional exam tip, or null",
      "latex": ["optional LaTeX snippet"]
    }
  ],
  "confidence": 0.85
}

Rules:
- confidence must be between 0.0 and 1.0. Use 0.0 if you are unsure.
- Keep each string under 200 characters.
- Return at least one step.
- Do NOT add any text before or after the JSON object.

Question:
""".strip()


# ─── Public dataclass returned to callers ────────────────────────────────────


@dataclass(frozen=True)
class LlamaTriviaSolveResult:
    """Thin wrapper so callers can distinguish the source of the answer."""

    answer: SolveAnswer
    steps: list[SolveStep]
    confidence: float
    warnings: list[str]
    model: str


# ─── Solver ──────────────────────────────────────────────────────────────────


class LlamaTriviaSolver:
    """Attempts to answer math trivia and concept questions using a local llama.cpp model."""

    def __init__(
        self,
        llama_client: LlamaClient | None = None,
    ) -> None:
        self.llama_client = llama_client or LlamaClient()

    @property
    def _model_name(self) -> str:
        return getattr(self.llama_client, "model", "local:llama-trivia")

    async def solve(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
    ) -> LlamaTriviaSolveResult | None:
        """Return a result or None if llama.cpp is unavailable, disabled, or not confident."""
        _ = (problem_type, difficulty)

        if not getattr(self.llama_client, "enabled", False):
            return None

        if self._should_skip(text, problem_type, difficulty):
            return None

        prompt = f"{LLAMA_TRIVIA_PROMPT}\n{text}"
        try:
            payload = await self.llama_client.generate_json(prompt=prompt)
        except Exception:
            return None

        return self._parse_payload(payload)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _should_skip(
        self, text: str, problem_type: ProblemType, difficulty: Difficulty
    ) -> bool:
        stripped = text.strip()
        if len(stripped) < 4:
            return True
        if difficulty is Difficulty.hard:
            return True
        if problem_type in {
            ProblemType.geometry,
            ProblemType.calculus,
            ProblemType.statistics,
            ProblemType.probability,
            ProblemType.trigonometry,
        }:
            return True
        return self._looks_like_computational_prompt(stripped)

    def _looks_like_computational_prompt(self, text: str) -> bool:
        return bool(
            re.search(r"(?:^|[^A-Za-z])(?:y|f\s*\(\s*x\s*\))\s*=", text, re.IGNORECASE)
            or re.search(r"[A-Za-z][A-Za-z0-9_]*.*(?:<=|>=|=|<|>)", text)
            or re.search(r"\b[a-zA-Z]\b\s*[-+*/^]", text)
            or re.search(r"[-+*/^]\s*\b[a-zA-Z]\b", text)
        )

    def _parse_payload(self, payload: dict[str, Any]) -> LlamaTriviaSolveResult | None:
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < LLAMA_TRIVIA_MIN_CONFIDENCE:
            return None

        answer_text = self._coerce_str(payload.get("answer_text", ""))
        if not answer_text:
            return None

        answer_latex = self._coerce_optional_str(payload.get("answer_latex"))
        answer = SolveAnswer(text=answer_text, latex=answer_latex)

        steps = self._parse_steps(payload.get("steps", []))
        if not steps:
            return None

        return LlamaTriviaSolveResult(
            answer=answer,
            steps=steps,
            confidence=min(1.0, max(0.0, confidence)),
            warnings=[],
            model=self._model_name,
        )

    def _parse_steps(self, raw_steps: Any) -> list[SolveStep]:
        if not isinstance(raw_steps, list):
            return []
        steps: list[SolveStep] = []
        for index, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                continue
            title = self._coerce_str(raw.get("title", ""))
            explanation = self._coerce_str(raw.get("explanation", ""))
            if not title or not explanation:
                continue
            steps.append(
                SolveStep(
                    index=index + 1,
                    title=title,
                    explanation=explanation,
                    why_it_happens=self._coerce_optional_str(raw.get("why_it_happens")),
                    hints=self._coerce_str_list(raw.get("hints", [])),
                    exam_tip=self._coerce_optional_str(raw.get("exam_tip")),
                    latex=self._coerce_str_list(raw.get("latex", [])),
                )
            )
        return steps

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if item is not None]
        return []


def trivia_result_to_local_solve_result(
    result: LlamaTriviaSolveResult,
    problem_type: ProblemType | None,
    original_text: str,
) -> LocalSolveResult:
    """Convert a LlamaTriviaSolveResult into the standard LocalSolveResult shape."""
    return LocalSolveResult(
        answer=result.answer,
        steps=result.steps,
        confidence=result.confidence,
        warnings=result.warnings,
        normalized_text=original_text,
        problem_type=problem_type,
        reason="answered a math trivia or concept question",
        detector_model=result.model,
    )
