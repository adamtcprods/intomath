"""Tests for LlamaTriviaSolver and its integration with LocalSolverSelector."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.schemas.common import Difficulty, ProblemType
from app.services.local_solver_selector import LocalSolverSelector
from app.services.llama_trivia_solver import (
    LLAMA_TRIVIA_MIN_CONFIDENCE,
    LlamaTriviaSolver,
)


# ─── Shared fakes ─────────────────────────────────────────────────────────────


class LocalSolverSettings:
    local_solver_first = True
    local_solver_llama_detection_enabled = False  # skip normalization path
    local_solver_llama_trivia_enabled = True


class LocalSolverSettingsTriviaDisabled:
    local_solver_first = True
    local_solver_llama_detection_enabled = False
    local_solver_llama_trivia_enabled = False


GOOD_TRIVIA_PAYLOAD: dict[str, Any] = {
    "answer_text": "Yes, 97 is a prime number because it has no divisors other than 1 and itself.",
    "answer_latex": None,
    "steps": [
        {
            "title": "Check divisibility",
            "explanation": "Test all primes up to sqrt(97) ≈ 9.8: 2, 3, 5, 7. None divide 97.",
            "why_it_happens": "A prime has exactly two factors: 1 and itself.",
            "hints": ["sqrt(97) is about 9.8, so only check primes up to 9."],
            "exam_tip": "You only need to check primes up to the square root of the number.",
            "latex": [],
        }
    ],
    "confidence": 0.88,
}

LOW_CONFIDENCE_PAYLOAD: dict[str, Any] = {
    "answer_text": "Maybe?",
    "answer_latex": None,
    "steps": [{"title": "Guess", "explanation": "Not sure."}],
    "confidence": 0.40,
}

EMPTY_STEPS_PAYLOAD: dict[str, Any] = {
    "answer_text": "Yes, 97 is prime.",
    "answer_latex": None,
    "steps": [],
    "confidence": 0.85,
}

NO_ANSWER_TEXT_PAYLOAD: dict[str, Any] = {
    "answer_text": "",
    "answer_latex": None,
    "steps": [{"title": "Step", "explanation": "Explanation."}],
    "confidence": 0.90,
}


class FakeLlamaClient:
    enabled = True
    model = "local:test-trivia"

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        raise_error: bool = False,
    ) -> None:
        self.payload = payload or {}
        self.raise_error = raise_error
        self.calls = 0
        self.last_prompt = ""

    async def generate_json(self, *, prompt: str) -> dict[str, Any]:
        self.calls += 1
        self.last_prompt = prompt
        if self.raise_error:
            raise RuntimeError("llama.cpp timed out")
        return self.payload


class DisabledLlamaClient(FakeLlamaClient):
    enabled = False


# ─── LlamaTriviaSolver unit tests ────────────────────────────────────────────


def test_trivia_solver_returns_result_for_valid_payload() -> None:
    client = FakeLlamaClient(GOOD_TRIVIA_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is not None
    assert "97 is a prime" in result.answer.text
    assert result.confidence == 0.88
    assert len(result.steps) == 1
    assert result.steps[0].title == "Check divisibility"
    assert client.calls == 1
    assert "Is 97 prime?" in client.last_prompt


def test_trivia_solver_rejects_low_confidence() -> None:
    client = FakeLlamaClient(LOW_CONFIDENCE_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is None
    assert LOW_CONFIDENCE_PAYLOAD["confidence"] < LLAMA_TRIVIA_MIN_CONFIDENCE


def test_trivia_solver_rejects_empty_steps() -> None:
    client = FakeLlamaClient(EMPTY_STEPS_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is None


def test_trivia_solver_rejects_missing_answer_text() -> None:
    client = FakeLlamaClient(NO_ANSWER_TEXT_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is None


def test_trivia_solver_silently_ignores_llama_error() -> None:
    client = FakeLlamaClient(raise_error=True)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is None
    assert client.calls == 1


def test_trivia_solver_skips_when_llama_disabled() -> None:
    client = DisabledLlamaClient(GOOD_TRIVIA_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve("Is 97 prime?", ProblemType.arithmetic, Difficulty.easy)
    )

    assert result is None
    assert client.calls == 0


def test_trivia_solver_skips_proof_questions() -> None:
    client = FakeLlamaClient(GOOD_TRIVIA_PAYLOAD)
    solver = LlamaTriviaSolver(llama_client=client)  # type: ignore[arg-type]

    result = asyncio.run(
        solver.solve(
            "Prove that there are infinitely many primes.",
            ProblemType.general,
            Difficulty.hard,
        )
    )

    assert result is None
    assert client.calls == 0


# ─── Integration with LocalSolverSelector ────────────────────────────────────


def test_selector_falls_through_to_trivia_solver_for_concept_question() -> None:
    trivia_client = FakeLlamaClient(GOOD_TRIVIA_PAYLOAD)
    trivia_solver = LlamaTriviaSolver(llama_client=trivia_client)  # type: ignore[arg-type]

    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=DisabledLlamaClient(),  # type: ignore[arg-type]
        trivia_solver=trivia_solver,
    )

    result = asyncio.run(
        selector.solve_if_supported(
            "Is 97 prime?",
            ProblemType.arithmetic,
            Difficulty.easy,
        )
    )

    assert result is not None
    assert "97 is a prime" in result.answer.text
    assert result.reason == "answered a math trivia or concept question"
    assert result.detector_model == "local:test-trivia"
    assert trivia_client.calls == 1


def test_selector_skips_trivia_solver_when_feature_disabled() -> None:
    trivia_client = FakeLlamaClient(GOOD_TRIVIA_PAYLOAD)
    trivia_solver = LlamaTriviaSolver(llama_client=trivia_client)  # type: ignore[arg-type]

    selector = LocalSolverSelector(
        settings=LocalSolverSettingsTriviaDisabled(),  # type: ignore[arg-type]
        llama_client=DisabledLlamaClient(),  # type: ignore[arg-type]
        trivia_solver=trivia_solver,
    )

    result = asyncio.run(
        selector.solve_if_supported(
            "Is 97 prime?",
            ProblemType.arithmetic,
            Difficulty.easy,
        )
    )

    assert result is None
    assert trivia_client.calls == 0


def test_selector_deterministic_solver_takes_priority_over_trivia() -> None:
    """Equations must still be solved deterministically, not handed to trivia."""
    trivia_client = FakeLlamaClient(GOOD_TRIVIA_PAYLOAD)
    trivia_solver = LlamaTriviaSolver(llama_client=trivia_client)  # type: ignore[arg-type]

    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=DisabledLlamaClient(),  # type: ignore[arg-type]
        trivia_solver=trivia_solver,
    )

    result = asyncio.run(
        selector.solve_if_supported(
            "2x + 5 = 3x - 1",
            ProblemType.algebra,
            Difficulty.easy,
        )
    )

    assert result is not None
    assert result.answer.latex == "x = 6"
    # The trivia solver must not have been called.
    assert trivia_client.calls == 0
