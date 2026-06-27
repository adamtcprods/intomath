import asyncio
from typing import Any

from app.schemas.common import Difficulty, ProblemType
from app.services.local_solver_selector import LocalSolverSelector


class LocalSolverSettings:
    local_solver_first = True
    local_solver_llama_detection_enabled = True


class FakeLlamaClient:
    enabled = True
    model = "local:test-detector"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"use_local_solver": False}
        self.calls = 0
        self.last_prompt = ""

    async def generate_json(self, *, prompt: str) -> dict[str, Any]:
        self.calls += 1
        self.last_prompt = prompt
        return self.payload


def test_selector_uses_deterministic_solver_before_llama() -> None:
    llama = FakeLlamaClient()
    selector = LocalSolverSelector(settings=LocalSolverSettings(), llama_client=llama)  # type: ignore[arg-type]

    result = asyncio.run(
        selector.solve_if_supported(
            "2x + 5 = 3x - 1",
            ProblemType.algebra,
            Difficulty.easy,
        )
    )

    assert result is not None
    assert result.answer.latex == "x = 6"
    assert result.problem_type is ProblemType.algebra
    assert result.detector_model is None
    assert llama.calls == 0


def test_selector_accepts_llama_normalized_supported_prompt() -> None:
    llama = FakeLlamaClient(
        {
            "use_local_solver": True,
            "normalized_prompt": "x + 5 = 17",
            "reason": "normalized worded linear equation",
        }
    )
    selector = LocalSolverSelector(settings=LocalSolverSettings(), llama_client=llama)  # type: ignore[arg-type]

    result = asyncio.run(
        selector.solve_if_supported(
            "Please solve x plus five equals seventeen.",
            ProblemType.algebra,
            Difficulty.easy,
        )
    )

    assert result is not None
    assert result.answer.latex == "x = 12"
    assert result.normalized_text == "x + 5 = 17"
    assert result.detector_model == "local:test-detector"
    assert "strict routing classifier" in llama.last_prompt


def test_selector_accepts_llama_normalized_parenthesized_equation() -> None:
    llama = FakeLlamaClient(
        {
            "use_local_solver": True,
            "normalized_prompt": "2(x + 3) = 14",
            "reason": "normalized worded parenthesized equation",
        }
    )
    selector = LocalSolverSelector(settings=LocalSolverSettings(), llama_client=llama)  # type: ignore[arg-type]

    result = asyncio.run(
        selector.solve_if_supported(
            "Twice the quantity x plus three equals fourteen.",
            ProblemType.algebra,
            Difficulty.easy,
        )
    )

    assert result is not None
    assert result.answer.latex == "x = 4"
    assert result.normalized_text == "2(x + 3) = 14"
    assert result.detector_model == "local:test-detector"
    assert "parentheses" in llama.last_prompt


def test_selector_rejects_llama_hint_if_deterministic_solver_cannot_solve_it() -> None:
    llama = FakeLlamaClient(
        {
            "use_local_solver": True,
            "normalized_prompt": "prove the triangle statement",
            "reason": "bad hint",
        }
    )
    selector = LocalSolverSelector(settings=LocalSolverSettings(), llama_client=llama)  # type: ignore[arg-type]

    result = asyncio.run(
        selector.solve_if_supported(
            "Please draw a triangle-like object with extra constraints.",
            ProblemType.geometry,
            Difficulty.medium,
        )
    )

    assert result is None
    assert llama.calls == 1
