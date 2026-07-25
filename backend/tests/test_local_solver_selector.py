import asyncio
from typing import Any

from app.schemas.common import Difficulty, ProblemType
from app.services.local_solver_selector import LocalSolverSelector
from app.services.model_router import SolveRoute


class LocalSolverSettings:
    local_solver_first = True
    local_solver_llama_trivia_enabled = False


class FakeLlamaClient:
    enabled = True
    available = True
    model = "local:test-router"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_json(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("exact execution must not call a model")


def test_exact_linear_equation_bypasses_all_models() -> None:
    llama = FakeLlamaClient()
    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=llama,  # type: ignore[arg-type]
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
    assert result.problem_type is ProblemType.algebra
    assert result.detector_model is None
    assert llama.calls == 0


def test_router_normalization_is_not_executable_input() -> None:
    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=FakeLlamaClient(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        selector.solve_selected_route(
            "Twice the quantity x plus three equals fourteen.",
            ProblemType.algebra,
            Difficulty.easy,
            SolveRoute.deterministic,
        )
    )

    assert result is None


def test_selected_deterministic_route_rechecks_original_exact_grammar() -> None:
    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=FakeLlamaClient(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        selector.solve_selected_route(
            "2(x + 3) = 14",
            ProblemType.algebra,
            Difficulty.easy,
            SolveRoute.deterministic,
        )
    )

    assert result is not None
    assert result.answer.latex == "x = 4"


def test_geometry_never_executes_through_deterministic_route() -> None:
    selector = LocalSolverSelector(
        settings=LocalSolverSettings(),  # type: ignore[arg-type]
        llama_client=FakeLlamaClient(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        selector.solve_selected_route(
            "Construct triangle ABC with AB = 3.",
            ProblemType.geometry,
            Difficulty.medium,
            SolveRoute.deterministic,
        )
    )

    assert result is None
