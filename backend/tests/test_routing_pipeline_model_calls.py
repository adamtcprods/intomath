from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.schemas.solve import SolveAnswer, SolveRequest, SolveStep
from app.services.cache import AsyncSingleFlight, TTLCache
from app.services.fallback_solver import FallbackSolver
from app.services.llama_trivia_solver import LlamaTriviaSolveResult
from app.services.local_solver_selector import LocalSolverSelector
from app.services.model_router import (
    EASY_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    RoutingDecision,
    SolveRoute,
)
from app.services.solver_pipeline.orchestration import solve_request
from app.services.solver_pipeline.response_builder import StructuredSolveDraft
from app.services.solver_pipeline.routing import (
    with_local_solver_routing,
    with_remote_solver_routing,
    with_structured_solver_routing,
)


@dataclass
class ModelCalls:
    routing: int = 0
    trivia_solving: int = 0
    remote_solving: int = 0
    visualization_generation: int = 0


class NoOCR:
    async def extract_problem_text(self, *_: object) -> None:
        return None


class CountingTriviaSolver:
    def __init__(self, calls: ModelCalls) -> None:
        self.calls = calls

    async def solve(
        self,
        *_: object,
    ) -> LlamaTriviaSolveResult:
        self.calls.trivia_solving += 1
        return LlamaTriviaSolveResult(
            answer=SolveAnswer(
                text="A prime number has exactly two positive divisors.",
                latex=None,
            ),
            steps=[
                SolveStep(
                    index=1,
                    title="Recall the definition",
                    explanation="Count the positive divisors.",
                )
            ],
            confidence=0.9,
            warnings=[],
            model="local:test-trivia",
        )


class UnifiedRouter:
    def __init__(self, calls: ModelCalls) -> None:
        self.calls = calls

    async def route_async(
        self,
        text: str,
        *,
        has_image: bool,
    ) -> RoutingDecision:
        self.calls.routing += 1
        vision_model = "test-ocr" if has_image else None
        if "prime number" in text.casefold():
            return RoutingDecision(
                problem_type=ProblemType.number_theory,
                difficulty=Difficulty.easy,
                parser_model=EASY_MODEL,
                solver_model="local:llama-trivia",
                vision_model=vision_model,
                visualization_environment=None,
                reason="one unified concept route",
                solve_route=SolveRoute.local_trivia,
                normalized_prompt=text,
            )
        if "triangle" in text.casefold():
            # An adversarial router proposal must still be rejected by the exact gate.
            return RoutingDecision(
                problem_type=ProblemType.geometry,
                difficulty=Difficulty.medium,
                parser_model=EASY_MODEL,
                solver_model="local:deterministic-solver",
                vision_model=vision_model,
                visualization_environment=VisualizationEnvironment.geometry_2d,
                reason="incorrect deterministic proposal",
                solve_route=SolveRoute.deterministic,
                normalized_prompt="2 + 2",
            )
        if "twice the quantity" in text.casefold():
            return RoutingDecision(
                problem_type=ProblemType.algebra,
                difficulty=Difficulty.easy,
                parser_model=EASY_MODEL,
                solver_model="local:deterministic-solver",
                vision_model=vision_model,
                visualization_environment=None,
                reason="normalized a word equation",
                solve_route=SolveRoute.deterministic,
                normalized_prompt="2(x + 3) = 14",
            )
        return RoutingDecision(
            problem_type=ProblemType.calculus,
            difficulty=Difficulty.medium,
            parser_model=EASY_MODEL,
            solver_model=EASY_MODEL,
            vision_model=vision_model,
            visualization_environment=None,
            reason="one unified remote route",
            solve_route=SolveRoute.remote,
            normalized_prompt=text,
        )


class CountingVisualizationExtractor:
    def __init__(self, calls: ModelCalls) -> None:
        self.calls = calls

    async def extract(self, *_: object, **__: object) -> None:
        self.calls.visualization_generation += 1
        raise RuntimeError("stop after proving visualization was requested")


class GuardedFallbackSolver(FallbackSolver):
    def solve(
        self,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
    ) -> tuple[SolveAnswer, list[SolveStep], float, list[str]]:
        if problem_type is ProblemType.geometry:
            raise AssertionError("geometry must never execute deterministically")
        return super().solve(text, problem_type, difficulty)


class PipelineService:
    def __init__(self) -> None:
        self.calls = ModelCalls()
        self.settings = SimpleNamespace(
            solve_request_timeout_seconds=2.0,
            max_solve_text_length=20_000,
            max_image_base64_length=14_000_000,
            max_decoded_image_bytes=10_485_760,
            max_image_width=8_192,
            max_image_height=8_192,
            local_solver_first=True,
            local_solver_llama_trivia_enabled=True,
        )
        self.db = None
        self.ocr_service = NoOCR()
        self.router = UnifiedRouter(self.calls)
        self.geometry_extractor = CountingVisualizationExtractor(self.calls)
        self.fallback_solver = GuardedFallbackSolver()
        self.local_solver_selector = LocalSolverSelector(
            fallback_solver=self.fallback_solver,
            llama_client=SimpleNamespace(enabled=False),
            settings=self.settings,
            trivia_solver=CountingTriviaSolver(self.calls),  # type: ignore[arg-type]
        )
        self.nvidia_client = SimpleNamespace(enabled=True)

    def try_solve_exact(self, text: str):
        return self.local_solver_selector.try_solve_exact(text)

    def _build_cache_key(self, normalized_text: str, request: SolveRequest) -> str:
        return f"{normalized_text}:{request.options.include_visualization}"

    def _with_local_solver_routing(self, routing, local_result, *, original_text):
        return with_local_solver_routing(
            routing,
            local_result,
            original_text=original_text,
        )

    def _with_remote_solver_routing(self, routing, *, reason):
        return with_remote_solver_routing(routing, reason=reason)

    def _with_structured_solver_routing(self, routing, *, solver_model):
        return with_structured_solver_routing(routing, solver_model=solver_model)

    def _without_backend_config_warnings(self, warnings: list[str]) -> list[str]:
        return warnings

    async def _solve_structured(self, **_: object) -> StructuredSolveDraft:
        self.calls.remote_solving += 1
        return StructuredSolveDraft(
            answer=SolveAnswer(text="Remote solution.", latex=None),
            steps=[
                SolveStep(
                    index=1,
                    title="Solve remotely",
                    explanation="Use the selected general solver.",
                )
            ],
            confidence=0.8,
            warnings=[],
            solver_model="remote:test-solver",
        )


def run_pipeline(
    text: str,
    *,
    include_visualization: bool = False,
) -> tuple[object, ModelCalls]:
    service = PipelineService()
    request = SolveRequest.model_validate(
        {
            "input": {"text": text},
            "options": {"include_visualization": include_visualization},
        }
    )
    response = asyncio.run(
        solve_request(
            service,  # type: ignore[arg-type]
            request,
            TTLCache(ttl_seconds=60, max_size=10),
            AsyncSingleFlight(),
        )
    )
    return response, service.calls


def test_simple_arithmetic_uses_zero_model_calls() -> None:
    response, calls = run_pipeline("12 * (3 + 4) - 5")

    assert calls == ModelCalls()
    assert response.routing.solver_model == "local:deterministic-solver"


def test_supported_linear_equation_uses_zero_model_calls() -> None:
    response, calls = run_pipeline("2(x + 3) = 14")

    assert calls == ModelCalls()
    assert response.answer.latex == "x = 4"


def test_supported_quadratic_graph_uses_zero_solving_model_calls() -> None:
    response, calls = run_pipeline("Graph y = x^2 - 4x + 3")

    assert calls == ModelCalls()
    assert (
        response.routing.visualization_environment
        is VisualizationEnvironment.graphing
    )


def test_visualization_runs_only_when_selected_and_requested() -> None:
    _, graph_calls = run_pipeline(
        "Graph y = x^2 - 4x + 3",
        include_visualization=True,
    )
    _, arithmetic_calls = run_pipeline(
        "2 + 2",
        include_visualization=True,
    )

    assert graph_calls == ModelCalls(visualization_generation=1)
    assert arithmetic_calls == ModelCalls()


def test_unsupported_problem_has_one_route_before_remote_solving() -> None:
    _, calls = run_pipeline("Find the indefinite integral of sin(x^2).")

    assert calls == ModelCalls(routing=1, remote_solving=1)


def test_concept_problem_has_one_route_plus_one_trivia_solve() -> None:
    response, calls = run_pipeline("What is a prime number?")

    assert calls == ModelCalls(routing=1, trivia_solving=1)
    assert response.routing.solver_model == "local:llama-trivia"


def test_geometry_rejects_deterministic_proposal_and_falls_through_remote() -> None:
    response, calls = run_pipeline("Construct triangle ABC with AB = 3.")

    assert calls == ModelCalls(routing=1, remote_solving=1)
    assert response.problem_type == "geometry"
    assert response.routing.solver_model == "remote:test-solver"
    assert "deterministic parser rejected the original prompt" in response.routing.reason


def test_model_normalization_cannot_broaden_deterministic_grammar() -> None:
    response, calls = run_pipeline(
        "Twice the quantity x plus three equals fourteen."
    )

    assert calls == ModelCalls(routing=1, remote_solving=1)
    assert response.routing.solver_model == "remote:test-solver"
    assert response.answer.text == "Remote solution."
