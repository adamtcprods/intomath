import asyncio
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from types import SimpleNamespace

from app.services.model_router import (
    EASY_MODEL,
    HARD_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    ModelRouter,
    remote_model_timeout_seconds,
    structured_model_endpoints,
)

VIETNAMESE_GEOMETRY_PROOF = r"""
Cho tam giác (ABC) ((AB < AC)) nội tiếp đường tròn ((O;R)) có đường kính (BC).
Trên cung nhỏ (AC) lấy điểm (D). Đường thẳng (BD) cắt (AC) tại (E).
Từ (E) kẻ (EF \perp BC) tại (F).

Chứng minh tứ giác (BAEF) nội tiếp một đường tròn.
""".strip()


class RoutingLlamaClient:
    enabled = True
    available = True
    model = "test-general-local-model"

    def __init__(self) -> None:
        self.calls = 0
        self.request: dict[str, object] = {}
        self.settings = SimpleNamespace(
            local_router_llama_model="test-tiny-router",
            local_router_llama_max_tokens=256,
            local_router_llama_timeout_seconds=3.0,
        )

    def is_model_available(self, model: str) -> bool:
        return model == "test-tiny-router"

    async def generate_json(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        self.request = kwargs
        return {
            "problem_type": "geometry",
            "difficulty": "medium",
            "solve_route": "remote",
            "normalized_prompt": "Visualize a tetrahedron!",
            "visualization_environment": "graphics_3d",
            "visualization_search_terms": ["Tetrahedron", "solid", "3D"],
            "reason": "a spatial geometry request needs remote solving",
        }


class NoVisualizationLlamaClient(RoutingLlamaClient):
    async def generate_json(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        self.request = kwargs
        return {
            "problem_type": "arithmetic",
            "difficulty": "easy",
            "solve_route": "deterministic",
            "normalized_prompt": "2 + 2",
            "visualization_environment": "none",
            "visualization_search_terms": [],
            "reason": "an exact arithmetic expression",
        }


class GeometryWithoutVisualizationLlamaClient(RoutingLlamaClient):
    async def generate_json(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        self.request = kwargs
        return {
            "problem_type": "geometry",
            "difficulty": "hard",
            "solve_route": "remote",
            "normalized_prompt": VIETNAMESE_GEOMETRY_PROOF,
            "visualization_environment": "none",
            "visualization_search_terms": [],
            "reason": "a geometry proof requires remote solving",
        }


def test_sync_router_leaves_subject_unclassified_without_ai() -> None:
    routing = ModelRouter().route(VIETNAMESE_GEOMETRY_PROOF, has_image=False)

    assert routing.problem_type is ProblemType.general
    assert routing.difficulty is Difficulty.medium
    assert routing.visualization_environment is None
    assert routing.solver_model == EASY_MODEL
    assert routing.parser_model == LOCAL_LLAMA_GEOMETRY_PARSER_MODEL
    assert "left unclassified" in routing.reason


def test_async_router_does_not_replace_failed_ai_with_keyword_detection() -> None:
    problem = (
        "Determine all pairs (a, b) of positive integers for which there exist "
        "positive integers g and N such that gcd(a^n+b, b^n+a) = g holds for "
        "all integers n ≥ N."
    )
    llama_client = RoutingLlamaClient()

    async def invalid_payload(**_: object) -> dict[str, object]:
        return {"problem_type": "not-valid"}

    llama_client.generate_json = invalid_payload  # type: ignore[method-assign]

    routing = asyncio.run(
        ModelRouter(llama_client=llama_client).route_async(problem, has_image=False)
    )

    assert routing.problem_type is ProblemType.general
    assert routing.difficulty is Difficulty.medium
    assert routing.visualization_environment is None
    assert "left unclassified" in routing.reason


def test_unified_router_selects_three_dimensional_environment() -> None:
    llama_client = RoutingLlamaClient()
    routing = asyncio.run(
        ModelRouter(llama_client=llama_client).route_async(
            "Visualize a tetrahedron!", has_image=False
        )
    )

    assert llama_client.calls == 1
    assert llama_client.request["model"] == "test-tiny-router"
    assert llama_client.request["thinking_budget_tokens"] == 0
    assert llama_client.request["json_schema"]["properties"][
        "visualization_environment"
    ]["enum"] == [
        "geometry_2d",
        "graphing",
        "graphics_3d",
        "cas",
        "probability",
        "statistics",
        "spreadsheet",
        "none",
    ]
    assert llama_client.request["json_schema"]["properties"]["solve_route"]["enum"] == [
        "deterministic",
        "local_trivia",
        "remote",
    ]
    assert "normalized_prompt" in llama_client.request["json_schema"]["properties"]
    assert "reason" in llama_client.request["json_schema"]["properties"]
    assert routing.problem_type is ProblemType.geometry
    assert routing.difficulty is Difficulty.medium
    assert (
        routing.visualization_environment
        is VisualizationEnvironment.graphics_3d
    )
    assert routing.solver_model == EASY_MODEL
    assert routing.visualization_search_terms == ("Tetrahedron", "solid", "3D")
    assert "test-tiny-router supplied classification" in routing.reason


def test_router_preserves_model_decision_that_no_visualization_is_useful() -> None:
    routing = asyncio.run(
        ModelRouter(llama_client=NoVisualizationLlamaClient()).route_async(
            "What is 2 + 2?", has_image=False
        )
    )

    assert routing.problem_type is ProblemType.arithmetic
    assert routing.difficulty is Difficulty.easy
    assert routing.visualization_environment is None
    assert routing.visualization_search_terms == ()


def test_router_reconciles_geometry_with_no_visualization() -> None:
    routing = asyncio.run(
        ModelRouter(
            llama_client=GeometryWithoutVisualizationLlamaClient()
        ).route_async(VIETNAMESE_GEOMETRY_PROOF, has_image=False)
    )

    assert routing.problem_type is ProblemType.geometry
    assert routing.visualization_environment is VisualizationEnvironment.geometry_2d
    assert "reconciled geometry classification" in routing.reason


def test_structured_endpoints_use_only_the_gpt_oss_models() -> None:
    endpoints = structured_model_endpoints(HARD_MODEL)

    assert [endpoint.model for endpoint in endpoints] == [HARD_MODEL, EASY_MODEL]


def test_only_gpt_oss_120b_receives_the_large_model_timeout() -> None:
    settings = SimpleNamespace(
        remote_model_attempt_timeout_seconds=25.0,
        nvidia_large_model_attempt_timeout_seconds=50.0,
    )

    assert (
        remote_model_timeout_seconds(
            settings, provider="nvidia_direct", model=HARD_MODEL
        )
        == 50.0
    )
    assert (
        remote_model_timeout_seconds(
            settings, provider="nvidia_direct", model=EASY_MODEL
        )
        == 25.0
    )
