import asyncio
from app.core.solve_metrics import SolveMetrics, bind_solve_metrics, reset_solve_metrics
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.semantic_router.contracts import AxisScore
from app.services.semantic_router import SemanticAbstention, SemanticClassification
from types import SimpleNamespace

from app.core.model_policy import (
    EASY_MODEL,
    HARD_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    remote_model_timeout_seconds,
    structured_model_endpoints,
)
from app.services.model_router import ModelRouter

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
    assert "solve_route" not in llama_client.request["json_schema"]["properties"]
    assert "normalized_prompt" not in llama_client.request["json_schema"]["properties"]
    assert "reason" in llama_client.request["json_schema"]["properties"]
    assert routing.problem_type is ProblemType.geometry
    assert routing.difficulty is Difficulty.medium
    assert (
        routing.visualization_environment
        is VisualizationEnvironment.graphics_3d
    )
    assert routing.solver_model == EASY_MODEL
    assert routing.visualization_search_terms == ("Tetrahedron", "solid", "3D")
    assert routing.solve_route.value == "remote"
    assert "LLM fallback router test-tiny-router supplied classification" in routing.reason


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
    assert routing.solve_route.value == "remote"


def test_router_preserves_nonvisual_geometry_classification() -> None:
    routing = asyncio.run(
        ModelRouter(
            llama_client=GeometryWithoutVisualizationLlamaClient()
        ).route_async(VIETNAMESE_GEOMETRY_PROOF, has_image=False)
    )

    assert routing.problem_type is ProblemType.geometry
    assert routing.visualization_environment is None
    assert "reconciled geometry classification" not in routing.reason


class StubSemanticRouter:
    def __init__(self, result: SemanticClassification | SemanticAbstention) -> None:
        self.result = result
        self.calls = 0

    async def classify_async(self, *_: object, **__: object) -> SemanticClassification | SemanticAbstention:
        self.calls += 1
        return self.result


def _score(label: str, runner_up: str, *, confident: bool = True) -> AxisScore:
    confidence = 0.92 if confident else 0.51
    runner_confidence = 0.03 if confident else 0.49
    return AxisScore(
        label=label,
        confidence=confidence,
        runner_up=runner_up,
        margin=confidence - runner_confidence,
        raw_similarity=0.80,
        scores=((label, confidence), (runner_up, runner_confidence)),
    )


def _embedding_classification(
    *,
    problem_type: ProblemType = ProblemType.geometry,
    difficulty: Difficulty = Difficulty.medium,
    environment: VisualizationEnvironment | None = VisualizationEnvironment.geometry_2d,
    confident_axes: frozenset[str] = frozenset(
        {"problem_type", "difficulty", "visualization_environment"}
    ),
) -> SemanticClassification:
    environment_label = environment.value if environment is not None else "none"
    return SemanticClassification(
        problem_type=problem_type,
        difficulty=difficulty,
        visualization_environment=environment,
        confidence=0.90,
        margin=0.80,
        reason="independent multilingual prototype classification",
        model_name="fake-embedding-model",
        used_fallback=False,
        language="mixed",
        latency_ms=1.0,
        axis_scores=(
            (
                "problem_type",
                _score(
                    problem_type.value,
                    "general",
                    confident="problem_type" in confident_axes,
                ),
            ),
            (
                "difficulty",
                _score(
                    difficulty.value,
                    "medium" if difficulty is not Difficulty.medium else "easy",
                    confident="difficulty" in confident_axes,
                ),
            ),
            (
                "visualization_environment",
                _score(
                    environment_label,
                    "none" if environment_label != "none" else "cas",
                    confident="visualization_environment" in confident_axes,
                ),
            ),
        ),
        confident_axes=confident_axes,
        visualization_search_terms=("Midpoint", "Perpendicular"),
    )


def test_high_confidence_embedding_classification_skips_llm() -> None:
    llama_client = RoutingLlamaClient()
    semantic_router = StubSemanticRouter(_embedding_classification())

    routing = asyncio.run(
        ModelRouter(
            llama_client=llama_client,
            semantic_router=semantic_router,  # type: ignore[arg-type]
        ).route_async(
            "Construct đường trung trực AB",
            has_image=False,
            language="mixed",
        )
    )

    assert semantic_router.calls == 1
    assert llama_client.calls == 0
    assert routing.problem_type is ProblemType.geometry
    assert routing.visualization_environment is VisualizationEnvironment.geometry_2d
    assert routing.visualization_search_terms == ("Midpoint", "Perpendicular")
    assert routing.solve_route.value == "remote"
    assert routing.routing_source == "embedding"
    assert "multilingual embedding router" in routing.reason


def test_embedding_and_fallback_provenance_are_recorded_in_metrics() -> None:
    metrics = SolveMetrics(request_id="semantic-metrics")
    token = bind_solve_metrics(metrics)
    try:
        asyncio.run(
            ModelRouter(
                llama_client=RoutingLlamaClient(),
                semantic_router=StubSemanticRouter(  # type: ignore[arg-type]
                    _embedding_classification()
                ),
            ).route_async("geometry", has_image=False)
        )
    finally:
        reset_solve_metrics(token)

    assert metrics.routing_source == "embedding"
    assert metrics.semantic_inference_count == 1
    assert metrics.llm_routing_fallback_count == 0
    assert metrics.model_call_count == 1
    assert metrics.routing_confidence is not None
    assert metrics.routing_margin is not None


def test_uncertain_difficulty_defaults_to_medium_without_llm() -> None:
    llama_client = RoutingLlamaClient()
    classification = _embedding_classification(
        problem_type=ProblemType.calculus,
        difficulty=Difficulty.hard,
        environment=None,
        confident_axes=frozenset(
            {"problem_type", "visualization_environment"}
        ),
    )

    routing = asyncio.run(
        ModelRouter(
            llama_client=llama_client,
            semantic_router=StubSemanticRouter(classification),  # type: ignore[arg-type]
        ).route_async("Explain a derivative", has_image=False)
    )

    assert llama_client.calls == 0
    assert routing.problem_type is ProblemType.calculus
    assert routing.difficulty is Difficulty.medium
    assert "difficulty defaulted safely to medium" in routing.reason


def test_uncertain_problem_type_uses_one_complete_llm_fallback() -> None:
    llama_client = RoutingLlamaClient()
    classification = _embedding_classification(
        confident_axes=frozenset({"difficulty", "visualization_environment"})
    )

    routing = asyncio.run(
        ModelRouter(
            llama_client=llama_client,
            semantic_router=StubSemanticRouter(classification),  # type: ignore[arg-type]
        ).route_async("ambiguous prompt", has_image=False)
    )

    assert llama_client.calls == 1
    assert routing.routing_source == "llm_fallback"
    assert routing.problem_type is ProblemType.geometry
    assert routing.visualization_environment is VisualizationEnvironment.graphics_3d
    assert routing.abstention_reason == "low_confidence:problem_type"


def test_uncertain_visualization_falls_back_only_when_requested() -> None:
    classification = _embedding_classification(
        problem_type=ProblemType.calculus,
        environment=VisualizationEnvironment.cas,
        confident_axes=frozenset({"problem_type", "difficulty"}),
    )
    requested_llama = RoutingLlamaClient()
    requested = asyncio.run(
        ModelRouter(
            llama_client=requested_llama,
            semantic_router=StubSemanticRouter(classification),  # type: ignore[arg-type]
        ).route_async("differentiate this", has_image=False)
    )
    skipped_llama = RoutingLlamaClient()
    not_requested = asyncio.run(
        ModelRouter(
            llama_client=skipped_llama,
            semantic_router=StubSemanticRouter(classification),  # type: ignore[arg-type]
        ).route_async(
            "differentiate this",
            has_image=False,
            visualization_requested=False,
        )
    )

    assert requested_llama.calls == 1
    assert requested.routing_source == "llm_fallback"
    assert skipped_llama.calls == 0
    assert not_requested.routing_source == "embedding"
    assert not_requested.visualization_environment is None
    assert not_requested.visualization_search_terms == ()


def test_out_of_distribution_uses_llm_and_disabled_fallback_is_safe() -> None:
    abstention = SemanticAbstention(
        reason="out_of_distribution",
        model_name="fake-embedding-model",
        used_fallback=False,
        language="en",
        latency_ms=1.0,
        inference_performed=True,
    )
    llama_client = RoutingLlamaClient()
    with_fallback = asyncio.run(
        ModelRouter(
            llama_client=llama_client,
            semantic_router=StubSemanticRouter(abstention),  # type: ignore[arg-type]
        ).route_async("unknown", has_image=False)
    )
    without_fallback = asyncio.run(
        ModelRouter(
            llama_client=RoutingLlamaClient(),
            semantic_router=StubSemanticRouter(abstention),  # type: ignore[arg-type]
            settings=SimpleNamespace(
                semantic_router_fallback_to_llm=False,
                semantic_router_min_confidence=0.60,
            ),
        ).route_async("unknown", has_image=False)
    )

    assert llama_client.calls == 1
    assert with_fallback.routing_source == "llm_fallback"
    assert without_fallback.routing_source == "safe_unclassified"
    assert "left unclassified" in without_fallback.reason


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
