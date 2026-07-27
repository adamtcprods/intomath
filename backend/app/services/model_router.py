from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.model_policy import (
    EASY_MODEL,
    HARD_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    SolveRoute,
    VISION_MODEL,
    derive_non_exact_solve_route,
)
from app.core.solve_metrics import current_solve_metrics, record_model_attempt
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.semantic_router import (
    SemanticAbstention,
    SemanticClassification,
    SemanticRouter,
)

if TYPE_CHECKING:
    from app.integrations.llama_client import LlamaClient

logger = logging.getLogger(__name__)

UNIFIED_ROUTING_PROMPT = """
Classify one math prompt. Return only the schema-constrained JSON object. Do not
solve the problem and do not select an execution route.

problem_type is exactly one of:
- arithmetic
- algebra
- number_theory
- geometry
- trigonometry
- calculus
- probability
- statistics
- general

difficulty is exactly one of easy, medium, or hard.

visualization_environment is exactly one of:
- graphics_3d: a useful/requested three-dimensional object, solid, surface, or spatial-coordinate scene
- geometry_2d: planar geometry or a geometric construction
- graphing: functions, equations, inequalities, loci, regions, or coordinate plots
- cas: symbolic algebra/calculus intended for a CAS view
- probability: probability distributions, trees, or diagrams
- statistics: statistical plots, charts, or data summaries
- spreadsheet: tabular/cell-based computation
- none: no useful interactive mathematical representation

visualization_search_terms is an array of 0 to 8 short English GeoGebra
operation terms or likely command names. Use an empty array when the environment
is none. Do not solve the prompt or invent objects, coordinates, or relationships.

reason is one short sentence explaining the classification.

Follow the environment definitions literally. A request to visualize, draw,
construct, graph, plot, chart, or use a named view must not be none. Function
graphs use graphing, not geometry_2d. Planar constructions use geometry_2d.
A three-dimensional solid uses graphics_3d, not geometry_2d.

Representative classifications:
- "Show a spatial solid" -> geometry, medium, graphics_3d
- "Construct a planar polygon" -> geometry, medium, geometry_2d
- "Graph f(x)=x^2" -> algebra, medium, graphing
- "What is a prime number?" -> number_theory, easy, none
- "Plot a histogram of these values" -> statistics, medium, statistics
- "Solve this word problem ..." -> general, medium, none

Difficulty guidance:
- Use hard for proofs, multi-step theorem reasoning, long multi-part prompts, or advanced topics.
- Use medium for graph/function/statistics/trigonometry/calculus prompts unless clearly advanced.
- Use easy for short routine arithmetic/algebra prompts.
""".strip()

UNIFIED_ROUTING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "problem_type": {
            "type": "string",
            "enum": [problem_type.value for problem_type in ProblemType],
        },
        "difficulty": {
            "type": "string",
            "enum": [difficulty.value for difficulty in Difficulty],
        },
        "visualization_environment": {
            "type": "string",
            "enum": [*[item.value for item in VisualizationEnvironment], "none"],
        },
        "visualization_search_terms": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 40},
            "maxItems": 8,
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
    "required": [
        "problem_type",
        "difficulty",
        "visualization_environment",
        "visualization_search_terms",
        "reason",
    ],
    "additionalProperties": False,
}


@dataclass
class RoutingDecision:
    problem_type: ProblemType
    difficulty: Difficulty
    parser_model: str
    solver_model: str
    vision_model: str | None
    visualization_environment: VisualizationEnvironment | None
    reason: str
    visualization_search_terms: tuple[str, ...] = ()
    solve_route: SolveRoute = SolveRoute.remote
    normalized_prompt: str = ""
    routing_source: str = "safe_unclassified"
    classification_confidence: float | None = None
    classification_margin: float | None = None
    abstention_reason: str | None = None


@dataclass(frozen=True)
class RouterClassification:
    problem_type: ProblemType
    difficulty: Difficulty
    visualization_environment: VisualizationEnvironment | None
    reason: str
    solve_route: SolveRoute = SolveRoute.remote
    normalized_prompt: str = ""
    model: str | None = None
    visualization_search_terms: tuple[str, ...] = ()
    source: str = "safe_unclassified"
    confidence: float | None = None
    margin: float | None = None
    abstention_reason: str | None = None


class ModelRouter:
    """Embedding-first router with one local-LLM uncertainty fallback."""

    def __init__(
        self,
        llama_client: "LlamaClient | None" = None,
        *,
        semantic_router: SemanticRouter | None = None,
        settings: Any | None = None,
    ) -> None:
        self.llama_client = llama_client
        self.semantic_router = semantic_router
        self.settings = settings or getattr(llama_client, "settings", None) or get_settings()

    def route(self, text: str, *, has_image: bool) -> RoutingDecision:
        classification = self._uncertain_classification(text)
        return self._build_decision(classification, has_image=has_image)

    async def route_async(
        self,
        text: str,
        *,
        has_image: bool,
        language: str | None = None,
        visualization_requested: bool = True,
    ) -> RoutingDecision:
        embedding_result = await self._classify_with_embeddings(text, language=language)
        classification, fallback_reason = self._embedding_policy(
            embedding_result,
            text=text,
            visualization_requested=visualization_requested,
        )
        if classification is None and bool(
            getattr(self.settings, "semantic_router_fallback_to_llm", True)
        ):
            metrics = current_solve_metrics()
            if metrics is not None:
                metrics.llm_routing_fallback_count += 1
            classification = await self._classify_with_llama(
                text,
                fallback_reason=fallback_reason or "embedding router unavailable",
            )
        if classification is None:
            classification = self._uncertain_classification(
                text,
                abstention_reason=fallback_reason,
            )
        self._record_routing_metrics(classification)
        return self._build_decision(classification, has_image=has_image)

    async def _classify_with_embeddings(
        self,
        text: str,
        *,
        language: str | None,
    ) -> SemanticClassification | SemanticAbstention | None:
        if self.semantic_router is None:
            return None
        try:
            result = await self.semantic_router.classify_async(text, language=language)
        except Exception:
            logger.debug("Embedding classification failed", exc_info=True)
            return SemanticAbstention(
                reason="embedding_error",
                model_name="semantic-router",
                used_fallback=False,
                language=language or "unknown",
                latency_ms=0.0,
            )
        metrics = current_solve_metrics()
        if metrics is not None:
            metrics.semantic_routing_latency_ms += result.latency_ms
            if isinstance(result, SemanticClassification) or result.inference_performed:
                metrics.semantic_inference_count += 1
        if isinstance(result, SemanticClassification) or result.inference_performed:
            record_model_attempt(
                provider="local_embedding",
                model=result.model_name,
                operation="semantic_routing",
            )
        return result

    def _embedding_policy(
        self,
        result: SemanticClassification | SemanticAbstention | None,
        *,
        text: str,
        visualization_requested: bool,
    ) -> tuple[RouterClassification | None, str | None]:
        if result is None:
            return None, "embedding_router_unavailable"
        if isinstance(result, SemanticAbstention):
            return None, result.reason
        if not result.is_confident("problem_type"):
            return None, "low_confidence:problem_type"

        environment = result.visualization_environment
        environment_reason: str | None = None
        if not result.is_confident("visualization_environment"):
            if not visualization_requested:
                environment = None
                environment_reason = "visualization not requested; uncertain environment defaulted to none"
            elif self._safe_nonvisual_embedding_result(result):
                environment = None
                environment_reason = "uncertain visualization safely defaulted to none"
            else:
                return None, "low_confidence:visualization_environment"

        difficulty = result.difficulty
        difficulty_reason: str | None = None
        if not result.is_confident("difficulty"):
            difficulty = Difficulty.medium
            difficulty_reason = "uncertain difficulty defaulted safely to medium"

        reason_parts = [result.reason]
        if difficulty_reason:
            reason_parts.append(difficulty_reason)
        if environment_reason:
            reason_parts.append(environment_reason)
        problem_score = result.axis_score("problem_type")
        visualization_score = result.axis_score("visualization_environment")
        return (
            RouterClassification(
                problem_type=result.problem_type,
                difficulty=difficulty,
                visualization_environment=environment,
                reason="; ".join(reason_parts),
                solve_route=derive_non_exact_solve_route(),
                normalized_prompt=text.strip(),
                model=result.model_name,
                visualization_search_terms=(
                    result.visualization_search_terms if environment is not None else ()
                ),
                source="embedding",
                confidence=min(
                    problem_score.confidence,
                    visualization_score.confidence,
                ),
                margin=min(problem_score.margin, visualization_score.margin),
            ),
            None,
        )

    def _safe_nonvisual_embedding_result(
        self,
        result: SemanticClassification,
    ) -> bool:
        visualization_score = result.axis_score("visualization_environment")
        minimum_confidence = float(
            getattr(self.settings, "semantic_router_min_confidence", 0.60)
        )
        return (
            result.problem_type in {ProblemType.arithmetic, ProblemType.number_theory}
            and visualization_score.label == "none"
            and visualization_score.confidence >= minimum_confidence
        )

    def _build_decision(
        self, classification: RouterClassification, *, has_image: bool
    ) -> RoutingDecision:
        parser_model = LOCAL_LLAMA_GEOMETRY_PARSER_MODEL
        solver_model = (
            HARD_MODEL if classification.difficulty is Difficulty.hard else EASY_MODEL
        )
        vision_model = VISION_MODEL if has_image else None
        reason_parts = []
        if has_image:
            reason_parts.append("visual input requires OCR before solving")
        reason_parts.append(f"classified as {classification.problem_type.value}")
        reason_parts.append(f"difficulty assessed as {classification.difficulty.value}")
        reason_parts.append(classification.reason)
        if classification.source == "embedding":
            reason_parts.append(
                f"multilingual embedding router {classification.model} supplied classification"
            )
        elif classification.source == "llm_fallback":
            reason_parts.append(
                f"LLM fallback router {classification.model} supplied classification"
            )
        if classification.difficulty is Difficulty.hard:
            reason_parts.append("escalated to the JSON-stable hard free model")
        else:
            reason_parts.append("kept on the lower-latency JSON-stable model")

        return RoutingDecision(
            problem_type=classification.problem_type,
            difficulty=classification.difficulty,
            parser_model=parser_model,
            solver_model=solver_model,
            vision_model=vision_model,
            visualization_environment=classification.visualization_environment,
            reason="; ".join(reason_parts),
            visualization_search_terms=classification.visualization_search_terms,
            solve_route=classification.solve_route,
            normalized_prompt=classification.normalized_prompt,
            routing_source=classification.source,
            classification_confidence=classification.confidence,
            classification_margin=classification.margin,
            abstention_reason=classification.abstention_reason,
        )

    async def _classify_with_llama(
        self,
        text: str,
        *,
        fallback_reason: str,
    ) -> RouterClassification | None:
        llama_client = self._get_llama_client()
        if llama_client is None or not getattr(llama_client, "enabled", False):
            return None
        router_model = str(
            getattr(
                getattr(llama_client, "settings", None),
                "local_router_llama_model",
                getattr(llama_client, "model", ""),
            )
        ).strip()
        model_available = getattr(llama_client, "is_model_available", None)
        if callable(model_available):
            if not model_available(router_model):
                return None
        elif not getattr(llama_client, "available", True):
            return None

        stripped = text.strip()
        if not stripped:
            return RouterClassification(
                problem_type=ProblemType.general,
                difficulty=Difficulty.easy,
                visualization_environment=None,
                reason="empty prompt",
                normalized_prompt="",
                visualization_search_terms=(),
                source="llm_fallback",
                abstention_reason=fallback_reason,
            )
        if len(stripped) > 4_000:
            return None

        prompt = f"{UNIFIED_ROUTING_PROMPT}\n\nProblem:\n{stripped}"
        try:
            payload = await llama_client.generate_json(
                prompt=prompt,
                model=router_model or None,
                max_tokens=int(
                    getattr(
                        getattr(llama_client, "settings", None),
                        "local_router_llama_max_tokens",
                        300,
                    )
                ),
                thinking_budget_tokens=0,
                timeout_seconds=float(
                    getattr(
                        getattr(llama_client, "settings", None),
                        "local_router_llama_timeout_seconds",
                        8.0,
                    )
                ),
                json_schema=UNIFIED_ROUTING_RESPONSE_SCHEMA,
                operation="local_unified_routing",
            )
        except Exception:
            logger.debug("Llama fallback classification failed", exc_info=True)
            return None

        problem_type = self._coerce_problem_type(payload.get("problem_type"))
        difficulty = self._coerce_difficulty(payload.get("difficulty"))
        supplied_reason = self._coerce_required_string(payload.get("reason"))
        environment_valid, visualization_environment = (
            self._coerce_visualization_environment(
                payload.get("visualization_environment")
            )
        )
        visualization_search_terms = self._coerce_search_terms(
            payload.get("visualization_search_terms")
        )
        if (
            problem_type is None
            or difficulty is None
            or supplied_reason is None
            or not environment_valid
            or visualization_search_terms is None
        ):
            return None

        return RouterClassification(
            problem_type=problem_type,
            difficulty=difficulty,
            visualization_environment=visualization_environment,
            reason=f"{supplied_reason}; embedding fallback cause: {fallback_reason}",
            solve_route=derive_non_exact_solve_route(),
            normalized_prompt=stripped,
            model=router_model or getattr(llama_client, "model", None),
            visualization_search_terms=visualization_search_terms,
            source="llm_fallback",
            abstention_reason=fallback_reason,
        )

    def _get_llama_client(self) -> "LlamaClient | None":
        if self.llama_client is not None:
            return self.llama_client
        try:
            from app.integrations.llama_client import LlamaClient
        except Exception:
            logger.debug("Failed to initialize LlamaClient", exc_info=True)
            return None
        self.llama_client = LlamaClient()
        return self.llama_client

    def _uncertain_classification(
        self,
        text: str,
        *,
        abstention_reason: str | None = None,
    ) -> RouterClassification:
        if not text.strip():
            return RouterClassification(
                problem_type=ProblemType.general,
                difficulty=Difficulty.easy,
                visualization_environment=None,
                reason="empty prompt",
                normalized_prompt="",
                visualization_search_terms=(),
                abstention_reason=abstention_reason,
            )
        cause = f" after {abstention_reason}" if abstention_reason else ""
        return RouterClassification(
            problem_type=ProblemType.general,
            difficulty=Difficulty.medium,
            visualization_environment=None,
            reason=(
                f"semantic and LLM routers unavailable{cause}; subject and "
                "visualization environment left unclassified"
            ),
            solve_route=derive_non_exact_solve_route(),
            normalized_prompt=text.strip(),
            visualization_search_terms=(),
            source="safe_unclassified",
            abstention_reason=abstention_reason,
        )

    def _record_routing_metrics(self, classification: RouterClassification) -> None:
        metrics = current_solve_metrics()
        if metrics is None:
            return
        metrics.routing_source = classification.source
        metrics.routing_abstention_reason = classification.abstention_reason
        metrics.routing_confidence = classification.confidence
        metrics.routing_margin = classification.margin

    def _coerce_required_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _coerce_search_terms(self, value: Any) -> tuple[str, ...] | None:
        if not isinstance(value, list):
            return None
        terms = tuple(
            item.strip()[:40]
            for item in value[:8]
            if isinstance(item, str) and item.strip()
        )
        return tuple(dict.fromkeys(terms))

    def _coerce_problem_type(self, value: Any) -> ProblemType | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "coordinate_geometry": ProblemType.geometry,
            "functions": ProblemType.algebra,
            "function": ProblemType.algebra,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return ProblemType(normalized)
        except ValueError:
            return None

    def _coerce_visualization_environment(
        self, value: Any
    ) -> tuple[bool, VisualizationEnvironment | None]:
        if not isinstance(value, str):
            return False, None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "none":
            return True, None
        try:
            return True, VisualizationEnvironment(normalized)
        except ValueError:
            return False, None

    def _coerce_difficulty(self, value: Any) -> Difficulty | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        try:
            return Difficulty(normalized)
        except ValueError:
            return None
