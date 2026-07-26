from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.model_policy import (
    EASY_MODEL,
    HARD_MODEL,
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    LOCAL_LLAMA_TRIVIA_MODEL,
    SolveRoute,
    VISION_MODEL,
)
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment

if TYPE_CHECKING:
    from app.integrations.llama_client import LlamaClient

logger = logging.getLogger(__name__)

UNIFIED_ROUTING_PROMPT = """
Make one routing decision for a math prompt. Return only the schema-constrained
JSON object. Do not solve the problem.

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

solve_route is exactly one of:
- deterministic: only when the ORIGINAL prompt already has one of these exact
  executable shapes: a numeric arithmetic expression; one linear equation or
  inequality in x; or y=/f(x)= quadratic graph analysis. Geometry, proofs, word
  problems, systems, and uncertain prompts must never use this route.
- local_trivia: a short, non-proof math fact, definition, or concept question
  that the local trivia tutor can answer.
- remote: every other problem.

normalized_prompt is a concise cleaned version of the original prompt. It is
advisory metadata only and cannot make an unsupported prompt deterministic.

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

reason is one short sentence explaining the route.

Follow the environment definitions literally. A request to visualize, draw,
construct, graph, plot, chart, or use a named view must not be none. Function
graphs use graphing, not geometry_2d. Planar constructions use geometry_2d.
A three-dimensional solid uses graphics_3d, not geometry_2d.

Representative classifications:
- "Show a spatial solid" -> geometry, medium, remote, graphics_3d
- "Construct a planar polygon" -> geometry, medium, remote, geometry_2d
- "Graph f(x)=x^2" -> algebra, medium, deterministic, graphing
- "What is a prime number?" -> number_theory, easy, local_trivia, none
- "Plot a histogram of these values" -> statistics, medium, remote, statistics
- "Solve this word problem ..." -> general, medium, remote, none

Difficulty guidance:
- easy
- medium
- hard

- Use geometry for Euclidean construction/proof/theorem problems, including coordinate or analytic geometry.
- Use algebra for equations, inequalities, simplification, factoring, symbolic manipulation, graphing functions, domains/ranges, vertices, intercepts, or y=/f(x)= style function analysis.
- Use number_theory for divisibility, gcd/lcm, primes, congruences, Diophantine equations, or statements quantified over integers.
- Use arithmetic for numeric-only calculations.
- Use hard for proofs, multi-step theorem reasoning, long multi-part prompts, or advanced topics.
- Use medium for graph/function/statistics/trigonometry/calculus prompts unless they are clearly advanced or proof-like.
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
        "solve_route": {
            "type": "string",
            "enum": [route.value for route in SolveRoute],
        },
        "normalized_prompt": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4_000,
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
        "solve_route",
        "normalized_prompt",
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


class ModelRouter:
    """Routes problems using a local llama.cpp classifier when available.

    The synchronous `route` method returns an explicitly uncertain route for tests
    and non-async callers. Production solve flow uses `route_async` and does not
    replace a failed model classification with keyword-based subject detection.
    """

    def __init__(self, llama_client: "LlamaClient | None" = None) -> None:
        self.llama_client = llama_client

    def route(self, text: str, *, has_image: bool) -> RoutingDecision:
        classification = self._uncertain_classification(text)
        return self._build_decision(classification, has_image=has_image)

    async def route_async(self, text: str, *, has_image: bool) -> RoutingDecision:
        classification = await self._classify_with_llama(text)
        if classification is None:
            classification = self._uncertain_classification(text)
        return self._build_decision(classification, has_image=has_image)

    def _build_decision(
        self, classification: RouterClassification, *, has_image: bool
    ) -> RoutingDecision:
        parser_model = LOCAL_LLAMA_GEOMETRY_PARSER_MODEL
        solver_model = (
            HARD_MODEL if classification.difficulty is Difficulty.hard else EASY_MODEL
        )
        if classification.solve_route is SolveRoute.deterministic:
            solver_model = LOCAL_DETERMINISTIC_SOLVER_MODEL
        elif classification.solve_route is SolveRoute.local_trivia:
            solver_model = LOCAL_LLAMA_TRIVIA_MODEL
        vision_model = VISION_MODEL if has_image else None
        reason_parts = []
        if has_image:
            reason_parts.append("visual input requires OCR before solving")
        reason_parts.append(f"classified as {classification.problem_type.value}")
        reason_parts.append(f"difficulty assessed as {classification.difficulty.value}")
        reason_parts.append(classification.reason)
        if classification.model:
            reason_parts.append(
                f"local llama.cpp router {classification.model} supplied classification"
            )
        if classification.solve_route is SolveRoute.deterministic:
            reason_parts.append("selected deterministic execution subject to parser validation")
        elif classification.solve_route is SolveRoute.local_trivia:
            reason_parts.append("selected the local trivia tutor")
        elif classification.difficulty is Difficulty.hard:
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
        )

    async def _classify_with_llama(self, text: str) -> RouterClassification | None:
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
            logger.debug(
                "Llama classification failed; using an unclassified route",
                exc_info=True,
            )
            return None

        problem_type = self._coerce_problem_type(payload.get("problem_type"))
        difficulty = self._coerce_difficulty(payload.get("difficulty"))
        solve_route = self._coerce_solve_route(payload.get("solve_route"))
        normalized_prompt = self._coerce_required_string(
            payload.get("normalized_prompt")
        )
        supplied_reason = self._coerce_required_string(payload.get("reason"))
        environment_valid, visualization_environment = (
            self._coerce_visualization_environment(
                payload.get("visualization_environment")
            )
        )
        if (
            problem_type is None
            or difficulty is None
            or solve_route is None
            or normalized_prompt is None
            or supplied_reason is None
            or not environment_valid
        ):
            return None
        classification_reason = supplied_reason
        if (
            problem_type is ProblemType.geometry
            and visualization_environment is None
        ):
            # A geometry classification denotes a planar geometry problem unless
            # the model identified a spatial structure above.  Do not let a
            # contradictory `none` decision suppress the requested diagram.
            visualization_environment = VisualizationEnvironment.geometry_2d
            classification_reason = (
                "classified by local router; reconciled geometry classification "
                "to geometry_2d visualization"
            )
        visualization_search_terms = self._coerce_search_terms(
            payload.get("visualization_search_terms")
        )
        if visualization_search_terms is None:
            return None

        return RouterClassification(
            problem_type=problem_type,
            difficulty=difficulty,
            visualization_environment=visualization_environment,
            reason=classification_reason,
            solve_route=solve_route,
            normalized_prompt=normalized_prompt,
            model=router_model or getattr(llama_client, "model", None),
            visualization_search_terms=visualization_search_terms,
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

    def _uncertain_classification(self, text: str) -> RouterClassification:
        if not text.strip():
            return RouterClassification(
                problem_type=ProblemType.general,
                difficulty=Difficulty.easy,
                visualization_environment=None,
                reason="empty prompt",
                normalized_prompt="",
                visualization_search_terms=(),
            )
        return RouterClassification(
            problem_type=ProblemType.general,
            difficulty=Difficulty.medium,
            visualization_environment=None,
            reason=(
                "local router unavailable; subject and visualization environment "
                "left unclassified"
            ),
            solve_route=SolveRoute.remote,
            normalized_prompt=text.strip(),
            visualization_search_terms=(),
        )

    def _coerce_solve_route(self, value: Any) -> SolveRoute | None:
        if not isinstance(value, str):
            return None
        try:
            return SolveRoute(value.strip().lower())
        except ValueError:
            return None

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
