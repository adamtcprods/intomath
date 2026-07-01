from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.schemas.common import Difficulty, ProblemType

if TYPE_CHECKING:
    from app.integrations.llama_client import LlamaClient

EASY_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
HARD_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
JSON_SECONDARY_FALLBACK_MODEL = EASY_MODEL
JSON_FALLBACK_MODEL = "openrouter/free"
LOCAL_DETERMINISTIC_SOLVER_MODEL = "local:deterministic-solver"
LOCAL_HEURISTIC_PARSER_MODEL = "local:heuristic-parser"
LOCAL_LLAMA_TRIVIA_MODEL = "local:llama-trivia"
VISION_MODEL = "local:deepseek-ai/deepseek-ocr-2"

ROUTER_CLASSIFICATION_PROMPT = """
You are IntoMath's strict math problem router.
Return exactly one JSON object and nothing else.

Classify the problem into exactly one problem_type:
- arithmetic
- algebra
- geometry
- trigonometry
- calculus
- probability
- statistics
- general

Also assess difficulty as exactly one of:
- easy
- medium
- hard

Guidelines:
- Use geometry for Euclidean construction/proof/theorem problems, including coordinate or analytic geometry.
- Use algebra for equations, inequalities, simplification, factoring, symbolic manipulation, graphing functions, domains/ranges, vertices, intercepts, or y=/f(x)= style function analysis.
- Use arithmetic for numeric-only calculations.
- Use hard for proofs, multi-step theorem reasoning, long multi-part prompts, or advanced topics.
- Use medium for graph/function/statistics/trigonometry/calculus prompts unless they are clearly advanced or proof-like.
- Use easy for short routine arithmetic/algebra prompts.

JSON schema:
{"problem_type": "algebra", "difficulty": "easy", "reason": "short reason"}
""".strip()


@dataclass
class RoutingDecision:
    problem_type: ProblemType
    difficulty: Difficulty
    parser_model: str
    solver_model: str
    vision_model: str | None
    reason: str


@dataclass(frozen=True)
class RouterClassification:
    problem_type: ProblemType
    difficulty: Difficulty
    reason: str
    model: str | None = None


class ModelRouter:
    """Routes problems using a tiny local classifier when available.

    The synchronous `route` method is kept as a structural fallback for tests and
    non-async callers. Production solve flow should use `route_async`, which asks
    the local llama.cpp model to classify the problem and falls back to structural
    signals if the local classifier is unavailable or returns invalid JSON.
    """

    def __init__(self, llama_client: "LlamaClient | None" = None) -> None:
        self.llama_client = llama_client

    def route(self, text: str, *, has_image: bool) -> RoutingDecision:
        classification = self._classify_structurally(text)
        return self._build_decision(classification, has_image=has_image)

    async def route_async(self, text: str, *, has_image: bool) -> RoutingDecision:
        classification = await self._classify_with_llama(text)
        if classification is None:
            classification = self._classify_structurally(text)
        return self._build_decision(classification, has_image=has_image)

    def _build_decision(
        self, classification: RouterClassification, *, has_image: bool
    ) -> RoutingDecision:
        parser_model = EASY_MODEL
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
        if classification.model:
            reason_parts.append(
                f"local llama.cpp router {classification.model} supplied classification"
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
            reason="; ".join(reason_parts),
        )

    async def _classify_with_llama(self, text: str) -> RouterClassification | None:
        llama_client = self._get_llama_client()
        if llama_client is None or not getattr(llama_client, "enabled", False):
            return None

        stripped = text.strip()
        if not stripped:
            return RouterClassification(
                problem_type=ProblemType.general,
                difficulty=Difficulty.easy,
                reason="empty prompt routed structurally",
            )
        if len(stripped) > 4_000:
            return None

        prompt = f"{ROUTER_CLASSIFICATION_PROMPT}\n\nProblem:\n{stripped}"
        try:
            payload = await llama_client.generate_json(prompt=prompt)
        except Exception:
            return None

        problem_type = self._coerce_problem_type(payload.get("problem_type"))
        difficulty = self._coerce_difficulty(payload.get("difficulty"))
        if problem_type is None or difficulty is None:
            return None

        reason = (
            self._coerce_reason(payload.get("reason")) or "classified by local router"
        )
        return RouterClassification(
            problem_type=problem_type,
            difficulty=difficulty,
            reason=reason,
            model=getattr(llama_client, "model", None),
        )

    def _get_llama_client(self) -> "LlamaClient | None":
        if self.llama_client is not None:
            return self.llama_client
        try:
            from app.integrations.llama_client import LlamaClient
        except Exception:
            return None
        self.llama_client = LlamaClient()
        return self.llama_client

    def _classify_structurally(self, text: str) -> RouterClassification:
        stripped = text.strip()
        if not stripped:
            return RouterClassification(
                problem_type=ProblemType.general,
                difficulty=Difficulty.easy,
                reason="empty prompt",
            )

        problem_type = self._infer_problem_type_from_structure(stripped)
        difficulty = self._infer_difficulty_from_structure(stripped, problem_type)
        return RouterClassification(
            problem_type=problem_type,
            difficulty=difficulty,
            reason="classified by structural fallback because local router was unavailable",
        )

    def _infer_problem_type_from_structure(self, text: str) -> ProblemType:
        if self._has_function_assignment(text):
            return ProblemType.algebra
        if self._has_coordinate_structure(text):
            return ProblemType.geometry
        if self._has_geometry_structure(text):
            return ProblemType.geometry
        if self._has_calculus_notation(text):
            return ProblemType.calculus
        if self._has_trigonometry_notation(text):
            return ProblemType.trigonometry
        if self._has_variable_relation(text) or self._has_symbolic_variable(text):
            return ProblemType.algebra
        if self._is_arithmetic_expression(text):
            return ProblemType.arithmetic
        return ProblemType.general

    def _infer_difficulty_from_structure(
        self, text: str, problem_type: ProblemType
    ) -> Difficulty:
        if self._has_proof_like_structure(text, problem_type):
            return Difficulty.hard
        if self._has_many_subquestions(text):
            return Difficulty.hard
        if problem_type is ProblemType.geometry:
            if len(self._point_labels(text)) >= 5:
                return Difficulty.hard
            return Difficulty.medium
        if len(text.split()) > 30:
            return Difficulty.medium
        if self._has_function_assignment(text):
            return Difficulty.medium
        if problem_type in {
            ProblemType.calculus,
            ProblemType.statistics,
            ProblemType.trigonometry,
        }:
            return Difficulty.medium
        return Difficulty.easy

    def _has_function_assignment(self, text: str) -> bool:
        return bool(
            re.search(r"(?:^|[^A-Za-z])(?:y|f\s*\(\s*x\s*\))\s*=", text, re.IGNORECASE)
        )

    def _has_coordinate_structure(self, text: str) -> bool:
        coordinate_pairs = re.findall(
            r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)",
            text,
        )
        return len(coordinate_pairs) >= 1 and (
            len(coordinate_pairs) >= 2 or bool(re.search(r"\b[A-Z]\s*\(", text))
        )

    def _has_geometry_structure(self, text: str) -> bool:
        if re.search(
            r"\\(?:perp|parallel|angle|triangle|circle|odot|cong|sim)\b", text
        ):
            return True
        labels = self._point_labels(text)
        if len(labels) >= 3 and re.search(r"\b[A-Z]{2,4}\b|\([A-Z]{2,4}\)", text):
            return True
        if len(labels) >= 4 and any(symbol in text for symbol in {"∠", "⊥", "∥", "°"}):
            return True
        return False

    def _has_calculus_notation(self, text: str) -> bool:
        return bool(
            "∫" in text
            or re.search(r"\\(?:int|lim)\b", text)
            or re.search(r"\bdy\s*/\s*dx\b|\bd\s*/\s*dx\b", text, re.IGNORECASE)
        )

    def _has_trigonometry_notation(self, text: str) -> bool:
        return bool(
            re.search(r"\\(?:sin|cos|tan)\b", text)
            or re.search(r"\b(?:sin|cos|tan)\s*\(", text, re.IGNORECASE)
        )

    def _has_variable_relation(self, text: str) -> bool:
        return bool(re.search(r"[A-Za-z][A-Za-z0-9_]*.*(?:<=|>=|=|<|>)", text))

    def _has_symbolic_variable(self, text: str) -> bool:
        return bool(re.search(r"(?<![A-Za-z])[a-z](?![A-Za-z])", text, re.IGNORECASE))

    def _is_arithmetic_expression(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        return bool(re.fullmatch(r"[-+*/^().\d]+", compact)) and bool(
            re.search(r"\d", compact)
        )

    def _has_proof_like_structure(self, text: str, problem_type: ProblemType) -> bool:
        if problem_type is not ProblemType.geometry:
            return False
        if re.search(r"(?:^|\n)\s*[a-z]\)", text, re.IGNORECASE):
            return True
        relation_count = len(re.findall(r"=|\\cdot|≅|∼|⊥|∥", text))
        return len(self._point_labels(text)) >= 5 and relation_count >= 2

    def _has_many_subquestions(self, text: str) -> bool:
        labels = re.findall(r"(?:^|\n)\s*(?:[a-z]|\d+)\)", text, flags=re.IGNORECASE)
        return len(labels) >= 2 or text.count("?") + text.count(";") > 6

    def _point_labels(self, text: str) -> set[str]:
        labels = set(re.findall(r"\b[A-Z]\b", text))
        for token in re.findall(r"\b[A-Z]{2,4}\b", text):
            labels.update(token)
        return labels

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

    def _coerce_difficulty(self, value: Any) -> Difficulty | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        try:
            return Difficulty(normalized)
        except ValueError:
            return None

    def _coerce_reason(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        reason = value.strip()
        return reason[:200] if reason else None
