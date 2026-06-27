from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.common import Difficulty, ProblemType

EASY_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
HARD_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
JSON_SECONDARY_FALLBACK_MODEL = EASY_MODEL
JSON_FALLBACK_MODEL = "openrouter/free"
LOCAL_DETERMINISTIC_SOLVER_MODEL = "local:deterministic-solver"
LOCAL_HEURISTIC_PARSER_MODEL = "local:heuristic-parser"
LOCAL_LLAMA_TRIVIA_MODEL = "local:llama-trivia"
VISION_MODEL = "local:deepseek-ai/deepseek-ocr-2"


@dataclass
class RoutingDecision:
    problem_type: ProblemType
    difficulty: Difficulty
    parser_model: str
    solver_model: str
    vision_model: str | None
    reason: str


class ModelRouter:
    PROOF_KEYWORDS = {
        "prove",
        "show that",
        "justify",
        "demonstrate",
        "theorem",
        "chứng minh",
        "chung minh",
        "chứng tỏ",
        "chung to",
    }
    GEOMETRY_KEYWORDS = {
        "triangle",
        "circle",
        "angle",
        "perpendicular",
        "parallel",
        "midpoint",
        "polygon",
        "line",
        "tam giác",
        "tam giac",
        "đường tròn",
        "duong tron",
        "đường kính",
        "duong kinh",
        "bán kính",
        "ban kinh",
        "góc",
        "goc",
        "vuông góc",
        "vuong goc",
        "song song",
        "trung điểm",
        "trung diem",
        "phân giác",
        "phan giac",
        "nội tiếp",
        "noi tiep",
        "tứ giác",
        "tu giac",
        "cung",
    }
    CALCULUS_KEYWORDS = {
        "derivative",
        "integral",
        "limit",
        "differentiate",
        "integrate",
        "tangent",
    }
    STATISTICS_KEYWORDS = {
        "mean",
        "median",
        "variance",
        "distribution",
        "probability",
        "histogram",
        "standard deviation",
    }
    FUNCTION_KEYWORDS = {
        "graph",
        "function",
        "vertex",
        "parabola",
        "domain",
        "range",
        "intersects",
    }
    TRIG_KEYWORDS = {"sin", "cos", "tan", "trigonometry", "angle of elevation"}

    def route(self, text: str, *, has_image: bool) -> RoutingDecision:
        normalized = text.lower().strip()
        problem_type = self.classify(normalized)
        difficulty = self.assess_difficulty(text, problem_type)
        parser_model = EASY_MODEL
        solver_model = HARD_MODEL if difficulty is Difficulty.hard else EASY_MODEL
        vision_model = VISION_MODEL if has_image else None

        reason_parts = []
        if has_image:
            reason_parts.append("visual input requires OCR before solving")
        reason_parts.append(f"classified as {problem_type.value}")
        reason_parts.append(f"difficulty assessed as {difficulty.value}")
        if difficulty is Difficulty.hard:
            reason_parts.append("escalated to the JSON-stable hard free model")
        else:
            reason_parts.append("kept on the lower-latency JSON-stable model")

        return RoutingDecision(
            problem_type=problem_type,
            difficulty=difficulty,
            parser_model=parser_model,
            solver_model=solver_model,
            vision_model=vision_model,
            reason="; ".join(reason_parts),
        )

    def classify(self, text: str) -> ProblemType:
        if not text:
            return ProblemType.general
        if self._contains_any(text, self.GEOMETRY_KEYWORDS):
            if "coordinate" in text or re.search(r"\([-\d\s,.]+\)", text):
                return ProblemType.coordinate_geometry
            return ProblemType.geometry
        if self._contains_any(text, self.CALCULUS_KEYWORDS):
            return ProblemType.calculus
        if self._contains_any(text, self.STATISTICS_KEYWORDS):
            if "probability" in text:
                return ProblemType.probability
            return ProblemType.statistics
        if self._contains_any(text, self.TRIG_KEYWORDS):
            return ProblemType.trigonometry
        if (
            self._contains_any(text, self.FUNCTION_KEYWORDS)
            or "f(x)" in text
            or "y =" in text
        ):
            return ProblemType.functions
        if any(char.isalpha() for char in text):
            return ProblemType.algebra
        return ProblemType.arithmetic

    def assess_difficulty(self, text: str, problem_type: ProblemType) -> Difficulty:
        lowered = text.lower()
        if self._contains_any(lowered, self.PROOF_KEYWORDS):
            return Difficulty.hard
        if problem_type in {ProblemType.geometry, ProblemType.coordinate_geometry}:
            standalone_points = re.findall(r"\b[A-Z]\b", text)
            compound_point_tokens = re.findall(r"\b[A-Z]{2,4}\b", text)
            point_labels = set(standalone_points)
            for token in compound_point_tokens:
                point_labels.update(token)
            if len(point_labels) >= 5:
                return Difficulty.hard
        if problem_type is ProblemType.calculus and any(
            word in lowered
            for word in ["prove", "show", "derive", "chứng minh", "chung minh"]
        ):
            return Difficulty.hard
        if text.count("?") + text.count(";") + text.count(",") > 6:
            return Difficulty.hard
        if len(text.split()) > 30:
            return Difficulty.medium
        if problem_type in {
            ProblemType.functions,
            ProblemType.calculus,
            ProblemType.statistics,
            ProblemType.trigonometry,
        }:
            return Difficulty.medium
        return Difficulty.easy

    def _contains_any(self, text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)
