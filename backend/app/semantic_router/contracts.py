"""Shared contracts for semantic-router data, artifacts, and runtime scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment

ARTIFACT_SCHEMA_VERSION = "1.0"
DATASET_SCHEMA_VERSION = "1.0"
VISUALIZATION_NONE = "none"


class ReviewStatus(str, Enum):
    needs_review = "needs_review"
    human_reviewed = "human_reviewed"


@dataclass(frozen=True)
class SemanticExample:
    id: str
    text: str
    language: str
    problem_type: ProblemType
    difficulty: Difficulty
    visualization_environment: VisualizationEnvironment | None
    group_id: str
    source: str
    review_status: ReviewStatus
    visualization_search_terms: tuple[str, ...] = ()
    in_scope: bool = True
    source_id: str | None = None

    @property
    def visualization_label(self) -> str:
        if self.visualization_environment is None:
            return VISUALIZATION_NONE
        return self.visualization_environment.value

    @property
    def label_tuple(self) -> tuple[str, str, str]:
        return (
            self.problem_type.value,
            self.difficulty.value,
            self.visualization_label,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "problem_type": self.problem_type.value,
            "difficulty": self.difficulty.value,
            "visualization_environment": self.visualization_label,
            "group_id": self.group_id,
            "source": self.source,
            "review_status": self.review_status.value,
            "in_scope": self.in_scope,
        }
        if self.visualization_search_terms:
            payload["visualization_search_terms"] = list(
                self.visualization_search_terms
            )
        if self.source_id:
            payload["source_id"] = self.source_id
        return payload


@dataclass(frozen=True)
class AxisScore:
    label: str
    confidence: float
    runner_up: str
    margin: float
    raw_similarity: float
    scores: tuple[tuple[str, float], ...]


AXIS_LABELS: dict[str, tuple[str, ...]] = {
    "problem_type": tuple(item.value for item in ProblemType),
    "difficulty": tuple(item.value for item in Difficulty),
    "visualization_environment": (
        *(item.value for item in VisualizationEnvironment),
        VISUALIZATION_NONE,
    ),
}

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "AXIS_LABELS",
    "AxisScore",
    "DATASET_SCHEMA_VERSION",
    "ReviewStatus",
    "SemanticExample",
    "VISUALIZATION_NONE",
]
