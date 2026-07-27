"""Deterministic, independent per-axis pair generation for optional training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Literal

from app.semantic_router.contracts import SemanticExample

TrainingAxis = Literal[
    "problem_type",
    "difficulty",
    "visualization_environment",
]

_HARD_NEGATIVE_LABEL_PAIRS: dict[TrainingAxis, set[frozenset[str]]] = {
    "problem_type": {
        frozenset({"arithmetic", "algebra"}),
        frozenset({"statistics", "probability"}),
        frozenset({"calculus", "algebra"}),
        frozenset({"geometry", "algebra"}),
    },
    "difficulty": {
        frozenset({"easy", "medium"}),
        frozenset({"medium", "hard"}),
    },
    "visualization_environment": {
        frozenset({"geometry_2d", "graphing"}),
        frozenset({"probability", "statistics"}),
        frozenset({"cas", "graphing"}),
        frozenset({"none", "cas"}),
    },
}


@dataclass(frozen=True)
class AxisTrainingPair:
    axis: TrainingAxis
    left_id: str
    right_id: str
    similarity: float
    kind: str


def axis_label(example: SemanticExample, axis: TrainingAxis) -> str:
    if axis == "problem_type":
        return example.problem_type.value
    if axis == "difficulty":
        return example.difficulty.value
    return example.visualization_label


def is_hard_negative(
    left: SemanticExample,
    right: SemanticExample,
    axis: TrainingAxis,
) -> bool:
    labels = frozenset({axis_label(left, axis), axis_label(right, axis)})
    return labels in _HARD_NEGATIVE_LABEL_PAIRS[axis]


def build_axis_training_pairs(
    examples: Iterable[SemanticExample],
    *,
    axis: TrainingAxis,
    seed: int = 42,
    max_positive_per_example: int = 8,
    max_negative_per_example: int = 8,
) -> tuple[AxisTrainingPair, ...]:
    """Build binary pairs for one axis without a cross-axis composite target."""
    rows = tuple(
        sorted((item for item in examples if item.in_scope), key=lambda item: item.id)
    )
    randomizer = random.Random(f"{seed}:{axis}")
    pairs: dict[tuple[str, str], AxisTrainingPair] = {}

    def add(left: SemanticExample, right: SemanticExample, kind: str) -> None:
        key = tuple(sorted((left.id, right.id)))
        similarity = float(axis_label(left, axis) == axis_label(right, axis))
        candidate = AxisTrainingPair(
            axis=axis,
            left_id=key[0],
            right_id=key[1],
            similarity=similarity,
            kind=kind,
        )
        current = pairs.get(key)
        priority = {"negative": 0, "positive": 1, "hard_negative": 2, "group_positive": 3}
        if current is None or priority[kind] > priority[current.kind]:
            pairs[key] = candidate

    by_group: dict[str, list[SemanticExample]] = {}
    for row in rows:
        by_group.setdefault(row.group_id, []).append(row)
    for group_rows in by_group.values():
        for left, right in combinations(group_rows, 2):
            add(left, right, "group_positive")

    for left in rows:
        positives = [
            right
            for right in rows
            if right.id != left.id
            and right.group_id != left.group_id
            and axis_label(right, axis) == axis_label(left, axis)
        ]
        negatives = [
            right
            for right in rows
            if axis_label(right, axis) != axis_label(left, axis)
        ]
        randomizer.shuffle(positives)
        randomizer.shuffle(negatives)
        for right in positives[:max_positive_per_example]:
            add(left, right, "positive")
        hard_negatives = [
            right for right in negatives if is_hard_negative(left, right, axis)
        ]
        ordinary_negatives = [
            right for right in negatives if right not in hard_negatives
        ]
        chosen_negatives = (
            hard_negatives[:max_negative_per_example]
            + ordinary_negatives[:max_negative_per_example]
        )
        for right in chosen_negatives:
            add(
                left,
                right,
                "hard_negative" if right in hard_negatives else "negative",
            )

    return tuple(
        sorted(
            pairs.values(),
            key=lambda item: (item.kind, item.left_id, item.right_id),
        )
    )


__all__ = [
    "AxisTrainingPair",
    "TrainingAxis",
    "axis_label",
    "build_axis_training_pairs",
    "is_hard_negative",
]
