"""Independent semantic prototype-bank construction and artifact validation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.semantic_router.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    AXIS_LABELS,
    SemanticExample,
)

DEFAULT_TEMPERATURES = {
    "problem_type": 0.10,
    "difficulty": 0.10,
    "visualization_environment": 0.10,
}


@dataclass(frozen=True)
class TermPrototype:
    example_id: str
    environment: str
    embedding: tuple[float, ...]
    terms: tuple[str, ...]


@dataclass(frozen=True)
class PrototypeArtifact:
    model_name: str
    embedding_dimension: int
    data_hash: str
    centroids: Mapping[str, Mapping[str, tuple[float, ...]]]
    label_counts: Mapping[str, Mapping[str, int]]
    temperatures: Mapping[str, float]
    recommended_confidence: Mapping[str, float]
    recommended_margin: Mapping[str, float]
    min_raw_similarity: float
    term_retrieval_min_similarity: float
    term_prototypes: tuple[TermPrototype, ...]
    artifact_version: str = ARTIFACT_SCHEMA_VERSION

    @property
    def identity(self) -> str:
        return f"{self.artifact_version}:{self.model_name}:{self.data_hash[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "data_hash": self.data_hash,
            "centroids": {
                axis: {label: list(vector) for label, vector in labels.items()}
                for axis, labels in self.centroids.items()
            },
            "label_counts": {
                axis: dict(labels) for axis, labels in self.label_counts.items()
            },
            "temperatures": dict(self.temperatures),
            "recommended_confidence": dict(self.recommended_confidence),
            "recommended_margin": dict(self.recommended_margin),
            "min_raw_similarity": self.min_raw_similarity,
            "term_retrieval_min_similarity": self.term_retrieval_min_similarity,
            "term_prototypes": [
                {
                    "example_id": item.example_id,
                    "environment": item.environment,
                    "embedding": list(item.embedding),
                    "terms": list(item.terms),
                }
                for item in self.term_prototypes
            ],
        }


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("embedding norm must be positive")
    return tuple(value / norm for value in values)


def mean_normalized(vectors: Iterable[Sequence[float]]) -> tuple[float, ...]:
    normalized = tuple(normalize_vector(vector) for vector in vectors)
    if not normalized:
        raise ValueError("cannot build a prototype from no examples")
    dimension = len(normalized[0])
    if any(len(vector) != dimension for vector in normalized):
        raise ValueError("prototype embeddings have inconsistent dimensions")
    return normalize_vector(
        [
            sum(vector[index] for vector in normalized) / len(normalized)
            for index in range(dimension)
        ]
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    normalized_left = normalize_vector(left)
    normalized_right = normalize_vector(right)
    if len(normalized_left) != len(normalized_right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(normalized_left, normalized_right, strict=True))


def _axis_label(example: SemanticExample, axis: str) -> str:
    if axis == "problem_type":
        return example.problem_type.value
    if axis == "difficulty":
        return example.difficulty.value
    if axis == "visualization_environment":
        return example.visualization_label
    raise ValueError(f"unsupported prototype axis: {axis}")


def build_prototype_artifact(
    examples: Iterable[SemanticExample],
    embeddings: Mapping[str, Sequence[float]],
    *,
    model_name: str,
    data_hash: str,
    temperatures: Mapping[str, float] | None = None,
    recommended_confidence: Mapping[str, float] | None = None,
    recommended_margin: Mapping[str, float] | None = None,
    min_raw_similarity: float = -1.0,
    term_retrieval_min_similarity: float = 0.55,
) -> PrototypeArtifact:
    rows = tuple(item for item in examples if item.in_scope)
    missing = sorted(item.id for item in rows if item.id not in embeddings)
    if missing:
        raise ValueError(f"missing embeddings for: {', '.join(missing[:5])}")
    if not rows:
        raise ValueError("prototype bank requires at least one in-scope example")

    first_vector = normalize_vector(embeddings[rows[0].id])
    dimension = len(first_vector)
    centroids: dict[str, dict[str, tuple[float, ...]]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    for axis, allowed_labels in AXIS_LABELS.items():
        by_label: dict[str, list[Sequence[float]]] = defaultdict(list)
        for row in rows:
            by_label[_axis_label(row, axis)].append(embeddings[row.id])
        missing_labels = [label for label in allowed_labels if not by_label[label]]
        if missing_labels:
            raise ValueError(
                f"prototype axis {axis} has no examples for: {', '.join(missing_labels)}"
            )
        centroids[axis] = {
            label: mean_normalized(by_label[label]) for label in allowed_labels
        }
        label_counts[axis] = {
            label: len(by_label[label]) for label in allowed_labels
        }

    term_prototypes = tuple(
        TermPrototype(
            example_id=row.id,
            environment=row.visualization_label,
            embedding=normalize_vector(embeddings[row.id]),
            terms=row.visualization_search_terms,
        )
        for row in rows
        if row.visualization_search_terms
    )
    return PrototypeArtifact(
        model_name=model_name,
        embedding_dimension=dimension,
        data_hash=data_hash,
        centroids=centroids,
        label_counts=label_counts,
        temperatures=dict(temperatures or DEFAULT_TEMPERATURES),
        recommended_confidence=dict(recommended_confidence or {}),
        recommended_margin=dict(recommended_margin or {}),
        min_raw_similarity=float(min_raw_similarity),
        term_retrieval_min_similarity=float(term_retrieval_min_similarity),
        term_prototypes=term_prototypes,
    )


def artifact_from_dict(payload: Mapping[str, Any]) -> PrototypeArtifact:
    if payload.get("artifact_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported semantic-router artifact version")
    model_name = payload.get("model_name")
    data_hash = payload.get("data_hash")
    dimension = payload.get("embedding_dimension")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("artifact model_name is required")
    if not isinstance(data_hash, str) or len(data_hash) != 64:
        raise ValueError("artifact data_hash must be a SHA-256 hex digest")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("artifact embedding_dimension must be positive")

    raw_centroids = payload.get("centroids")
    if not isinstance(raw_centroids, Mapping):
        raise ValueError("artifact centroids must be an object")
    centroids: dict[str, dict[str, tuple[float, ...]]] = {}
    for axis, allowed_labels in AXIS_LABELS.items():
        raw_labels = raw_centroids.get(axis)
        if not isinstance(raw_labels, Mapping):
            raise ValueError(f"artifact is missing centroid axis {axis}")
        if set(raw_labels) != set(allowed_labels):
            raise ValueError(f"artifact centroid labels do not match axis {axis}")
        centroids[axis] = {}
        for label in allowed_labels:
            raw_vector = raw_labels[label]
            if not isinstance(raw_vector, list):
                raise ValueError(f"artifact centroid {axis}/{label} must be an array")
            vector = normalize_vector(raw_vector)
            if len(vector) != dimension:
                raise ValueError(f"artifact centroid {axis}/{label} dimension mismatch")
            centroids[axis][label] = vector

    raw_counts = payload.get("label_counts", {})
    if not isinstance(raw_counts, Mapping):
        raise ValueError("artifact label_counts must be an object")
    label_counts = {
        axis: {label: int(raw_counts.get(axis, {}).get(label, 0)) for label in labels}
        for axis, labels in AXIS_LABELS.items()
    }
    temperatures = _float_mapping(payload.get("temperatures", {}), "temperatures")
    confidence = _float_mapping(
        payload.get("recommended_confidence", {}), "recommended_confidence"
    )
    margin = _float_mapping(
        payload.get("recommended_margin", {}), "recommended_margin"
    )

    raw_terms = payload.get("term_prototypes", [])
    if not isinstance(raw_terms, list):
        raise ValueError("artifact term_prototypes must be an array")
    term_prototypes: list[TermPrototype] = []
    for item in raw_terms:
        if not isinstance(item, Mapping):
            raise ValueError("term prototype must be an object")
        vector = normalize_vector(item.get("embedding", []))
        if len(vector) != dimension:
            raise ValueError("term prototype dimension mismatch")
        terms = item.get("terms", [])
        if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
            raise ValueError("term prototype terms must be strings")
        term_prototypes.append(
            TermPrototype(
                example_id=str(item.get("example_id", "")),
                environment=str(item.get("environment", "")),
                embedding=vector,
                terms=tuple(term.strip() for term in terms if term.strip()),
            )
        )

    return PrototypeArtifact(
        model_name=model_name.strip(),
        embedding_dimension=dimension,
        data_hash=data_hash,
        centroids=centroids,
        label_counts=label_counts,
        temperatures=temperatures or DEFAULT_TEMPERATURES,
        recommended_confidence=confidence,
        recommended_margin=margin,
        min_raw_similarity=float(payload.get("min_raw_similarity", -1.0)),
        term_retrieval_min_similarity=float(
            payload.get("term_retrieval_min_similarity", 0.55)
        ),
        term_prototypes=tuple(term_prototypes),
    )


def _float_mapping(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact {field} must be an object")
    return {str(key): float(item) for key, item in value.items()}


__all__ = [
    "DEFAULT_TEMPERATURES",
    "PrototypeArtifact",
    "TermPrototype",
    "artifact_from_dict",
    "build_prototype_artifact",
    "cosine_similarity",
    "mean_normalized",
    "normalize_vector",
]
