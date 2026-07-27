"""Optional multilingual embedding router with independent prototype banks."""

from __future__ import annotations

import asyncio
import json
import math
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from app.core.config import get_settings
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.semantic_router.contracts import AxisScore
from app.semantic_router.dataset import load_dataset, normalize_text
from app.semantic_router.prototypes import (
    PrototypeArtifact,
    artifact_from_dict,
    build_prototype_artifact,
    cosine_similarity,
    normalize_vector,
)


class EmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEmbeddingBackend:
    """Local-files-only sentence-transformers adapter.

    Model acquisition is an explicit setup/training action. Runtime requests never
    download a model from Hugging Face or call another external service.
    """

    def __init__(self, source: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers runtime extra is not installed") from exc
        self._model = SentenceTransformer(
            source,
            device=device,
            local_files_only=True,
        )
        self._model_name = source
        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("embedding model did not report a valid dimension")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._model.encode(
            list(texts),
            batch_size=min(64, max(1, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True)
class SemanticClassification:
    problem_type: ProblemType
    difficulty: Difficulty
    visualization_environment: VisualizationEnvironment | None
    confidence: float
    margin: float
    reason: str
    model_name: str
    used_fallback: bool
    language: str
    latency_ms: float
    axis_scores: tuple[tuple[str, AxisScore], ...]
    confident_axes: frozenset[str]
    visualization_search_terms: tuple[str, ...] = ()

    def axis_score(self, axis: str) -> AxisScore:
        for name, score in self.axis_scores:
            if name == axis:
                return score
        raise KeyError(axis)

    def is_confident(self, axis: str) -> bool:
        return axis in self.confident_axes


@dataclass(frozen=True)
class SemanticAbstention:
    reason: str
    model_name: str
    used_fallback: bool
    language: str
    latency_ms: float


SemanticRouterResult = SemanticClassification | SemanticAbstention


@dataclass(frozen=True)
class SemanticRouterStatus:
    enabled: bool
    state: str
    model_source: str
    model_path: str | None
    model_name: str | None
    model_version: str | None
    artifact_path: str | None
    embedding_dimension: int | None
    prototype_count: int
    model_load_count: int
    prototype_load_count: int
    load_time_ms: float | None
    approximate_memory_mb: float | None
    error_category: str | None


BackendFactory = Callable[[str, str], EmbeddingBackend]


class SemanticRouter:
    """Process-local multilingual classifier backed by learned examples."""

    def __init__(
        self,
        settings: Any | None = None,
        *,
        backend_factory: BackendFactory | None = None,
        prototype_artifact: PrototypeArtifact | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._backend_factory = backend_factory or SentenceTransformerEmbeddingBackend
        self._provided_artifact = prototype_artifact
        self._data_dir = Path(
            data_dir
            or getattr(
                self.settings,
                "semantic_router_data_path",
                Path(__file__).resolve().parents[2] / "data" / "semantic_router",
            )
        )
        configured_path = str(
            getattr(self.settings, "semantic_router_model_path", "") or ""
        ).strip()
        configured_model = str(
            getattr(
                self.settings,
                "semantic_router_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        ).strip()
        self._model_path = configured_path or None
        self._model_source = configured_path or configured_model
        self._artifact_path = self._resolve_artifact_path()
        self._state = (
            "uninitialized"
            if bool(getattr(self.settings, "semantic_router_enabled", True))
            else "disabled"
        )
        self._backend: EmbeddingBackend | None = None
        self._artifact: PrototypeArtifact | None = None
        self._error_category: str | None = None
        self._load_time_ms: float | None = None
        self._approximate_memory_mb: float | None = None
        self._model_load_count = 0
        self._prototype_load_count = 0
        self._initialization_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "semantic_router_enabled", True))

    @property
    def artifact_identity(self) -> str:
        artifact = self._artifact
        return artifact.identity if artifact is not None else "unavailable"

    def initialize(self) -> SemanticRouterStatus:
        if not self.enabled:
            return self.status()
        with self._initialization_lock:
            if self._state in {"ready", "unavailable"}:
                return self.status()
            self._state = "loading"
            started = time.monotonic()
            memory_before = _maximum_resident_memory_mb()
            try:
                if self._model_path and not Path(self._model_path).is_dir():
                    raise FileNotFoundError("configured semantic-router model path is missing")
                backend = self._backend_factory(
                    self._model_source,
                    str(getattr(self.settings, "semantic_router_device", "cpu")),
                )
                self._model_load_count += 1
                artifact = self._load_or_build_artifact(backend)
                if backend.dimension != artifact.embedding_dimension:
                    raise ValueError("embedding model and prototype dimensions do not match")
                self._backend = backend
                self._artifact = artifact
                self._state = "ready"
            except Exception as exc:
                self._backend = None
                self._artifact = None
                self._state = "unavailable"
                self._error_category = type(exc).__name__
            finally:
                self._load_time_ms = (time.monotonic() - started) * 1000.0
                memory_after = _maximum_resident_memory_mb()
                self._approximate_memory_mb = max(0.0, memory_after - memory_before)
            return self.status()

    async def initialize_async(self) -> SemanticRouterStatus:
        return await asyncio.to_thread(self.initialize)

    def status(self) -> SemanticRouterStatus:
        artifact = self._artifact
        backend = self._backend
        return SemanticRouterStatus(
            enabled=self.enabled,
            state=self._state,
            model_source=self._model_source,
            model_path=self._model_path,
            model_name=backend.model_name if backend is not None else None,
            model_version=artifact.identity if artifact is not None else None,
            artifact_path=str(self._artifact_path) if self._artifact_path else None,
            embedding_dimension=(
                artifact.embedding_dimension if artifact is not None else None
            ),
            prototype_count=(
                sum(len(labels) for labels in artifact.centroids.values())
                if artifact is not None
                else 0
            ),
            model_load_count=self._model_load_count,
            prototype_load_count=self._prototype_load_count,
            load_time_ms=self._load_time_ms,
            approximate_memory_mb=self._approximate_memory_mb,
            error_category=self._error_category,
        )

    def classify(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> SemanticRouterResult:
        started = time.monotonic()
        normalized = normalize_text(text)
        language_label = (language or "unknown").strip() or "unknown"
        if not normalized:
            return self._abstain("empty_input", language_label, started)
        max_chars = int(getattr(self.settings, "semantic_router_max_text_chars", 4_000))
        if len(normalized) > max_chars:
            return self._abstain("too_long", language_label, started)
        status = self.initialize()
        if status.state != "ready":
            return self._abstain("model_unavailable", language_label, started)

        backend = self._backend
        artifact = self._artifact
        if backend is None or artifact is None:
            return self._abstain("model_unavailable", language_label, started)
        try:
            with self._encode_lock:
                encoded = backend.encode([normalized])
            if len(encoded) != 1:
                raise ValueError("embedding backend returned an unexpected batch size")
            query = normalize_vector(encoded[0])
            if len(query) != artifact.embedding_dimension:
                raise ValueError("query embedding dimension does not match prototypes")
            axis_scores = tuple(
                (axis, self._score_axis(query, artifact, axis))
                for axis in (
                    "problem_type",
                    "difficulty",
                    "visualization_environment",
                )
            )
        except Exception:
            return self._abstain("embedding_error", language_label, started)

        score_map = dict(axis_scores)
        nearest_similarity = score_map["problem_type"].raw_similarity
        minimum_similarity = max(
            float(getattr(self.settings, "semantic_router_min_raw_similarity", 0.20)),
            artifact.min_raw_similarity,
        )
        if nearest_similarity < minimum_similarity:
            return self._abstain(
                "out_of_distribution",
                language_label,
                started,
                model_name=backend.model_name,
            )

        confident_axes = frozenset(
            axis
            for axis, score in axis_scores
            if self._axis_is_confident(axis, score, artifact)
        )
        problem_type = ProblemType(score_map["problem_type"].label)
        difficulty = Difficulty(score_map["difficulty"].label)
        environment_label = score_map["visualization_environment"].label
        environment = (
            None
            if environment_label == "none"
            else VisualizationEnvironment(environment_label)
        )
        uncertain_axes = [
            axis for axis, _ in axis_scores if axis not in confident_axes
        ]
        reason = "independent multilingual prototype classification"
        if uncertain_axes:
            reason += f"; uncertain axes: {', '.join(uncertain_axes)}"
        terms = self._retrieve_terms(query, environment, artifact)
        required_scores = tuple(score_map.values())
        return SemanticClassification(
            problem_type=problem_type,
            difficulty=difficulty,
            visualization_environment=environment,
            confidence=min(score.confidence for score in required_scores),
            margin=min(score.margin for score in required_scores),
            reason=reason,
            model_name=backend.model_name,
            used_fallback=False,
            language=language_label,
            latency_ms=(time.monotonic() - started) * 1000.0,
            axis_scores=axis_scores,
            confident_axes=confident_axes,
            visualization_search_terms=terms,
        )

    async def classify_async(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> SemanticRouterResult:
        return await asyncio.to_thread(self.classify, text, language=language)

    def _load_or_build_artifact(
        self,
        backend: EmbeddingBackend,
    ) -> PrototypeArtifact:
        self._prototype_load_count += 1
        if self._provided_artifact is not None:
            return self._provided_artifact
        if self._artifact_path is not None and self._artifact_path.is_file():
            payload = json.loads(self._artifact_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("semantic-router prototype artifact must be an object")
            return artifact_from_dict(payload)

        dataset = load_dataset(self._data_dir)
        train = dataset.splits["train"]
        with self._encode_lock:
            vectors = backend.encode([row.text for row in train])
        if len(vectors) != len(train):
            raise ValueError("embedding backend returned an unexpected prototype batch")
        embeddings = {
            row.id: vector for row, vector in zip(train, vectors, strict=True)
        }
        return build_prototype_artifact(
            train,
            embeddings,
            model_name=backend.model_name,
            data_hash=dataset.data_hash,
            min_raw_similarity=float(
                getattr(self.settings, "semantic_router_min_raw_similarity", 0.20)
            ),
            term_retrieval_min_similarity=float(
                getattr(self.settings, "semantic_router_term_min_similarity", 0.55)
            ),
        )

    def _resolve_artifact_path(self) -> Path | None:
        configured = str(
            getattr(self.settings, "semantic_router_artifact_path", "") or ""
        ).strip()
        if configured:
            return Path(configured)
        if not self._model_path:
            return None
        model_path = Path(self._model_path)
        candidates = (
            model_path / "prototypes.json",
            model_path.parent / "prototypes.json",
        )
        return next((path for path in candidates if path.is_file()), candidates[-1])

    def _score_axis(
        self,
        query: Sequence[float],
        artifact: PrototypeArtifact,
        axis: str,
    ) -> AxisScore:
        similarities = {
            label: cosine_similarity(query, centroid)
            for label, centroid in artifact.centroids[axis].items()
        }
        temperature = max(1e-6, float(artifact.temperatures.get(axis, 0.10)))
        maximum_logit = max(similarities.values()) / temperature
        exponentials = {
            label: math.exp(similarity / temperature - maximum_logit)
            for label, similarity in similarities.items()
        }
        denominator = sum(exponentials.values())
        probabilities = {
            label: value / denominator for label, value in exponentials.items()
        }
        ordered = sorted(
            probabilities.items(), key=lambda item: (-item[1], item[0])
        )
        top_label, top_probability = ordered[0]
        runner_label, runner_probability = ordered[1]
        return AxisScore(
            label=top_label,
            confidence=top_probability,
            runner_up=runner_label,
            margin=top_probability - runner_probability,
            raw_similarity=similarities[top_label],
            scores=tuple(ordered),
        )

    def _axis_is_confident(
        self,
        axis: str,
        score: AxisScore,
        artifact: PrototypeArtifact,
    ) -> bool:
        confidence_threshold = max(
            float(getattr(self.settings, "semantic_router_min_confidence", 0.60)),
            float(artifact.recommended_confidence.get(axis, 0.0)),
        )
        margin_threshold = max(
            float(getattr(self.settings, "semantic_router_min_margin", 0.05)),
            float(artifact.recommended_margin.get(axis, 0.0)),
        )
        return (
            score.confidence >= confidence_threshold
            and score.margin >= margin_threshold
        )

    def _retrieve_terms(
        self,
        query: Sequence[float],
        environment: VisualizationEnvironment | None,
        artifact: PrototypeArtifact,
    ) -> tuple[str, ...]:
        if environment is None:
            return ()
        threshold = max(
            artifact.term_retrieval_min_similarity,
            float(getattr(self.settings, "semantic_router_term_min_similarity", 0.55)),
        )
        candidates = sorted(
            (
                (cosine_similarity(query, item.embedding), item)
                for item in artifact.term_prototypes
                if item.environment == environment.value
            ),
            key=lambda item: (-item[0], item[1].example_id),
        )
        terms: list[str] = []
        for similarity, prototype in candidates[:3]:
            if similarity < threshold:
                continue
            for term in prototype.terms:
                if term not in terms:
                    terms.append(term)
                if len(terms) == 8:
                    return tuple(terms)
        return tuple(terms)

    def _abstain(
        self,
        reason: str,
        language: str,
        started: float,
        *,
        model_name: str | None = None,
    ) -> SemanticAbstention:
        return SemanticAbstention(
            reason=reason,
            model_name=model_name or self._model_source,
            used_fallback=False,
            language=language,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )


def _maximum_resident_memory_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. IntoMath currently deploys on Linux,
    # but keep the approximation portable for local evaluation.
    return usage / (1024.0 * 1024.0) if usage > 10_000_000 else usage / 1024.0


__all__ = [
    "EmbeddingBackend",
    "SemanticAbstention",
    "SemanticClassification",
    "SemanticRouter",
    "SemanticRouterResult",
    "SemanticRouterStatus",
    "SentenceTransformerEmbeddingBackend",
]
