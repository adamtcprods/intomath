from __future__ import annotations

import asyncio
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Sequence

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from app.api.v1.endpoints.health import _semantic_router_health
from app.semantic_router.contracts import AXIS_LABELS
from app.semantic_router.prototypes import (
    PrototypeArtifact,
    TermPrototype,
    normalize_vector,
)
from app.services.semantic_router import (
    SemanticAbstention,
    SemanticClassification,
    SemanticRouter,
)

DIMENSION = 21
PROBLEM_OFFSET = 0
DIFFICULTY_OFFSET = 9
ENVIRONMENT_OFFSET = 12


def _one_hot(index: int) -> tuple[float, ...]:
    values = [0.0] * DIMENSION
    values[index] = 1.0
    return tuple(values)


def _query(
    problem_type: str,
    difficulty: str,
    environment: str,
) -> tuple[float, ...]:
    values = [0.0] * DIMENSION
    values[PROBLEM_OFFSET + AXIS_LABELS["problem_type"].index(problem_type)] = 1.0
    values[DIFFICULTY_OFFSET + AXIS_LABELS["difficulty"].index(difficulty)] = 1.0
    values[
        ENVIRONMENT_OFFSET
        + AXIS_LABELS["visualization_environment"].index(environment)
    ] = 1.0
    return normalize_vector(values)


def _artifact(*, with_terms: bool = True) -> PrototypeArtifact:
    centroids = {
        "problem_type": {
            label: _one_hot(PROBLEM_OFFSET + index)
            for index, label in enumerate(AXIS_LABELS["problem_type"])
        },
        "difficulty": {
            label: _one_hot(DIFFICULTY_OFFSET + index)
            for index, label in enumerate(AXIS_LABELS["difficulty"])
        },
        "visualization_environment": {
            label: _one_hot(ENVIRONMENT_OFFSET + index)
            for index, label in enumerate(
                AXIS_LABELS["visualization_environment"]
            )
        },
    }
    terms = (
        TermPrototype(
            example_id="geometry-term",
            environment="geometry_2d",
            embedding=_query("geometry", "medium", "geometry_2d"),
            terms=("Midpoint", "Perpendicular", "Segment"),
        ),
    ) if with_terms else ()
    return PrototypeArtifact(
        model_name="fake-multilingual-model",
        embedding_dimension=DIMENSION,
        data_hash="0" * 64,
        centroids=centroids,
        label_counts={
            axis: {label: 3 for label in labels}
            for axis, labels in AXIS_LABELS.items()
        },
        temperatures={axis: 0.10 for axis in AXIS_LABELS},
        recommended_confidence={},
        recommended_margin={},
        min_raw_similarity=0.20,
        term_retrieval_min_similarity=0.50,
        term_prototypes=terms,
    )


class FakeEmbeddingBackend:
    model_name = "fake-multilingual-model"
    dimension = DIMENSION

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.encode_calls = 0

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.encode_calls += 1
        return [self.vectors[text] for text in texts]


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "semantic_router_enabled": True,
        "semantic_router_model": "fake-multilingual-model",
        "semantic_router_model_path": "",
        "semantic_router_artifact_path": "",
        "semantic_router_device": "cpu",
        "semantic_router_max_text_chars": 4_000,
        "semantic_router_min_confidence": 0.60,
        "semantic_router_min_margin": 0.05,
        "semantic_router_min_raw_similarity": 0.20,
        "semantic_router_term_min_similarity": 0.50,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _router(
    vectors: dict[str, tuple[float, ...]],
    *,
    artifact: PrototypeArtifact | None = None,
    settings: SimpleNamespace | None = None,
) -> tuple[SemanticRouter, FakeEmbeddingBackend, list[int]]:
    backend = FakeEmbeddingBackend(vectors)
    factory_calls: list[int] = []

    def factory(_: str, __: str) -> FakeEmbeddingBackend:
        factory_calls.append(1)
        return backend

    return (
        SemanticRouter(
            settings or _settings(),
            backend_factory=factory,
            prototype_artifact=artifact or _artifact(),
        ),
        backend,
        factory_calls,
    )


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Construct a perpendicular bisector", "en"),
        ("Dựng đường trung trực", "vi"),
        ("Construct đường trung trực AB", "mixed"),
        ("dung duong trung truc AB nhe", "vi"),
        ("AB ⟂ CD và M=(A+B)/2", "mixed"),
    ],
)
def test_multilingual_notation_typo_and_mixed_prompts_classify(
    text: str,
    language: str,
) -> None:
    vector = _query("geometry", "medium", "geometry_2d")
    router, _, _ = _router({text: vector})

    result = router.classify(text, language=language)

    assert isinstance(result, SemanticClassification)
    assert result.problem_type.value == "geometry"
    assert result.difficulty.value == "medium"
    assert result.visualization_environment.value == "geometry_2d"
    assert result.confident_axes == {
        "problem_type",
        "difficulty",
        "visualization_environment",
    }
    assert result.language == language
    assert result.used_fallback is False


@pytest.mark.parametrize(
    ("first", "second", "expected_first", "expected_second"),
    [
        (
            _query("geometry", "medium", "geometry_2d"),
            _query("algebra", "medium", "graphing"),
            ("geometry", "geometry_2d"),
            ("algebra", "graphing"),
        ),
        (
            _query("arithmetic", "easy", "none"),
            _query("algebra", "easy", "none"),
            ("arithmetic", None),
            ("algebra", None),
        ),
        (
            _query("statistics", "medium", "statistics"),
            _query("probability", "medium", "probability"),
            ("statistics", "statistics"),
            ("probability", "probability"),
        ),
    ],
)
def test_confused_classes_are_scored_independently(
    first: tuple[float, ...],
    second: tuple[float, ...],
    expected_first: tuple[str, str | None],
    expected_second: tuple[str, str | None],
) -> None:
    router, _, _ = _router({"first": first, "second": second})

    first_result = router.classify("first")
    second_result = router.classify("second")

    assert isinstance(first_result, SemanticClassification)
    assert isinstance(second_result, SemanticClassification)
    assert first_result.problem_type.value == expected_first[0]
    assert (
        first_result.visualization_environment.value
        if first_result.visualization_environment
        else None
    ) == expected_first[1]
    assert second_result.problem_type.value == expected_second[0]
    assert (
        second_result.visualization_environment.value
        if second_result.visualization_environment
        else None
    ) == expected_second[1]


def test_low_confidence_and_top_score_margin_are_axis_specific() -> None:
    values = [0.0] * DIMENSION
    for index in range(len(AXIS_LABELS["problem_type"])):
        values[PROBLEM_OFFSET + index] = 1.0
    values[DIFFICULTY_OFFSET + 1] = 1.0
    values[ENVIRONMENT_OFFSET + 7] = 1.0
    router, _, _ = _router({"ambiguous": normalize_vector(values)})

    result = router.classify("ambiguous")

    assert isinstance(result, SemanticClassification)
    problem_score = result.axis_score("problem_type")
    assert problem_score.margin == pytest.approx(0.0)
    assert problem_score.confidence == pytest.approx(1 / 9)
    assert result.is_confident("problem_type") is False
    assert result.is_confident("difficulty") is True
    assert result.is_confident("visualization_environment") is True


def test_uncertain_difficulty_does_not_abstain_other_axes() -> None:
    values = list(_query("calculus", "medium", "cas"))
    values[DIFFICULTY_OFFSET] = values[DIFFICULTY_OFFSET + 1]
    router, _, _ = _router({"derivative": normalize_vector(values)})

    result = router.classify("derivative")

    assert isinstance(result, SemanticClassification)
    assert result.is_confident("problem_type") is True
    assert result.is_confident("visualization_environment") is True
    assert result.is_confident("difficulty") is False


def test_out_of_distribution_and_overlong_inputs_abstain() -> None:
    ood = _one_hot(DIMENSION - 1)
    router, backend, _ = _router({"weather": ood})

    ood_result = router.classify("weather")
    long_result = router.classify("x" * 4_001)

    assert isinstance(ood_result, SemanticAbstention)
    assert ood_result.reason == "out_of_distribution"
    assert isinstance(long_result, SemanticAbstention)
    assert long_result.reason == "too_long"
    assert backend.encode_calls == 1


def test_model_unavailable_is_cached_and_not_retried() -> None:
    calls = 0

    def unavailable(_: str, __: str) -> FakeEmbeddingBackend:
        nonlocal calls
        calls += 1
        raise RuntimeError("missing local model")

    router = SemanticRouter(
        _settings(),
        backend_factory=unavailable,
        prototype_artifact=_artifact(),
    )

    first = router.classify("one")
    second = router.classify("two")

    assert isinstance(first, SemanticAbstention)
    assert isinstance(second, SemanticAbstention)
    assert first.reason == second.reason == "model_unavailable"
    assert calls == 1
    assert router.status().state == "unavailable"


def test_model_and_prototypes_load_once_under_concurrency() -> None:
    vector = _query("algebra", "medium", "graphing")
    backend = FakeEmbeddingBackend({f"item-{index}": vector for index in range(12)})
    factory_calls = 0

    def delayed_factory(_: str, __: str) -> FakeEmbeddingBackend:
        nonlocal factory_calls
        factory_calls += 1
        time.sleep(0.02)
        return backend

    router = SemanticRouter(
        _settings(),
        backend_factory=delayed_factory,
        prototype_artifact=_artifact(),
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(router.classify, [f"item-{index}" for index in range(12)])
        )

    assert all(isinstance(result, SemanticClassification) for result in results)
    assert factory_calls == 1
    assert router.status().model_load_count == 1
    assert router.status().prototype_load_count == 1


def test_search_term_retrieval_is_separate_from_classification() -> None:
    vector = _query("geometry", "medium", "geometry_2d")
    with_terms, _, _ = _router({"geometry": vector})
    without_terms, _, _ = _router(
        {"geometry": vector}, artifact=_artifact(with_terms=False)
    )

    retrieved = with_terms.classify("geometry")
    absent = without_terms.classify("geometry")

    assert isinstance(retrieved, SemanticClassification)
    assert isinstance(absent, SemanticClassification)
    assert retrieved.visualization_search_terms == (
        "Midpoint",
        "Perpendicular",
        "Segment",
    )
    assert absent.visualization_search_terms == ()
    assert retrieved.problem_type == absent.problem_type
    assert retrieved.confident_axes == absent.confident_axes


def test_embedding_classification_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _query("probability", "medium", "probability")
    router, _, _ = _router({"local only": vector})

    def forbidden_socket(*_: object, **__: object) -> socket.socket:
        raise AssertionError("network access is forbidden")

    async def classify_without_network() -> object:
        monkeypatch.setattr(socket, "socket", forbidden_socket)
        return await router.classify_async("local only", language="en")

    result = asyncio.run(classify_without_network())

    assert isinstance(result, SemanticClassification)
    assert result.problem_type.value == "probability"


def _health_request(router: SemanticRouter) -> Request:
    app = FastAPI()
    app.state.model_clients = SimpleNamespace(semantic_router=router)
    return Request({"type": "http", "app": app})


def test_health_reports_ready_unavailable_and_disabled_states() -> None:
    vector = _query("geometry", "medium", "geometry_2d")
    ready_router, _, _ = _router({"ready": vector})
    ready_router.classify("ready")
    ready, ready_degraded = _semantic_router_health(_health_request(ready_router))

    def unavailable(_: str, __: str) -> FakeEmbeddingBackend:
        raise RuntimeError("missing")

    unavailable_router = SemanticRouter(
        _settings(),
        backend_factory=unavailable,
        prototype_artifact=_artifact(),
    )
    unavailable_router.initialize()
    missing, missing_degraded = _semantic_router_health(
        _health_request(unavailable_router)
    )

    disabled_router = SemanticRouter(
        _settings(semantic_router_enabled=False),
        prototype_artifact=_artifact(),
    )
    disabled, disabled_degraded = _semantic_router_health(
        _health_request(disabled_router)
    )

    assert ready.status == "ready"
    assert ready.model_loaded is True
    assert ready.model_path == "fake-multilingual-model"
    assert ready.model_version
    assert ready_degraded is False
    assert missing.status == "unavailable"
    assert missing.model_loaded is False
    assert missing_degraded is True
    assert disabled.status == "disabled"
    assert disabled_degraded is False
