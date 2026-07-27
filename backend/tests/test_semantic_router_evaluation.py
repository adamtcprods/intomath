from __future__ import annotations

import json
from pathlib import Path

from app.semantic_router.contracts import AXIS_LABELS, SemanticExample
from app.semantic_router.dataset import load_dataset
from app.semantic_router.evaluation import (
    calibrate_artifact,
    evaluate_embeddings,
    evaluate_promotion,
)
from app.semantic_router.prototypes import (
    build_prototype_artifact,
    normalize_vector,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semantic_router"
DIMENSION = 21


def _embedding(example: SemanticExample) -> tuple[float, ...]:
    if not example.in_scope:
        values = [0.0] * DIMENSION
        values[-1] = 1.0
        return tuple(values)
    values = [0.0] * DIMENSION
    values[AXIS_LABELS["problem_type"].index(example.problem_type.value)] = 1.0
    values[9 + AXIS_LABELS["difficulty"].index(example.difficulty.value)] = 1.0
    values[
        12
        + AXIS_LABELS["visualization_environment"].index(
            example.visualization_label
        )
    ] = 1.0
    return normalize_vector(values)


def test_base_prototype_calibration_is_validation_only_and_reproducible() -> None:
    dataset = load_dataset(DATA_DIR)
    embeddings = {
        split_name: {row.id: _embedding(row) for row in rows}
        for split_name, rows in dataset.splits.items()
    }
    base = build_prototype_artifact(
        dataset.splits["train"],
        embeddings["train"],
        model_name="fake-untouched-base",
        data_hash=dataset.data_hash,
        min_raw_similarity=0.20,
    )

    first = calibrate_artifact(
        base,
        dataset.splits["validation"],
        embeddings["validation"],
    )
    second = calibrate_artifact(
        base,
        dataset.splits["validation"],
        embeddings["validation"],
    )
    report = evaluate_embeddings(
        dataset.splits["test"],
        embeddings["test"],
        first,
    )

    assert first.to_dict() == second.to_dict()
    assert set(first.temperatures) == set(AXIS_LABELS)
    assert set(first.recommended_confidence) == set(AXIS_LABELS)
    assert set(first.recommended_margin) == set(AXIS_LABELS)
    assert first.min_raw_similarity > 0.0
    assert report["total_examples"] == 36
    assert report["out_of_scope_examples"] == 3
    assert report["ood_recall"] == 1.0
    assert report["chosen_coverage"] == 0.70
    assert report["realized_chosen_coverage"] >= 0.70
    assert 0.0 <= report["selective_accuracy_at_chosen_coverage"] <= 1.0
    assert set(report["accuracy_by_language"]) == {"en", "mixed", "vi"}
    assert set(report["axes"]) == set(AXIS_LABELS)
    assert set(report["axes"]["problem_type"]["confusion_matrix"]) == set(
        AXIS_LABELS["problem_type"]
    )


def _promotion_report(value: float) -> dict[str, object]:
    return {
        "axes": {
            "problem_type": {"macro_f1": value},
            "visualization_environment": {"macro_f1": value},
        },
        "accuracy_by_language": {
            "en": value,
            "vi": value,
            "mixed": value,
        },
        "selective_accuracy_at_chosen_coverage": value,
    }


def test_promotion_requires_every_quality_and_resource_gate() -> None:
    promoted = evaluate_promotion(
        _promotion_report(0.70),
        _promotion_report(0.75),
        base_performance={"warm_latency_ms": 10.0, "approximate_memory_mb": 100.0},
        tuned_performance={"warm_latency_ms": 11.0, "approximate_memory_mb": 108.0},
    )
    latency_regression = evaluate_promotion(
        _promotion_report(0.70),
        _promotion_report(0.75),
        base_performance={"warm_latency_ms": 10.0, "approximate_memory_mb": 100.0},
        tuned_performance={"warm_latency_ms": 13.0, "approximate_memory_mb": 108.0},
    )
    quality_regression = evaluate_promotion(
        _promotion_report(0.70),
        _promotion_report(0.70),
        base_performance={"warm_latency_ms": 10.0, "approximate_memory_mb": 100.0},
        tuned_performance={"warm_latency_ms": 10.0, "approximate_memory_mb": 100.0},
    )

    assert promoted["promoted"] is True
    assert promoted["decision"] == "promote"
    assert promoted["runtime_configuration_changed"] is False
    assert json.loads(json.dumps(promoted))["decision"] == "promote"
    assert latency_regression["promoted"] is False
    assert latency_regression["resource_checks"]["latency_ratio_acceptable"] is False
    assert quality_regression["promoted"] is False
    assert not all(quality_regression["quality_checks"].values())
