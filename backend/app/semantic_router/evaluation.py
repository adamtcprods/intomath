"""Offline calibration, metrics, and promotion gates for semantic routing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from app.semantic_router.contracts import AXIS_LABELS, AxisScore, SemanticExample
from app.semantic_router.prototypes import PrototypeArtifact, score_axis

EVALUATION_REPORT_VERSION = "1.0"


@dataclass(frozen=True)
class ExamplePrediction:
    example: SemanticExample
    scores: Mapping[str, AxisScore]
    out_of_distribution: bool
    accepted: bool

    @property
    def joint_correct(self) -> bool:
        return (
            self.scores["problem_type"].label == self.example.problem_type.value
            and self.scores["visualization_environment"].label
            == self.example.visualization_label
        )


def evaluate_embeddings(
    examples: Iterable[SemanticExample],
    embeddings: Mapping[str, Sequence[float]],
    artifact: PrototypeArtifact,
    *,
    minimum_confidence: float = 0.60,
    minimum_margin: float = 0.05,
    chosen_coverage: float = 0.70,
) -> dict[str, Any]:
    predictions = tuple(
        _predict(
            example,
            embeddings[example.id],
            artifact,
            minimum_confidence=minimum_confidence,
            minimum_margin=minimum_margin,
        )
        for example in examples
    )
    in_scope = tuple(item for item in predictions if item.example.in_scope)
    out_of_scope = tuple(item for item in predictions if not item.example.in_scope)
    accepted = tuple(item for item in in_scope if item.accepted)

    axes = {
        axis: _axis_metrics(in_scope, axis, labels)
        for axis, labels in AXIS_LABELS.items()
    }
    by_language: dict[str, float] = {}
    for language in sorted({item.example.language for item in in_scope}):
        subset = tuple(
            item for item in in_scope if item.example.language == language
        )
        by_language[language] = _ratio(
            sum(item.joint_correct for item in subset), len(subset)
        )

    by_problem_type = {
        label: _ratio(
            sum(
                item.scores["problem_type"].label == label
                for item in in_scope
                if item.example.problem_type.value == label
            ),
            sum(
                item.example.problem_type.value == label for item in in_scope
            ),
        )
        for label in AXIS_LABELS["problem_type"]
    }
    by_environment = {
        label: _ratio(
            sum(
                item.scores["visualization_environment"].label == label
                for item in in_scope
                if item.example.visualization_label == label
            ),
            sum(item.example.visualization_label == label for item in in_scope),
        )
        for label in AXIS_LABELS["visualization_environment"]
    }
    fallback_count = sum(not item.accepted for item in predictions)
    target_count = (
        min(len(in_scope), max(1, math.ceil(len(in_scope) * chosen_coverage)))
        if in_scope
        else 0
    )
    ranked_for_selection = sorted(
        in_scope,
        key=lambda item: (
            -min(
                item.scores["problem_type"].confidence,
                item.scores["visualization_environment"].confidence,
            ),
            -min(
                item.scores["problem_type"].margin,
                item.scores["visualization_environment"].margin,
            ),
            item.example.id,
        ),
    )
    selected_at_coverage = tuple(ranked_for_selection[:target_count])
    return {
        "total_examples": len(predictions),
        "in_scope_examples": len(in_scope),
        "out_of_scope_examples": len(out_of_scope),
        "axes": axes,
        "joint_problem_visualization_accuracy": _ratio(
            sum(item.joint_correct for item in in_scope), len(in_scope)
        ),
        "accuracy_by_language": by_language,
        "accuracy_by_problem_type": by_problem_type,
        "accuracy_by_visualization_environment": by_environment,
        "coverage": _ratio(len(accepted), len(in_scope)),
        "selective_accuracy": _ratio(
            sum(item.joint_correct for item in accepted), len(accepted)
        ),
        "chosen_coverage": chosen_coverage,
        "realized_chosen_coverage": _ratio(target_count, len(in_scope)),
        "selective_accuracy_at_chosen_coverage": _ratio(
            sum(item.joint_correct for item in selected_at_coverage),
            len(selected_at_coverage),
        ),
        "abstention_rate": _ratio(
            sum(not item.accepted for item in in_scope), len(in_scope)
        ),
        "llm_fallback_rate": _ratio(fallback_count, len(predictions)),
        "ood_recall": _ratio(
            sum(item.out_of_distribution for item in out_of_scope),
            len(out_of_scope),
        ),
        "ood_false_reject_rate": _ratio(
            sum(item.out_of_distribution for item in in_scope), len(in_scope)
        ),
    }


def calibrate_artifact(
    artifact: PrototypeArtifact,
    validation_examples: Iterable[SemanticExample],
    validation_embeddings: Mapping[str, Sequence[float]],
    *,
    minimum_confidence: float = 0.60,
    minimum_margin: float = 0.05,
    minimum_coverage: float = 0.70,
) -> PrototypeArtifact:
    rows = tuple(validation_examples)
    in_scope = tuple(item for item in rows if item.in_scope)
    temperatures = {
        axis: _fit_temperature(
            artifact,
            in_scope,
            validation_embeddings,
            axis,
        )
        for axis in AXIS_LABELS
    }
    calibrated = replace(artifact, temperatures=temperatures)
    confidence: dict[str, float] = {}
    margin: dict[str, float] = {}
    for axis in AXIS_LABELS:
        confidence[axis], margin[axis] = _select_thresholds(
            calibrated,
            in_scope,
            validation_embeddings,
            axis,
            minimum_confidence=minimum_confidence,
            minimum_margin=minimum_margin,
            minimum_coverage=minimum_coverage,
        )
    ood_threshold = _select_ood_threshold(
        calibrated,
        rows,
        validation_embeddings,
    )
    return replace(
        calibrated,
        recommended_confidence=confidence,
        recommended_margin=margin,
        min_raw_similarity=ood_threshold,
    )


def evaluate_promotion(
    base_report: Mapping[str, Any],
    tuned_report: Mapping[str, Any],
    *,
    base_performance: Mapping[str, float],
    tuned_performance: Mapping[str, float],
    maximum_latency_ratio: float = 1.20,
    maximum_memory_ratio: float = 1.15,
) -> dict[str, Any]:
    quality_checks = {
        "problem_type_macro_f1": _metric(tuned_report, "axes", "problem_type", "macro_f1")
        > _metric(base_report, "axes", "problem_type", "macro_f1"),
        "visualization_environment_macro_f1": _metric(
            tuned_report, "axes", "visualization_environment", "macro_f1"
        )
        > _metric(base_report, "axes", "visualization_environment", "macro_f1"),
        "english_accuracy": _metric(tuned_report, "accuracy_by_language", "en")
        > _metric(base_report, "accuracy_by_language", "en"),
        "vietnamese_accuracy": _metric(tuned_report, "accuracy_by_language", "vi")
        > _metric(base_report, "accuracy_by_language", "vi"),
        "mixed_language_accuracy": _metric(
            tuned_report, "accuracy_by_language", "mixed"
        )
        > _metric(base_report, "accuracy_by_language", "mixed"),
        "selective_accuracy_at_chosen_coverage": float(
            tuned_report.get("selective_accuracy_at_chosen_coverage", 0.0)
        )
        > float(base_report.get("selective_accuracy_at_chosen_coverage", 0.0)),
    }
    latency_ratio = _safe_ratio(
        float(tuned_performance.get("warm_latency_ms", math.inf)),
        float(base_performance.get("warm_latency_ms", 0.0)),
    )
    memory_ratio = _safe_ratio(
        float(
            tuned_performance.get(
                "model_parameter_memory_mb",
                tuned_performance.get("approximate_memory_mb", math.inf),
            )
        ),
        float(
            base_performance.get(
                "model_parameter_memory_mb",
                base_performance.get("approximate_memory_mb", 0.0),
            )
        ),
    )
    resource_checks = {
        "latency_ratio_acceptable": latency_ratio <= maximum_latency_ratio,
        "memory_ratio_acceptable": memory_ratio <= maximum_memory_ratio,
    }
    promoted = all(quality_checks.values()) and all(resource_checks.values())
    return {
        "promoted": promoted,
        "decision": "promote" if promoted else "keep_base_model",
        "quality_checks": quality_checks,
        "resource_checks": resource_checks,
        "latency_ratio": latency_ratio,
        "memory_ratio": memory_ratio,
        "maximum_latency_ratio": maximum_latency_ratio,
        "maximum_memory_ratio": maximum_memory_ratio,
        "runtime_configuration_changed": False,
    }


def _predict(
    example: SemanticExample,
    embedding: Sequence[float],
    artifact: PrototypeArtifact,
    *,
    minimum_confidence: float,
    minimum_margin: float,
) -> ExamplePrediction:
    scores = {
        axis: score_axis(embedding, artifact, axis) for axis in AXIS_LABELS
    }
    out_of_distribution = (
        scores["problem_type"].raw_similarity < artifact.min_raw_similarity
    )
    required_axes = ("problem_type", "visualization_environment")
    accepted = not out_of_distribution and all(
        scores[axis].confidence
        >= max(
            minimum_confidence,
            float(artifact.recommended_confidence.get(axis, 0.0)),
        )
        and scores[axis].margin
        >= max(
            minimum_margin,
            float(artifact.recommended_margin.get(axis, 0.0)),
        )
        for axis in required_axes
    )
    return ExamplePrediction(
        example=example,
        scores=scores,
        out_of_distribution=out_of_distribution,
        accepted=accepted,
    )


def _axis_metrics(
    predictions: Sequence[ExamplePrediction],
    axis: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    matrix = {
        gold: {predicted: 0 for predicted in labels} for gold in labels
    }
    correct = 0
    for item in predictions:
        gold = _gold_label(item.example, axis)
        predicted = item.scores[axis].label
        matrix[gold][predicted] += 1
        correct += gold == predicted
    f1_by_label: dict[str, float] = {}
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[other][label] for other in labels if other != label
        )
        false_negative = sum(
            matrix[label][other] for other in labels if other != label
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1_by_label[label] = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
    return {
        "accuracy": _ratio(correct, len(predictions)),
        "macro_f1": _ratio(sum(f1_by_label.values()), len(f1_by_label)),
        "f1_by_label": f1_by_label,
        "confusion_matrix": matrix,
    }


def _fit_temperature(
    artifact: PrototypeArtifact,
    examples: Sequence[SemanticExample],
    embeddings: Mapping[str, Sequence[float]],
    axis: str,
) -> float:
    candidates = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50)
    best_temperature = candidates[0]
    best_loss = math.inf
    for temperature in candidates:
        candidate_artifact = replace(
            artifact,
            temperatures={**artifact.temperatures, axis: temperature},
        )
        loss = 0.0
        for example in examples:
            score = score_axis(embeddings[example.id], candidate_artifact, axis)
            probabilities = dict(score.scores)
            loss -= math.log(max(1e-12, probabilities[_gold_label(example, axis)]))
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _select_thresholds(
    artifact: PrototypeArtifact,
    examples: Sequence[SemanticExample],
    embeddings: Mapping[str, Sequence[float]],
    axis: str,
    *,
    minimum_confidence: float,
    minimum_margin: float,
    minimum_coverage: float,
) -> tuple[float, float]:
    scores = tuple(
        (example, score_axis(embeddings[example.id], artifact, axis))
        for example in examples
    )
    confidence_candidates = sorted(
        {minimum_confidence, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
    )
    margin_candidates = sorted(
        {minimum_margin, 0.08, 0.10, 0.15, 0.20, 0.30}
    )
    best = (minimum_confidence, minimum_margin)
    best_rank = (-1.0, -1.0, -math.inf, -math.inf)
    for confidence in confidence_candidates:
        for margin in margin_candidates:
            accepted = tuple(
                (example, score)
                for example, score in scores
                if score.confidence >= confidence and score.margin >= margin
            )
            coverage = _ratio(len(accepted), len(scores))
            if coverage < minimum_coverage:
                continue
            accuracy = _ratio(
                sum(score.label == _gold_label(example, axis) for example, score in accepted),
                len(accepted),
            )
            rank = (accuracy, coverage, -confidence, -margin)
            if rank > best_rank:
                best_rank = rank
                best = (confidence, margin)
    return best


def _select_ood_threshold(
    artifact: PrototypeArtifact,
    examples: Sequence[SemanticExample],
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    if not any(not example.in_scope for example in examples):
        return artifact.min_raw_similarity
    values = tuple(
        (
            score_axis(embeddings[example.id], artifact, "problem_type").raw_similarity,
            example.in_scope,
        )
        for example in examples
    )
    candidates = sorted(
        {
            -1.0,
            0.0,
            *(value for value, _ in values),
            *(math.nextafter(value, math.inf) for value, _ in values),
        }
    )
    best_threshold = artifact.min_raw_similarity
    best_rank = (-1.0, -1.0)
    for threshold in candidates:
        in_scope_kept = _ratio(
            sum(value >= threshold for value, in_scope in values if in_scope),
            sum(in_scope for _, in_scope in values),
        )
        ood_rejected = _ratio(
            sum(value < threshold for value, in_scope in values if not in_scope),
            sum(not in_scope for _, in_scope in values),
        )
        rank = ((in_scope_kept + ood_rejected) / 2.0, in_scope_kept)
        if rank > best_rank:
            best_rank = rank
            best_threshold = threshold
    return best_threshold


def _gold_label(example: SemanticExample, axis: str) -> str:
    if axis == "problem_type":
        return example.problem_type.value
    if axis == "difficulty":
        return example.difficulty.value
    return example.visualization_label


def _metric(payload: Mapping[str, Any], *path: str) -> float:
    value: Any = payload
    for part in path:
        value = value[part]
    return float(value)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else math.inf
    return numerator / denominator


__all__ = [
    "EVALUATION_REPORT_VERSION",
    "ExamplePrediction",
    "calibrate_artifact",
    "evaluate_embeddings",
    "evaluate_promotion",
]
