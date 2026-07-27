#!/usr/bin/env python3
"""Optionally fine-tune the multilingual encoder with independent axis losses."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.semantic_router.contracts import AXIS_LABELS, ReviewStatus  # noqa: E402
from app.semantic_router.dataset import load_dataset  # noqa: E402
from app.semantic_router.evaluation import (  # noqa: E402
    EVALUATION_REPORT_VERSION,
    calibrate_artifact,
    evaluate_embeddings,
    evaluate_promotion,
)
from app.semantic_router.prototypes import (  # noqa: E402
    PrototypeArtifact,
    build_prototype_artifact,
    score_axis,
)
from app.semantic_router.training import (  # noqa: E402
    build_training_objectives,
    set_deterministic_seed,
    train_independent_axis_losses,
)

DEFAULT_BASE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "semantic_router",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-pairs-per-axis", type=int, default=1_200)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow this explicit training command to fetch the base model.",
    )
    parser.add_argument(
        "--allow-needs-review",
        action="store_true",
        help="Acknowledge experimental training on rows that still need human review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_arguments(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")

    dataset = load_dataset(args.data_dir)
    review_counts = Counter(row.review_status.value for row in dataset.examples)
    if review_counts.get(ReviewStatus.needs_review.value, 0) and not args.allow_needs_review:
        raise SystemExit(
            "dataset contains needs_review rows; pass --allow-needs-review only for "
            "an explicitly experimental run"
        )
    set_deterministic_seed(args.seed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        result = _run_training(
            args,
            dataset,
            staging,
            output_dir,
            review_counts,
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_training(
    args: argparse.Namespace,
    dataset: Any,
    staging: Path,
    output_dir: Path,
    review_counts: Mapping[str, int],
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed; install "
            "backend[semantic-router-train]"
        ) from exc

    memory_before = _resident_memory_mb()
    load_started = time.perf_counter()
    model = SentenceTransformer(
        args.base_model,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    load_time_ms = (time.perf_counter() - load_started) * 1000.0
    parameter_memory_mb = _parameter_memory_mb(model)
    memory_after_load = _resident_memory_mb()

    cold_text = dataset.splits["validation"][0].text
    cold_started = time.perf_counter()
    _encode(model, [cold_text], batch_size=1)
    cold_latency_ms = (time.perf_counter() - cold_started) * 1000.0

    base_embeddings = _encode_splits(model, dataset.splits, args.batch_size)
    base_artifact = build_prototype_artifact(
        dataset.splits["train"],
        base_embeddings["train"],
        model_name=args.base_model,
        data_hash=dataset.data_hash,
        min_raw_similarity=0.20,
        term_retrieval_min_similarity=0.55,
    )
    base_artifact = calibrate_artifact(
        base_artifact,
        dataset.splits["validation"],
        base_embeddings["validation"],
    )
    base_validation = evaluate_embeddings(
        dataset.splits["validation"],
        base_embeddings["validation"],
        base_artifact,
    )
    base_test = evaluate_embeddings(
        dataset.splits["test"],
        base_embeddings["test"],
        base_artifact,
    )
    base_performance = {
        "model_load_time_ms": load_time_ms,
        "cold_embedding_latency_ms": cold_latency_ms,
        "warm_latency_ms": _benchmark(model, cold_text, base_artifact, args.warm_runs),
        "approximate_memory_mb": max(0.0, memory_after_load - memory_before),
        "model_parameter_memory_mb": parameter_memory_mb,
        "peak_process_rss_mb": _resident_memory_mb(),
        "device": args.device,
    }

    objectives = build_training_objectives(
        dataset.splits["train"],
        seed=args.seed,
        max_pairs_per_axis=args.max_pairs_per_axis,
    )
    pair_report = {
        axis: {
            "generated": selection.generated_count,
            "selected": selection.selected_count,
            "dropped": selection.dropped_count,
            "generated_by_kind": dict(selection.generated_by_kind),
            "selected_by_kind": dict(selection.selected_by_kind),
        }
        for axis, selection in objectives.items()
    }
    for axis, report in pair_report.items():
        print(
            f"axis={axis} generated_pairs={report['generated']} "
            f"selected_pairs={report['selected']} dropped_pairs={report['dropped']}",
            file=sys.stderr,
        )

    training_started = time.perf_counter()
    training_config = train_independent_axis_losses(
        model,
        dataset.splits["train"],
        objectives,
        output_path=staging / "model",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    training_time_seconds = time.perf_counter() - training_started
    training_config["pair_selection"] = pair_report
    training_config["max_pairs_per_axis"] = args.max_pairs_per_axis

    tuned_embeddings = _encode_splits(model, dataset.splits, args.batch_size)
    tuned_artifact = build_prototype_artifact(
        dataset.splits["train"],
        tuned_embeddings["train"],
        model_name=str(output_dir / "model"),
        data_hash=dataset.data_hash,
        min_raw_similarity=0.20,
        term_retrieval_min_similarity=0.55,
    )
    tuned_artifact = calibrate_artifact(
        tuned_artifact,
        dataset.splits["validation"],
        tuned_embeddings["validation"],
    )
    tuned_validation = evaluate_embeddings(
        dataset.splits["validation"],
        tuned_embeddings["validation"],
        tuned_artifact,
    )
    tuned_test = evaluate_embeddings(
        dataset.splits["test"],
        tuned_embeddings["test"],
        tuned_artifact,
    )
    tuned_performance = {
        "warm_latency_ms": _benchmark(
            model,
            cold_text,
            tuned_artifact,
            args.warm_runs,
        ),
        "approximate_memory_mb": max(0.0, memory_after_load - memory_before),
        "model_parameter_memory_mb": _parameter_memory_mb(model),
        "peak_process_rss_mb": _resident_memory_mb(),
        "device": args.device,
    }
    promotion = evaluate_promotion(
        base_test,
        tuned_test,
        base_performance=base_performance,
        tuned_performance=tuned_performance,
    )

    fine_tuning_date = datetime.now(UTC).isoformat()
    warning = (
        "This run used a small starter corpus whose rows require human review. "
        "Metrics are not production accuracy claims."
    )
    evaluation = {
        "report_version": EVALUATION_REPORT_VERSION,
        "fine_tuning_date": fine_tuning_date,
        "dataset_warning": warning,
        "data_hash": dataset.data_hash,
        "base_model": {
            "name": args.base_model,
            "validation": base_validation,
            "test": base_test,
            "performance": base_performance,
        },
        "fine_tuned": {
            "validation": tuned_validation,
            "test": tuned_test,
            "performance": tuned_performance,
            "training_time_seconds": training_time_seconds,
        },
        "promotion": promotion,
    }
    metadata = {
        "artifact_version": tuned_artifact.artifact_version,
        "base_model": args.base_model,
        "fine_tuning_date": fine_tuning_date,
        "training_data_hash": dataset.data_hash,
        "dataset_sizes": {
            name: len(rows) for name, rows in dataset.splits.items()
        },
        "dataset_groups": {
            name: len({row.group_id for row in rows})
            for name, rows in dataset.splits.items()
        },
        "languages_represented": sorted(
            {row.language for row in dataset.examples}
        ),
        "label_names": {axis: list(labels) for axis, labels in AXIS_LABELS.items()},
        "embedding_dimension": tuned_artifact.embedding_dimension,
        "training_configuration": training_config,
        "review_status_counts": dict(sorted(review_counts.items())),
        "validation_metrics": tuned_validation,
        "test_metrics": tuned_test,
        "base_validation_metrics": base_validation,
        "base_test_metrics": base_test,
        "recommended_confidence_thresholds": dict(
            tuned_artifact.recommended_confidence
        ),
        "recommended_margin_thresholds": dict(tuned_artifact.recommended_margin),
        "recommended_min_raw_similarity": tuned_artifact.min_raw_similarity,
        "promotion": promotion,
        "limitations": warning,
    }
    _write_json(staging / "prototypes.json", tuned_artifact.to_dict())
    _write_json(staging / "metadata.json", metadata)
    _write_json(staging / "evaluation.json", evaluation)
    return evaluation


def _encode_splits(
    model: Any,
    splits: Mapping[str, Sequence[Any]],
    batch_size: int,
) -> dict[str, dict[str, Sequence[float]]]:
    encoded: dict[str, dict[str, Sequence[float]]] = {}
    for split_name, rows in splits.items():
        vectors = _encode(
            model,
            [row.text for row in rows],
            batch_size=batch_size,
        )
        if len(vectors) != len(rows):
            raise RuntimeError(f"model returned wrong batch size for {split_name}")
        encoded[split_name] = {
            row.id: vector for row, vector in zip(rows, vectors, strict=True)
        }
    return encoded


def _encode(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> Sequence[Sequence[float]]:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _benchmark(
    model: Any,
    text: str,
    artifact: PrototypeArtifact,
    runs: int,
) -> float:
    started = time.perf_counter()
    for _ in range(runs):
        vector = _encode(model, [text], batch_size=1)[0]
        for axis in AXIS_LABELS:
            score_axis(vector, artifact, axis)
    return (time.perf_counter() - started) * 1000.0 / runs


def _parameter_memory_mb(model: Any) -> float:
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return total_bytes / (1024.0 * 1024.0)


def _resident_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0 or args.max_pairs_per_axis <= 0:
        raise SystemExit("--batch-size and --max-pairs-per-axis must be positive")
    if args.learning_rate <= 0 or args.warm_runs <= 0:
        raise SystemExit("--learning-rate and --warm-runs must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
