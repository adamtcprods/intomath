#!/usr/bin/env python3
"""Evaluate the untouched multilingual base encoder or an exported local model."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.semantic_router.contracts import AXIS_LABELS  # noqa: E402
from app.semantic_router.dataset import dataset_summary, load_dataset  # noqa: E402
from app.semantic_router.evaluation import (  # noqa: E402
    EVALUATION_REPORT_VERSION,
    calibrate_artifact,
    evaluate_embeddings,
)
from app.semantic_router.prototypes import (  # noqa: E402
    artifact_from_dict,
    build_prototype_artifact,
    score_axis,
)

DEFAULT_BASE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class SentenceTransformerEvaluator:
    def __init__(
        self,
        source: str,
        *,
        device: str,
        local_files_only: bool,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is not installed; install "
                "backend[semantic-router] before evaluation"
            ) from exc
        self.source = source
        self.model = SentenceTransformer(
            source,
            device=device,
            local_files_only=local_files_only,
        )
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = self.model.get_sentence_embedding_dimension
        dimension = dimension_getter()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("model did not report an embedding dimension")
        self.dimension = dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "semantic_router",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warm-runs", type=int, default=20)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow this explicit evaluation command to fetch the configured model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.warm_runs <= 0:
        raise SystemExit("--batch-size and --warm-runs must be positive")

    dataset = load_dataset(args.data_dir)
    summary = dataset_summary(dataset)
    model_source, exported_artifact = _resolve_model(args.model_dir, args.base_model)
    memory_before = _resident_memory_mb()
    load_started = time.perf_counter()
    encoder = SentenceTransformerEvaluator(
        model_source,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    load_time_ms = (time.perf_counter() - load_started) * 1000.0
    memory_after_load = _resident_memory_mb()

    cold_text = dataset.splits["validation"][0].text
    cold_started = time.perf_counter()
    encoder.encode([cold_text], batch_size=1)
    cold_embedding_latency_ms = (time.perf_counter() - cold_started) * 1000.0

    embeddings: dict[str, dict[str, Sequence[float]]] = {}
    for split_name, rows in dataset.splits.items():
        vectors = encoder.encode(
            [row.text for row in rows],
            batch_size=args.batch_size,
        )
        if len(vectors) != len(rows):
            raise RuntimeError(f"model returned wrong batch size for {split_name}")
        embeddings[split_name] = {
            row.id: vector for row, vector in zip(rows, vectors, strict=True)
        }

    if exported_artifact is None:
        artifact = build_prototype_artifact(
            dataset.splits["train"],
            embeddings["train"],
            model_name=args.base_model,
            data_hash=dataset.data_hash,
            min_raw_similarity=0.20,
            term_retrieval_min_similarity=0.55,
        )
        artifact = calibrate_artifact(
            artifact,
            dataset.splits["validation"],
            embeddings["validation"],
        )
        model_kind = "untouched_base_encoder"
    else:
        artifact = exported_artifact
        if artifact.data_hash != dataset.data_hash:
            raise RuntimeError(
                "exported prototype data hash does not match the evaluation dataset"
            )
        if artifact.embedding_dimension != encoder.dimension:
            raise RuntimeError("exported model and prototype dimensions do not match")
        model_kind = "exported_local_model"

    validation_report = evaluate_embeddings(
        dataset.splits["validation"],
        embeddings["validation"],
        artifact,
    )
    test_report = evaluate_embeddings(
        dataset.splits["test"],
        embeddings["test"],
        artifact,
    )

    warm_started = time.perf_counter()
    for _ in range(args.warm_runs):
        vector = encoder.encode([cold_text], batch_size=1)[0]
        for axis in AXIS_LABELS:
            score_axis(vector, artifact, axis)
    warm_latency_ms = (
        (time.perf_counter() - warm_started) * 1000.0 / args.warm_runs
    )
    memory_after_evaluation = _resident_memory_mb()
    performance = {
        "model_load_time_ms": load_time_ms,
        "cold_embedding_latency_ms": cold_embedding_latency_ms,
        "warm_latency_ms": warm_latency_ms,
        "approximate_memory_mb": max(0.0, memory_after_load - memory_before),
        "model_parameter_memory_mb": _parameter_memory_mb(encoder.model),
        "peak_process_rss_mb": memory_after_evaluation,
        "device": args.device,
        "warm_runs": args.warm_runs,
        "batch_size": args.batch_size,
    }

    warning = (
        "All starter examples require human review. Metrics are development "
        "baselines, not production accuracy claims."
    )
    evaluation = {
        "report_version": EVALUATION_REPORT_VERSION,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset_warning": warning,
        "data_hash": dataset.data_hash,
        "model_kind": model_kind,
        "model_source": model_source,
        "base_model": args.base_model,
        "validation": validation_report,
        "test": test_report,
        "performance": performance,
        "recommended_thresholds": {
            "confidence": dict(artifact.recommended_confidence),
            "margin": dict(artifact.recommended_margin),
            "min_raw_similarity": artifact.min_raw_similarity,
        },
        "fine_tuned": None,
        "promotion": {
            "promoted": False,
            "decision": "not_evaluated",
            "reason": "optional fine-tuning has not been run",
            "runtime_configuration_changed": False,
        },
    }
    metadata = {
        "artifact_version": artifact.artifact_version,
        "model_kind": model_kind,
        "base_model": args.base_model,
        "model_source": model_source,
        "evaluated_at": evaluation["evaluated_at"],
        "training_data_hash": dataset.data_hash,
        "dataset": summary,
        "languages_represented": sorted(summary["languages"]),
        "label_names": {axis: list(labels) for axis, labels in AXIS_LABELS.items()},
        "embedding_dimension": artifact.embedding_dimension,
        "prototype_label_counts": {
            axis: dict(counts) for axis, counts in artifact.label_counts.items()
        },
        "review_status_counts": dict(
            Counter(row.review_status.value for row in dataset.examples)
        ),
        "training_configuration": None,
        "validation_metrics": validation_report,
        "test_metrics": test_report,
        "recommended_confidence_thresholds": dict(
            artifact.recommended_confidence
        ),
        "recommended_margin_thresholds": dict(artifact.recommended_margin),
        "recommended_min_raw_similarity": artifact.min_raw_similarity,
        "limitations": warning,
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "prototypes.json", artifact.to_dict())
        _write_json(args.output_dir / "metadata.json", metadata)
        _write_json(args.output_dir / "evaluation.json", evaluation)

    print(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _resolve_model(
    model_dir: Path | None,
    base_model: str,
) -> tuple[str, Any | None]:
    if model_dir is None:
        return base_model, None
    model_path = model_dir / "model"
    artifact_path = model_dir / "prototypes.json"
    if not model_path.is_dir() or not artifact_path.is_file():
        raise SystemExit(
            "--model-dir must contain model/ and prototypes.json; use --base-model "
            "without --model-dir for the untouched baseline"
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("prototypes.json must contain an object")
    return str(model_path), artifact_from_dict(payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parameter_memory_mb(model: Any) -> float:
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return total_bytes / (1024.0 * 1024.0)


def _resident_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
