from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.semantic_router.contracts import ReviewStatus
from app.semantic_router.dataset import (
    DatasetValidationError,
    dataset_summary,
    example_from_payload,
    load_dataset,
    load_jsonl,
    validate_splits,
)
from app.semantic_router.pairs import build_axis_training_pairs
from app.semantic_router.prototypes import build_prototype_artifact

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semantic_router"


def test_starter_dataset_is_grouped_balanced_and_needs_review() -> None:
    dataset = load_dataset(DATA_DIR)
    summary = dataset_summary(dataset)

    assert summary["sizes"] == {"train": 144, "validation": 36, "test": 36}
    assert summary["groups"] == {"train": 48, "validation": 12, "test": 12}
    assert summary["languages"] == {"en": 72, "mixed": 72, "vi": 72}
    assert summary["problem_types"] == {
        "algebra": 24,
        "arithmetic": 24,
        "calculus": 24,
        "general": 24,
        "geometry": 24,
        "number_theory": 24,
        "probability": 24,
        "statistics": 24,
        "trigonometry": 24,
    }
    assert summary["review_statuses"] == {"needs_review": 216}
    assert summary["in_scope"] == 210
    assert summary["out_of_scope"] == 6
    assert all(
        example.review_status is ReviewStatus.needs_review
        for example in dataset.examples
    )
    assert len(dataset.data_hash) == 64


def test_starter_dataset_represents_every_label_in_each_split() -> None:
    dataset = load_dataset(DATA_DIR)

    for rows in dataset.splits.values():
        assert {row.problem_type.value for row in rows} == {
            "arithmetic",
            "algebra",
            "number_theory",
            "geometry",
            "trigonometry",
            "calculus",
            "statistics",
            "probability",
            "general",
        }
        assert {row.difficulty.value for row in rows} == {"easy", "medium", "hard"}
        assert {row.language for row in rows} == {"en", "vi", "mixed"}
        assert {row.visualization_label for row in rows} == {
            "none",
            "geometry_2d",
            "graphing",
            "graphics_3d",
            "cas",
            "probability",
            "statistics",
            "spreadsheet",
        }


def test_starter_dataset_has_no_group_leakage() -> None:
    dataset = load_dataset(DATA_DIR)
    group_splits: dict[str, set[str]] = {}
    for split_name, rows in dataset.splits.items():
        for row in rows:
            group_splits.setdefault(row.group_id, set()).add(split_name)

    assert all(len(splits) == 1 for splits in group_splits.values())
    assert validate_splits(dataset.splits) == ()


def test_loader_requires_review_status_and_valid_enums(tmp_path: Path) -> None:
    payload = {
        "id": "bad-1",
        "text": "Dựng đường trung trực AB",
        "language": "vi",
        "problem_type": "geometry",
        "difficulty": "medium",
        "visualization_environment": "geometry_2d",
        "group_id": "bad-group",
        "source": "test",
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="review_status"):
        load_jsonl(path)

    payload["review_status"] = "needs_review"
    payload["problem_type"] = "not-a-label"
    with pytest.raises(ValueError, match="problem_type"):
        example_from_payload(payload)


def test_duplicate_and_conflicting_examples_are_detected() -> None:
    dataset = load_dataset(DATA_DIR)
    row = dataset.splits["train"][0]
    duplicate = replace(row, id="duplicate-id")
    conflict = replace(
        row,
        id="conflict-id",
        text="A unique conflicting group variant",
        difficulty=dataset.splits["test"][0].difficulty,
    )
    splits = {
        "train": (row, conflict),
        "validation": (duplicate,),
        "test": (),
    }

    issues = validate_splits(splits)

    assert any("duplicate normalized text" in issue for issue in issues)
    assert any("crosses splits" in issue for issue in issues)
    assert any("conflicting labels" in issue for issue in issues)


def test_axis_pairs_are_binary_independent_and_reproducible() -> None:
    train = load_dataset(DATA_DIR).splits["train"]

    first = build_axis_training_pairs(train, axis="problem_type", seed=17)
    second = build_axis_training_pairs(train, axis="problem_type", seed=17)

    assert first == second
    assert {pair.similarity for pair in first} == {0.0, 1.0}
    assert all(pair.axis == "problem_type" for pair in first)
    assert any(pair.kind == "group_positive" for pair in first)
    assert any(pair.kind == "hard_negative" for pair in first)


def test_prototype_artifact_builds_each_axis_and_keeps_term_retrieval_separate() -> None:
    dataset = load_dataset(DATA_DIR)
    train = dataset.splits["train"]
    embeddings = {
        row.id: (
            float(index + 1),
            float((index % 7) + 1),
            float((index % 11) + 2),
            1.0,
        )
        for index, row in enumerate(train)
    }

    artifact = build_prototype_artifact(
        train,
        embeddings,
        model_name="fake-multilingual-model",
        data_hash=dataset.data_hash,
    )

    assert set(artifact.centroids) == {
        "problem_type",
        "difficulty",
        "visualization_environment",
    }
    assert set(artifact.centroids["problem_type"]) == {
        row.problem_type.value for row in train
    }
    assert artifact.embedding_dimension == 4
    assert artifact.term_prototypes
    assert all(item.terms for item in artifact.term_prototypes)
