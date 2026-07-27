"""Semantic-router JSONL loading, validation, hashing, and summaries."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.semantic_router.contracts import (
    DATASET_SCHEMA_VERSION,
    ReviewStatus,
    SemanticExample,
    VISUALIZATION_NONE,
)

SPLIT_NAMES = ("train", "validation", "test")
_REQUIRED_FIELDS = (
    "id",
    "text",
    "language",
    "problem_type",
    "difficulty",
    "visualization_environment",
    "group_id",
    "source",
    "review_status",
)


class DatasetValidationError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("Semantic-router dataset validation failed:\n- " + "\n- ".join(self.issues))


@dataclass(frozen=True)
class LoadedDataset:
    splits: Mapping[str, tuple[SemanticExample, ...]]
    data_hash: str

    @property
    def examples(self) -> tuple[SemanticExample, ...]:
        return tuple(
            example
            for split_name in SPLIT_NAMES
            for example in self.splits[split_name]
        )


def normalize_text(text: str) -> str:
    """Normalize harmless formatting without changing mathematical semantics."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def duplicate_key(text: str) -> str:
    return normalize_text(text).casefold()


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _enum_value(enum_type: type[Any], payload: Mapping[str, Any], field: str) -> Any:
    value = _required_string(payload, field)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field} must be one of: {allowed}") from exc


def example_from_payload(payload: Mapping[str, Any]) -> SemanticExample:
    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    environment_value = _required_string(payload, "visualization_environment")
    if environment_value == VISUALIZATION_NONE:
        environment = None
    else:
        try:
            environment = VisualizationEnvironment(environment_value)
        except ValueError as exc:
            allowed = ", ".join(
                [*(item.value for item in VisualizationEnvironment), VISUALIZATION_NONE]
            )
            raise ValueError(
                f"visualization_environment must be one of: {allowed}"
            ) from exc

    raw_terms = payload.get("visualization_search_terms", [])
    if not isinstance(raw_terms, list) or any(
        not isinstance(term, str) for term in raw_terms
    ):
        raise ValueError("visualization_search_terms must be an array of strings")
    terms = tuple(
        dict.fromkeys(term.strip() for term in raw_terms if term.strip())
    )
    if len(terms) > 8 or any(len(term) > 40 for term in terms):
        raise ValueError(
            "visualization_search_terms must contain at most 8 terms of 40 characters"
        )
    if environment is None and terms:
        raise ValueError(
            "visualization_search_terms must be empty when environment is none"
        )

    in_scope = payload.get("in_scope", True)
    if not isinstance(in_scope, bool):
        raise ValueError("in_scope must be a boolean")

    source_id_value = payload.get("source_id")
    if source_id_value is not None and (
        not isinstance(source_id_value, str) or not source_id_value.strip()
    ):
        raise ValueError("source_id must be a non-empty string when supplied")

    return SemanticExample(
        id=_required_string(payload, "id"),
        text=_required_string(payload, "text"),
        language=_required_string(payload, "language"),
        problem_type=_enum_value(ProblemType, payload, "problem_type"),
        difficulty=_enum_value(Difficulty, payload, "difficulty"),
        visualization_environment=environment,
        group_id=_required_string(payload, "group_id"),
        source=_required_string(payload, "source"),
        review_status=_enum_value(ReviewStatus, payload, "review_status"),
        visualization_search_terms=terms,
        in_scope=in_scope,
        source_id=(source_id_value.strip() if source_id_value else None),
    )


def load_jsonl(path: Path) -> tuple[SemanticExample, ...]:
    issues: list[str] = []
    examples: list[SemanticExample] = []
    if not path.is_file():
        raise DatasetValidationError([f"missing dataset file: {path}"])
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            issues.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{path.name}:{line_number}: row must be a JSON object")
            continue
        try:
            examples.append(example_from_payload(payload))
        except ValueError as exc:
            issues.append(f"{path.name}:{line_number}: {exc}")
    if issues:
        raise DatasetValidationError(issues)
    return tuple(examples)


def validate_splits(
    splits: Mapping[str, Iterable[SemanticExample]],
) -> tuple[str, ...]:
    issues: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_text: dict[str, tuple[str, SemanticExample]] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_labels: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)

    for split_name in SPLIT_NAMES:
        if split_name not in splits:
            issues.append(f"missing split: {split_name}")
            continue
        for example in splits[split_name]:
            previous_split = seen_ids.get(example.id)
            if previous_split is not None:
                issues.append(
                    f"duplicate id {example.id!r} in {previous_split} and {split_name}"
                )
            else:
                seen_ids[example.id] = split_name

            key = duplicate_key(example.text)
            previous_text = seen_text.get(key)
            if previous_text is not None:
                other_split, other = previous_text
                conflict = " with conflicting labels" if other.label_tuple != example.label_tuple else ""
                issues.append(
                    f"duplicate normalized text {example.id!r}/{other.id!r} "
                    f"in {split_name}/{other_split}{conflict}"
                )
            else:
                seen_text[key] = (split_name, example)

            group_splits[example.group_id].add(split_name)
            group_labels[example.group_id].add(example.label_tuple)
            if example.source_id:
                source_splits[example.source_id].add(split_name)

    for group_id, assigned_splits in sorted(group_splits.items()):
        if len(assigned_splits) > 1:
            issues.append(
                f"group {group_id!r} crosses splits: {', '.join(sorted(assigned_splits))}"
            )
        if len(group_labels[group_id]) > 1:
            issues.append(f"group {group_id!r} contains conflicting labels")
    for source_id, assigned_splits in sorted(source_splits.items()):
        if len(assigned_splits) > 1:
            issues.append(
                f"source_id {source_id!r} crosses splits: "
                f"{', '.join(sorted(assigned_splits))}"
            )
    return tuple(issues)


def load_dataset(data_dir: str | Path) -> LoadedDataset:
    directory = Path(data_dir)
    splits = {
        split_name: load_jsonl(directory / f"{split_name}.jsonl")
        for split_name in SPLIT_NAMES
    }
    issues = validate_splits(splits)
    if issues:
        raise DatasetValidationError(issues)
    return LoadedDataset(splits=splits, data_hash=dataset_hash(splits))


def dataset_hash(splits: Mapping[str, Iterable[SemanticExample]]) -> str:
    digest = hashlib.sha256()
    digest.update(DATASET_SCHEMA_VERSION.encode("utf-8"))
    for split_name in SPLIT_NAMES:
        digest.update(split_name.encode("utf-8"))
        for example in sorted(splits[split_name], key=lambda item: item.id):
            canonical = json.dumps(
                example.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            digest.update(canonical.encode("utf-8"))
    return digest.hexdigest()


def dataset_summary(dataset: LoadedDataset) -> dict[str, Any]:
    examples = dataset.examples
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "data_hash": dataset.data_hash,
        "sizes": {name: len(dataset.splits[name]) for name in SPLIT_NAMES},
        "groups": {
            name: len({item.group_id for item in dataset.splits[name]})
            for name in SPLIT_NAMES
        },
        "languages": dict(sorted(Counter(item.language for item in examples).items())),
        "problem_types": dict(
            sorted(Counter(item.problem_type.value for item in examples).items())
        ),
        "difficulties": dict(
            sorted(Counter(item.difficulty.value for item in examples).items())
        ),
        "visualization_environments": dict(
            sorted(Counter(item.visualization_label for item in examples).items())
        ),
        "review_statuses": dict(
            sorted(Counter(item.review_status.value for item in examples).items())
        ),
        "sources": dict(sorted(Counter(item.source for item in examples).items())),
        "in_scope": sum(item.in_scope for item in examples),
        "out_of_scope": sum(not item.in_scope for item in examples),
    }


def slugify_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")


__all__ = [
    "DatasetValidationError",
    "LoadedDataset",
    "SPLIT_NAMES",
    "dataset_hash",
    "dataset_summary",
    "duplicate_key",
    "example_from_payload",
    "load_dataset",
    "load_jsonl",
    "normalize_text",
    "slugify_identifier",
    "validate_splits",
]
