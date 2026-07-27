"""Optional deterministic multi-task training helpers."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.semantic_router.contracts import AXIS_LABELS, SemanticExample
from app.semantic_router.pairs import (
    AxisTrainingPair,
    TrainingAxis,
    build_axis_training_pairs,
)


@dataclass(frozen=True)
class PairSelection:
    pairs: tuple[AxisTrainingPair, ...]
    generated_count: int
    selected_count: int
    dropped_count: int
    generated_by_kind: Mapping[str, int]
    selected_by_kind: Mapping[str, int]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "semantic-router training dependencies are not installed"
        ) from exc
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_training_objectives(
    examples: Iterable[SemanticExample],
    *,
    seed: int,
    max_pairs_per_axis: int,
) -> dict[TrainingAxis, PairSelection]:
    rows = tuple(examples)
    objectives: dict[TrainingAxis, PairSelection] = {}
    for axis in AXIS_LABELS:
        pairs = build_axis_training_pairs(
            rows,
            axis=axis,  # type: ignore[arg-type]
            seed=seed,
        )
        objectives[axis] = select_axis_pairs(
            pairs,
            max_pairs=max_pairs_per_axis,
            seed=seed,
        )
    return objectives


def select_axis_pairs(
    pairs: Sequence[AxisTrainingPair],
    *,
    max_pairs: int,
    seed: int,
) -> PairSelection:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    generated_by_kind = Counter(pair.kind for pair in pairs)
    if len(pairs) <= max_pairs:
        selected = tuple(pairs)
    else:
        by_kind: dict[str, list[AxisTrainingPair]] = {}
        for pair in pairs:
            by_kind.setdefault(pair.kind, []).append(pair)
        randomizer = random.Random(f"{seed}:{pairs[0].axis if pairs else 'empty'}")
        for kind_pairs in by_kind.values():
            randomizer.shuffle(kind_pairs)

        selected_list: list[AxisTrainingPair] = []
        ordered_kinds = (
            "group_positive",
            "positive",
            "hard_negative",
            "negative",
        )
        while len(selected_list) < max_pairs and any(
            by_kind.get(kind) for kind in ordered_kinds
        ):
            for kind in ordered_kinds:
                candidates = by_kind.get(kind, [])
                if not candidates or len(selected_list) >= max_pairs:
                    continue
                selected_list.append(candidates.pop())
        selected = tuple(
            sorted(
                selected_list,
                key=lambda item: (item.kind, item.left_id, item.right_id),
            )
        )
    selected_by_kind = Counter(pair.kind for pair in selected)
    return PairSelection(
        pairs=selected,
        generated_count=len(pairs),
        selected_count=len(selected),
        dropped_count=len(pairs) - len(selected),
        generated_by_kind=dict(sorted(generated_by_kind.items())),
        selected_by_kind=dict(sorted(selected_by_kind.items())),
    )


def train_independent_axis_losses(
    model: Any,
    examples: Iterable[SemanticExample],
    objectives: Mapping[TrainingAxis, PairSelection],
    *,
    output_path: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    try:
        import torch
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "semantic-router training dependencies are not installed"
        ) from exc

    rows = {example.id: example for example in examples}
    train_objectives = []
    loader_sizes: dict[str, int] = {}
    for axis_index, axis in enumerate(AXIS_LABELS):
        selection = objectives[axis]  # type: ignore[index]
        input_examples = [
            InputExample(
                texts=[rows[pair.left_id].text, rows[pair.right_id].text],
                label=pair.similarity,
            )
            for pair in selection.pairs
        ]
        generator = torch.Generator()
        generator.manual_seed(seed + axis_index)
        loader = DataLoader(
            input_examples,
            shuffle=True,
            batch_size=batch_size,
            num_workers=0,
            generator=generator,
        )
        train_objectives.append((loader, losses.CosineSimilarityLoss(model)))
        loader_sizes[axis] = len(loader)

    if not train_objectives or any(size == 0 for size in loader_sizes.values()):
        raise ValueError("every independent training axis requires at least one batch")
    steps_per_epoch = min(loader_sizes.values())
    warmup_steps = max(
        1,
        round(steps_per_epoch * len(train_objectives) * epochs * 0.10),
    )
    output_path.mkdir(parents=True, exist_ok=False)
    legacy_training_method = getattr(model, "old_fit", None)
    uses_legacy_training = callable(legacy_training_method)
    training_method = legacy_training_method if uses_legacy_training else model.fit
    training_method(
        train_objectives=train_objectives,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_path),
        save_best_model=False,
        show_progress_bar=False,
        use_amp=False,
    )
    return {
        "loss_design": "separate_cosine_similarity_loss_per_axis",
        "training_method": (
            "sentence_transformers.old_fit"
            if uses_legacy_training
            else "sentence_transformers.fit"
        ),
        "axes": list(AXIS_LABELS),
        "loader_batches": loader_sizes,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }


__all__ = [
    "PairSelection",
    "build_training_objectives",
    "select_axis_pairs",
    "set_deterministic_seed",
    "train_independent_axis_losses",
]
