from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from app.semantic_router.contracts import AXIS_LABELS
from app.semantic_router.dataset import load_dataset
from app.semantic_router.training import (
    build_training_objectives,
    train_independent_axis_losses,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semantic_router"
TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_semantic_router.py"


def test_training_cli_refuses_to_overwrite_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--output-dir", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "refusing to overwrite existing output directory" in completed.stderr


def test_training_objectives_are_reproducible_train_only_and_capped() -> None:
    dataset = load_dataset(DATA_DIR)
    first = build_training_objectives(
        dataset.splits["train"],
        seed=29,
        max_pairs_per_axis=40,
    )
    second = build_training_objectives(
        dataset.splits["train"],
        seed=29,
        max_pairs_per_axis=40,
    )
    train_ids = {row.id for row in dataset.splits["train"]}
    locked_ids = {
        row.id
        for split in ("validation", "test")
        for row in dataset.splits[split]
    }

    assert first == second
    assert set(first) == set(AXIS_LABELS)
    for axis, selection in first.items():
        assert selection.selected_count == 40
        assert selection.generated_count > selection.selected_count
        assert selection.dropped_count == (
            selection.generated_count - selection.selected_count
        )
        assert all(pair.axis == axis for pair in selection.pairs)
        assert all(pair.similarity in {0.0, 1.0} for pair in selection.pairs)
        pair_ids = {
            pair_id
            for pair in selection.pairs
            for pair_id in (pair.left_id, pair.right_id)
        }
        assert pair_ids <= train_ids
        assert pair_ids.isdisjoint(locked_ids)
        assert selection.selected_by_kind.get("group_positive", 0) > 0
        assert selection.selected_by_kind.get("hard_negative", 0) > 0


def test_training_uses_one_separate_loss_and_loader_per_axis(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(DATA_DIR)
    objectives = build_training_objectives(
        dataset.splits["train"],
        seed=7,
        max_pairs_per_axis=24,
    )
    created_losses: list[object] = []

    class FakeGenerator:
        def manual_seed(self, seed: int) -> None:
            self.seed = seed

    class FakeDataLoader:
        def __init__(
            self,
            examples,
            *,
            shuffle: bool,
            batch_size: int,
            num_workers: int,
            generator: FakeGenerator,
        ) -> None:
            self.examples = tuple(examples)
            self.batch_size = batch_size
            self.generator = generator

        def __len__(self) -> int:
            return (len(self.examples) + self.batch_size - 1) // self.batch_size

    class FakeInputExample:
        def __init__(self, *, texts: list[str], label: float) -> None:
            self.texts = texts
            self.label = label

    class FakeLoss:
        def __init__(self, model: object) -> None:
            self.model = model
            created_losses.append(self)

    torch = ModuleType("torch")
    torch.Generator = FakeGenerator  # type: ignore[attr-defined]
    torch_utils = ModuleType("torch.utils")
    torch_utils_data = ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = FakeDataLoader  # type: ignore[attr-defined]
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.InputExample = FakeInputExample  # type: ignore[attr-defined]
    sentence_transformers.losses = SimpleNamespace(  # type: ignore[attr-defined]
        CosineSimilarityLoss=FakeLoss
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    class FakeModel:
        def fit(self, **kwargs) -> None:
            self.fit_kwargs = kwargs

    model = FakeModel()
    output_path = tmp_path / "model"
    report = train_independent_axis_losses(
        model,
        dataset.splits["train"],
        objectives,
        output_path=output_path,
        seed=7,
        epochs=2,
        batch_size=8,
        learning_rate=1e-5,
    )

    assert output_path.is_dir()
    assert len(created_losses) == len(AXIS_LABELS)
    assert len({id(loss) for loss in created_losses}) == len(AXIS_LABELS)
    assert len(model.fit_kwargs["train_objectives"]) == len(AXIS_LABELS)
    assert report["loss_design"] == "separate_cosine_similarity_loss_per_axis"
    assert report["training_method"] == "sentence_transformers.fit"
    assert report["axes"] == list(AXIS_LABELS)
    assert model.fit_kwargs["steps_per_epoch"] == min(
        report["loader_batches"].values()
    )
