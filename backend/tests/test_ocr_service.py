import asyncio
import re
import tomllib
from pathlib import Path

import pytest

from app.services.ocr_service import (
    OCRInputError,
    OCRService,
    OCRUnavailableError,
    _clean_for_solver,
)


def test_clean_for_solver_preserves_math_labels() -> None:
    text = "Let l be a line through circle center O. Find O and prove l ∥ m."

    assert _clean_for_solver(text) == text


def test_clean_for_solver_still_normalizes_typographic_operators() -> None:
    assert _clean_for_solver("6 × 3 − 4 ÷ 2") == "6 * 3 - 4 / 2"


def test_ocr_service_converts_runner_failure_to_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = OCRService()

    async def fail_ocr(**_: str) -> str:
        raise RuntimeError("CUDA runtime unavailable")

    monkeypatch.setattr(service.local_ocr, "extract_text", fail_ocr)

    with pytest.raises(OCRUnavailableError, match="OCR is temporarily unavailable"):
        asyncio.run(service.extract_problem_text("aW1hZ2U=", "image/png"))


@pytest.mark.parametrize("raw_text", ["", "   \n\t"])
def test_ocr_service_rejects_blank_output(
    monkeypatch: pytest.MonkeyPatch, raw_text: str
) -> None:
    service = OCRService()

    async def return_blank(**_: str) -> str:
        return raw_text

    monkeypatch.setattr(service.local_ocr, "extract_text", return_blank)

    with pytest.raises(OCRInputError, match="could not extract usable text"):
        asyncio.run(service.extract_problem_text("aW1hZ2U=", "image/png"))


def _dependency_names(specifications: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\s\[]", specification, maxsplit=1)[0].lower()
        for specification in specifications
    }


def test_local_ocr_dependencies_have_one_canonical_manifest() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    project = tomllib.loads((backend_dir / "pyproject.toml").read_text(encoding="utf-8"))

    runtime_dependencies = _dependency_names(project["project"]["dependencies"])
    ocr_dependencies = _dependency_names(
        project["project"]["optional-dependencies"]["ocr"]
    )

    assert {"pillow", "torch", "transformers"} <= runtime_dependencies
    assert "flash-attn" in ocr_dependencies

    requirements = (backend_dir / "requirements.txt").read_text(encoding="utf-8")
    assert "-e ./backend[dev]" in requirements
    assert not ({"pillow", "torch", "transformers", "flash-attn"} & _dependency_names(requirements.splitlines()))

    repository_root = backend_dir.parent
    setup_docs = (repository_root / "docs" / "SETUP.md").read_text(encoding="utf-8")
    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    for documentation in (setup_docs, readme):
        assert './backend[dev]' in documentation
        assert './backend[ocr]' in documentation
        assert "--no-build-isolation" in documentation
