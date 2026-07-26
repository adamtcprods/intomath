from __future__ import annotations

import ast
import tomllib
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_integrations_do_not_import_service_implementations() -> None:
    violations: list[str] = []
    for path in sorted((BACKEND_ROOT / "app" / "integrations").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).startswith("app.services"):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app.services"):
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")

    assert violations == []


def test_requirements_mirror_canonical_core_dependencies() -> None:
    pyproject = tomllib.loads(
        (BACKEND_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_dependencies = set(pyproject["project"]["dependencies"])
    requirements_dependencies = {
        line
        for raw_line in (BACKEND_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }

    assert requirements_dependencies == project_dependencies


def test_model_router_does_not_own_shared_provider_policy() -> None:
    source = (BACKEND_ROOT / "app" / "services" / "model_router.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            [*node.targets] if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "NVIDIA_DIRECT_FALLBACK_MODELS" not in assigned_names
    assert "NVIDIA_GPT_OSS_MODELS" not in assigned_names
    assert "remote_model_timeout_seconds" not in {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
