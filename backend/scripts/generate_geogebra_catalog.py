#!/usr/bin/env python3
"""Regenerate the local command catalog from an explicit GeoGebra manual source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPOSITORY = "https://github.com/geogebra/manual"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "geogebra_commands.json"
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.services.geogebra_support_policy import load_runtime_acceptance  # noqa: E402
from app.services.geogebra_command_registry import evaluate_overload_support  # noqa: E402


GENERATOR_VERSION = "1.4.0"
CATALOG_SCHEMA_VERSION = "1.4"
_SIGNATURE_LINE = re.compile(
    r"^([A-Za-z][A-Za-z0-9]*)\s*\(([^\n]*)\)\s*::\s*$", re.MULTILINE
)
_CATEGORY_XREF = re.compile(r"xref:/?commands/([A-Za-z0-9_]+)\.adoc")
_EXAMPLE = re.compile(r"`\+\+(.+?)\+\+`", re.DOTALL)

_CATEGORY_NAMES = {
    "3D": "3d",
    "Algebra": "algebra",
    "CAS_Restricted": "cas_restricted",
    "CAS_Specific": "cas_specific",
    "CAS_View_Supported_Geometry": "cas_supported_geometry",
    "Chart": "chart",
    "Conic": "conic",
    "Discrete_Math": "discrete_math",
    "Financial": "financial",
    "Functions_and_Calculus": "functions_and_calculus",
    "Geometry": "geometry",
    "GeoGebra": "geogebra_general",
    "List": "list",
    "Logic": "logic",
    "Logical": "logic",
    "Optimization": "optimization",
    "Probability": "probability",
    "Scripting": "scripting",
    "Spreadsheet": "spreadsheet",
    "Statistics": "statistics",
    "Text": "text",
    "Transformation": "transformation",
    "Vector_and_Matrix": "vector_and_matrix",
}

# Upstream categories describe where a command appears in the GeoGebra manual.
# Product families are a smaller, stable taxonomy used to plan validation and
# runtime rollout. A command can belong to more than one family because the
# upstream category pages overlap (for example, Circle is both 2D and 3D).
COMMAND_FAMILY_ORDER = (
    "geometry_2d",
    "transformations",
    "graphing_calculus",
    "graphics_3d",
    "cas",
    "statistics",
    "probability",
    "spreadsheet",
    "lists",
    "vector_matrix",
    "discrete_math",
    "financial",
    "logic",
    "optimization",
    "text",
    "scripting",
    "general",
    "other",
)
_COMMAND_FAMILY_INDEX = {
    family: index for index, family in enumerate(COMMAND_FAMILY_ORDER)
}
_CATEGORY_TO_FAMILY = {
    "3d": "graphics_3d",
    "algebra": "graphing_calculus",
    "cas": "cas",
    "cas_restricted": "cas",
    "cas_specific": "cas",
    "cas_supported_geometry": "cas",
    "chart": "statistics",
    "conic": "geometry_2d",
    "discrete_math": "discrete_math",
    "financial": "financial",
    "functions_and_calculus": "graphing_calculus",
    "geogebra_general": "general",
    "geometry": "geometry_2d",
    "list": "lists",
    "logic": "logic",
    "optimization": "optimization",
    "other": "other",
    "probability": "probability",
    "scripting": "scripting",
    "spreadsheet": "spreadsheet",
    "statistics": "statistics",
    "text": "text",
    "transformation": "transformations",
    "vector_and_matrix": "vector_matrix",
}


def command_families_for_categories(categories: Iterable[str]) -> list[str]:
    """Map overlapping manual categories to stable product command families."""

    families = {
        _CATEGORY_TO_FAMILY.get(category, "other") for category in categories
    }
    return sorted(
        families or {"other"},
        key=lambda family: _COMMAND_FAMILY_INDEX[family],
    )


def _family_counts(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    command_names: dict[str, set[str]] = defaultdict(set)
    overloads: dict[str, int] = defaultdict(int)
    for entry in entries:
        for family in entry["families"]:
            command_names[family].add(entry["command_name"])
            overloads[family] += 1
    return {
        family: {
            "command_names": len(command_names[family]),
            "overloads": overloads[family],
        }
        for family in COMMAND_FAMILY_ORDER
        if family in command_names
    }


def _support_status_counts(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    command_names: dict[str, set[str]] = defaultdict(set)
    overloads: dict[str, int] = defaultdict(int)
    for entry in entries:
        status = entry["support_status"]
        command_names[status].add(entry["command_name"])
        overloads[status] += 1
    return {
        status: {
            "command_names": len(command_names[status]),
            "overloads": overloads[status],
        }
        for status in ("supported", "experimental", "blocked")
        if status in command_names
    }


def _runtime_eligibility_counts(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for label, eligible in (("eligible", True), ("blocked", False)):
        selected = [
            entry for entry in entries if entry["runtime_eligible"] is eligible
        ]
        result[label] = {
            "command_names": len({entry["command_name"] for entry in selected}),
            "overloads": len(selected),
        }
    return result


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _checkout(repository: str, ref: str, destination: Path) -> tuple[Path, str]:
    _run("git", "init", "--quiet", str(destination))
    _run("git", "remote", "add", "origin", repository, cwd=destination)
    _run("git", "fetch", "--quiet", "--depth", "1", "origin", ref, cwd=destination)
    _run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=destination)
    return destination, _run("git", "rev-parse", "HEAD", cwd=destination)


def _resolve_commit(manual_path: Path, explicit_commit: str | None) -> str:
    if explicit_commit:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", explicit_commit):
            raise ValueError("--commit must be a full 40-character Git SHA.")
        return explicit_commit.lower()
    try:
        return _run("git", "rev-parse", "HEAD", cwd=manual_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError(
            "A local manual path must be a Git checkout or be paired with --commit."
        ) from exc


def _commands_directory(manual_path: Path) -> Path:
    candidates = (
        manual_path / "en/modules/ROOT/pages/commands",
        manual_path / "modules/ROOT/pages/commands",
        manual_path,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.adoc")):
            return candidate
    raise ValueError(
        "Could not find en/modules/ROOT/pages/commands under the manual path."
    )


def _category_map(commands_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for page in sorted(commands_dir.glob("*_Commands.adoc")):
        prefix = page.stem.removesuffix("_Commands")
        category = _CATEGORY_NAMES.get(prefix)
        if category is None:
            continue
        text = page.read_text(encoding="utf-8")
        for target in _CATEGORY_XREF.findall(text):
            result[target].add(category)
    return result


def _clean_markup(value: str) -> str:
    value = re.sub(r"xref:[^\[]+\[([^]]+)\]", r"\1", value)
    value = re.sub(r"https?://[^\[]+\[([^]]+)\]", r"\1", value)
    value = re.sub(r"image:[^\[]+\[[^]]*\]", "", value)
    value = value.replace("_", "").replace("`++", "").replace("++`", "")
    return " ".join(value.split())


def _description_after(text: str, end: int) -> str:
    lines: list[str] = []
    for raw_line in text[end:].splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith(("[", "=", "//")) or _SIGNATURE_LINE.match(stripped):
            break
        if stripped.startswith(":"):
            break
        lines.append(stripped)
    return _clean_markup(" ".join(lines))[:1_000]


def _examples_for(text: str, command_name: str) -> list[str]:
    examples: list[str] = []
    for match in _EXAMPLE.findall(text):
        candidate = " ".join(match.split())
        if command_name.casefold() not in candidate.casefold():
            continue
        if candidate not in examples:
            examples.append(candidate[:500])
        if len(examples) == 3:
            break
    return examples


def _search_text_for(entry: dict[str, Any]) -> str:
    categories = entry["all_categories"]
    examples = entry["examples"]
    syntax = entry["syntax"]
    description = entry["description"]
    return " | ".join(
        part
        for part in (
            entry["command_name"],
            f"[{', '.join(categories)}]",
            f"{syntax}: {description}" if description else syntax,
            f"Examples: {'; '.join(examples)}" if examples else "",
        )
        if part
    )


def parse_manual(
    manual_path: Path,
    *,
    repository: str,
    commit: str,
) -> tuple[list[dict[str, Any]], int]:
    commands_dir = _commands_directory(manual_path)
    categories_by_page = _category_map(commands_dir)
    runtime_acceptance = load_runtime_acceptance()
    entries: list[dict[str, Any]] = []
    parsed_pages = 0

    for page in sorted(commands_dir.glob("*.adoc"), key=lambda item: item.name.casefold()):
        text = page.read_text(encoding="utf-8")
        signatures = list(_SIGNATURE_LINE.finditer(text))
        if not signatures:
            continue
        parsed_pages += 1
        categories = sorted(categories_by_page.get(page.stem, {"other"}))
        families = command_families_for_categories(categories)
        for signature in signatures:
            command_name = signature.group(1)
            syntax = f"{command_name}({signature.group(2)})"
            description = _description_after(text, signature.end())
            category = categories[0]
            examples = _examples_for(text, command_name)
            acceptance = runtime_acceptance.get(
                (command_name.casefold(), syntax)
            )
            runtime_environments = sorted(
                acceptance.environments if acceptance is not None else ()
            )
            support = evaluate_overload_support(
                command_name=command_name,
                syntax=syntax,
                categories=categories,
                families=families,
                is_cas=any(category.startswith("cas") for category in categories),
                runtime_environment_values=runtime_environments,
                runtime_output_type_value=(
                    acceptance.output_type if acceptance is not None else None
                ),
                runtime_output_type_strategy=(
                    acceptance.output_type_strategy
                    if acceptance is not None
                    else None
                ),
            )
            entry = {
                "command_name": command_name,
                "syntax": syntax,
                "description": description,
                "examples": examples,
                "is_cas": any(category.startswith("cas") for category in categories),
                "notes": [],
                "category": category,
                "all_categories": categories,
                "families": families,
                "support_status": support.status.value,
                "support_requirements": list(support.requirements),
                "runtime_eligible": support.runtime_eligible,
                "runtime_accepted_environments": runtime_environments,
                "capabilities": sorted(
                    capability.value for capability in support.capabilities
                ),
                "output_type": support.output_type.value,
                "output_type_strategy": support.output_type_strategy,
                "source": {
                    "repository": repository,
                    "path": f"en/modules/ROOT/pages/commands/{page.name}",
                    "commit": commit,
                },
                "web_compatibility": "unknown",
            }
            entry["search_text"] = _search_text_for(entry)
            entries.append(entry)

    entries.sort(
        key=lambda entry: (
            entry["command_name"].casefold(),
            entry["syntax"].casefold(),
            entry["source"]["path"],
        )
    )
    return entries, parsed_pages


def _catalog_bytes(
    entries: list[dict[str, Any]],
    *,
    repository: str,
    commit: str,
    parsed_pages: int,
) -> bytes:
    payload = {
        "metadata": {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "upstream_repository": repository,
            "upstream_commit": commit,
            "command_pages_parsed": parsed_pages,
            "command_names": len({entry["command_name"] for entry in entries}),
            "overloads": len(entries),
            "families": _family_counts(entries),
            "support_statuses": _support_status_counts(entries),
            "runtime_eligibility": _runtime_eligibility_counts(entries),
        },
        "commands": entries,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def generate_catalog(
    manual_path: Path,
    *,
    repository: str,
    commit: str,
    output: Path,
) -> dict[str, Any]:
    entries, parsed_pages = parse_manual(
        manual_path, repository=repository, commit=commit
    )
    if not entries or parsed_pages < 1:
        raise ValueError("No command signatures were parsed; the existing catalog was not changed.")
    content = _catalog_bytes(
        entries,
        repository=repository,
        commit=commit,
        parsed_pages=parsed_pages,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(content)
    try:
        # Import only for the development command so application startup never
        # reaches the network or trusts an unvalidated replacement file.
        from app.services.geogebra_command_registry import GeoGebraCommandRegistry

        registry = GeoGebraCommandRegistry(temporary)
        if registry.overload_count != len(entries):
            raise ValueError("Generated registry validation changed the overload count.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": hashlib.sha256(content).hexdigest(),
        "upstream_repository": repository,
        "upstream_commit": commit,
        "generator_version": GENERATOR_VERSION,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "command_pages_parsed": parsed_pages,
        "command_names": len({entry["command_name"] for entry in entries}),
        "overloads": len(entries),
        "families": _family_counts(entries),
        "support_statuses": _support_status_counts(entries),
        "runtime_eligibility": _runtime_eligibility_counts(entries),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-path", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.manual_path:
        manual_path = args.manual_path.resolve()
        commit = _resolve_commit(manual_path, args.commit)
        metadata = generate_catalog(
            manual_path,
            repository=args.repository,
            commit=commit,
            output=args.output.resolve(),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="geogebra-manual-") as temporary:
            manual_path, commit = _checkout(
                args.repository, args.ref, Path(temporary)
            )
            metadata = generate_catalog(
                manual_path,
                repository=args.repository,
                commit=commit,
                output=args.output.resolve(),
            )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
