from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class SupportStatus(str, Enum):
    supported = "supported"
    experimental = "experimental"
    blocked = "blocked"


@dataclass(frozen=True)
class RuntimeAcceptance:
    command_name: str
    signature: str
    environments: frozenset[str]
    fixture_id: str
    output_type: str | None
    output_type_strategy: str | None


DEFAULT_RUNTIME_ACCEPTANCE_PATH = (
    Path(__file__).resolve().parents[2] / "geogebra_runtime_acceptance.json"
)

_UNSAFE_EXACT: dict[str, str] = {
    "Button": "action objects are outside the construction-command trust boundary",
    "CopyFreeObject": "object copying can bypass dependency tracking",
    "Delete": "destructive state mutation is not a construction command",
    "Execute": "it executes command strings from text",
    "ExportImage": "export and URL behavior is not supported",
    "InputBox": "action objects are outside the construction-command trust boundary",
    "Open": "external navigation is not supported",
    "ParseToFunction": "it parses an unrestricted expression from text",
    "ParseToNumber": "it parses an unrestricted expression from text",
    "PlaySound": "media and external URL behavior is not supported",
    "RunClickScript": "GeoGebra scripting is disabled",
    "RunUpdateScript": "GeoGebra scripting is disabled",
    "StartAnimation": "animation is exposed only through structured render controls",
    "ToolImage": "media extraction is outside the construction-command trust boundary",
    "UpdateConstruction": "global construction mutation is not supported",
}

_KNOWN_OUTPUT_TYPES = frozenset(
    {
        "Point",
        "Line",
        "Segment",
        "Ray",
        "Circle",
        "Conic",
        "Polygon",
        "Function",
        "Number",
        "Angle",
        "List",
        "Vector",
        "Matrix",
        "Text",
        "Boolean",
        "Equation",
        "Interval",
        "Plane",
        "Surface",
        "Solid",
    }
)
_KNOWN_OUTPUT_TYPE_STRATEGIES = frozenset({"same_as_first_argument"})


def permanent_block_reason(
    command_name: str, families: Iterable[str] = ()
) -> str | None:
    family_set = {family.casefold() for family in families}
    if "scripting" in family_set:
        return "the scripting command family is permanently blocked"
    if command_name in _UNSAFE_EXACT:
        return _UNSAFE_EXACT[command_name]
    if command_name.startswith("Set"):
        return "state and styling changes must use structured applet API operations"
    if "Script" in command_name or "JavaScript" in command_name:
        return "GeoGebra and JavaScript scripting are disabled"
    return None


def load_runtime_acceptance(
    path: Path = DEFAULT_RUNTIME_ACCEPTANCE_PATH,
) -> dict[tuple[str, str], RuntimeAcceptance]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load GeoGebra runtime-acceptance policy at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise RuntimeError("GeoGebra runtime-acceptance policy must use schema 1.0.")
    raw_entries = payload.get("accepted_overloads")
    if not isinstance(raw_entries, list):
        raise RuntimeError("GeoGebra runtime-acceptance policy requires accepted_overloads.")

    result: dict[tuple[str, str], RuntimeAcceptance] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise RuntimeError("Runtime-acceptance entries must be objects.")
        command_name = str(raw.get("command_name", "")).strip()
        signature = str(raw.get("signature", "")).strip()
        fixture_id = str(raw.get("fixture_id", "")).strip()
        output_type = str(raw.get("output_type", "")).strip() or None
        output_type_strategy = (
            str(raw.get("output_type_strategy", "")).strip() or None
        )
        environments = raw.get("environments")
        if (
            not command_name
            or not signature
            or not fixture_id
            or not isinstance(environments, list)
            or not environments
            or any(not isinstance(item, str) or not item for item in environments)
        ):
            raise RuntimeError("Runtime-acceptance entry fields are incomplete.")
        if output_type not in _KNOWN_OUTPUT_TYPES and output_type is not None:
            raise RuntimeError(
                f"Runtime acceptance has unknown output type {output_type!r}."
            )
        if (
            output_type_strategy not in _KNOWN_OUTPUT_TYPE_STRATEGIES
            and output_type_strategy is not None
        ):
            raise RuntimeError(
                "Runtime acceptance has an unknown output-type strategy."
            )
        if (output_type is None) == (output_type_strategy is None):
            raise RuntimeError(
                "Runtime acceptance requires exactly one output type or output-type strategy."
            )
        key = (command_name.casefold(), signature)
        if key in result:
            raise RuntimeError(
                f"Duplicate runtime-acceptance entry for {command_name}: {signature}"
            )
        result[key] = RuntimeAcceptance(
            command_name=command_name,
            signature=signature,
            environments=frozenset(environments),
            fixture_id=fixture_id,
            output_type=output_type,
            output_type_strategy=output_type_strategy,
        )
    return result

