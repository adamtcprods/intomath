from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from app.core.config import get_settings
from app.schemas.geometry_dsl import VisualizationEnvironment


class GeoGebraObjectType(str, Enum):
    POINT = "Point"
    LINE = "Line"
    SEGMENT = "Segment"
    RAY = "Ray"
    CIRCLE = "Circle"
    CONIC = "Conic"
    POLYGON = "Polygon"
    FUNCTION = "Function"
    NUMBER = "Number"
    ANGLE = "Angle"
    LIST = "List"
    VECTOR = "Vector"
    MATRIX = "Matrix"
    TEXT = "Text"
    BOOLEAN = "Boolean"
    EQUATION = "Equation"
    INTERVAL = "Interval"
    PLANE = "Plane"
    SURFACE = "Surface"
    SOLID = "Solid"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class CommandSignature:
    original: str
    command_name: str
    argument_labels: tuple[str, ...]
    expected_types: tuple[frozenset[GeoGebraObjectType], ...]
    min_arguments: int
    max_arguments: int | None
    variadic: bool
    normalization_certain: bool


@dataclass(frozen=True)
class CommandOverload:
    command_name: str
    signature: CommandSignature
    description: str
    examples: tuple[str, ...]
    category: str
    all_categories: tuple[str, ...]
    is_cas: bool
    capabilities: frozenset[VisualizationEnvironment]
    source: dict[str, str] | None
    web_compatibility: str


@dataclass(frozen=True)
class CommandDefinition:
    name: str
    overloads: tuple[CommandOverload, ...]
    unsafe_reason: str | None = None

    @property
    def capabilities(self) -> frozenset[VisualizationEnvironment]:
        return frozenset(
            capability
            for overload in self.overloads
            for capability in overload.capabilities
        )


@dataclass(frozen=True)
class RetrievedCommand:
    name: str
    score: float
    signatures: tuple[str, ...]
    description: str


_SIGNATURE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*\((.*)\)\s*$", re.DOTALL)
_PLACEHOLDER = re.compile(r"<([^<>]+)>")
_TOKEN = re.compile(r"[a-z][a-z0-9_]+")
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "and",
        "construct",
        "create",
        "diagram",
        "draw",
        "find",
        "from",
        "given",
        "into",
        "math",
        "point",
        "show",
        "that",
        "the",
        "then",
        "through",
        "using",
        "visualize",
        "with",
    }
)

GLOBAL_CORE_COMMANDS = frozenset(
    {
        "Angle",
        "AngleBisector",
        "Circle",
        "Distance",
        "Intersect",
        "Length",
        "Line",
        "Midpoint",
        "ParallelLine",
        "PerpendicularBisector",
        "PerpendicularLine",
        "Point",
        "Polygon",
        "Ray",
        "Reflect",
        "Rotate",
        "Segment",
        "Tangent",
        "Translate",
        "Vector",
    }
)
# This is the production rollout boundary, not a statement about catalog coverage.
# Additional families stay discoverable in the registry but must gain focused
# signature and runtime tests before being added here.
ROLLED_OUT_GENERIC_COMMANDS = GLOBAL_CORE_COMMANDS

_ALIASES: dict[str, tuple[str, ...]] = {
    "bisect": ("AngleBisector", "PerpendicularBisector"),
    "bisector": ("AngleBisector", "PerpendicularBisector"),
    "circumcircle": ("Circle",),
    "differentiate": ("Derivative",),
    "gradient": ("Derivative",),
    "integrate": ("Integral",),
    "mirror": ("Reflect",),
    "perpendicular": ("PerpendicularLine", "PerpendicularBisector"),
    "project": ("ClosestPoint",),
    "reflection": ("Reflect",),
    "rotation": ("Rotate",),
    "tangent": ("Tangent",),
}

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
    "UpdateConstruction": "global construction mutation is not supported",
}

_THREE_D_TYPES = frozenset(
    {
        GeoGebraObjectType.PLANE,
        GeoGebraObjectType.SURFACE,
        GeoGebraObjectType.SOLID,
    }
)
_THREE_D_COMMANDS = frozenset(
    {
        "Bottom",
        "Cone",
        "Cube",
        "Cylinder",
        "Dodecahedron",
        "Icosahedron",
        "InfiniteCone",
        "InfiniteCylinder",
        "IntersectConic",
        "Net",
        "Octahedron",
        "OrthogonalPlane",
        "Plane",
        "Prism",
        "Pyramid",
        "Side",
        "Sphere",
        "Surface",
        "Tetrahedron",
        "Top",
        "Volume",
    }
)

_OUTPUT_TYPES: dict[str, GeoGebraObjectType] = {
    "Angle": GeoGebraObjectType.ANGLE,
    "AngleBisector": GeoGebraObjectType.LINE,
    "Area": GeoGebraObjectType.NUMBER,
    "Circle": GeoGebraObjectType.CIRCLE,
    "ClosestPoint": GeoGebraObjectType.POINT,
    "Derivative": GeoGebraObjectType.FUNCTION,
    "Distance": GeoGebraObjectType.NUMBER,
    "Intersect": GeoGebraObjectType.POINT,
    "Integral": GeoGebraObjectType.FUNCTION,
    "Length": GeoGebraObjectType.NUMBER,
    "Line": GeoGebraObjectType.LINE,
    "Midpoint": GeoGebraObjectType.POINT,
    "ParallelLine": GeoGebraObjectType.LINE,
    "PerpendicularBisector": GeoGebraObjectType.LINE,
    "PerpendicularLine": GeoGebraObjectType.LINE,
    "Point": GeoGebraObjectType.POINT,
    "Polygon": GeoGebraObjectType.POLYGON,
    "Ray": GeoGebraObjectType.RAY,
    "Reflect": GeoGebraObjectType.UNKNOWN,
    "Rotate": GeoGebraObjectType.UNKNOWN,
    "Segment": GeoGebraObjectType.SEGMENT,
    "Tangent": GeoGebraObjectType.LINE,
    "Translate": GeoGebraObjectType.UNKNOWN,
    "Vector": GeoGebraObjectType.VECTOR,
}


def _expected_types(label: str) -> frozenset[GeoGebraObjectType]:
    normalized = re.sub(r"\bxref:[^\[]+\[([^]]+)\]", r"\1", label).lower()
    found: set[GeoGebraObjectType] = set()
    patterns: tuple[tuple[tuple[str, ...], GeoGebraObjectType], ...] = (
        (("matrix",), GeoGebraObjectType.MATRIX),
        (("list", "data"), GeoGebraObjectType.LIST),
        (("point", "vertex"), GeoGebraObjectType.POINT),
        (("segment",), GeoGebraObjectType.SEGMENT),
        (("ray",), GeoGebraObjectType.RAY),
        (("line", "axis"), GeoGebraObjectType.LINE),
        (("circle",), GeoGebraObjectType.CIRCLE),
        (("conic", "ellipse", "parabola", "hyperbola"), GeoGebraObjectType.CONIC),
        (("polygon",), GeoGebraObjectType.POLYGON),
        (("function", "curve"), GeoGebraObjectType.FUNCTION),
        (("number", "integer", "radius", "x-value", "value"), GeoGebraObjectType.NUMBER),
        (("angle",), GeoGebraObjectType.ANGLE),
        (("vector", "direction"), GeoGebraObjectType.VECTOR),
        (("text", "string"), GeoGebraObjectType.TEXT),
        (("boolean",), GeoGebraObjectType.BOOLEAN),
        (("equation",), GeoGebraObjectType.EQUATION),
        (("interval",), GeoGebraObjectType.INTERVAL),
        (("plane",), GeoGebraObjectType.PLANE),
        (("surface",), GeoGebraObjectType.SURFACE),
        (("solid", "sphere", "cone", "cylinder", "prism", "pyramid"), GeoGebraObjectType.SOLID),
    )
    for keywords, object_type in patterns:
        if any(keyword in normalized for keyword in keywords):
            found.add(object_type)
    if not found or any(word in normalized for word in ("object", "geo element", "anything")):
        found.add(GeoGebraObjectType.UNKNOWN)
    return frozenset(found)


def parse_signature(syntax: str, fallback_name: str) -> CommandSignature:
    match = _SIGNATURE.fullmatch(syntax.strip())
    if not match:
        return CommandSignature(
            original=syntax,
            command_name=fallback_name,
            argument_labels=(),
            expected_types=(),
            min_arguments=0,
            max_arguments=None,
            variadic=False,
            normalization_certain=False,
        )

    command_name, arguments_text = match.groups()
    labels = tuple(part.strip() for part in _PLACEHOLDER.findall(arguments_text))
    variadic = "..." in arguments_text
    unexplained = _PLACEHOLDER.sub("", arguments_text)
    unexplained = re.sub(r"[\s,.\[\]]", "", unexplained)
    certain = not unexplained and command_name.lower() == fallback_name.lower()
    return CommandSignature(
        original=syntax,
        command_name=command_name,
        argument_labels=labels,
        expected_types=tuple(_expected_types(label) for label in labels),
        min_arguments=len(labels),
        max_arguments=None if variadic else len(labels),
        variadic=variadic,
        normalization_certain=certain,
    )


def _category_capabilities(
    *,
    command_name: str,
    categories: Iterable[str],
    signature: CommandSignature,
    is_cas: bool,
) -> frozenset[VisualizationEnvironment]:
    if is_cas:
        return frozenset({VisualizationEnvironment.cas})
    expected = {item for group in signature.expected_types for item in group}
    if command_name in _THREE_D_COMMANDS or expected.intersection(_THREE_D_TYPES):
        return frozenset({VisualizationEnvironment.graphics_3d})

    result: set[VisualizationEnvironment] = set()
    for category in categories:
        normalized = category.lower()
        if normalized == "3d":
            if command_name in GLOBAL_CORE_COMMANDS:
                result.add(VisualizationEnvironment.geometry_2d)
            else:
                result.add(VisualizationEnvironment.graphics_3d)
        elif normalized == "cas":
            result.add(VisualizationEnvironment.cas)
        elif normalized == "spreadsheet":
            result.add(VisualizationEnvironment.spreadsheet)
        elif normalized == "probability":
            result.add(VisualizationEnvironment.probability)
        elif normalized in {"statistics", "chart"}:
            result.add(VisualizationEnvironment.statistics)
        elif normalized in {"geometry", "conic", "transformation"}:
            result.add(VisualizationEnvironment.geometry_2d)
        elif normalized in {
            "algebra",
            "functions_and_calculus",
            "list",
            "vector_and_matrix",
        }:
            result.add(VisualizationEnvironment.graphing)

    if not result:
        result.add(VisualizationEnvironment.graphing)
    return frozenset(result)


def _unsafe_reason(command_name: str) -> str | None:
    if command_name in _UNSAFE_EXACT:
        return _UNSAFE_EXACT[command_name]
    if command_name.startswith("Set"):
        return "state and styling changes must use structured applet API operations"
    if "Script" in command_name or "JavaScript" in command_name:
        return "GeoGebra and JavaScript scripting are disabled"
    return None


class GeoGebraCommandRegistry:
    def __init__(self, catalog_path: Path | None = None) -> None:
        self.catalog_path = catalog_path or get_settings().geogebra_catalog_path
        self.metadata: dict[str, Any] = {}
        self._definitions = self._load(self.catalog_path)
        self._lower_names = {name.lower(): name for name in self._definitions}

    @classmethod
    @lru_cache(maxsize=4)
    def cached(cls, catalog_path: str | None = None) -> "GeoGebraCommandRegistry":
        return cls(Path(catalog_path) if catalog_path else None)

    def _load(self, path: Path) -> dict[str, CommandDefinition]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load GeoGebra command catalog at {path}: {exc}") from exc

        if isinstance(payload, dict):
            entries = payload.get("commands", [])
            self.metadata = dict(payload.get("metadata", {}))
        else:
            entries = payload
        if not isinstance(entries, list):
            raise RuntimeError("GeoGebra command catalog must contain a command list.")

        grouped: dict[str, list[CommandOverload]] = defaultdict(list)
        canonical_names: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("command_name", "")).strip()
            syntax = str(entry.get("syntax", "")).strip()
            if not name or not syntax:
                continue
            canonical = canonical_names.setdefault(name.lower(), name)
            signature = parse_signature(syntax, canonical)
            categories = tuple(
                dict.fromkeys(
                    str(category)
                    for category in (entry.get("all_categories") or [entry.get("category", "other")])
                    if category
                )
            ) or ("other",)
            is_cas = bool(entry.get("is_cas", False))
            grouped[canonical].append(
                CommandOverload(
                    command_name=canonical,
                    signature=signature,
                    description=str(entry.get("description", "")).strip(),
                    examples=tuple(str(example) for example in entry.get("examples", [])[:3]),
                    category=str(entry.get("category", categories[0])),
                    all_categories=categories,
                    is_cas=is_cas,
                    capabilities=_category_capabilities(
                        command_name=canonical,
                        categories=categories,
                        signature=signature,
                        is_cas=is_cas,
                    ),
                    source=entry.get("source") if isinstance(entry.get("source"), dict) else None,
                    web_compatibility=str(entry.get("web_compatibility", "unknown")),
                )
            )

        return {
            name: CommandDefinition(
                name=name,
                overloads=tuple(
                    sorted(overloads, key=lambda item: item.signature.original.casefold())
                ),
                unsafe_reason=_unsafe_reason(name),
            )
            for name, overloads in sorted(grouped.items(), key=lambda item: item[0].casefold())
        }

    def __len__(self) -> int:
        return len(self._definitions)

    @property
    def overload_count(self) -> int:
        return sum(len(definition.overloads) for definition in self._definitions.values())

    def lookup(self, command_name: str) -> CommandDefinition | None:
        canonical = self._lower_names.get(command_name.strip().lower())
        return self._definitions.get(canonical) if canonical else None

    def by_capability(
        self, environment: VisualizationEnvironment, *, include_unsafe: bool = False
    ) -> list[CommandDefinition]:
        return [
            definition
            for definition in self._definitions.values()
            if environment in definition.capabilities
            and (include_unsafe or definition.unsafe_reason is None)
        ]

    def output_type(self, command_name: str) -> GeoGebraObjectType:
        return _OUTPUT_TYPES.get(command_name, GeoGebraObjectType.UNKNOWN)

    def search(
        self,
        query: str,
        environment: VisualizationEnvironment,
        *,
        limit: int = 12,
        overload_limit: int = 4,
    ) -> list[RetrievedCommand]:
        limit = max(1, min(limit, 20))
        query_lower = query.casefold()
        tokens = {token for token in _TOKEN.findall(query_lower) if token not in _STOP_WORDS}
        alias_targets = {
            target
            for token in tokens
            for target in _ALIASES.get(token, ())
        }
        scored: list[tuple[float, CommandDefinition]] = []

        for definition in self.by_capability(environment):
            name_lower = definition.name.casefold()
            name_tokens = set(_TOKEN.findall(name_lower))
            score = 0.0
            if name_lower in query_lower:
                score += 100.0
            if definition.name in alias_targets:
                score += 80.0
            score += 12.0 * len(tokens.intersection(name_tokens))

            searchable = " ".join(
                (
                    definition.name,
                    *(overload.description for overload in definition.overloads[:4]),
                    *(" ".join(overload.all_categories) for overload in definition.overloads[:4]),
                )
            ).casefold()
            overlap = sum(1 for token in tokens if token in searchable)
            score += min(overlap, 6) * 3.0
            if definition.name in GLOBAL_CORE_COMMANDS:
                score += 1.0
            if score > 3.0:
                scored.append((score, definition))

        scored.sort(key=lambda item: (-item[0], item[1].name.casefold()))
        if scored and scored[0][0] >= 80.0:
            # Explicit command/alias matches are strong enough that low-overlap
            # documentation hits only add noise to a small local-model prompt.
            scored = [item for item in scored if item[0] >= 40.0]
        results: list[RetrievedCommand] = []
        for score, definition in scored[:limit]:
            overloads = [
                overload
                for overload in definition.overloads
                if environment in overload.capabilities
            ][:overload_limit]
            if not overloads:
                continue
            results.append(
                RetrievedCommand(
                    name=definition.name,
                    score=round(score, 3),
                    signatures=tuple(item.signature.original for item in overloads),
                    description=next(
                        (item.description for item in overloads if item.description), ""
                    )[:240],
                )
            )
        return results
