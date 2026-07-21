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
from app.services.geogebra_support_policy import (
    SupportStatus,
    load_runtime_acceptance,
    permanent_block_reason,
)


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
    families: tuple[str, ...]
    support_status: SupportStatus
    runtime_accepted_environments: frozenset[VisualizationEnvironment]
    support_requirements: tuple[str, ...]
    is_cas: bool
    capabilities: frozenset[VisualizationEnvironment]
    output_type: GeoGebraObjectType
    output_type_strategy: str | None
    source: dict[str, str] | None
    web_compatibility: str

    def is_supported_in(self, environment: VisualizationEnvironment) -> bool:
        return (
            self.support_status is SupportStatus.supported
            and environment in self.runtime_accepted_environments
            and environment in self.capabilities
        )

    def is_runtime_eligible_in(
        self, environment: VisualizationEnvironment
    ) -> bool:
        """Allow safe experimental overloads to rely on browser validation."""

        return (
            self.support_status is not SupportStatus.blocked
            and self.signature.normalization_certain
            and environment in self.capabilities
        )


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

    @property
    def families(self) -> frozenset[str]:
        return frozenset(
            family for overload in self.overloads for family in overload.families
        )

    @property
    def support_status(self) -> SupportStatus:
        statuses = {overload.support_status for overload in self.overloads}
        if SupportStatus.supported in statuses:
            return SupportStatus.supported
        if statuses == {SupportStatus.blocked}:
            return SupportStatus.blocked
        return SupportStatus.experimental


    def runtime_eligible_overloads(
        self, environment: VisualizationEnvironment
    ) -> tuple[CommandOverload, ...]:
        return tuple(
            overload
            for overload in self.overloads
            if overload.is_runtime_eligible_in(environment)
        )


@dataclass(frozen=True)
class RetrievedCommand:
    name: str
    score: float
    signatures: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class OverloadSupportEvaluation:
    status: SupportStatus
    runtime_eligible: bool
    runtime_environments: frozenset[VisualizationEnvironment]
    capabilities: frozenset[VisualizationEnvironment]
    output_type: GeoGebraObjectType
    output_type_strategy: str | None
    requirements: tuple[str, ...]
    unsafe_reason: str | None


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

_FAMILY_ENVIRONMENTS: dict[str, frozenset[VisualizationEnvironment]] = {
    "geometry_2d": frozenset({VisualizationEnvironment.geometry_2d}),
    "transformations": frozenset({VisualizationEnvironment.geometry_2d}),
    "graphing_calculus": frozenset({VisualizationEnvironment.graphing}),
    "graphics_3d": frozenset({VisualizationEnvironment.graphics_3d}),
    "cas": frozenset({VisualizationEnvironment.cas}),
    "statistics": frozenset({VisualizationEnvironment.statistics}),
    "probability": frozenset({VisualizationEnvironment.probability}),
    "spreadsheet": frozenset({VisualizationEnvironment.spreadsheet}),
}
_DEFAULT_FAMILY_ENVIRONMENTS = frozenset({VisualizationEnvironment.graphing})


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
    variadic = "..." in arguments_text
    labels: list[str] = []
    optional_arguments = 0
    certain = command_name.casefold() == fallback_name.casefold()
    for raw_part in arguments_text.split(","):
        part = raw_part.strip()
        if not part or part == "...":
            continue
        part = part.replace("...", "").strip()
        is_optional = bool(re.search(r"\(\s*optional\s*\)", part, re.IGNORECASE))
        part = re.sub(r"\(\s*optional\s*\)", "", part, flags=re.IGNORECASE).strip()
        placeholders = _PLACEHOLDER.findall(part)
        if placeholders:
            part_labels = [label.strip() for label in placeholders if label.strip()]
            remainder = _PLACEHOLDER.sub("", part).strip().strip('"\'')
            if remainder:
                certain = False
        else:
            # A small set of official manual signatures use unwrapped labels,
            # quoted placeholders, or the literal independent variable x.
            part_labels = [part.strip().strip('"\'').strip()]
        if not part_labels or any(not label for label in part_labels):
            certain = False
            continue
        labels.extend(part_labels)
        optional_arguments += int(is_optional)

    label_values = tuple(labels)
    return CommandSignature(
        original=syntax,
        command_name=command_name,
        argument_labels=label_values,
        expected_types=tuple(_expected_types(label) for label in label_values),
        min_arguments=max(0, len(label_values) - optional_arguments),
        max_arguments=None if variadic else len(label_values),
        variadic=variadic,
        normalization_certain=certain,
    )


def _family_capabilities(
    families: Iterable[str], *, is_cas: bool
) -> frozenset[VisualizationEnvironment]:
    """Derive runtime views from generated catalog families, never command names."""

    result: set[VisualizationEnvironment] = set()
    for family in families:
        result.update(
            _FAMILY_ENVIRONMENTS.get(
                family.casefold(), _DEFAULT_FAMILY_ENVIRONMENTS
            )
        )
    if is_cas:
        result.add(VisualizationEnvironment.cas)
    return frozenset(result or _DEFAULT_FAMILY_ENVIRONMENTS)


def _support_requirements(
    *,
    signature: CommandSignature,
    capabilities: frozenset[VisualizationEnvironment],
    runtime_environments: frozenset[VisualizationEnvironment],
    output_type: GeoGebraObjectType,
    output_type_strategy: str | None,
    unsafe_reason: str | None,
) -> tuple[str, ...]:
    if unsafe_reason:
        return ("permanently_blocked",)

    requirements: list[str] = []
    if not signature.normalization_certain:
        requirements.append("safely_normalized_signature")
    if (
        len(signature.expected_types) != len(signature.argument_labels)
        or any(not expected for expected in signature.expected_types)
    ):
        requirements.append("representable_typed_arguments")
    if output_type is GeoGebraObjectType.UNKNOWN and output_type_strategy is None:
        requirements.append("known_output_type")
    if not capabilities or not runtime_environments.issubset(capabilities):
        requirements.append("correct_environment_metadata")
    if not runtime_environments:
        requirements.append("runtime_acceptance_test")
    return tuple(requirements)


def _effective_support_status(
    *, unsafe_reason: str | None, requirements: tuple[str, ...]
) -> SupportStatus:
    if unsafe_reason:
        return SupportStatus.blocked
    if not requirements:
        return SupportStatus.supported
    return SupportStatus.experimental


def evaluate_overload_support(
    *,
    command_name: str,
    syntax: str,
    categories: Iterable[str],
    families: Iterable[str],
    is_cas: bool,
    runtime_environment_values: Iterable[str],
    runtime_output_type_value: str | None = None,
    runtime_output_type_strategy: str | None = None,
) -> OverloadSupportEvaluation:
    """Derive support from all rollout prerequisites for one exact overload."""

    _ = tuple(categories)
    family_values = tuple(families)
    signature = parse_signature(syntax, command_name)
    capabilities = _family_capabilities(family_values, is_cas=is_cas)
    try:
        runtime_environments = frozenset(
            VisualizationEnvironment(value) for value in runtime_environment_values
        )
        output_type = (
            GeoGebraObjectType(runtime_output_type_value)
            if runtime_output_type_value
            else GeoGebraObjectType.UNKNOWN
        )
    except ValueError as exc:
        raise ValueError(
            f"Runtime acceptance for '{command_name}' has unknown metadata."
        ) from exc
    if runtime_output_type_strategy not in {None, "same_as_first_argument"}:
        raise ValueError(
            f"Runtime acceptance for '{command_name}' has an unknown output strategy."
        )
    if output_type is not GeoGebraObjectType.UNKNOWN and runtime_output_type_strategy:
        raise ValueError(
            f"Runtime acceptance for '{command_name}' declares two output strategies."
        )
    unsafe_reason = permanent_block_reason(command_name, family_values)
    requirements = _support_requirements(
        signature=signature,
        capabilities=capabilities,
        runtime_environments=runtime_environments,
        output_type=output_type,
        output_type_strategy=runtime_output_type_strategy,
        unsafe_reason=unsafe_reason,
    )
    return OverloadSupportEvaluation(
        status=_effective_support_status(
            unsafe_reason=unsafe_reason,
            requirements=requirements,
        ),
        runtime_eligible=(
            unsafe_reason is None
            and signature.normalization_certain
            and bool(capabilities)
        ),
        runtime_environments=runtime_environments,
        capabilities=capabilities,
        output_type=output_type,
        output_type_strategy=runtime_output_type_strategy,
        requirements=requirements,
        unsafe_reason=unsafe_reason,
    )


def _schema_at_least(metadata: dict[str, Any], major: int, minor: int) -> bool:
    value = str(metadata.get("schema_version", "0.0"))
    try:
        current_major, current_minor = (int(part) for part in value.split(".", 1))
    except (TypeError, ValueError):
        return False
    return (current_major, current_minor) >= (major, minor)


class GeoGebraCommandRegistry:
    def __init__(self, catalog_path: Path | None = None) -> None:
        self._enforce_all_acceptance = catalog_path is None
        self.catalog_path = catalog_path or get_settings().geogebra_catalog_path
        self.metadata: dict[str, Any] = {}
        self.runtime_acceptance = load_runtime_acceptance()
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

        require_support_metadata = _schema_at_least(self.metadata, 1, 2)
        require_generated_metadata = _schema_at_least(self.metadata, 1, 4)
        grouped: dict[str, list[CommandOverload]] = defaultdict(list)
        canonical_names: dict[str, str] = {}
        unsafe_reasons: dict[str, str | None] = {}
        matched_acceptance_keys: set[tuple[str, str]] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("command_name", "")).strip()
            syntax = str(entry.get("syntax", "")).strip()
            if not name or not syntax:
                continue
            canonical = canonical_names.setdefault(name.lower(), name)
            categories = tuple(
                dict.fromkeys(
                    str(category)
                    for category in (entry.get("all_categories") or [entry.get("category", "other")])
                    if category
                )
            ) or ("other",)
            families = tuple(
                dict.fromkeys(
                    str(family) for family in entry.get("families", []) if family
                )
            ) or ("other",)
            is_cas = bool(entry.get("is_cas", False))
            signature = parse_signature(syntax, canonical)
            acceptance_key = (canonical.casefold(), signature.original)
            runtime_acceptance = self.runtime_acceptance.get(acceptance_key)
            expected_runtime_values = (
                runtime_acceptance.environments
                if runtime_acceptance is not None
                else frozenset()
            )
            if runtime_acceptance is not None:
                matched_acceptance_keys.add(acceptance_key)
            try:
                evaluation = evaluate_overload_support(
                    command_name=canonical,
                    syntax=syntax,
                    categories=categories,
                    families=families,
                    is_cas=is_cas,
                    runtime_environment_values=expected_runtime_values,
                    runtime_output_type_value=(
                        runtime_acceptance.output_type
                        if runtime_acceptance is not None
                        else None
                    ),
                    runtime_output_type_strategy=(
                        runtime_acceptance.output_type_strategy
                        if runtime_acceptance is not None
                        else None
                    ),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid support metadata for '{canonical}': {signature.original}"
                ) from exc
            raw_runtime_values = entry.get("runtime_accepted_environments", [])
            if not isinstance(raw_runtime_values, list):
                raw_runtime_values = []
            if require_support_metadata and set(raw_runtime_values) != set(
                expected_runtime_values
            ):
                raise RuntimeError(
                    f"Catalog runtime acceptance is stale for '{canonical}': "
                    f"{signature.original}"
                )
            raw_capabilities = entry.get("capabilities", [])
            if not isinstance(raw_capabilities, list):
                raw_capabilities = []
            if require_generated_metadata and set(raw_capabilities) != {
                capability.value for capability in evaluation.capabilities
            }:
                raise RuntimeError(
                    f"Catalog capabilities are stale for '{canonical}': "
                    f"{signature.original}"
                )
            raw_output_type = str(entry.get("output_type", "Unknown"))
            raw_output_strategy = entry.get("output_type_strategy")
            if require_generated_metadata and (
                raw_output_type != evaluation.output_type.value
                or raw_output_strategy != evaluation.output_type_strategy
            ):
                raise RuntimeError(
                    f"Catalog output metadata is stale for '{canonical}': "
                    f"{signature.original}"
                )

            unsafe_reason = evaluation.unsafe_reason
            unsafe_reasons[canonical] = unsafe_reasons.get(canonical) or unsafe_reason
            effective_status = evaluation.status
            raw_runtime_eligible = entry.get("runtime_eligible", False)
            if (
                require_support_metadata
                and raw_runtime_eligible is not evaluation.runtime_eligible
            ):
                raise RuntimeError(
                    f"Catalog runtime eligibility is stale for '{canonical}': "
                    f"{signature.original}"
                )
            raw_requirements = entry.get("support_requirements", [])
            if not isinstance(raw_requirements, list):
                raw_requirements = []
            if require_support_metadata and tuple(raw_requirements) != evaluation.requirements:
                raise RuntimeError(
                    f"Catalog support requirements are stale for '{canonical}': "
                    f"{signature.original}"
                )
            raw_status = entry.get("support_status", SupportStatus.experimental.value)
            try:
                declared_status = SupportStatus(raw_status)
            except ValueError as exc:
                raise RuntimeError(
                    f"Catalog overload has invalid support status {raw_status!r}."
                ) from exc
            if require_support_metadata and declared_status is not effective_status:
                raise RuntimeError(
                    f"Catalog support status is stale for '{canonical}': "
                    f"{signature.original}; expected {effective_status.value}."
                )
            grouped[canonical].append(
                CommandOverload(
                    command_name=canonical,
                    signature=signature,
                    description=str(entry.get("description", "")).strip(),
                    examples=tuple(str(example) for example in entry.get("examples", [])[:3]),
                    category=str(entry.get("category", categories[0])),
                    all_categories=categories,
                    families=families,
                    support_status=effective_status,
                    runtime_accepted_environments=evaluation.runtime_environments,
                    support_requirements=evaluation.requirements,
                    is_cas=is_cas,
                    capabilities=evaluation.capabilities,
                    output_type=evaluation.output_type,
                    output_type_strategy=evaluation.output_type_strategy,
                    source=entry.get("source") if isinstance(entry.get("source"), dict) else None,
                    web_compatibility=str(entry.get("web_compatibility", "unknown")),
                )
            )

        unmatched_acceptance = set(self.runtime_acceptance).difference(
            matched_acceptance_keys
        )
        catalog_command_keys = set(canonical_names)
        relevant_unmatched_acceptance = {
            key
            for key in unmatched_acceptance
            if self._enforce_all_acceptance or key[0] in catalog_command_keys
        }
        if relevant_unmatched_acceptance:
            command_name, signature = sorted(relevant_unmatched_acceptance)[0]
            raise RuntimeError(
                "Runtime acceptance names an overload absent from the catalog: "
                f"{command_name}: {signature}"
            )

        return {
            name: CommandDefinition(
                name=name,
                overloads=tuple(
                    sorted(overloads, key=lambda item: item.signature.original.casefold())
                ),
                unsafe_reason=unsafe_reasons.get(name),
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

    def by_family(
        self, family: str, *, include_unsafe: bool = False
    ) -> list[CommandDefinition]:
        normalized = family.strip().casefold()
        return [
            definition
            for definition in self._definitions.values()
            if any(item.casefold() == normalized for item in definition.families)
            and (include_unsafe or definition.unsafe_reason is None)
        ]

    def by_support_status(
        self, status: SupportStatus
    ) -> list[CommandDefinition]:
        return [
            definition
            for definition in self._definitions.values()
            if definition.support_status is status
        ]

    def output_type(self, command_name: str) -> GeoGebraObjectType:
        definition = self.lookup(command_name)
        if definition is None:
            return GeoGebraObjectType.UNKNOWN
        output_types = {
            overload.output_type
            for overload in definition.overloads
            if overload.output_type is not GeoGebraObjectType.UNKNOWN
        }
        return (
            next(iter(output_types))
            if len(output_types) == 1
            else GeoGebraObjectType.UNKNOWN
        )

    def output_type_strategy(self, command_name: str) -> str | None:
        definition = self.lookup(command_name)
        if definition is None:
            return None
        strategies = {
            overload.output_type_strategy
            for overload in definition.overloads
            if overload.output_type_strategy is not None
        }
        return next(iter(strategies)) if len(strategies) == 1 else None

    def search(
        self,
        query: str,
        environment: VisualizationEnvironment,
        *,
        limit: int = 12,
        overload_limit: int = 12,
    ) -> list[RetrievedCommand]:
        limit = max(1, min(limit, 20))
        query_lower = query.casefold()
        tokens = {token for token in _TOKEN.findall(query_lower) if token not in _STOP_WORDS}
        scored: list[tuple[float, CommandDefinition]] = []

        for definition in self.by_capability(environment):
            eligible_overloads = definition.runtime_eligible_overloads(environment)
            if not eligible_overloads:
                continue
            name_lower = definition.name.casefold()
            name_tokens = set(_TOKEN.findall(name_lower))
            score = 0.0
            if name_lower in query_lower:
                score += 100.0
            score += 12.0 * len(tokens.intersection(name_tokens))

            searchable = " ".join(
                (
                    definition.name,
                    *(overload.description for overload in eligible_overloads),
                    *(" ".join(overload.all_categories) for overload in eligible_overloads),
                    *(" ".join(overload.families) for overload in eligible_overloads),
                )
            ).casefold()
            overlap = sum(1 for token in tokens if token in searchable)
            score += min(overlap, 6) * 3.0
            if score > 3.0:
                scored.append((score, definition))

        scored.sort(key=lambda item: (-item[0], item[1].name.casefold()))
        if scored and scored[0][0] >= 80.0:
            # Explicit command matches are strong enough that low-overlap
            # documentation hits only add noise to a small local-model prompt.
            scored = [item for item in scored if item[0] >= 40.0]
        results: list[RetrievedCommand] = []
        for score, definition in scored[:limit]:
            overloads = list(definition.runtime_eligible_overloads(environment))[
                : max(1, min(overload_limit, 12))
            ]
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
