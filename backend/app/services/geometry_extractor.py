from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.integrations.errors import (
    IntegrationFailureCategory,
    exception_diagnostics,
)
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.schemas.geometry_dsl import (
    GeometryAction,
    GeometryActionType,
    GeometryDSL,
    GeoGebraValidationIssue,
    RenderHints,
    ValidationSeverity,
    VisualizationEnvironment,
)
from app.services.geogebra_command_registry import (
    CommandSignature,
    GeoGebraCommandRegistry,
    GeoGebraObjectType,
    RetrievedCommand,
)
from app.services.geogebra_validator import GeoGebraDSLValidator
from app.services.model_router import (
    NVIDIA_DIRECT_FALLBACK_MODELS,
    NVIDIA_GPT_OSS_20B_MODEL,
    remote_model_timeout_seconds,
)


logger = logging.getLogger(__name__)
GEOMETRY_RETRIEVAL_LIMIT = 10
LOCAL_GEOMETRY_MAX_PROMPT_CHARS = 4_000


def _ordered_geometry_fallback_models(parser_model: str) -> tuple[str, ...]:
    """Return each eligible NVIDIA model once, preferred model first."""

    preferred = (
        (parser_model,)
        if parser_model in NVIDIA_DIRECT_FALLBACK_MODELS
        else (NVIDIA_GPT_OSS_20B_MODEL,)
    )
    return tuple(dict.fromkeys((*preferred, *NVIDIA_DIRECT_FALLBACK_MODELS)))




LOCAL_GEOMETRY_EXTRACTION_PROMPT = """
Convert the math problem into a minimal GeoGebra construction plan.
Return exactly one JSON object and nothing else. Do not solve or prove the problem.
Use only the top-level keys summary and dsl. Keep summary at 200 characters or fewer.
Do not invent mathematical relationships that the problem does not give. When the
user explicitly requests a visualization but omits placement or scale, you may choose
simple finite display coordinates or dimensions solely to make the requested object visible.
Preserve labels exactly. References may be emitted out of order; trusted code will sort them.

Allowed actions:
CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT,
MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION,
DEFINE_OBJECT, and EXECUTE_COMMAND only for a retrieved command listed below.

Required fields for each action (never omit them):
- CREATE_POINT: action, label; add coordinates only when explicitly given.
- CREATE_LINE: action, label, points with exactly two existing point labels.
- CREATE_CIRCLE: action, label, center and radius; or through with two existing points.
- CREATE_POLYGON: action, label, points with at least three existing point labels.
- INTERSECT: action, label, metadata.objects with two existing object labels.
- MIDPOINT: action, label, points with exactly two existing point labels.
- PERPENDICULAR or PARALLEL: action, label, line, metadata.through_point.
- ANGLE_BISECTOR: action, label, points with exactly three existing point labels.
- CREATE_FUNCTION: action, label, equation.
- DEFINE_OBJECT: action, output, object_type, value. Prefer this over
  CREATE_FUNCTION for explicit function definitions such as f(x) = x^2.
- EXECUTE_COMMAND: action, command, arguments, and normally output. Every argument
  must use a typed kind; never put raw GeoGebra syntax in a string.
When retrieved commands are provided, prefer one minimal EXECUTE_COMMAND action
that directly represents the request. Match one listed overload exactly. If the
prompt omits display placement or scale, supply simple finite typed arguments for
that overload so the requested object is still visible; never leave required
arguments empty.

Use an empty actions list when no faithful construction can be extracted.
Omit only optional fields. Never shorten or summarize action objects.
For a named center, vertex, or endpoint, emit CREATE_POINT before using its label.
Use construction actions such as MIDPOINT instead of inventing coordinates for derived points.
""".strip()


def _action_schema(
    action: str,
    required_fields: list[str],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "const": action},
            "label": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$"},
            **properties,
        },
        "required": ["action", "label", *required_fields],
        "additionalProperties": False,
    }


_POINT_LABELS = {
    "type": "array",
    "items": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$"},
}
_OBJECT_PAIR_METADATA = {
    "type": "object",
    "properties": {
        "objects": {**_POINT_LABELS, "minItems": 2, "maxItems": 2},
    },
    "required": ["objects"],
    "additionalProperties": False,
}
_THROUGH_POINT_METADATA = {
    "type": "object",
    "properties": {
        "through_point": {
            "type": "string",
            "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$",
        }
    },
    "required": ["through_point"],
    "additionalProperties": False,
}

_REFERENCE_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "reference"},
        "value": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$"},
    },
    "required": ["kind", "value"],
    "additionalProperties": False,
}
_NUMBER_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "number"},
        "value": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
    },
    "required": ["kind", "value"],
    "additionalProperties": False,
}
_ANGLE_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "angle"},
        "value": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
        "unit": {"type": "string", "enum": ["degree", "radian"]},
    },
    "required": ["kind", "value", "unit"],
    "additionalProperties": False,
}


def _coordinate_argument_schema(kind: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": kind},
            "x": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
            "y": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
            "z": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
        },
        "required": ["kind", "x", "y"],
        "additionalProperties": False,
    }


_TEXT_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "text"},
        "value": {"type": "string", "maxLength": 500},
    },
    "required": ["kind", "value"],
    "additionalProperties": False,
}
_BOOLEAN_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "boolean"},
        "value": {"type": "boolean"},
    },
    "required": ["kind", "value"],
    "additionalProperties": False,
}


def _string_argument_schema(kind: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": kind},
            "value": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": ["kind", "value"],
        "additionalProperties": False,
    }


_INTERVAL_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "interval"},
        "lower": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
        "upper": {"type": "number", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
        "lower_inclusive": {"type": "boolean"},
        "upper_inclusive": {"type": "boolean"},
    },
    "required": ["kind", "lower", "upper", "lower_inclusive", "upper_inclusive"],
    "additionalProperties": False,
}
_NON_LIST_ARGUMENT_SCHEMAS = [
    _REFERENCE_ARGUMENT_SCHEMA,
    _NUMBER_ARGUMENT_SCHEMA,
    _ANGLE_ARGUMENT_SCHEMA,
    _coordinate_argument_schema("point"),
    _coordinate_argument_schema("vector"),
    _TEXT_ARGUMENT_SCHEMA,
    _BOOLEAN_ARGUMENT_SCHEMA,
    _string_argument_schema("expression"),
    _string_argument_schema("equation"),
    _INTERVAL_ARGUMENT_SCHEMA,
]
def _list_argument_schema(depth: int) -> dict[str, Any]:
    item_schemas = list(_NON_LIST_ARGUMENT_SCHEMAS)
    if depth > 1:
        item_schemas.append(_list_argument_schema(depth - 1))
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "list"},
            "items": {
                "type": "array",
                "maxItems": 32,
                "items": {"oneOf": item_schemas},
            },
        },
        "required": ["kind", "items"],
        "additionalProperties": False,
    }


_LIST_ARGUMENT_SCHEMA = _list_argument_schema(2)


def _generic_action_schema(command_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "const": "EXECUTE_COMMAND"},
            "output": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$"},
            "command": {"type": "string", "enum": command_names},
            "arguments": {
                "type": "array",
                "maxItems": 16,
                "items": {"oneOf": [*_NON_LIST_ARGUMENT_SCHEMAS, _LIST_ARGUMENT_SCHEMA]},
            },
        },
        "required": ["action", "command", "arguments"],
        "additionalProperties": False,
    }


_ARGUMENT_SCHEMA_BY_OBJECT_TYPE: dict[GeoGebraObjectType, dict[str, Any]] = {
    GeoGebraObjectType.POINT: _coordinate_argument_schema("point"),
    GeoGebraObjectType.NUMBER: _NUMBER_ARGUMENT_SCHEMA,
    GeoGebraObjectType.ANGLE: _ANGLE_ARGUMENT_SCHEMA,
    GeoGebraObjectType.VECTOR: _coordinate_argument_schema("vector"),
    GeoGebraObjectType.TEXT: _TEXT_ARGUMENT_SCHEMA,
    GeoGebraObjectType.BOOLEAN: _BOOLEAN_ARGUMENT_SCHEMA,
    GeoGebraObjectType.EQUATION: _string_argument_schema("equation"),
    GeoGebraObjectType.LIST: _LIST_ARGUMENT_SCHEMA,
    GeoGebraObjectType.MATRIX: _LIST_ARGUMENT_SCHEMA,
    GeoGebraObjectType.INTERVAL: _INTERVAL_ARGUMENT_SCHEMA,
}
_SELF_CONTAINED_ARGUMENT_TYPES = frozenset(_ARGUMENT_SCHEMA_BY_OBJECT_TYPE)


def _signature_is_self_contained(signature: CommandSignature) -> bool:
    return (
        signature.normalization_certain
        and signature.max_arguments is not None
        and all(
            bool(expected.intersection(_SELF_CONTAINED_ARGUMENT_TYPES))
            for expected in signature.expected_types
        )
    )


def _commands_have_self_contained_overloads(
    command_signatures: dict[str, tuple[CommandSignature, ...]],
) -> bool:
    return bool(command_signatures) and all(
        any(_signature_is_self_contained(signature) for signature in signatures)
        for signatures in command_signatures.values()
    )


def _expected_argument_schema(
    expected_types: frozenset[GeoGebraObjectType],
) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for object_type in sorted(expected_types, key=lambda item: item.value):
        schema = _ARGUMENT_SCHEMA_BY_OBJECT_TYPE.get(
            object_type, _REFERENCE_ARGUMENT_SCHEMA
        )
        fingerprint = json.dumps(schema, sort_keys=True)
        if fingerprint not in seen:
            seen.add(fingerprint)
            options.append(schema)
    if not options or GeoGebraObjectType.UNKNOWN in expected_types:
        return {
            "oneOf": [*_NON_LIST_ARGUMENT_SCHEMAS, _LIST_ARGUMENT_SCHEMA]
        }
    return options[0] if len(options) == 1 else {"oneOf": options}


def _typed_generic_action_schemas(
    command_signatures: dict[str, tuple[CommandSignature, ...]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for command_name in sorted(command_signatures, key=str.casefold):
        for signature in command_signatures[command_name]:
            if (
                not signature.normalization_certain
                or signature.max_arguments is None
                or len(signature.expected_types) != signature.max_arguments
            ):
                continue
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "const": "EXECUTE_COMMAND",
                        },
                        "output": {
                            "type": "string",
                            "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$",
                        },
                        "command": {"type": "string", "const": command_name},
                        "arguments": {
                            "type": "array",
                            "minItems": signature.min_arguments,
                            "maxItems": signature.max_arguments,
                            "prefixItems": [
                                _expected_argument_schema(expected)
                                for expected in signature.expected_types
                            ],
                        },
                    },
                    "required": ["action", "command", "arguments"],
                    "additionalProperties": False,
                }
            )
    return variants


def _definition_action_schemas() -> list[dict[str, Any]]:
    value_schemas = {
        "function": _string_argument_schema("equation"),
        "equation": _string_argument_schema("equation"),
        "expression": _string_argument_schema("expression"),
        "number": _NUMBER_ARGUMENT_SCHEMA,
        "point": _coordinate_argument_schema("point"),
        "vector": _coordinate_argument_schema("vector"),
        "list": _LIST_ARGUMENT_SCHEMA,
        "text": _TEXT_ARGUMENT_SCHEMA,
        "boolean": _BOOLEAN_ARGUMENT_SCHEMA,
        "interval": _INTERVAL_ARGUMENT_SCHEMA,
    }
    return [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "DEFINE_OBJECT"},
                "output": {
                    "type": "string",
                    "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$",
                },
                "object_type": {"type": "string", "const": object_type},
                "value": value_schema,
            },
            "required": ["action", "output", "object_type", "value"],
            "additionalProperties": False,
        }
        for object_type, value_schema in value_schemas.items()
    ]


def geometry_response_schema(
    command_names: list[str] | None = None,
    environment: VisualizationEnvironment = VisualizationEnvironment.geometry_2d,
    *,
    generic_commands_only: bool = False,
    command_signatures: dict[str, tuple[CommandSignature, ...]] | None = None,
) -> dict[str, Any]:
    command_names = sorted(set(command_names or []), key=str.casefold)
    high_level_action_schemas = [
        _action_schema(
            "CREATE_POINT",
            [],
            {
                "coordinates": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
        ),
        _action_schema(
            "CREATE_LINE",
            ["points"],
            {"points": {**_POINT_LABELS, "minItems": 2, "maxItems": 2}},
        ),
        _action_schema(
            "CREATE_CIRCLE",
            ["center", "radius"],
            {
                "center": {"type": "string"},
                "radius": {"type": "number", "exclusiveMinimum": 0},
            },
        ),
        _action_schema(
            "CREATE_CIRCLE",
            ["through"],
            {"through": {**_POINT_LABELS, "minItems": 2, "maxItems": 3}},
        ),
        _action_schema(
            "CREATE_POLYGON",
            ["points"],
            {"points": {**_POINT_LABELS, "minItems": 3}},
        ),
        _action_schema("INTERSECT", ["metadata"], {"metadata": _OBJECT_PAIR_METADATA}),
        _action_schema(
            "MIDPOINT",
            ["points"],
            {"points": {**_POINT_LABELS, "minItems": 2, "maxItems": 2}},
        ),
        _action_schema(
            "PERPENDICULAR",
            ["line", "metadata"],
            {"line": {"type": "string"}, "metadata": _THROUGH_POINT_METADATA},
        ),
        _action_schema(
            "PARALLEL",
            ["line", "metadata"],
            {"line": {"type": "string"}, "metadata": _THROUGH_POINT_METADATA},
        ),
        _action_schema(
            "ANGLE_BISECTOR",
            ["points"],
            {"points": {**_POINT_LABELS, "minItems": 3, "maxItems": 3}},
        ),
        _action_schema(
            "CREATE_FUNCTION",
            ["equation"],
            {"equation": {"type": "string", "minLength": 1, "maxLength": 200}},
        ),
        *_definition_action_schemas(),
    ]
    action_schemas = [] if generic_commands_only else high_level_action_schemas
    typed_generic_schemas = _typed_generic_action_schemas(command_signatures or {})
    if typed_generic_schemas:
        action_schemas.extend(typed_generic_schemas)
    elif command_names:
        action_schemas.append(_generic_action_schema(command_names))
    return {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 200},
        "dsl": {
            "type": "object",
            "properties": {
                "version": {"type": "string", "const": "1.1"},
                "space": {
                    "type": "string",
                    "const": "euclidean_3d" if environment is VisualizationEnvironment.graphics_3d else "euclidean_2d",
                },
                "environment": {"type": "string", "const": environment.value},
                "actions": {
                    "type": "array",
                    "maxItems": 12 if generic_commands_only else 40,
                    "items": {
                        "oneOf": action_schemas
                    },
                },
                "render_hints": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            "required": ["version", "space", "environment", "actions", "render_hints"],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "dsl"],
    "additionalProperties": False,
    }




def geometry_repair_schema(
    command_names: list[str],
    *,
    replacement_count: int,
) -> dict[str, Any]:
    action_schema = geometry_response_schema(command_names)["properties"]["dsl"][
        "properties"
    ]["actions"]["items"]
    return {
        "type": "object",
        "properties": {
            "replacements": {
                "type": "array",
                "minItems": replacement_count,
                "maxItems": replacement_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "action_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 39,
                        },
                        "action": action_schema,
                    },
                    "required": ["action_index", "action"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["replacements"],
        "additionalProperties": False,
    }


class GeometryProvider(str, Enum):
    local_llama = "llama.cpp"
    nvidia = "NVIDIA"


class GeometryOperation(str, Enum):
    extraction = "geometry_extraction"
    repair = "geometry_repair"


class GeometryAttemptOutcome(str, Enum):
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class GeometryFailureCategory(str, Enum):
    local_unavailable = "local_unavailable"
    timeout = "timeout"
    connectivity = "connectivity"
    rate_limit = "rate_limit"
    invalid_json = "invalid_json"
    invalid_schema = "invalid_schema"
    invalid_dsl = "invalid_dsl"
    provider_error = "provider_error"
    deadline_exceeded = "deadline_exceeded"
    unknown = "unknown"


class GeometryFailureAction(str, Enum):
    preferred_remote = "preferred_remote"
    alternate_remote = "alternate_remote"
    repair_once = "repair_once"
    stop = "stop"


GEOMETRY_FAILURE_ACTIONS: dict[
    tuple[GeometryProvider, GeometryFailureCategory], GeometryFailureAction
] = {
    **{
        (GeometryProvider.local_llama, category): GeometryFailureAction.preferred_remote
        for category in (
            GeometryFailureCategory.local_unavailable,
            GeometryFailureCategory.timeout,
            GeometryFailureCategory.connectivity,
            GeometryFailureCategory.invalid_json,
            GeometryFailureCategory.invalid_schema,
            GeometryFailureCategory.invalid_dsl,
            GeometryFailureCategory.provider_error,
            GeometryFailureCategory.unknown,
        )
    },
    **{
        (GeometryProvider.nvidia, category): GeometryFailureAction.alternate_remote
        for category in (
            GeometryFailureCategory.rate_limit,
            GeometryFailureCategory.timeout,
            GeometryFailureCategory.connectivity,
            GeometryFailureCategory.invalid_json,
            GeometryFailureCategory.invalid_schema,
            GeometryFailureCategory.provider_error,
        )
    },
    (
        GeometryProvider.nvidia,
        GeometryFailureCategory.invalid_dsl,
    ): GeometryFailureAction.repair_once,
    (
        GeometryProvider.local_llama,
        GeometryFailureCategory.deadline_exceeded,
    ): GeometryFailureAction.stop,
    (
        GeometryProvider.nvidia,
        GeometryFailureCategory.deadline_exceeded,
    ): GeometryFailureAction.stop,
    (
        GeometryProvider.nvidia,
        GeometryFailureCategory.unknown,
    ): GeometryFailureAction.stop,
}


@dataclass(frozen=True)
class GeometryAttemptRecord:
    provider: GeometryProvider
    model: str
    operation: GeometryOperation
    duration_seconds: float
    outcome: GeometryAttemptOutcome
    failure_category: GeometryFailureCategory | None = None


@dataclass(frozen=True)
class _GeometryAttemptKey:
    provider: GeometryProvider
    model: str
    operation: GeometryOperation


@dataclass
class _GeometryExecution:
    deadline: float
    attempts: list[GeometryAttemptRecord] = field(default_factory=list)
    attempted_keys: set[_GeometryAttemptKey] = field(default_factory=set)
    repaired_proposals: set[tuple[GeometryProvider, str]] = field(
        default_factory=set
    )


@dataclass(frozen=True)
class _GeometryFailure:
    category: GeometryFailureCategory
    error: Exception


class GeometryProposalValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        category: GeometryFailureCategory,
        dsl: GeometryDSL | None = None,
        summary: str | None = None,
        issues: tuple[GeoGebraValidationIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_category = category
        self.dsl = dsl
        self.summary = summary
        self.issues = issues


_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_STRICT_ACTION_FIELDS: dict[
    GeometryActionType, tuple[frozenset[str], frozenset[str]]
] = {
    GeometryActionType.CREATE_POINT: (
        frozenset({"action", "label", "coordinates"}),
        frozenset({"action", "label"}),
    ),
    GeometryActionType.CREATE_LINE: (
        frozenset({"action", "label", "points"}),
        frozenset({"action", "label", "points"}),
    ),
    GeometryActionType.CREATE_POLYGON: (
        frozenset({"action", "label", "points"}),
        frozenset({"action", "label", "points"}),
    ),
    GeometryActionType.INTERSECT: (
        frozenset({"action", "label", "metadata"}),
        frozenset({"action", "label", "metadata"}),
    ),
    GeometryActionType.MIDPOINT: (
        frozenset({"action", "label", "points"}),
        frozenset({"action", "label", "points"}),
    ),
    GeometryActionType.PERPENDICULAR: (
        frozenset({"action", "label", "line", "metadata"}),
        frozenset({"action", "label", "line", "metadata"}),
    ),
    GeometryActionType.PARALLEL: (
        frozenset({"action", "label", "line", "metadata"}),
        frozenset({"action", "label", "line", "metadata"}),
    ),
    GeometryActionType.ANGLE_BISECTOR: (
        frozenset({"action", "label", "points"}),
        frozenset({"action", "label", "points"}),
    ),
    GeometryActionType.CREATE_FUNCTION: (
        frozenset({"action", "label", "equation"}),
        frozenset({"action", "label", "equation"}),
    ),
    GeometryActionType.DEFINE_OBJECT: (
        frozenset({"action", "output", "object_type", "value"}),
        frozenset({"action", "output", "object_type", "value"}),
    ),
    GeometryActionType.EXECUTE_COMMAND: (
        frozenset({"action", "output", "command", "arguments"}),
        frozenset({"action", "command", "arguments"}),
    ),
}
_STRICT_ARGUMENT_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "reference": (
        frozenset({"kind", "value"}),
        frozenset({"kind", "value"}),
    ),
    "number": (frozenset({"kind", "value"}), frozenset({"kind", "value"})),
    "angle": (
        frozenset({"kind", "value", "unit"}),
        frozenset({"kind", "value", "unit"}),
    ),
    "point": (
        frozenset({"kind", "x", "y", "z"}),
        frozenset({"kind", "x", "y"}),
    ),
    "vector": (
        frozenset({"kind", "x", "y", "z"}),
        frozenset({"kind", "x", "y"}),
    ),
    "text": (frozenset({"kind", "value"}), frozenset({"kind", "value"})),
    "boolean": (
        frozenset({"kind", "value"}),
        frozenset({"kind", "value"}),
    ),
    "expression": (
        frozenset({"kind", "value"}),
        frozenset({"kind", "value"}),
    ),
    "equation": (
        frozenset({"kind", "value"}),
        frozenset({"kind", "value"}),
    ),
    "interval": (
        frozenset(
            {"kind", "lower", "upper", "lower_inclusive", "upper_inclusive"}
        ),
        frozenset(
            {"kind", "lower", "upper", "lower_inclusive", "upper_inclusive"}
        ),
    ),
    "list": (frozenset({"kind", "items"}), frozenset({"kind", "items"})),
}


@dataclass
class GeometryExtractionResult:
    dsl: GeometryDSL
    summary: str | None
    warnings: list[str]
    allowed_commands: frozenset[str] = field(default_factory=frozenset)
    retrieved_commands: tuple[RetrievedCommand, ...] = ()
    attempts: tuple[GeometryAttemptRecord, ...] = ()


class GeometryExtractor:
    def __init__(
        self,
        llama_client: LlamaClient | None = None,
        settings: Any | None = None,
        nvidia_client: NvidiaClient | None = None,
    ) -> None:
        self.llama_client = llama_client or LlamaClient()
        self.settings = settings or get_settings()
        self.nvidia_client = nvidia_client or NvidiaClient(self.settings)
        catalog_path = getattr(self.settings, "geogebra_catalog_path", None)
        self.registry = GeoGebraCommandRegistry.cached(
            str(catalog_path) if catalog_path is not None else None
        )
        self.validator = GeoGebraDSLValidator(self.registry)

    async def extract(
        self,
        text: str,
        parser_model: str,
        *,
        environment: VisualizationEnvironment,
        semantic_query_terms: tuple[str, ...] = (),
        request_id: str | None = None,
        request_deadline: float | None = None,
    ) -> GeometryExtractionResult:
        retrieved = self._retrieve_commands(
            text,
            environment,
            semantic_query_terms=semantic_query_terms,
        )
        execution = self._new_execution(request_deadline)
        local_model = str(
            getattr(self.llama_client, "model", "local:llama-geometry-parser")
        )
        local_eligible = (
            self._local_geometry_parser_enabled()
            and len(text.strip()) <= LOCAL_GEOMETRY_MAX_PROMPT_CHARS
        )

        if local_eligible:
            local_result, local_failure = await self._attempt_extraction(
                execution,
                provider=GeometryProvider.local_llama,
                model=local_model,
                text=text,
                environment=environment,
                retrieved=retrieved,
                request_id=request_id,
            )
            if local_result is not None:
                return self._finish_extraction(local_result, execution)
        else:
            local_failure = self._record_skipped_local(execution, local_model)

        if (
            self._failure_action(
                GeometryProvider.local_llama, local_failure.category
            )
            is not GeometryFailureAction.preferred_remote
            or not getattr(self.nvidia_client, "enabled", False)
        ):
            return self._empty_extraction(
                environment,
                retrieved,
                self._failure_warning(local_failure.category),
                attempts=tuple(execution.attempts),
            )

        for remote_model in _ordered_geometry_fallback_models(parser_model):
            remote_result, remote_failure = await self._attempt_extraction(
                execution,
                provider=GeometryProvider.nvidia,
                model=remote_model,
                text=text,
                environment=environment,
                retrieved=retrieved,
                request_id=request_id,
            )
            if remote_result is not None:
                remote_result.warnings.insert(
                    0, self._fallback_success_warning(local_failure.category)
                )
                return self._finish_extraction(remote_result, execution)

            action = self._fallback_after_remote_failure(remote_failure.category)
            if action is GeometryFailureAction.repair_once:
                repaired_result, _ = await self._attempt_repair(
                    execution,
                    model=remote_model,
                    proposal_error=remote_failure.error,
                    text=text,
                    environment=environment,
                    retrieved=retrieved,
                    request_id=request_id,
                )
                if repaired_result is not None:
                    repaired_result.warnings.insert(
                        0, self._fallback_success_warning(local_failure.category)
                    )
                    return self._finish_extraction(repaired_result, execution)
                action = GeometryFailureAction.alternate_remote

            if (
                action is not GeometryFailureAction.alternate_remote
                or self._remaining_seconds(execution) <= 0
            ):
                break

        final_failure = (
            execution.attempts[-1].failure_category
            if execution.attempts
            and execution.attempts[-1].failure_category is not None
            else local_failure.category
        )
        return self._empty_extraction(
            environment,
            retrieved,
            self._failure_warning(final_failure),
            attempts=tuple(execution.attempts),
        )

    @staticmethod
    def _empty_extraction(
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        warning: str,
        *,
        attempts: tuple[GeometryAttemptRecord, ...] = (),
    ) -> GeometryExtractionResult:
        return GeometryExtractionResult(
            dsl=GeometryDSL(
                version="1.1",
                space=(
                    "euclidean_3d"
                    if environment is VisualizationEnvironment.graphics_3d
                    else "euclidean_2d"
                ),
                environment=environment,
                actions=[],
                render_hints=RenderHints(),
            ),
            summary=None,
            warnings=[warning],
            allowed_commands=frozenset(command.name for command in retrieved),
            retrieved_commands=retrieved,
            attempts=attempts,
        )

    def _local_geometry_parser_enabled(self) -> bool:
        return bool(
            getattr(self.settings, "local_llama_geometry_extraction_enabled", False)
            and getattr(self.llama_client, "enabled", False)
            and getattr(self.llama_client, "available", True)
        )

    def _new_execution(self, request_deadline: float | None) -> _GeometryExecution:
        deadline = (
            request_deadline
            if request_deadline is not None
            else time.monotonic()
            + float(getattr(self.settings, "solve_request_timeout_seconds", 70.0))
        )
        return _GeometryExecution(deadline=deadline)

    @staticmethod
    def _remaining_seconds(execution: _GeometryExecution) -> float:
        return max(0.0, execution.deadline - time.monotonic())

    @staticmethod
    def _finish_extraction(
        result: GeometryExtractionResult, execution: _GeometryExecution
    ) -> GeometryExtractionResult:
        result.attempts = tuple(execution.attempts)
        return result

    @staticmethod
    def _failure_action(
        provider: GeometryProvider, category: GeometryFailureCategory
    ) -> GeometryFailureAction:
        return GEOMETRY_FAILURE_ACTIONS.get(
            (provider, category), GeometryFailureAction.stop
        )

    def _record_skipped_local(
        self, execution: _GeometryExecution, model: str
    ) -> _GeometryFailure:
        key = _GeometryAttemptKey(
            GeometryProvider.local_llama,
            model,
            GeometryOperation.extraction,
        )
        execution.attempted_keys.add(key)
        failure = _GeometryFailure(
            GeometryFailureCategory.local_unavailable,
            RuntimeError("The validated local geometry parser is unavailable."),
        )
        execution.attempts.append(
            GeometryAttemptRecord(
                provider=key.provider,
                model=key.model,
                operation=key.operation,
                duration_seconds=0.0,
                outcome=GeometryAttemptOutcome.skipped,
                failure_category=failure.category,
            )
        )
        return failure

    async def _attempt_extraction(
        self,
        execution: _GeometryExecution,
        *,
        provider: GeometryProvider,
        model: str,
        text: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        request_id: str | None,
    ) -> tuple[GeometryExtractionResult | None, _GeometryFailure]:
        key = _GeometryAttemptKey(provider, model, GeometryOperation.extraction)
        if key in execution.attempted_keys:
            return None, _GeometryFailure(
                GeometryFailureCategory.unknown,
                RuntimeError("Duplicate geometry attempt was suppressed."),
            )
        execution.attempted_keys.add(key)

        remaining = self._remaining_seconds(execution)
        if remaining <= 0:
            failure = _GeometryFailure(
                GeometryFailureCategory.deadline_exceeded,
                TimeoutError("The geometry extraction deadline was exhausted."),
            )
            execution.attempts.append(
                GeometryAttemptRecord(
                    provider=provider,
                    model=model,
                    operation=key.operation,
                    duration_seconds=0.0,
                    outcome=GeometryAttemptOutcome.skipped,
                    failure_category=failure.category,
                )
            )
            return None, failure

        configured_timeout = (
            float(
                getattr(
                    self.settings, "local_llama_geometry_timeout_seconds", 30.0
                )
            )
            if provider is GeometryProvider.local_llama
            else remote_model_timeout_seconds(
                self.settings, provider="nvidia_direct", model=model
            )
        )
        timeout_seconds = min(configured_timeout, remaining)
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_seconds):
                if provider is GeometryProvider.local_llama:
                    result = await self._extract_with_local_llama(
                        text,
                        environment,
                        retrieved,
                        request_id=request_id,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    result = await self._extract_with_llm(
                        text,
                        model,
                        environment,
                        retrieved,
                        request_id=request_id,
                        completion_client=self.nvidia_client,
                        provider_name=provider.value,
                        timeout_seconds=timeout_seconds,
                    )
        except Exception as exc:
            category = self._failure_category(exc)
            if (
                category is GeometryFailureCategory.timeout
                and self._remaining_seconds(execution) <= 0
            ):
                category = GeometryFailureCategory.deadline_exceeded
            failure = _GeometryFailure(category, exc)
            execution.attempts.append(
                GeometryAttemptRecord(
                    provider=provider,
                    model=model,
                    operation=key.operation,
                    duration_seconds=time.monotonic() - started,
                    outcome=GeometryAttemptOutcome.failed,
                    failure_category=category,
                )
            )
            self._log_attempt_failure(
                request_id=request_id,
                environment=environment,
                provider=provider,
                model=model,
                operation=key.operation,
                category=category,
                error=exc,
            )
            return None, failure

        execution.attempts.append(
            GeometryAttemptRecord(
                provider=provider,
                model=model,
                operation=key.operation,
                duration_seconds=time.monotonic() - started,
                outcome=GeometryAttemptOutcome.succeeded,
            )
        )
        return result, _GeometryFailure(
            GeometryFailureCategory.unknown,
            RuntimeError("No failure."),
        )

    async def _attempt_repair(
        self,
        execution: _GeometryExecution,
        *,
        model: str,
        proposal_error: Exception,
        text: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        request_id: str | None,
    ) -> tuple[GeometryExtractionResult | None, _GeometryFailure]:
        proposal_key = (GeometryProvider.nvidia, model)
        key = _GeometryAttemptKey(
            GeometryProvider.nvidia, model, GeometryOperation.repair
        )
        if (
            proposal_key in execution.repaired_proposals
            or key in execution.attempted_keys
            or not isinstance(proposal_error, GeometryProposalValidationError)
            or proposal_error.dsl is None
            or not proposal_error.issues
        ):
            return None, _GeometryFailure(
                GeometryFailureCategory.invalid_dsl,
                RuntimeError("The proposal was not eligible for action-scoped repair."),
            )

        execution.repaired_proposals.add(proposal_key)
        execution.attempted_keys.add(key)
        remaining = self._remaining_seconds(execution)
        if remaining <= 0:
            failure = _GeometryFailure(
                GeometryFailureCategory.deadline_exceeded,
                TimeoutError("The geometry repair deadline was exhausted."),
            )
            execution.attempts.append(
                GeometryAttemptRecord(
                    provider=key.provider,
                    model=key.model,
                    operation=key.operation,
                    duration_seconds=0.0,
                    outcome=GeometryAttemptOutcome.skipped,
                    failure_category=failure.category,
                )
            )
            return None, failure

        timeout_seconds = min(
            remote_model_timeout_seconds(
                self.settings, provider="nvidia_direct", model=model
            ),
            remaining,
        )
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_seconds):
                repaired_dsl = await self._repair_remote_dsl(
                    proposal_error.dsl,
                    proposal_error.issues,
                    parser_model=model,
                    environment=environment,
                    retrieved=retrieved,
                    request_id=request_id,
                    completion_client=self.nvidia_client,
                    provider_name=GeometryProvider.nvidia.value,
                    timeout_seconds=timeout_seconds,
                )
                result = self._validated_remote_result(
                    text,
                    repaired_dsl,
                    summary=proposal_error.summary or "",
                    environment=environment,
                    retrieved=retrieved,
                    source=GeometryProvider.nvidia.value,
                )
        except Exception as exc:
            category = self._failure_category(exc)
            if (
                category is GeometryFailureCategory.timeout
                and self._remaining_seconds(execution) <= 0
            ):
                category = GeometryFailureCategory.deadline_exceeded
            failure = _GeometryFailure(category, exc)
            execution.attempts.append(
                GeometryAttemptRecord(
                    provider=key.provider,
                    model=key.model,
                    operation=key.operation,
                    duration_seconds=time.monotonic() - started,
                    outcome=GeometryAttemptOutcome.failed,
                    failure_category=category,
                )
            )
            self._log_attempt_failure(
                request_id=request_id,
                environment=environment,
                provider=key.provider,
                model=key.model,
                operation=key.operation,
                category=category,
                error=exc,
            )
            return None, failure

        execution.attempts.append(
            GeometryAttemptRecord(
                provider=key.provider,
                model=key.model,
                operation=key.operation,
                duration_seconds=time.monotonic() - started,
                outcome=GeometryAttemptOutcome.succeeded,
            )
        )
        return result, _GeometryFailure(
            GeometryFailureCategory.unknown,
            RuntimeError("No failure."),
        )

    @staticmethod
    def _failure_category(error: Exception) -> GeometryFailureCategory:
        explicit = getattr(error, "failure_category", None)
        if isinstance(explicit, GeometryFailureCategory):
            return explicit
        integration_mapping = {
            IntegrationFailureCategory.timeout: GeometryFailureCategory.timeout,
            IntegrationFailureCategory.connectivity: GeometryFailureCategory.connectivity,
            IntegrationFailureCategory.rate_limit: GeometryFailureCategory.rate_limit,
            IntegrationFailureCategory.http_error: GeometryFailureCategory.provider_error,
            IntegrationFailureCategory.invalid_response: GeometryFailureCategory.invalid_json,
        }
        if isinstance(explicit, IntegrationFailureCategory):
            return integration_mapping[explicit]
        diagnostics = exception_diagnostics(error)
        if diagnostics.status_code == 429:
            return GeometryFailureCategory.rate_limit
        chain: list[BaseException] = []
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            current = current.__cause__
        if any(
            isinstance(item, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException))
            for item in chain
        ):
            return GeometryFailureCategory.timeout
        if any(isinstance(item, json.JSONDecodeError) for item in chain):
            return GeometryFailureCategory.invalid_json
        if any(
            isinstance(item, (httpx.TransportError, ConnectionError))
            for item in chain
        ):
            return GeometryFailureCategory.connectivity
        if diagnostics.status_code is not None:
            return GeometryFailureCategory.provider_error
        return GeometryFailureCategory.unknown

    @staticmethod
    def _fallback_success_warning(
        local_failure: GeometryFailureCategory,
    ) -> str:
        if local_failure in {
            GeometryFailureCategory.invalid_json,
            GeometryFailureCategory.invalid_schema,
            GeometryFailureCategory.invalid_dsl,
        }:
            return (
                "The local visualization parser produced an invalid proposal; a "
                "deterministically validated NVIDIA proposal was used instead."
            )
        return (
            "The local visualization parser was unavailable; a deterministically "
            "validated NVIDIA proposal was used instead."
        )

    @staticmethod
    def _failure_warning(category: GeometryFailureCategory) -> str:
        if category is GeometryFailureCategory.deadline_exceeded:
            return (
                "The visualization request deadline was exhausted before a validated "
                "plan was produced."
            )
        if category in {
            GeometryFailureCategory.invalid_json,
            GeometryFailureCategory.invalid_schema,
            GeometryFailureCategory.invalid_dsl,
        }:
            return (
                "Every model-generated visualization proposal failed deterministic "
                "validation; no visualization was generated."
            )
        return (
            "No available visualization model produced a validated plan; no "
            "visualization was generated."
        )

    @staticmethod
    def _log_attempt_failure(
        *,
        request_id: str | None,
        environment: VisualizationEnvironment,
        provider: GeometryProvider,
        model: str,
        operation: GeometryOperation,
        category: GeometryFailureCategory,
        error: Exception,
    ) -> None:
        diagnostics = exception_diagnostics(error)
        logger.warning(
            "Geometry model attempt failed request_id=%s provider=%s model=%s "
            "operation=%s environment=%s failure_category=%s error_type=%s "
            "error_message=%s status_code=%s response_body=%s",
            request_id,
            provider.value,
            model,
            operation.value,
            environment.value,
            category.value,
            diagnostics.error_type,
            diagnostics.error_message,
            diagnostics.status_code,
            diagnostics.response_body,
        )

    def _retrieve_commands(
        self,
        text: str,
        environment: VisualizationEnvironment,
        *,
        semantic_query_terms: tuple[str, ...] = (),
    ) -> tuple[RetrievedCommand, ...]:
        """Search original text plus bounded semantic terms selected by the router AI."""

        terms = tuple(
            term.strip()
            for term in semantic_query_terms[:12]
            if isinstance(term, str) and term.strip()
        )
        expanded_query = f"{text}\n{' '.join(terms)}" if terms else text

        return tuple(
            self.registry.search(
                expanded_query,
                environment,
                limit=GEOMETRY_RETRIEVAL_LIMIT,
            )
        )

    async def _extract_with_local_llama(
        self,
        text: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        *,
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> GeometryExtractionResult:
        command_names = sorted(
            {command.name for command in retrieved}, key=str.casefold
        )
        discovery_context = self._format_command_context(retrieved)
        command_signatures: dict[str, tuple[CommandSignature, ...]] = {}
        for command in retrieved:
            definition = self.registry.lookup(command.name)
            if definition is None:
                continue
            retrieved_signatures = set(command.signatures)
            command_signatures[command.name] = tuple(
                overload.signature
                for overload in definition.runtime_eligible_overloads(environment)
                if overload.signature.original in retrieved_signatures
            )
        generic_commands_only = (
            environment
            not in {
                VisualizationEnvironment.geometry_2d,
                VisualizationEnvironment.graphing,
            }
            and _commands_have_self_contained_overloads(command_signatures)
        )
        schema_signatures = (
            {
                command_name: tuple(
                    signature
                    for signature in signatures
                    if _signature_is_self_contained(signature)
                )
                for command_name, signatures in command_signatures.items()
            }
            if generic_commands_only
            else command_signatures
        )
        payload = await self.llama_client.generate_json(
            prompt=(
                f"{LOCAL_GEOMETRY_EXTRACTION_PROMPT}\n\n"
                f"Selected environment: {environment.value}\n"
                f"Retrieved generic commands (use no others):\n{discovery_context}\n\n"
                f"Problem:\n{text.strip()}"
            ),
            max_tokens=int(
                getattr(self.settings, "local_llama_geometry_max_tokens", 1_200)
            ),
            thinking_budget_tokens=0,
            timeout_seconds=float(
                timeout_seconds
                if timeout_seconds is not None
                else getattr(
                    self.settings, "local_llama_geometry_timeout_seconds", 8.0
                )
            ),
            json_schema=geometry_response_schema(
                command_names,
                environment,
                generic_commands_only=generic_commands_only,
                command_signatures=schema_signatures,
            ),
            operation="local_geometry_extraction",
            trace_id=request_id,
        )

        try:
            dsl, summary = self._parse_geometry_payload(
                payload,
                environment=environment,
                source="Local llama-server",
                allowed_command_names=set(command_names),
            )
        except (ValidationError, ValueError) as exc:
            raise GeometryProposalValidationError(
                "Local llama-server output did not match the required geometry schema.",
                category=GeometryFailureCategory.invalid_schema,
            ) from exc
        validation = self.validator.validate(
            dsl, allowed_command_names={command.name for command in retrieved}
        )
        issues = [
            issue.message
            for issue in validation.issues
            if issue.severity is ValidationSeverity.error
        ]
        issues.extend(self._validate_local_prompt_grounding(text, dsl))
        issues.extend(self._validate_intent_alignment(text, dsl))
        if issues:
            raise GeometryProposalValidationError(
                "Invalid local geometry DSL: " + "; ".join(issues),
                category=GeometryFailureCategory.invalid_dsl,
                dsl=dsl,
                summary=summary,
            )
        dsl.actions = list(validation.actions)

        return GeometryExtractionResult(
            dsl=dsl,
            summary=summary,
            warnings=[],
            allowed_commands=frozenset(command.name for command in retrieved),
            retrieved_commands=retrieved,
        )

    async def _extract_with_llm(
        self,
        text: str,
        parser_model: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        *,
        request_id: str | None = None,
        completion_client: Any | None = None,
        provider_name: str = "NVIDIA",
        timeout_seconds: float | None = None,
    ) -> GeometryExtractionResult:
        allowed_command_names = {command.name for command in retrieved}
        command_names = sorted(allowed_command_names, key=str.casefold)
        discovery_context = self._format_command_context(retrieved)
        selected_client = completion_client or self.nvidia_client
        payload = await selected_client.complete_json(
            model=parser_model,
            system_prompt=(
                "Extract only visualization intents for a math problem. Output JSON with exactly the "
                "top-level keys summary and dsl; keep summary at 200 characters or fewer. Use DSL "
                "version 1.1 with version, space, environment, actions, render_hints. "
                "Supported actions: CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT, "
                "MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION, DEFINE_OBJECT, "
                "EXECUTE_COMMAND. Prefer DEFINE_OBJECT for explicit function definitions. "
                "EXECUTE_COMMAND arguments must be typed objects and its command must appear in the "
                "retrieved list. Never return raw GeoGebra commands or JavaScript. When an explicit "
                "visualization request omits placement or scale, choose simple finite display "
                "coordinates or dimensions without implying extra mathematical relationships.\n\n"
                f"Environment: {environment.value}\nRetrieved commands:\n{discovery_context}"
            ),
            user_prompt=text,
            temperature=0.1,
            json_schema=geometry_response_schema(command_names, environment),
            schema_name="geometry_visualization",
            max_tokens=int(
                getattr(self.settings, "geometry_extraction_max_tokens", 1_200)
            ),
            require_parameters=False,
            allow_schema_downgrade=False,
            repair_invalid_json=False,
            timeout_seconds=(
                float(timeout_seconds)
                if timeout_seconds is not None
                else remote_model_timeout_seconds(
                    self.settings,
                    provider="nvidia_direct",
                    model=parser_model,
                )
            ),
            operation="geometry_extraction",
            trace_id=request_id,
        )
        try:
            dsl, summary = self._parse_geometry_payload(
                payload,
                environment=environment,
                source=provider_name,
                allowed_command_names=set(command_names),
            )
        except (ValidationError, ValueError) as exc:
            schema_issue_counts = (
                Counter(
                    f"schema_{str(item.get('type', 'invalid')).replace('.', '_')}"
                    for item in exc.errors(include_url=False)
                )
                if isinstance(exc, ValidationError)
                else Counter({"schema_payload_shape": 1})
            )
            logger.warning(
                "Remote geometry schema validation failed request_id=%s model=%s "
                "environment=%s issue_codes=%s",
                request_id,
                parser_model,
                environment.value,
                json.dumps(dict(sorted(schema_issue_counts.items())), sort_keys=True),
            )
            raise GeometryProposalValidationError(
                f"{provider_name} geometry output did not match the required response schema.",
                category=GeometryFailureCategory.invalid_schema,
            ) from exc

        return self._validated_remote_result(
            text,
            dsl,
            summary=summary,
            environment=environment,
            retrieved=retrieved,
            source=provider_name,
            parser_model=parser_model,
            request_id=request_id,
        )

    def _validated_remote_result(
        self,
        text: str,
        dsl: GeometryDSL,
        *,
        summary: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        source: str,
        parser_model: str | None = None,
        request_id: str | None = None,
    ) -> GeometryExtractionResult:
        allowed_command_names = {command.name for command in retrieved}
        validation = self.validator.validate(
            dsl, allowed_command_names=allowed_command_names
        )
        errors = tuple(
            issue
            for issue in validation.issues
            if issue.severity is ValidationSeverity.error
        )
        semantic_issues = self._validate_intent_alignment(text, dsl)
        if errors or not dsl.actions:
            issue_codes = Counter(issue.code for issue in errors)
            if not dsl.actions:
                issue_codes["empty_proposal"] += 1
            raise GeometryProposalValidationError(
                f"{source} geometry proposal failed deterministic validation: "
                + json.dumps(dict(sorted(issue_codes.items())), sort_keys=True),
                category=GeometryFailureCategory.invalid_dsl,
                dsl=dsl,
                summary=summary,
                issues=errors,
            )

        dsl.actions = list(validation.actions)
        warnings: list[str] = []
        if semantic_issues:
            logger.warning(
                "Remote geometry proposal accepted with incomplete intent alignment "
                "request_id=%s model=%s environment=%s action_count=%s issue_count=%s",
                request_id,
                parser_model or source,
                environment.value,
                len(dsl.actions),
                len(semantic_issues),
            )
            warnings.append(
                "The interactive construction may omit some stated relationships: "
                + " ".join(semantic_issues)
            )
        return GeometryExtractionResult(
            dsl=dsl,
            summary=summary,
            warnings=warnings,
            allowed_commands=frozenset(allowed_command_names),
            retrieved_commands=retrieved,
        )

    async def _repair_remote_dsl(
        self,
        dsl: GeometryDSL,
        validation_issues: tuple[GeoGebraValidationIssue, ...],
        *,
        parser_model: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        request_id: str | None = None,
        completion_client: Any | None = None,
        provider_name: str = "NVIDIA",
        timeout_seconds: float | None = None,
    ) -> GeometryDSL:
        issues_by_index: dict[int, list[GeoGebraValidationIssue]] = {}
        for issue in validation_issues:
            index = issue.action_index
            if index is None or index < 0 or index >= len(dsl.actions):
                continue
            issues_by_index.setdefault(index, []).append(issue)
        if not issues_by_index:
            logger.warning(
                "Remote geometry repair skipped request_id=%s model=%s reason=no_action_scoped_issues",
                request_id,
                parser_model,
            )
            raise GeometryProposalValidationError(
                "The invalid proposal had no action-scoped repair target.",
                category=GeometryFailureCategory.invalid_dsl,
                dsl=dsl,
                issues=validation_issues,
            )

        failing_fragments = [
            {
                "action_index": index,
                "action": dsl.actions[index].model_dump(mode="json", exclude_none=True),
                "validation_issues": [
                    issue.model_dump(mode="json") for issue in issues_by_index[index]
                ],
            }
            for index in sorted(issues_by_index)
        ]
        command_names = sorted(
            {command.name for command in retrieved}, key=str.casefold
        )
        discovery_context = self._format_command_context(retrieved)
        selected_client = completion_client or self.nvidia_client
        payload = await selected_client.complete_json(
            model=parser_model,
            system_prompt=(
                "Repair only the supplied failing Geometry DSL actions. Return one JSON "
                "object with a replacements array. Every item must contain the original "
                "zero-based action_index and one complete corrected typed action. Do not "
                "return non-failing actions, prose, raw GeoGebra syntax, scripts, or "
                "JavaScript. EXECUTE_COMMAND may use only the retrieved commands below.\n\n"
                "Exact format example:\n"
                '{"replacements":[{"action_index":2,"action":{"action":'
                '"CREATE_LINE","label":"lAB","points":["A","B"]}}]}\n\n'
                f"Environment: {environment.value}\nRetrieved commands:\n{discovery_context}"
            ),
            user_prompt=json.dumps(
                {"failing_actions": failing_fragments},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            temperature=0.0,
            json_schema=geometry_repair_schema(
                command_names, replacement_count=len(issues_by_index)
            ),
            schema_name="geometry_action_repair",
            max_tokens=int(
                getattr(self.settings, "geometry_repair_max_tokens", 800)
            ),
            require_parameters=False,
            allow_schema_downgrade=False,
            repair_invalid_json=False,
            timeout_seconds=(
                float(timeout_seconds)
                if timeout_seconds is not None
                else remote_model_timeout_seconds(
                    self.settings,
                    provider="nvidia_direct",
                    model=parser_model,
                )
            ),
            operation="geometry_repair",
            trace_id=request_id,
        )
        try:
            replacements = self._parse_repair_replacements(
                payload,
                expected_indices=tuple(sorted(issues_by_index)),
                allowed_command_names=set(command_names),
            )
        except (ValidationError, ValueError) as exc:
            raise GeometryProposalValidationError(
                "The geometry repair output did not match the repair schema.",
                category=GeometryFailureCategory.invalid_schema,
                dsl=dsl,
                issues=validation_issues,
            ) from exc

        indices = sorted(issues_by_index)
        if not replacements:
            logger.warning(
                "Remote geometry repair rejected request_id=%s provider=%s model=%s "
                "reason=no_parseable_replacements expected=%s actual=0",
                request_id,
                provider_name,
                parser_model,
                len(indices),
            )
            raise GeometryProposalValidationError(
                "The geometry repair contained no valid replacements.",
                category=GeometryFailureCategory.invalid_schema,
                dsl=dsl,
                issues=validation_issues,
            )

        merged = dsl.model_copy(deep=True)
        for index, replacement in replacements.items():
            merged.actions[index] = replacement
        logger.info(
            "Remote geometry repair parsed request_id=%s provider=%s model=%s "
            "expected=%s recovered=%s partial=%s",
            request_id,
            provider_name,
            parser_model,
            len(indices),
            len(replacements),
            len(replacements) != len(indices),
        )
        return merged

    def _fallback_after_remote_failure(
        self, category: GeometryFailureCategory
    ) -> GeometryFailureAction:
        """Map one typed remote failure to its only permitted next action."""

        return self._failure_action(GeometryProvider.nvidia, category)

    def _validate_strict_argument_payload(
        self,
        raw_argument: Any,
        *,
        source: str,
        remaining_list_depth: int = 2,
    ) -> None:
        if not isinstance(raw_argument, dict):
            raise ValueError(f"{source} command argument must be an object.")
        kind = raw_argument.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{source} command argument kind must be a string.")
        contract = _STRICT_ARGUMENT_FIELDS.get(kind)
        if contract is None or (kind == "list" and remaining_list_depth < 1):
            raise ValueError(f"{source} command argument has unsupported kind {kind!r}.")
        allowed_fields, required_fields = contract
        actual_fields = set(raw_argument)
        unexpected = actual_fields - allowed_fields
        missing = required_fields - actual_fields
        if unexpected or missing:
            raise ValueError(
                f"{source} {kind} argument fields do not match the strict schema "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})."
            )
        if any(value is None for value in raw_argument.values()):
            raise ValueError(f"{source} {kind} argument contains a null field.")
        if kind == "list":
            items = raw_argument.get("items")
            if not isinstance(items, list) or len(items) > 32:
                raise ValueError(f"{source} list argument must contain at most 32 items.")
            for item in items:
                self._validate_strict_argument_payload(
                    item,
                    source=source,
                    remaining_list_depth=remaining_list_depth - 1,
                )

    def _parse_strict_proposal_action(
        self,
        raw_action: Any,
        *,
        source: str,
        allowed_command_names: set[str],
    ) -> GeometryAction:
        if not isinstance(raw_action, dict):
            raise ValueError(f"{source} geometry action must be an object.")
        try:
            action_type = GeometryActionType(raw_action.get("action"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} geometry action has an unsupported action type."
            ) from exc

        if action_type is GeometryActionType.CREATE_CIRCLE:
            if "through" in raw_action:
                allowed_fields = frozenset({"action", "label", "through"})
                required_fields = allowed_fields
            else:
                allowed_fields = frozenset({"action", "label", "center", "radius"})
                required_fields = allowed_fields
        else:
            allowed_fields, required_fields = _STRICT_ACTION_FIELDS[action_type]

        actual_fields = set(raw_action)
        unexpected = actual_fields - allowed_fields
        missing = required_fields - actual_fields
        if unexpected or missing:
            raise ValueError(
                f"{source} {action_type.value} fields do not match the strict schema "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})."
            )
        if any(value is None for value in raw_action.values()):
            raise ValueError(f"{source} {action_type.value} contains a null field.")

        if action_type in {
            GeometryActionType.CREATE_LINE,
            GeometryActionType.MIDPOINT,
        }:
            points = raw_action.get("points")
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError(f"{source} {action_type.value} requires exactly two points.")
        elif action_type is GeometryActionType.CREATE_POLYGON:
            points = raw_action.get("points")
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError(f"{source} CREATE_POLYGON requires at least three points.")
        elif action_type is GeometryActionType.ANGLE_BISECTOR:
            points = raw_action.get("points")
            if not isinstance(points, list) or len(points) != 3:
                raise ValueError(f"{source} ANGLE_BISECTOR requires exactly three points.")
        elif action_type is GeometryActionType.CREATE_CIRCLE and "through" in raw_action:
            through = raw_action.get("through")
            if not isinstance(through, list) or not 2 <= len(through) <= 3:
                raise ValueError(f"{source} CREATE_CIRCLE through requires two or three points.")
        elif action_type is GeometryActionType.INTERSECT:
            metadata = raw_action.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) != {"objects"}:
                raise ValueError(f"{source} INTERSECT metadata must contain only objects.")
            objects = metadata.get("objects")
            if (
                not isinstance(objects, list)
                or len(objects) != 2
                or any(
                    not isinstance(item, str) or not _SAFE_LABEL.fullmatch(item)
                    for item in objects
                )
            ):
                raise ValueError(
                    f"{source} INTERSECT requires exactly two safe object labels."
                )
        elif action_type in {
            GeometryActionType.PERPENDICULAR,
            GeometryActionType.PARALLEL,
        }:
            metadata = raw_action.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) != {"through_point"}:
                raise ValueError(
                    f"{source} {action_type.value} metadata must contain only through_point."
                )
            through_point = metadata.get("through_point")
            if not isinstance(through_point, str) or not _SAFE_LABEL.fullmatch(through_point):
                raise ValueError(
                    f"{source} {action_type.value} through_point must be a safe label."
                )
        elif action_type is GeometryActionType.EXECUTE_COMMAND:
            command = raw_action.get("command")
            if command not in allowed_command_names:
                raise ValueError(
                    f"{source} EXECUTE_COMMAND used a command outside the retrieved allowlist."
                )
            arguments = raw_action.get("arguments")
            if not isinstance(arguments, list) or len(arguments) > 16:
                raise ValueError(
                    f"{source} EXECUTE_COMMAND requires at most 16 typed arguments."
                )
            for argument in arguments:
                self._validate_strict_argument_payload(argument, source=source)
        elif action_type is GeometryActionType.DEFINE_OBJECT:
            self._validate_strict_argument_payload(
                raw_action.get("value"), source=source
            )

        return GeometryAction.model_validate(raw_action)

    def _parse_geometry_payload(
        self,
        payload: dict[str, Any],
        *,
        environment: VisualizationEnvironment,
        source: str,
        allowed_command_names: set[str],
    ) -> tuple[GeometryDSL, str]:
        required_payload_keys = {"summary", "dsl"}
        if not required_payload_keys.issubset(payload):
            nested_candidates = [
                value
                for value in payload.values()
                if isinstance(value, dict)
                and required_payload_keys.issubset(value)
            ]
            if len(nested_candidates) == 1:
                payload = nested_candidates[0]
        if not required_payload_keys.issubset(payload):
            raise ValueError(
                f"{source} geometry response must contain summary and dsl."
            )
        summary = payload.get("summary")
        if not isinstance(summary, str):
            raise ValueError(f"{source} geometry summary must be a string.")
        summary = summary.strip()
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."
        raw_dsl = payload.get("dsl")
        if not isinstance(raw_dsl, dict):
            raise ValueError(f"{source} geometry dsl must be an object.")
        required_dsl_keys = {
            "version",
            "space",
            "environment",
            "actions",
            "render_hints",
        }
        if not required_dsl_keys.issubset(raw_dsl):
            raise ValueError(
                f"{source} geometry dsl must contain the required version 1.1 fields."
            )
        expected_space = (
            "euclidean_3d"
            if environment is VisualizationEnvironment.graphics_3d
            else "euclidean_2d"
        )
        if raw_dsl.get("version") != "1.1":
            raise ValueError(f"{source} geometry dsl must use version 1.1.")
        if raw_dsl.get("space") != expected_space:
            raise ValueError(f"{source} geometry dsl used the wrong coordinate space.")
        if raw_dsl.get("environment") != environment.value:
            raise ValueError(f"{source} geometry dsl used the wrong environment.")
        if raw_dsl.get("render_hints") != {}:
            raise ValueError(f"{source} geometry render_hints must be an empty object.")
        raw_actions = raw_dsl.get("actions")
        if not isinstance(raw_actions, list):
            raise ValueError(f"{source} geometry actions must be an array.")
        parsed_actions = [
            self._parse_strict_proposal_action(
                action,
                source=source,
                allowed_command_names=allowed_command_names,
            )
            for action in raw_actions
        ]
        strict_dsl = {key: raw_dsl[key] for key in required_dsl_keys}
        strict_dsl["actions"] = parsed_actions
        return GeometryDSL.model_validate(strict_dsl), summary.strip()

    def _parse_repair_replacements(
        self,
        payload: dict[str, Any],
        *,
        expected_indices: tuple[int, ...],
        allowed_command_names: set[str],
    ) -> dict[int, GeometryAction]:
        if set(payload) != {"replacements"}:
            raise ValueError(
                "Geometry repair response must contain exactly the replacements array."
            )
        raw_replacements = payload.get("replacements")
        if not isinstance(raw_replacements, list):
            raise ValueError("Geometry repair replacements must be an array.")

        expected = set(expected_indices)
        replacements: dict[int, GeometryAction] = {}
        for item in raw_replacements:
            if not isinstance(item, dict) or set(item) != {"action_index", "action"}:
                continue
            index = item.get("action_index")
            if type(index) is not int or index not in expected or index in replacements:
                continue
            try:
                replacements[index] = self._parse_strict_proposal_action(
                    item.get("action"),
                    source="Geometry repair",
                    allowed_command_names=allowed_command_names,
                )
            except (ValidationError, ValueError):
                continue
        return replacements

    def _format_command_context(
        self, retrieved: tuple[RetrievedCommand, ...]
    ) -> str:
        if not retrieved:
            return "- No generic commands retrieved; use high-level actions only."
        lines: list[str] = []
        for command in retrieved:
            lines.append(
                f"- {command.name}: {'; '.join(command.signatures)}"
                + (f" — {command.description}" if command.description else "")
            )
        return "\n".join(lines)

    def _validate_local_prompt_grounding(
        self, text: str, dsl: GeometryDSL
    ) -> list[str]:
        """Reject ungrounded local-model coordinates without rewriting its plan."""

        issues: list[str] = []
        number = r"[-+]?\d+(?:\.\d+)?"

        for action in dsl.actions:
            if (
                action.action is not GeometryActionType.CREATE_POINT
                or action.coordinates is None
                or not action.label
            ):
                continue

            label = re.escape(action.label)
            coordinate_match = re.search(
                rf"(?:point\s+|center\s+)?\b{label}\b\s*(?:at|=)?\s*"
                rf"\(\s*({number})\s*,\s*({number})\s*\)",
                text,
                flags=re.IGNORECASE,
            )
            if coordinate_match is None:
                issues.append(
                    f"Point '{action.label}' has coordinates that are not explicitly "
                    "stated in the prompt."
                )

        return issues

    def _validate_intent_alignment(self, text: str, dsl: GeometryDSL) -> list[str]:
        action_types = {action.action for action in dsl.actions}
        has_function_definition = any(
            action.action is GeometryActionType.DEFINE_OBJECT
            and action.object_type is not None
            and action.object_type.value == "function"
            for action in dsl.actions
        )
        generic_commands = {
            (action.command or "").casefold()
            for action in dsl.actions
            if action.action is GeometryActionType.EXECUTE_COMMAND
        }
        line_edges = {
            frozenset(action.points)
            for action in dsl.actions
            if action.action is GeometryActionType.CREATE_LINE
            and len(action.points) == 2
        }
        line_vertices = set().union(*line_edges) if line_edges else set()
        has_triangle_edges = any(
            all(
                frozenset(edge) in line_edges
                for edge in combinations(vertices, 2)
            )
            for vertices in combinations(line_vertices, 3)
        )
        expected_actions: list[tuple[GeometryActionType, str]] = []

        if re.search(
            r"(?:y|f\s*\(\s*x\s*\))\s*=",
            text,
            flags=re.IGNORECASE,
        ):
            expected_actions.append(
                (GeometryActionType.CREATE_FUNCTION, "an explicit function assignment")
            )
        if re.search(r"\btriangle\b|tam\s+gi(?:á|a)c", text, flags=re.IGNORECASE):
            expected_actions.append((GeometryActionType.CREATE_POLYGON, "a triangle"))
        if re.search(
            r"\bcircle\b|đường\s+tròn|duong\s+tron", text, flags=re.IGNORECASE
        ):
            expected_actions.append((GeometryActionType.CREATE_CIRCLE, "a circle"))
        if re.search(
            r"\bmidpoint\b|\bhalfway\b|trung\s+điểm|trung\s+diem",
            text,
            flags=re.IGNORECASE,
        ):
            expected_actions.append((GeometryActionType.MIDPOINT, "a midpoint"))
        if re.search(
            r"\bperpendicular\b|vuông\s+góc|vuong\s+goc|\\perp|⊥",
            text,
            flags=re.IGNORECASE,
        ):
            expected_actions.append((GeometryActionType.PERPENDICULAR, "a perpendicular"))
        if re.search(
            r"\bparallel\b|song\s+song|\\parallel|∥",
            text,
            flags=re.IGNORECASE,
        ):
            expected_actions.append((GeometryActionType.PARALLEL, "a parallel"))

        generic_equivalents = {
            GeometryActionType.CREATE_POLYGON: "polygon",
            GeometryActionType.CREATE_CIRCLE: "circle",
            GeometryActionType.MIDPOINT: "midpoint",
            GeometryActionType.PERPENDICULAR: "perpendicularline",
            GeometryActionType.PARALLEL: "parallelline",
            GeometryActionType.CREATE_FUNCTION: "function",
        }

        issues: list[str] = []
        for action, description in expected_actions:
            represented = (
                action in action_types
                or generic_equivalents.get(action) in generic_commands
                or (
                    action is GeometryActionType.CREATE_FUNCTION
                    and has_function_definition
                )
                or (
                    action is GeometryActionType.CREATE_POLYGON
                    and has_triangle_edges
                )
            )
            if not represented:
                issues.append(
                    f"The prompt requests {description}, but the construction plan "
                    f"has no action representing it ({action.value})."
                )
        return issues



    def _coerce_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text[:200] or None
