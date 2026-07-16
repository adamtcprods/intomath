from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError

from app.core.config import get_settings
from app.integrations.errors import exception_diagnostics
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import (
    GeometryAction,
    GeometryActionType,
    GeometryDSL,
    GeoGebraValidationIssue,
    ValidationSeverity,
    VisualizationEnvironment,
)
from app.services.geogebra_command_registry import (
    GLOBAL_CORE_COMMANDS,
    GeoGebraCommandRegistry,
    ROLLED_OUT_GENERIC_COMMANDS,
    RetrievedCommand,
)
from app.services.geogebra_validator import GeoGebraDSLValidator
from app.services.model_router import (
    NVIDIA_DIRECT_FALLBACK_MODELS,
    remote_model_timeout_seconds,
)


logger = logging.getLogger(__name__)
GEOMETRY_RETRIEVAL_LIMIT = 10


LOCAL_GEOMETRY_EXTRACTION_PROMPT = """
Convert the math problem into a minimal GeoGebra construction plan.
Return exactly one JSON object and nothing else. Do not solve or prove the problem.
Do not invent coordinates, lengths, labels, or relationships that the problem does not give.
Preserve labels exactly. References may be emitted out of order; trusted code will sort them.

Allowed actions:
CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT,
MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION,
and EXECUTE_COMMAND only for a retrieved command listed below.

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
- EXECUTE_COMMAND: action, command, arguments, and normally output. Every argument
  must use a typed kind; never put raw GeoGebra syntax in a string.

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
_LIST_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "const": "list"},
        "items": {
            "type": "array",
            "maxItems": 32,
            "items": {"oneOf": _NON_LIST_ARGUMENT_SCHEMAS},
        },
    },
    "required": ["kind", "items"],
    "additionalProperties": False,
}


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


def geometry_response_schema(
    command_names: list[str] | None = None,
    environment: VisualizationEnvironment = VisualizationEnvironment.geometry_2d,
) -> dict[str, Any]:
    command_names = sorted(set(command_names or []), key=str.casefold)
    action_schemas = [
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
    ]
    if command_names:
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
                    "maxItems": 40,
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


LOCAL_GEOMETRY_RESPONSE_SCHEMA: dict[str, Any] = geometry_response_schema()


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


class GeometryProposalValidationError(ValueError):
    failure_code = "model_validation_failure"


_MAX_LOCAL_ACTIONS = 40
_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_SAFE_FUNCTION_EXPRESSION = re.compile(r"^[0-9A-Za-z_+\-*/^()., ]{1,200}$")
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


class VisualizationCapability(str, Enum):
    no_visualization = "none"
    geometry_2d = "geometry_2d"
    graphing = "graphing"
    graphics_3d = "graphics_3d"
    cas = "cas"
    probability = "probability"
    statistics = "statistics"
    spreadsheet = "spreadsheet"


@dataclass(frozen=True)
class VisualizationClassification:
    capability: VisualizationCapability
    environment: VisualizationEnvironment | None
    reason: str

    @property
    def visualizable(self) -> bool:
        return self.environment is not None


def classify_visualization_capability(text: str) -> VisualizationClassification:
    """Classify only visual structure stated in the prompt, never its answer shape."""

    stripped = text.strip()
    lowered = stripped.casefold()
    if not stripped:
        return VisualizationClassification(
            VisualizationCapability.no_visualization,
            None,
            "the problem statement is empty",
        )

    if "spreadsheet" in lowered or re.search(r"\bcell\s+[a-z]+\d+\b", lowered):
        return VisualizationClassification(
            VisualizationCapability.spreadsheet,
            VisualizationEnvironment.spreadsheet,
            "the problem explicitly requests spreadsheet structure",
        )

    if re.search(
        r"\b(histogram|box(?:-and-whisker)?\s+plot|boxplot|scatter\s*plot|"
        r"bar\s+(?:chart|graph)|pie\s+chart|frequency\s+polygon)\b",
        lowered,
    ):
        return VisualizationClassification(
            VisualizationCapability.statistics,
            VisualizationEnvironment.statistics,
            "the problem explicitly names a statistical visualization",
        )

    if re.search(
        r"\b(probability\s+(?:tree|distribution|diagram)|tree\s+diagram|"
        r"normal\s+distribution\s+(?:curve|graph))\b",
        lowered,
    ):
        return VisualizationClassification(
            VisualizationCapability.probability,
            VisualizationEnvironment.probability,
            "the problem explicitly names a probability visualization",
        )

    geometry_terms = re.search(
        r"\b(?:point|line|segment|ray|triangle|quadrilateral|rectangle|square|"
        r"polygon|circle|midpoint|diameter|radius|chord|tangent|secant|angle|"
        r"perpendicular|parallel|bisector|vertex|vertices|sphere|plane|solid|"
        r"surface|prism|pyramid|cone|cylinder)\b|"
        r"tam\s+gi(?:á|a)c|đường\s+tròn|duong\s+tron|đường\s+thẳng|"
        r"duong\s+thang|đoạn\s+thẳng|doan\s+thang|\bđiểm\b|\bdiem\b|"
        r"trung\s+điểm|trung\s+diem|đường\s+kính|duong\s+kinh|"
        r"vuông\s+góc|vuong\s+goc|nội\s+tiếp|noi\s+tiep|\\perp|"
        r"\\parallel|[⊥∥∠]",
        stripped,
        flags=re.IGNORECASE,
    )
    has_named_geometry = bool(
        geometry_terms
        or re.search(
            r"\b(?:construct|draw)\s+(?:the\s+)?(?:point|line|circle|triangle|polygon)\b",
            lowered,
        )
    )
    has_3d_structure = bool(
        re.search(
            r"\b(?:3d|three[- ]dimensional|sphere|solid|surface|prism|pyramid|"
            r"cone|cylinder|plane\s+in\s+space)\b",
            lowered,
        )
    )
    if has_named_geometry and has_3d_structure:
        return VisualizationClassification(
            VisualizationCapability.graphics_3d,
            VisualizationEnvironment.graphics_3d,
            "the problem states three-dimensional geometric structure",
        )
    if has_named_geometry:
        return VisualizationClassification(
            VisualizationCapability.geometry_2d,
            VisualizationEnvironment.geometry_2d,
            "the problem states geometric objects or relations",
        )

    explicit_function = re.search(
        r"(?:^|[^A-Za-z])(?:y|[A-Za-z][A-Za-z0-9_]*\s*\(\s*[A-Za-z]\s*\))\s*=",
        stripped,
        flags=re.IGNORECASE,
    )
    explicit_graph_request = re.search(
        r"\b(?:graph|plot|sketch)\b", lowered
    ) and re.search(r"[=<>≤≥]|\b(?:function|relation|inequality)\b", lowered)
    explicit_bivariate_relation = bool(
        re.search(r"=", stripped)
        and re.search(r"(?<![A-Za-z])x(?![A-Za-z])", stripped, re.IGNORECASE)
        and re.search(r"(?<![A-Za-z])y(?![A-Za-z])", stripped, re.IGNORECASE)
    )
    inequality_region = re.search(
        r"\b(?:shade|region|locus|feasible\s+set|solution\s+set)\b",
        lowered,
    ) and re.search(r"(?:<=|>=|<|>|≤|≥)", stripped)
    positional_coordinates = re.search(
        r"\b(?:point|coordinate|plot|graph)\b.{0,40}"
        r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    implied_named_coordinates = re.search(
        r"\b[A-Z][A-Za-z0-9_]{0,31}\s*"
        r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)",
        stripped,
    )
    if (
        explicit_function
        or explicit_graph_request
        or explicit_bivariate_relation
        or inequality_region
        or positional_coordinates
        or implied_named_coordinates
    ):
        return VisualizationClassification(
            VisualizationCapability.graphing,
            VisualizationEnvironment.graphing,
            "the problem states a function, graphable relation, region, locus, or coordinates",
        )

    if re.search(r"\b(?:use|in|with)\s+(?:a\s+)?cas\b", lowered):
        return VisualizationClassification(
            VisualizationCapability.cas,
            VisualizationEnvironment.cas,
            "the problem explicitly requests a CAS environment",
        )

    return VisualizationClassification(
        VisualizationCapability.no_visualization,
        None,
        "the problem states no geometric, functional, positional, or chart structure",
    )


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
        problem_type: ProblemType,
        parser_model: str,
        *,
        request_id: str | None = None,
    ) -> GeometryExtractionResult:
        classification = self.classify_visualization(text)
        if not classification.visualizable:
            logger.info(
                "Visualization extraction skipped request_id=%s capability=%s reason=%s",
                request_id,
                classification.capability.value,
                classification.reason,
            )
            return GeometryExtractionResult(dsl=GeometryDSL(), summary=None, warnings=[])

        environment = classification.environment
        if environment is None:  # Kept explicit for static type narrowing.
            return GeometryExtractionResult(dsl=GeometryDSL(), summary=None, warnings=[])
        retrieved = tuple(
            command
            for command in self.registry.search(
                text, environment, limit=GEOMETRY_RETRIEVAL_LIMIT
            )
            if command.name in ROLLED_OUT_GENERIC_COMMANDS
        )
        allowed_commands = frozenset(command.name for command in retrieved)

        if parser_model.startswith("local:"):
            if problem_type is ProblemType.algebra:
                deterministic = self._extract_heuristically(
                    text,
                    problem_type,
                    environment=environment,
                    allowed_commands=allowed_commands,
                    retrieved_commands=retrieved,
                )
                if deterministic.dsl.actions:
                    return deterministic
            if self._local_geometry_parser_enabled() and len(text.strip()) <= 4_000:
                try:
                    return await self._extract_with_local_llama(
                        text, environment, retrieved, request_id=request_id
                    )
                except ValueError:
                    fallback = self._extract_heuristically(
                        text,
                        problem_type,
                        environment=environment,
                        allowed_commands=allowed_commands,
                        retrieved_commands=retrieved,
                    )
                    fallback.warnings.insert(
                        0,
                        "The model-generated visualization plan failed validation; "
                        "a deterministic fallback was used.",
                    )
                    return fallback
                except Exception as exc:
                    diagnostics = exception_diagnostics(exc)
                    logger.warning(
                        "Local geometry extraction failed request_id=%s model=%s "
                        "error_type=%s error_message=%s status_code=%s response_body=%s",
                        request_id,
                        getattr(self.llama_client, "model", None),
                        diagnostics.error_type,
                        diagnostics.error_message,
                        diagnostics.status_code,
                        diagnostics.response_body,
                    )
            return self._extract_heuristically(
                text,
                problem_type,
                environment=environment,
                allowed_commands=allowed_commands,
                retrieved_commands=retrieved,
            )

        if self.nvidia_client.enabled:
            try:
                return await self._extract_with_llm(
                    text,
                    parser_model,
                    environment,
                    retrieved,
                    request_id=request_id,
                    completion_client=self.nvidia_client,
                    provider_name="NVIDIA",
                )
            except Exception as exc:
                failure_code = self._remote_failure_code(exc)
                diagnostics = exception_diagnostics(exc)
                logger.warning(
                    "Remote geometry extraction failed request_id=%s model=%s "
                    "environment=%s failure_code=%s error_type=%s error_message=%s "
                    "status_code=%s response_body=%s",
                    request_id,
                    parser_model,
                    environment.value,
                    failure_code,
                    diagnostics.error_type,
                    diagnostics.error_message,
                    diagnostics.status_code,
                    diagnostics.response_body,
                )
                return await self._fallback_after_remote_failure(
                    text,
                    problem_type,
                    environment,
                    retrieved,
                    request_id=request_id,
                    parser_model=parser_model,
                    failure_code=failure_code,
                )

        if self._local_geometry_parser_enabled():
            logger.info(
                "NVIDIA geometry primary skipped request_id=%s model=%s "
                "reason=disabled_or_key_missing",
                request_id,
                parser_model,
            )
            return await self._fallback_after_remote_failure(
                text,
                problem_type,
                environment,
                retrieved,
                request_id=request_id,
                parser_model=parser_model,
                failure_code="remote_request_failure",
            )

        return self._extract_heuristically(
            text,
            problem_type,
            environment=environment,
            allowed_commands=allowed_commands,
            retrieved_commands=retrieved,
        )

    def classify_environment(
        self, text: str, problem_type: ProblemType
    ) -> VisualizationEnvironment | None:
        _ = problem_type
        return self.classify_visualization(text).environment

    def classify_visualization(self, text: str) -> VisualizationClassification:
        return classify_visualization_capability(text)

    def _local_geometry_parser_enabled(self) -> bool:
        return bool(
            getattr(self.settings, "local_llama_geometry_extraction_enabled", False)
            and getattr(self.llama_client, "enabled", False)
            and getattr(self.llama_client, "available", True)
        )

    async def _extract_with_local_llama(
        self,
        text: str,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        *,
        request_id: str | None = None,
    ) -> GeometryExtractionResult:
        command_names = sorted(
            {command.name for command in retrieved}.union(GLOBAL_CORE_COMMANDS),
            key=str.casefold,
        )
        discovery_context = self._format_command_context(retrieved)
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
            timeout_seconds=float(
                getattr(self.settings, "local_llama_geometry_timeout_seconds", 8.0)
            ),
            json_schema=geometry_response_schema(command_names, environment),
            operation="local_geometry_extraction",
            trace_id=request_id,
        )

        dsl, summary = self._parse_geometry_payload(
            payload,
            environment=environment,
            source="Local llama-server",
            allowed_command_names=set(command_names),
        )
        dsl = self._sanitize_local_dsl(text, dsl)
        validation = self.validator.validate(
            dsl, allowed_command_names={command.name for command in retrieved}
        )
        issues = [
            issue.message
            for issue in validation.issues
            if issue.severity is ValidationSeverity.error
        ]
        issues.extend(self._validate_intent_alignment(text, dsl))
        if issues:
            raise ValueError("Invalid local geometry DSL: " + "; ".join(issues))
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
        command_names = sorted(
            allowed_command_names.union(GLOBAL_CORE_COMMANDS),
            key=str.casefold,
        )
        discovery_context = self._format_command_context(retrieved)
        selected_client = completion_client or self.nvidia_client
        payload = await selected_client.complete_json(
            model=parser_model,
            system_prompt=(
                "Extract only visualization intents for a math problem. Output JSON with keys: "
                "summary, dsl. Use DSL version 1.1 with version, space, environment, actions, render_hints. "
                "Supported actions: CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT, "
                "MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION, EXECUTE_COMMAND. "
                "EXECUTE_COMMAND arguments must be typed objects and its command must appear in the "
                "retrieved list. Never return raw GeoGebra commands or JavaScript.\n\n"
                f"Environment: {environment.value}\nRetrieved commands:\n{discovery_context}"
            ),
            user_prompt=text,
            temperature=0.1,
            json_schema=geometry_response_schema(command_names, environment),
            schema_name="geometry_visualization",
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
                f"{provider_name} geometry output did not match the required response schema."
            ) from exc

        validation = self.validator.validate(
            dsl, allowed_command_names=allowed_command_names
        )
        errors = tuple(
            issue
            for issue in validation.issues
            if issue.severity is ValidationSeverity.error
        )
        repair_attempted = False
        if errors:
            self._log_remote_validation_outcome(
                request_id=request_id,
                parser_model=parser_model,
                environment=environment,
                issues=errors,
                repair_attempted=False,
                repair_succeeded=False,
            )
            repair_attempted = True
            repaired_dsl = await self._repair_remote_dsl(
                dsl,
                errors,
                parser_model=parser_model,
                environment=environment,
                retrieved=retrieved,
                request_id=request_id,
                completion_client=selected_client,
                provider_name=provider_name,
            )
            if repaired_dsl is not None:
                dsl = repaired_dsl
                validation = self.validator.validate(
                    dsl, allowed_command_names=allowed_command_names
                )
                errors = tuple(
                    issue
                    for issue in validation.issues
                    if issue.severity is ValidationSeverity.error
                )

        semantic_issues = self._validate_intent_alignment(text, dsl)
        if not dsl.actions:
            semantic_issues.insert(
                0, "The geometry proposal contained no construction actions."
            )
        self._log_remote_validation_outcome(
            request_id=request_id,
            parser_model=parser_model,
            environment=environment,
            issues=errors,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_attempted and not errors and not semantic_issues,
        )
        if errors or semantic_issues:
            issue_codes = Counter(issue.code for issue in errors)
            if semantic_issues:
                issue_codes["semantic_mismatch"] += len(semantic_issues)
            raise GeometryProposalValidationError(
                "The model-generated visualization plan failed deterministic validation "
                "after at most one repair attempt: "
                + json.dumps(dict(sorted(issue_codes.items())), sort_keys=True)
            )

        dsl.actions = list(validation.actions)
        return GeometryExtractionResult(
            dsl=dsl,
            summary=summary,
            warnings=[],
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
    ) -> GeometryDSL | None:
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
            return None

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
            {command.name for command in retrieved}.union(GLOBAL_CORE_COMMANDS),
            key=str.casefold,
        )
        discovery_context = self._format_command_context(retrieved)
        try:
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
                require_parameters=False,
                allow_schema_downgrade=False,
                repair_invalid_json=False,
                timeout_seconds=remote_model_timeout_seconds(
                    self.settings,
                    provider="nvidia_direct",
                    model=parser_model,
                ),
                operation="geometry_repair",
                trace_id=request_id,
            )
            replacements = self._parse_repair_replacements(
                payload,
                expected_indices=tuple(sorted(issues_by_index)),
                allowed_command_names=set(command_names),
            )
        except Exception as exc:
            diagnostics = exception_diagnostics(exc)
            logger.warning(
                "Remote geometry repair failed request_id=%s provider=%s model=%s "
                "error_type=%s error_message=%s status_code=%s response_body=%s",
                request_id,
                provider_name,
                parser_model,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            return None

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
            return None

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

    async def _fallback_after_remote_failure(
        self,
        text: str,
        problem_type: ProblemType,
        environment: VisualizationEnvironment,
        retrieved: tuple[RetrievedCommand, ...],
        *,
        request_id: str | None,
        parser_model: str,
        failure_code: str,
    ) -> GeometryExtractionResult:
        provider_unavailable = failure_code == "structured_output_provider_unavailable"
        model_output_invalid = failure_code in {
            "raw_parse_failure",
            "schema_validation_failure",
            "model_validation_failure",
        }
        if self._local_geometry_parser_enabled() and len(text.strip()) <= 4_000:
            try:
                fallback = await self._extract_with_local_llama(
                    text, environment, retrieved, request_id=request_id
                )
                fallback.warnings.insert(
                    0,
                    (
                    "No eligible remote provider supported the required visualization "
                    "schema; the validated local geometry parser was used instead."
                        if provider_unavailable
                        else (
                            "The remote visualization model produced an invalid plan; the "
                            "validated local geometry parser was used instead."
                            if model_output_invalid
                            else "The remote visualization parser failed before it produced a valid "
                            "plan; the validated local geometry parser was used instead."
                        )
                    ),
                )
                logger.info(
                    "Remote geometry fallback succeeded request_id=%s remote_model=%s "
                    "fallback=local_llama",
                    request_id,
                    parser_model,
                )
                return fallback
            except Exception as exc:
                diagnostics = exception_diagnostics(exc)
                logger.warning(
                    "Remote geometry local fallback failed request_id=%s remote_model=%s "
                    "local_model=%s error_type=%s error_message=%s status_code=%s "
                    "response_body=%s",
                    request_id,
                    parser_model,
                    getattr(self.llama_client, "model", None),
                    diagnostics.error_type,
                    diagnostics.error_message,
                    diagnostics.status_code,
                    diagnostics.response_body,
                )

        if self.nvidia_client.enabled:
            for attempt_number, native_model in enumerate(
                NVIDIA_DIRECT_FALLBACK_MODELS, start=1
            ):
                try:
                    timeout_seconds = remote_model_timeout_seconds(
                        self.settings,
                        provider="nvidia_direct",
                        model=native_model,
                    )
                    logger.info(
                        "NVIDIA direct geometry fallback started request_id=%s model=%s "
                        "operation=geometry_extraction attempt=%s/%s timeout_seconds=%.1f",
                        request_id,
                        native_model,
                        attempt_number,
                        len(NVIDIA_DIRECT_FALLBACK_MODELS),
                        timeout_seconds,
                    )
                    fallback = await self._extract_with_llm(
                        text,
                        native_model,
                        environment,
                        retrieved,
                        request_id=request_id,
                        completion_client=self.nvidia_client,
                        provider_name="NVIDIA",
                        timeout_seconds=timeout_seconds,
                    )
                    fallback.warnings.insert(
                        0,
                        (
                            "No eligible remote provider supported the required "
                            "visualization schema; a deterministically validated NVIDIA "
                            "direct proposal was used instead."
                            if provider_unavailable
                            else (
                                "The remote visualization model produced an invalid plan; a "
                                "deterministically validated NVIDIA direct proposal was used instead."
                                if model_output_invalid
                                else "Remote visualization extraction failed; a deterministically "
                                "validated NVIDIA direct proposal was used instead."
                            )
                        ),
                    )
                    return fallback
                except Exception as exc:
                    diagnostics = exception_diagnostics(exc)
                    logger.warning(
                        "NVIDIA direct geometry fallback failed request_id=%s model=%s "
                        "operation=geometry_extraction attempt=%s/%s error_type=%s "
                        "error_message=%s status_code=%s response_body=%s",
                        request_id,
                        native_model,
                        attempt_number,
                        len(NVIDIA_DIRECT_FALLBACK_MODELS),
                        diagnostics.error_type,
                        diagnostics.error_message,
                        diagnostics.status_code,
                        diagnostics.response_body,
                    )

        deterministic = self._extract_heuristically(
            text,
            problem_type,
            environment=environment,
            allowed_commands=frozenset(command.name for command in retrieved),
            retrieved_commands=retrieved,
        )
        if deterministic.dsl.actions:
            deterministic.warnings.insert(
                0,
                (
                    "No eligible remote provider supported the required visualization "
                    "schema, and the model fallbacks were unavailable; a limited deterministic "
                    "construction was used."
                    if provider_unavailable
                    else (
                        "The model-generated visualization plan failed validation; a limited "
                        "deterministic construction was used."
                        if model_output_invalid
                        else "Model-backed visualization extraction was unavailable; a limited "
                        "deterministic construction was used."
                    )
                ),
            )
            return deterministic

        return GeometryExtractionResult(
            dsl=GeometryDSL(
                version="1.1",
                space=(
                    "euclidean_3d"
                    if environment is VisualizationEnvironment.graphics_3d
                    else "euclidean_2d"
                ),
                environment=environment,
            ),
            summary=None,
            warnings=[
                (
                    "No eligible remote provider supported the required visualization "
                    "schema, and no validated fallback produced a plan; no visualization "
                    "was generated."
                    if provider_unavailable
                    else (
                        "The model-generated visualization plan failed validation, and no validated "
                        "fallback produced one; no visualization was generated."
                        if model_output_invalid
                        else "The remote visualization model failed before producing a valid plan, "
                        "and no validated fallback produced one; no visualization was generated."
                    )
                )
            ],
            allowed_commands=frozenset(command.name for command in retrieved),
            retrieved_commands=retrieved,
        )

    def _log_remote_validation_outcome(
        self,
        *,
        request_id: str | None,
        parser_model: str,
        environment: VisualizationEnvironment,
        issues: tuple[GeoGebraValidationIssue, ...],
        repair_attempted: bool,
        repair_succeeded: bool,
    ) -> None:
        issue_counts = Counter(issue.code for issue in issues)
        log = logger.warning if issues else logger.info
        log(
            "Remote geometry validation outcome request_id=%s model=%s environment=%s "
            "passed=%s issue_codes=%s repair_attempted=%s repair_succeeded=%s",
            request_id,
            parser_model,
            environment.value,
            not issues,
            json.dumps(dict(sorted(issue_counts.items())), sort_keys=True),
            repair_attempted,
            repair_succeeded,
        )

    def _remote_failure_code(self, error: Exception) -> str:
        explicit_failure_code = getattr(error, "failure_code", None)
        if isinstance(explicit_failure_code, str):
            return explicit_failure_code
        diagnostics = exception_diagnostics(error)
        message = diagnostics.error_message.casefold()
        response_body = (diagnostics.response_body or "").casefold()
        combined = f"{message} {response_body}"
        if "invalid json" in message:
            return "raw_parse_failure"
        if "geometry dsl schema" in message or "schema validation" in message:
            return "schema_validation_failure"
        provider_markers = (
            "no eligible",
            "no endpoint",
            "requested parameters",
            "require_parameters",
            "structured output provider unavailable",
        )
        if (
            diagnostics.status_code == 404
            and any(marker in combined for marker in provider_markers)
        ) or any(
            marker in combined
            for marker in (
                "no eligible structured-output provider",
                "no eligible provider",
            )
        ):
            return "structured_output_provider_unavailable"
        return "remote_request_failure"

    def _validate_strict_argument_payload(
        self,
        raw_argument: Any,
        *,
        source: str,
        allow_list: bool = True,
    ) -> None:
        if not isinstance(raw_argument, dict):
            raise ValueError(f"{source} command argument must be an object.")
        kind = raw_argument.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{source} command argument kind must be a string.")
        contract = _STRICT_ARGUMENT_FIELDS.get(kind)
        if contract is None or (kind == "list" and not allow_list):
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
                    allow_list=False,
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

        return GeometryAction.model_validate(raw_action)

    def _parse_geometry_payload(
        self,
        payload: dict[str, Any],
        *,
        environment: VisualizationEnvironment,
        source: str,
        allowed_command_names: set[str],
    ) -> tuple[GeometryDSL, str]:
        if set(payload) != {"summary", "dsl"}:
            raise ValueError(
                f"{source} geometry response must contain exactly summary and dsl."
            )
        summary = payload.get("summary")
        if not isinstance(summary, str) or len(summary) > 200:
            raise ValueError(f"{source} geometry summary must be a string of at most 200 characters.")
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
        if set(raw_dsl) != required_dsl_keys:
            raise ValueError(
                f"{source} geometry dsl must contain exactly the required version 1.1 fields."
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
        strict_dsl = dict(raw_dsl)
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

    def _sanitize_local_dsl(self, text: str, dsl: GeometryDSL) -> GeometryDSL:
        sanitized = dsl.model_copy(deep=True)
        number = r"[-+]?\d+(?:\.\d+)?"

        for action in sanitized.actions:
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
            action.coordinates = (
                (float(coordinate_match.group(1)), float(coordinate_match.group(2)))
                if coordinate_match
                else None
            )

        created_points = {
            action.label
            for action in sanitized.actions
            if action.action is GeometryActionType.CREATE_POINT and action.label
        }
        used_labels = {action.label for action in sanitized.actions if action.label}

        center_match = re.search(
            rf"(?:circle\s+with\s+)?center\s+([A-Z])\s+at\s*"
            rf"\(\s*({number})\s*,\s*({number})\s*\)",
            text,
            flags=re.IGNORECASE,
        )
        if center_match:
            center_label = center_match.group(1).upper()
            circle_actions = [
                action
                for action in sanitized.actions
                if action.action is GeometryActionType.CREATE_CIRCLE
            ]
            if circle_actions:
                if center_label not in created_points:
                    center_action = GeometryAction(
                        action=GeometryActionType.CREATE_POINT,
                        label=center_label,
                        coordinates=(
                            float(center_match.group(2)),
                            float(center_match.group(3)),
                        ),
                    )
                    first_circle_index = sanitized.actions.index(circle_actions[0])
                    sanitized.actions.insert(first_circle_index, center_action)
                    created_points.add(center_label)
                    used_labels.add(center_label)

                for circle in circle_actions:
                    circle.center = center_label
                    if circle.label == center_label:
                        circle.label = self._unique_label("c", used_labels)
                        used_labels.add(circle.label)

        triangle_candidates = [
            re.search(
                r"(?:triangle|tam\s+gi(?:á|a)c)\s*\(?([A-Z])([A-Z])([A-Z])\)?",
                text,
                flags=re.IGNORECASE,
            ),
            re.search(
                r"\b([A-Z])([A-Z])([A-Z])\b.{0,20}\btriangle\b",
                text,
                flags=re.IGNORECASE,
            ),
        ]
        triangle_match = next(
            (
                candidate
                for candidate in triangle_candidates
                if candidate
                and all(
                    label.upper() in created_points for label in candidate.groups()
                )
            ),
            None,
        )
        has_polygon = any(
            action.action is GeometryActionType.CREATE_POLYGON
            for action in sanitized.actions
        )
        if triangle_match and not has_polygon:
            triangle_points = [label.upper() for label in triangle_match.groups()]
            if all(label in created_points for label in triangle_points):
                polygon_label = self._unique_label(
                    "poly" + "".join(triangle_points), used_labels
                )
                sanitized.actions.append(
                    GeometryAction(
                        action=GeometryActionType.CREATE_POLYGON,
                        label=polygon_label,
                        points=triangle_points,
                    )
                )

        return sanitized

    def _unique_label(self, preferred: str, used_labels: set[str]) -> str:
        if preferred not in used_labels:
            return preferred
        suffix = 1
        while f"{preferred}{suffix}" in used_labels:
            suffix += 1
        return f"{preferred}{suffix}"

    def _validate_local_dsl(self, dsl: GeometryDSL) -> list[str]:
        issues: list[str] = []
        created_labels: set[str] = set()

        if dsl.space != "euclidean_2d":
            issues.append("Only euclidean_2d constructions are supported.")
        if not dsl.actions:
            issues.append("A visualizable construction must contain at least one action.")
        elif len(dsl.actions) > _MAX_LOCAL_ACTIONS:
            issues.append(
                f"A construction may contain at most {_MAX_LOCAL_ACTIONS} actions."
            )

        for index, action in enumerate(dsl.actions, start=1):
            label = (action.label or "").strip()
            if not _SAFE_LABEL.fullmatch(label):
                issues.append(f"Action {index} has an invalid or missing label.")
                continue
            if label in created_labels:
                issues.append(f"Action {index} redefines label '{label}'.")
                continue

            references: list[str] = []
            if action.action is GeometryActionType.CREATE_POINT:
                if action.coordinates and not all(
                    math.isfinite(value) for value in action.coordinates
                ):
                    issues.append(f"Point '{label}' has non-finite coordinates.")

            elif action.action is GeometryActionType.CREATE_LINE:
                references = list(action.points or action.through or [])[:2]
                if len(references) < 2:
                    issues.append(f"Line '{label}' requires two points.")

            elif action.action is GeometryActionType.CREATE_CIRCLE:
                if action.center and action.radius is not None:
                    references = [action.center]
                    if not math.isfinite(action.radius) or action.radius <= 0:
                        issues.append(f"Circle '{label}' requires a positive finite radius.")
                else:
                    references = list(action.through or [])[:2]
                    if len(references) < 2:
                        issues.append(
                            f"Circle '{label}' requires a center/radius or two points."
                        )

            elif action.action is GeometryActionType.CREATE_POLYGON:
                references = list(action.points)
                if len(references) < 3:
                    issues.append(f"Polygon '{label}' requires at least three points.")

            elif action.action is GeometryActionType.INTERSECT:
                objects = action.metadata.get("objects", [])
                references = list(objects[:2]) if isinstance(objects, list) else []
                if len(references) < 2:
                    issues.append(f"Intersection '{label}' requires two objects.")

            elif action.action is GeometryActionType.MIDPOINT:
                references = list(action.points)[:2]
                if len(references) < 2:
                    issues.append(f"Midpoint '{label}' requires two points.")

            elif action.action in {
                GeometryActionType.PERPENDICULAR,
                GeometryActionType.PARALLEL,
            }:
                through_point = action.metadata.get("through_point")
                reference_line = action.line or action.metadata.get("reference_line")
                references = [
                    value
                    for value in (through_point, reference_line)
                    if isinstance(value, str) and value
                ]
                if len(references) < 2:
                    issues.append(
                        f"{action.action.value} '{label}' requires a point and a line."
                    )

            elif action.action is GeometryActionType.ANGLE_BISECTOR:
                references = list(action.points)[:3]
                if len(references) < 3:
                    issues.append(f"Angle bisector '{label}' requires three points.")

            elif action.action is GeometryActionType.CREATE_FUNCTION:
                equation = (action.equation or "").strip()
                if not _SAFE_FUNCTION_EXPRESSION.fullmatch(equation):
                    issues.append(f"Function '{label}' has an unsafe expression.")
                else:
                    identifiers = {
                        token.lower()
                        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", equation)
                    }
                    unsupported = identifiers - {
                        "x",
                        "sin",
                        "cos",
                        "tan",
                        "sqrt",
                        "abs",
                        "exp",
                        "log",
                        "ln",
                        "pi",
                        "e",
                    }
                    if unsupported:
                        issues.append(
                            f"Function '{label}' uses unsupported identifiers: "
                            + ", ".join(sorted(unsupported))
                        )

            for reference in references:
                if not isinstance(reference, str) or not _SAFE_LABEL.fullmatch(reference):
                    issues.append(f"Action {index} has an invalid object reference.")
                elif reference not in created_labels:
                    issues.append(
                        f"Action {index} references undefined object '{reference}'."
                    )

            created_labels.add(label)

        return issues

    def _validate_intent_alignment(self, text: str, dsl: GeometryDSL) -> list[str]:
        action_types = {action.action for action in dsl.actions}
        generic_commands = {
            (action.command or "").casefold()
            for action in dsl.actions
            if action.action is GeometryActionType.EXECUTE_COMMAND
        }
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

        return [
            f"The prompt requests {description}, but the DSL has no {action.value} action."
            for action, description in expected_actions
            if action not in action_types
            and {
                GeometryActionType.CREATE_CIRCLE: "circle",
                GeometryActionType.MIDPOINT: "midpoint",
                GeometryActionType.PERPENDICULAR: "perpendicularline",
                GeometryActionType.PARALLEL: "parallelline",
                GeometryActionType.CREATE_FUNCTION: "function",
            }.get(action)
            not in generic_commands
        ]



    def _coerce_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text[:200] or None

    def _extract_heuristically(
        self,
        text: str,
        problem_type: ProblemType,
        *,
        environment: VisualizationEnvironment | None = None,
        allowed_commands: frozenset[str] = frozenset(),
        retrieved_commands: tuple[RetrievedCommand, ...] = (),
    ) -> GeometryExtractionResult:
        environment = (
            environment
            or self.classify_environment(text, problem_type)
            or VisualizationEnvironment.geometry_2d
        )
        lowered = text.lower()
        actions: list[GeometryAction] = []
        warnings: list[str] = []
        summary: str | None = None

        def has_point(label: str) -> bool:
            return any(
                action.action is GeometryActionType.CREATE_POINT
                and action.label == label.upper()
                for action in actions
            )

        def ensure_point(label: str) -> None:
            label = label.upper()
            if not has_point(label):
                actions.append(
                    GeometryAction(action=GeometryActionType.CREATE_POINT, label=label)
                )

        function_match = re.search(
            r"(?:y|f\s*\(\s*x\s*\))\s*=\s*([0-9xX+\-*/^²().\s]+)",
            text,
            flags=re.IGNORECASE,
        )
        if function_match:
            equation = self._clean_equation(function_match.group(1))
            actions.append(
                GeometryAction(
                    action=GeometryActionType.CREATE_FUNCTION,
                    label="f",
                    equation=equation,
                )
            )
            summary = "Interactive function graph"

        triangle_match = re.search(
            r"(?:triangle|tam\s+gi(?:á|a)c)\s*\(?([A-Z])([A-Z])([A-Z])\)?",
            text,
            flags=re.IGNORECASE,
        )
        if triangle_match:
            points = [point.upper() for point in triangle_match.groups()]
            for point in points:
                actions.append(
                    GeometryAction(action=GeometryActionType.CREATE_POINT, label=point)
                )
            actions.append(
                GeometryAction(
                    action=GeometryActionType.CREATE_POLYGON,
                    label="poly1",
                    points=points,
                )
            )
            summary = summary or "Triangle construction"

        circle_match = re.search(
            r"circle\s+(?:with|has)\s+center\s+([A-Z])(?:\s+at\s*\(([-\d.]+),\s*([-\d.]+)\))?(?:\s+and)?\s+radius\s*([-\d.]+)",
            text,
            flags=re.IGNORECASE,
        )
        if circle_match:
            center_label = circle_match.group(1).upper()
            x_coord = circle_match.group(2)
            y_coord = circle_match.group(3)
            radius = float(circle_match.group(4))
            point_action = GeometryAction(
                action=GeometryActionType.CREATE_POINT, label=center_label
            )
            if x_coord and y_coord:
                point_action.coordinates = (float(x_coord), float(y_coord))
            actions.append(point_action)
            actions.append(
                GeometryAction(
                    action=GeometryActionType.CREATE_CIRCLE,
                    label="c",
                    center=center_label,
                    radius=radius,
                )
            )
            summary = summary or "Circle construction"

        has_circle = any(
            action.action is GeometryActionType.CREATE_CIRCLE for action in actions
        )
        if not has_circle:
            center_match = re.search(
                r"(?:\(\(?\s*([A-Z])\s*[;,.]\s*R\s*\)?\)?|center\s+([A-Z]))",
                text,
                flags=re.IGNORECASE,
            )
            center_label = None
            if center_match:
                center_label = center_match.group(1) or center_match.group(2)
            if center_label:
                center_label = center_label.upper()
                ensure_point(center_label)
                through_point = "B" if has_point("B") else None
                if through_point:
                    actions.append(
                        GeometryAction(
                            action=GeometryActionType.CREATE_CIRCLE,
                            label="c",
                            through=[center_label, through_point],
                        )
                    )
                    summary = summary or "Circle construction"

        explicit_point_pattern = re.finditer(
            r"point\s+([A-Z])\s+(?:at|=)\s*\(([-\d.]+),\s*([-\d.]+)\)",
            text,
            flags=re.IGNORECASE,
        )
        for match in explicit_point_pattern:
            actions.append(
                GeometryAction(
                    action=GeometryActionType.CREATE_POINT,
                    label=match.group(1).upper(),
                    coordinates=(float(match.group(2)), float(match.group(3))),
                )
            )

        midpoint_match = re.search(
            r"midpoint\s+of\s+([A-Z])([A-Z])", text, flags=re.IGNORECASE
        )
        if midpoint_match:
            p1, p2 = midpoint_match.group(1).upper(), midpoint_match.group(2).upper()
            ensure_point(p1)
            ensure_point(p2)
            actions.append(
                GeometryAction(
                    action=GeometryActionType.MIDPOINT,
                    label="M",
                    points=[p1, p2],
                )
            )
            summary = summary or "Midpoint construction"

        if "perpendicular bisector" in lowered:
            segment_match = re.search(
                r"perpendicular bisector of\s+([A-Z])([A-Z])", text, flags=re.IGNORECASE
            )
            if segment_match:
                p1, p2 = segment_match.group(1).upper(), segment_match.group(2).upper()
                ensure_point(p1)
                ensure_point(p2)
                actions.append(
                    GeometryAction(
                        action=GeometryActionType.MIDPOINT, label="M", points=[p1, p2]
                    )
                )
                actions.append(
                    GeometryAction(
                        action=GeometryActionType.CREATE_LINE,
                        label="l1",
                        points=[p1, p2],
                    )
                )
                actions.append(
                    GeometryAction(
                        action=GeometryActionType.PERPENDICULAR,
                        label="pb",
                        line="l1",
                        metadata={"through_point": "M"},
                    )
                )
                summary = summary or "Perpendicular bisector construction"

        if not actions and problem_type is ProblemType.geometry:
            warnings.append(
                "No deterministic geometry pattern was recognized, so no visualization was generated."
            )

        if (
            not actions
            and problem_type is ProblemType.algebra
            and self._looks_graphable(text)
        ):
            warnings.append("No graphable expression was detected in the prompt.")

        return GeometryExtractionResult(
            dsl=GeometryDSL(
                space=(
                    "euclidean_3d"
                    if environment is VisualizationEnvironment.graphics_3d
                    else "euclidean_2d"
                ),
                environment=environment,
                actions=actions,
            ),
            summary=summary,
            warnings=warnings,
            allowed_commands=allowed_commands,
            retrieved_commands=retrieved_commands,
        )

    def _looks_like_geometry_problem(self, text: str) -> bool:
        geometry_terms = re.search(
            r"\btriangle\b|\bcircle\b|\bmidpoint\b|\bdiameter\b|"
            r"tam\s+gi(?:á|a)c|đường\s+tròn|duong\s+tron|trung\s+điểm|"
            r"đường\s+kính|duong\s+kinh|vuông\s+góc|vuong\s+goc|"
            r"nội\s+tiếp|noi\s+tiep|\\perp|⊥",
            text,
            flags=re.IGNORECASE,
        )
        point_labels = {
            label
            for token in re.findall(r"\b[A-Z]{1,8}\b", text)
            for label in token
        }
        return bool(geometry_terms and len(point_labels) >= 2)

    def _looks_graphable(self, text: str) -> bool:
        return bool(
            re.search(
                r"(?:y|f\s*\(\s*x\s*\))\s*=\s*[0-9xX+\-*/^²().\s]+",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _clean_equation(self, expression: str) -> str:
        return expression.replace("²", "^2").replace("−", "-").strip(" .,:;?\n\t")
