from __future__ import annotations

import math
import re
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_GEOMETRY_ACTIONS = 40
MAX_COMMAND_ARGUMENTS = 16
MAX_LIST_ITEMS = 32
MAX_ARGUMENT_DEPTH = 3
MAX_TEXT_LENGTH = 500
MAX_EXPRESSION_LENGTH = 200
MAX_ABS_NUMBER = 1_000_000_000.0

LABEL_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,31}$"
COMMAND_PATTERN = r"^[A-Za-z][A-Za-z0-9]{0,63}$"
_SAFE_EXPRESSION_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^()., \[\]π°]+$")
_FUNCTION_CALL = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(")
_IDENTIFIER_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_SAFE_EXPRESSION_FUNCTIONS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "floor",
        "ln",
        "log",
        "max",
        "min",
        "round",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)

FiniteNumber: TypeAlias = Annotated[
    float,
    Field(
        strict=True,
        allow_inf_nan=False,
        ge=-MAX_ABS_NUMBER,
        le=MAX_ABS_NUMBER,
    ),
]
GeoGebraLabel: TypeAlias = Annotated[str, Field(pattern=LABEL_PATTERN)]


class VisualizationEnvironment(str, Enum):
    geometry_2d = "geometry_2d"
    graphing = "graphing"
    graphics_3d = "graphics_3d"
    cas = "cas"
    probability = "probability"
    statistics = "statistics"
    spreadsheet = "spreadsheet"


class ValidationSeverity(str, Enum):
    error = "error"
    warning = "warning"


class GeoGebraValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    action_index: int | None = None
    command: str | None = None
    output_label: str | None = None
    message: str
    severity: ValidationSeverity = ValidationSeverity.error


class GeometryActionType(str, Enum):
    CREATE_POINT = "CREATE_POINT"
    CREATE_LINE = "CREATE_LINE"
    CREATE_CIRCLE = "CREATE_CIRCLE"
    CREATE_POLYGON = "CREATE_POLYGON"
    INTERSECT = "INTERSECT"
    MIDPOINT = "MIDPOINT"
    PERPENDICULAR = "PERPENDICULAR"
    PARALLEL = "PARALLEL"
    ANGLE_BISECTOR = "ANGLE_BISECTOR"
    CREATE_FUNCTION = "CREATE_FUNCTION"
    EXECUTE_COMMAND = "EXECUTE_COMMAND"


def _validate_expression(value: str, *, equation_side: bool = False) -> str:
    value = value.strip()
    if not value or len(value) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression must contain 1 to {MAX_EXPRESSION_LENGTH} characters."
        )
    if not _SAFE_EXPRESSION_CHARS.fullmatch(value):
        raise ValueError("Expression contains unsupported characters.")
    if any(token in value for token in (";", "\n", "\r", '"', "'", "//")):
        raise ValueError("Expression contains a command separator or quote.")
    for function_name in _FUNCTION_CALL.findall(value):
        if function_name.lower() not in _SAFE_EXPRESSION_FUNCTIONS:
            raise ValueError(
                f"Function call '{function_name}' is not allowed in an expression."
            )
    function_calls = {name.lower() for name in _FUNCTION_CALL.findall(value)}
    for identifier in _IDENTIFIER_TOKEN.findall(value):
        if identifier.lower() not in function_calls.union({"e", "pi", "x", "y", "z"}):
            raise ValueError(
                f"Identifier '{identifier}' is not allowed in a self-contained expression; "
                "use a reference argument for construction labels."
            )
    if value.count("(") != value.count(")") or value.count("[") != value.count("]"):
        raise ValueError("Expression delimiters are unbalanced.")
    if not equation_side and "=" in value:
        raise ValueError("Use an equation argument for values containing '='.")
    return value


class ReferenceArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["reference"]
    value: GeoGebraLabel


class NumberArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["number"]
    value: FiniteNumber


class AngleArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["angle"]
    value: FiniteNumber
    unit: Literal["degree", "radian"]


class PointArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["point"]
    x: FiniteNumber
    y: FiniteNumber
    z: FiniteNumber | None = None


class VectorArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["vector"]
    x: FiniteNumber
    y: FiniteNumber
    z: FiniteNumber | None = None


class TextArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["text"]
    value: str = Field(min_length=0, max_length=MAX_TEXT_LENGTH)

    @field_validator("value")
    @classmethod
    def reject_control_and_script_text(cls, value: str) -> str:
        if any(character in value for character in ("\n", "\r", "\0", ";")):
            raise ValueError("Text contains a command separator or control character.")
        if value.strip().lower().startswith("javascript:"):
            raise ValueError("JavaScript URLs are not allowed.")
        return value


class BooleanArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["boolean"]
    value: bool


class ExpressionArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["expression"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_expression(cls, value: str) -> str:
        return _validate_expression(value)


class EquationArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["equation"]
    value: str

    @field_validator("value")
    @classmethod
    def validate_equation(cls, value: str) -> str:
        if value.count("=") != 1:
            raise ValueError("An equation must contain exactly one '=' sign.")
        left, right = value.split("=", maxsplit=1)
        return (
            f"{_validate_expression(left, equation_side=True)} = "
            f"{_validate_expression(right, equation_side=True)}"
        )


class IntervalArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["interval"]
    lower: FiniteNumber
    upper: FiniteNumber
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> "IntervalArgument":
        if self.lower > self.upper:
            raise ValueError("Interval lower bound must not exceed its upper bound.")
        return self


class ListArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["list"]
    items: list["GeoGebraArgument"] = Field(max_length=MAX_LIST_ITEMS)


GeoGebraArgument: TypeAlias = Annotated[
    ReferenceArgument
    | NumberArgument
    | AngleArgument
    | PointArgument
    | VectorArgument
    | TextArgument
    | BooleanArgument
    | ExpressionArgument
    | EquationArgument
    | ListArgument
    | IntervalArgument,
    Field(discriminator="kind"),
]

ListArgument.model_rebuild(_types_namespace={"GeoGebraArgument": GeoGebraArgument})


def argument_depth(argument: GeoGebraArgument) -> int:
    if not isinstance(argument, ListArgument) or not argument.items:
        return 1
    return 1 + max(argument_depth(item) for item in argument.items)


class ObjectStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: GeoGebraLabel
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    line_thickness: int | None = Field(default=None, ge=1, le=13)
    line_style: int | None = Field(default=None, ge=0, le=4)
    point_size: int | None = Field(default=None, ge=1, le=9)
    label_visible: bool | None = None
    visible: bool | None = None
    fixed: bool | None = None
    caption: str | None = Field(default=None, max_length=120)

    @field_validator("caption")
    @classmethod
    def validate_caption(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(character in value for character in ("\n", "\r", "\0")):
            raise ValueError("Caption contains a control character.")
        return value


class ViewportHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: FiniteNumber | None = None
    x_max: FiniteNumber | None = None
    y_min: FiniteNumber | None = None
    y_max: FiniteNumber | None = None
    z_min: FiniteNumber | None = None
    z_max: FiniteNumber | None = None
    axes_visible: bool | None = None
    grid_visible: bool | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ViewportHints":
        pairs = (
            (self.x_min, self.x_max, "x"),
            (self.y_min, self.y_max, "y"),
            (self.z_min, self.z_max, "z"),
        )
        for lower, upper, axis in pairs:
            if (lower is None) != (upper is None):
                raise ValueError(f"Both {axis}_min and {axis}_max are required.")
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError(f"{axis}_min must be less than {axis}_max.")
        return self


class InteractionHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movable_points: list[GeoGebraLabel] = Field(default_factory=list, max_length=20)
    animated_objects: list[GeoGebraLabel] = Field(default_factory=list, max_length=20)
    animation: Literal["start", "stop"] | None = None


class RenderHints(BaseModel):
    # Older cached DSL payloads may contain display-only hints unknown to v1.1.
    model_config = ConfigDict(extra="ignore")

    perspective: VisualizationEnvironment | None = None
    styles: list[ObjectStyle] = Field(default_factory=list, max_length=40)
    viewport: ViewportHints | None = None
    interaction: InteractionHints | None = None


class GeometryAction(BaseModel):
    # Existing high-level actions intentionally retain their extensible metadata shape.
    model_config = ConfigDict(extra="allow")

    action: GeometryActionType
    label: GeoGebraLabel | None = None
    output: GeoGebraLabel | None = None
    command: Annotated[str, Field(pattern=COMMAND_PATTERN)] | None = None
    arguments: list[GeoGebraArgument] = Field(
        default_factory=list, max_length=MAX_COMMAND_ARGUMENTS
    )
    points: list[GeoGebraLabel] = Field(default_factory=list, max_length=32)
    coordinates: tuple[FiniteNumber, FiniteNumber] | None = None
    # Kept as a plain string so the existing local-output sanitizer can repair
    # legacy coordinate-shaped center values before semantic validation.
    center: str | None = Field(default=None, max_length=64)
    radius: FiniteNumber | None = None
    through: list[GeoGebraLabel] | None = Field(default=None, max_length=32)
    line: GeoGebraLabel | None = None
    equation: str | None = Field(default=None, max_length=MAX_EXPRESSION_LENGTH)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_generic_shape_and_limits(self) -> "GeometryAction":
        if self.action is GeometryActionType.EXECUTE_COMMAND:
            if not self.command:
                raise ValueError("EXECUTE_COMMAND requires a command name.")
            if self.label is not None:
                raise ValueError(
                    "EXECUTE_COMMAND uses 'output'; 'label' is reserved for high-level actions."
                )
            for argument in self.arguments:
                if argument_depth(argument) > MAX_ARGUMENT_DEPTH:
                    raise ValueError(
                        f"Argument nesting may not exceed {MAX_ARGUMENT_DEPTH} levels."
                    )
        elif self.output is not None or self.command is not None or self.arguments:
            raise ValueError(
                "command, output, and arguments are only valid for EXECUTE_COMMAND."
            )
        if self.equation is not None:
            self.equation = _validate_expression(self.equation)
        if self.radius is not None and (not math.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("Radius must be a positive finite number.")
        return self

    @property
    def output_label(self) -> str | None:
        return self.output if self.action is GeometryActionType.EXECUTE_COMMAND else self.label


class GeometryDSL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0", "1.1"] = "1.0"
    space: Literal["euclidean_2d", "euclidean_3d"] = "euclidean_2d"
    environment: VisualizationEnvironment = VisualizationEnvironment.geometry_2d
    actions: list[GeometryAction] = Field(
        default_factory=list, max_length=MAX_GEOMETRY_ACTIONS
    )
    render_hints: RenderHints = Field(default_factory=RenderHints)

    @model_validator(mode="after")
    def validate_version_and_space(self) -> "GeometryDSL":
        if self.version == "1.0" and any(
            action.action is GeometryActionType.EXECUTE_COMMAND
            for action in self.actions
        ):
            raise ValueError("EXECUTE_COMMAND requires DSL version 1.1.")
        if (
            self.environment is VisualizationEnvironment.graphics_3d
            and self.space != "euclidean_3d"
        ):
            raise ValueError("graphics_3d requires space='euclidean_3d'.")
        if (
            self.space == "euclidean_3d"
            and self.environment is not VisualizationEnvironment.graphics_3d
        ):
            raise ValueError("euclidean_3d requires environment='graphics_3d'.")
        return self
