from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.schemas.geometry_dsl import (
    AngleArgument,
    BooleanArgument,
    EquationArgument,
    ExpressionArgument,
    GeoGebraArgument,
    GeoGebraValidationIssue,
    GeometryAction,
    GeometryActionType,
    GeometryDSL,
    IntervalArgument,
    ListArgument,
    NumberArgument,
    PointArgument,
    ReferenceArgument,
    TextArgument,
    VectorArgument,
)
from app.services.geogebra_command_registry import GeoGebraCommandRegistry
from app.services.geogebra_validator import GeoGebraDSLValidator


@dataclass(frozen=True)
class TranslationResult:
    commands: list[str]
    command_string: str
    validation_passed: bool
    issues: list[GeoGebraValidationIssue]

    @property
    def issue_messages(self) -> list[str]:
        return [issue.message for issue in self.issues]


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".15g")


def _escape_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class GeoGebraTranslator:
    def __init__(self, registry: GeoGebraCommandRegistry | None = None) -> None:
        self.registry = registry or GeoGebraCommandRegistry.cached()
        self.validator = GeoGebraDSLValidator(self.registry)
        self.default_coordinates = {
            "A": (-4.0, 1.5),
            "B": (4.0, 1.5),
            "C": (0.0, 5.5),
            "D": (1.5, 3.0),
            "E": (-1.5, 3.0),
            "O": (0.0, 0.0),
        }

    def translate(
        self,
        dsl: GeometryDSL,
        *,
        allowed_command_names: Iterable[str] | None = None,
        normalized_problem_text: str | None = None,
        solved_answer_text: str | None = None,
    ) -> TranslationResult:
        validation = self.validator.validate(
            dsl,
            allowed_command_names=allowed_command_names,
            normalized_problem_text=normalized_problem_text,
            solved_answer_text=solved_answer_text,
        )
        issues = list(validation.issues)
        if not validation.passed:
            return TranslationResult(
                commands=[],
                command_string="",
                validation_passed=False,
                issues=issues,
            )

        commands = [self._emit_action(action) for action in validation.actions]
        return TranslationResult(
            commands=commands,
            command_string="; ".join(commands),
            validation_passed=True,
            issues=issues,
        )

    def _emit_action(self, action: GeometryAction) -> str:
        if action.action is GeometryActionType.EXECUTE_COMMAND:
            return self._emit_generic_command(action)

        label = action.label or ""
        if action.action is GeometryActionType.CREATE_POINT:
            coordinates = action.coordinates or self.default_coordinates.get(
                label, (0.0, 0.0)
            )
            # Keep v1.0 numeric formatting stable for cached clients and tests.
            return f"{label} = ({coordinates[0]}, {coordinates[1]})"

        if action.action is GeometryActionType.CREATE_LINE:
            points = action.points or action.through or []
            return f"{label} = Line({points[0]}, {points[1]})"

        if action.action is GeometryActionType.CREATE_CIRCLE:
            if action.center and action.radius is not None:
                return f"{label} = Circle({action.center}, {action.radius})"
            points = action.through or []
            return f"{label} = Circle({', '.join(points)})"

        if action.action is GeometryActionType.CREATE_POLYGON:
            return f"{label} = Polygon({', '.join(action.points)})"

        if action.action is GeometryActionType.INTERSECT:
            objects = action.metadata.get("objects", [])
            return f"{label} = Intersect({objects[0]}, {objects[1]})"

        if action.action is GeometryActionType.MIDPOINT:
            return f"{label} = Midpoint({action.points[0]}, {action.points[1]})"

        if action.action is GeometryActionType.PERPENDICULAR:
            through_point = action.metadata["through_point"]
            return f"{label} = PerpendicularLine({through_point}, {action.line})"

        if action.action is GeometryActionType.PARALLEL:
            through_point = action.metadata["through_point"]
            return f"{label} = ParallelLine({through_point}, {action.line})"

        if action.action is GeometryActionType.ANGLE_BISECTOR:
            return (
                f"{label} = AngleBisector("
                f"{action.points[0]}, {action.points[1]}, {action.points[2]})"
            )

        if action.action is GeometryActionType.CREATE_FUNCTION:
            return f"{label}(x) = {action.equation or 'x'}"

        raise ValueError(f"Unsupported validated action: {action.action.value}")

    def _emit_generic_command(self, action: GeometryAction) -> str:
        definition = self.registry.lookup(action.command or "")
        if definition is None or definition.unsafe_reason:
            raise ValueError("Generic command reached translation without validation.")
        arguments = ", ".join(
            self.serialize_argument(argument) for argument in action.arguments
        )
        expression = f"{definition.name}({arguments})"
        return f"{action.output} = {expression}" if action.output else expression

    def serialize_argument(self, argument: GeoGebraArgument) -> str:
        if isinstance(argument, ReferenceArgument):
            return argument.value
        if isinstance(argument, NumberArgument):
            return _format_number(argument.value)
        if isinstance(argument, AngleArgument):
            value = _format_number(argument.value)
            return f"{value}°" if argument.unit == "degree" else f"({value} rad)"
        if isinstance(argument, PointArgument):
            coordinates = [argument.x, argument.y]
            if argument.z is not None:
                coordinates.append(argument.z)
            return f"({', '.join(_format_number(value) for value in coordinates)})"
        if isinstance(argument, VectorArgument):
            coordinates = [argument.x, argument.y]
            if argument.z is not None:
                coordinates.append(argument.z)
            serialized = ", ".join(_format_number(value) for value in coordinates)
            return f"Vector(({serialized}))"
        if isinstance(argument, TextArgument):
            return f'"{_escape_text(argument.value)}"'
        if isinstance(argument, BooleanArgument):
            return "true" if argument.value else "false"
        if isinstance(argument, ExpressionArgument):
            return argument.value
        if isinstance(argument, EquationArgument):
            return f"({argument.value})"
        if isinstance(argument, ListArgument):
            return "{" + ", ".join(
                self.serialize_argument(item) for item in argument.items
            ) + "}"
        if isinstance(argument, IntervalArgument):
            left = "≤" if argument.lower_inclusive else "<"
            right = "≤" if argument.upper_inclusive else "<"
            return (
                f"({_format_number(argument.lower)} {left} x {right} "
                f"{_format_number(argument.upper)})"
            )
        raise TypeError(f"Unsupported GeoGebra argument type: {type(argument)!r}")
