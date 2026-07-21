from __future__ import annotations

import heapq
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from app.schemas.geometry_dsl import (
    AngleArgument,
    BooleanArgument,
    EquationArgument,
    ExpressionArgument,
    DefinitionObjectType,
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
    ValidationSeverity,
    VectorArgument,
    VisualizationEnvironment,
    LABEL_PATTERN,
)
from app.services.geogebra_command_registry import (
    CommandOverload,
    GeoGebraCommandRegistry,
    GeoGebraObjectType,
)
from app.services.geogebra_support_policy import SupportStatus


logger = logging.getLogger(__name__)
_NUMBER_TOKEN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_NUMERIC_PAIR = re.compile(
    rf"\(\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\)"
)


@dataclass(frozen=True)
class ValidationResult:
    actions: tuple[GeometryAction, ...]
    object_types: dict[str, GeoGebraObjectType]
    issues: tuple[GeoGebraValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(
            issue.severity is ValidationSeverity.error for issue in self.issues
        )


_HIGH_LEVEL_OUTPUT_TYPES: dict[GeometryActionType, GeoGebraObjectType] = {
    GeometryActionType.CREATE_POINT: GeoGebraObjectType.POINT,
    GeometryActionType.CREATE_LINE: GeoGebraObjectType.LINE,
    GeometryActionType.CREATE_CIRCLE: GeoGebraObjectType.CIRCLE,
    GeometryActionType.CREATE_POLYGON: GeoGebraObjectType.POLYGON,
    GeometryActionType.INTERSECT: GeoGebraObjectType.POINT,
    GeometryActionType.MIDPOINT: GeoGebraObjectType.POINT,
    GeometryActionType.PERPENDICULAR: GeoGebraObjectType.LINE,
    GeometryActionType.PARALLEL: GeoGebraObjectType.LINE,
    GeometryActionType.ANGLE_BISECTOR: GeoGebraObjectType.LINE,
    GeometryActionType.CREATE_FUNCTION: GeoGebraObjectType.FUNCTION,
}

_DEFINITION_OUTPUT_TYPES: dict[DefinitionObjectType, GeoGebraObjectType] = {
    DefinitionObjectType.function: GeoGebraObjectType.FUNCTION,
    DefinitionObjectType.equation: GeoGebraObjectType.EQUATION,
    DefinitionObjectType.expression: GeoGebraObjectType.FUNCTION,
    DefinitionObjectType.number: GeoGebraObjectType.NUMBER,
    DefinitionObjectType.point: GeoGebraObjectType.POINT,
    DefinitionObjectType.vector: GeoGebraObjectType.VECTOR,
    DefinitionObjectType.list: GeoGebraObjectType.LIST,
    DefinitionObjectType.text: GeoGebraObjectType.TEXT,
    DefinitionObjectType.boolean: GeoGebraObjectType.BOOLEAN,
    DefinitionObjectType.interval: GeoGebraObjectType.INTERVAL,
}

_COMPATIBLE_ACTUAL_TYPES: dict[GeoGebraObjectType, frozenset[GeoGebraObjectType]] = {
    GeoGebraObjectType.CIRCLE: frozenset(
        {GeoGebraObjectType.CIRCLE, GeoGebraObjectType.CONIC}
    ),
    GeoGebraObjectType.SEGMENT: frozenset(
        {GeoGebraObjectType.SEGMENT, GeoGebraObjectType.LINE}
    ),
    GeoGebraObjectType.RAY: frozenset(
        {GeoGebraObjectType.RAY, GeoGebraObjectType.LINE}
    ),
}


def _issue(
    code: str,
    index: int | None,
    action: GeometryAction | None,
    message: str,
    *,
    severity: ValidationSeverity = ValidationSeverity.error,
) -> GeoGebraValidationIssue:
    return GeoGebraValidationIssue(
        code=code,
        action_index=index,
        command=action.command if action else None,
        output_label=action.output_label if action else None,
        message=message,
        severity=severity,
    )


def _argument_references(argument: GeoGebraArgument) -> list[str]:
    if isinstance(argument, ReferenceArgument):
        return [argument.value]
    if isinstance(argument, ListArgument):
        return [
            reference
            for item in argument.items
            for reference in _argument_references(item)
        ]
    return []


def _action_references(action: GeometryAction) -> list[str]:
    if action.action is GeometryActionType.EXECUTE_COMMAND:
        return [
            reference
            for argument in action.arguments
            for reference in _argument_references(argument)
        ]
    if action.action is GeometryActionType.DEFINE_OBJECT and action.value is not None:
        return _argument_references(action.value)
    if action.action is GeometryActionType.CREATE_LINE:
        return list(action.points or action.through or [])[:2]
    if action.action is GeometryActionType.CREATE_CIRCLE:
        if action.center and action.radius is not None:
            return [action.center]
        return list(action.through or [])[:3]
    if action.action in {
        GeometryActionType.CREATE_POLYGON,
        GeometryActionType.MIDPOINT,
        GeometryActionType.ANGLE_BISECTOR,
    }:
        return list(action.points)
    if action.action is GeometryActionType.INTERSECT:
        objects = action.metadata.get("objects", [])
        return list(objects) if isinstance(objects, list) else []
    if action.action in {
        GeometryActionType.PERPENDICULAR,
        GeometryActionType.PARALLEL,
    }:
        through_point = action.metadata.get("through_point")
        return [
            reference
            for reference in (through_point, action.line)
            if isinstance(reference, str)
        ]
    return []


def _argument_type(
    argument: GeoGebraArgument, object_types: dict[str, GeoGebraObjectType]
) -> GeoGebraObjectType:
    if isinstance(argument, ReferenceArgument):
        return object_types.get(argument.value, GeoGebraObjectType.UNKNOWN)
    if isinstance(argument, NumberArgument):
        return GeoGebraObjectType.NUMBER
    if isinstance(argument, AngleArgument):
        return GeoGebraObjectType.ANGLE
    if isinstance(argument, PointArgument):
        return GeoGebraObjectType.POINT
    if isinstance(argument, VectorArgument):
        return GeoGebraObjectType.VECTOR
    if isinstance(argument, TextArgument):
        return GeoGebraObjectType.TEXT
    if isinstance(argument, BooleanArgument):
        return GeoGebraObjectType.BOOLEAN
    if isinstance(argument, EquationArgument):
        return GeoGebraObjectType.EQUATION
    if isinstance(argument, ListArgument):
        if argument.items and all(
            isinstance(item, ListArgument) for item in argument.items
        ):
            return GeoGebraObjectType.MATRIX
        return GeoGebraObjectType.LIST
    if isinstance(argument, IntervalArgument):
        return GeoGebraObjectType.INTERVAL
    if isinstance(argument, ExpressionArgument):
        return GeoGebraObjectType.UNKNOWN
    return GeoGebraObjectType.UNKNOWN


def _type_matches(
    actual: GeoGebraObjectType, expected: frozenset[GeoGebraObjectType]
) -> bool:
    if GeoGebraObjectType.UNKNOWN in expected or actual is GeoGebraObjectType.UNKNOWN:
        return True
    actual_options = _COMPATIBLE_ACTUAL_TYPES.get(actual, frozenset({actual}))
    return bool(actual_options.intersection(expected))


def _count_matches(overload: CommandOverload, count: int) -> bool:
    signature = overload.signature
    if not signature.normalization_certain:
        return False
    if count < signature.min_arguments:
        return False
    return signature.max_arguments is None or count <= signature.max_arguments


def _types_match(
    overload: CommandOverload,
    arguments: list[GeoGebraArgument],
    object_types: dict[str, GeoGebraObjectType],
) -> bool:
    expected_types = overload.signature.expected_types
    if not expected_types and arguments:
        return False
    for index, argument in enumerate(arguments):
        expected = expected_types[min(index, len(expected_types) - 1)]
        if not _type_matches(_argument_type(argument, object_types), expected):
            return False
    return True


def _canonical_number(value: str | float) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number.normalize() if number else Decimal(0)


def _numeric_pairs(text: str | None) -> set[tuple[Decimal, Decimal]]:
    if not text:
        return set()
    pairs: set[tuple[Decimal, Decimal]] = set()
    for match in _NUMERIC_PAIR.finditer(text):
        x = _canonical_number(match.group(1))
        y = _canonical_number(match.group(2))
        if x is not None and y is not None:
            pairs.add((x, y))
    return pairs


class GeoGebraDSLValidator:
    def __init__(self, registry: GeoGebraCommandRegistry | None = None) -> None:
        self.registry = registry or GeoGebraCommandRegistry.cached()

    def validate(
        self,
        dsl: GeometryDSL,
        *,
        allowed_command_names: Iterable[str] | None = None,
        normalized_problem_text: str | None = None,
        solved_answer_text: str | None = None,
    ) -> ValidationResult:
        actions = [action.model_copy(deep=True) for action in dsl.actions]
        issues: list[GeoGebraValidationIssue] = []
        output_to_index: dict[str, int] = {}
        object_types: dict[str, GeoGebraObjectType] = {}
        allowed = (
            {name.casefold() for name in allowed_command_names}
            if allowed_command_names is not None
            else None
        )

        for index, action in enumerate(actions):
            if action.action is not GeometryActionType.EXECUTE_COMMAND:
                self._validate_high_level_action(action, index, issues)
            output = action.output_label
            if action.action is not GeometryActionType.EXECUTE_COMMAND and output is None:
                issues.append(
                    _issue(
                        "missing_output_label",
                        index,
                        action,
                        f"{action.action.value} requires a validated output label.",
                    )
                )
            if output is not None:
                if output in output_to_index:
                    issues.append(
                        _issue(
                            "duplicate_label",
                            index,
                            action,
                            f"Output label '{output}' is already produced by action "
                            f"{output_to_index[output]}.",
                        )
                    )
                else:
                    output_to_index[output] = index
                    object_types[output] = self._declared_output_type(
                        action, object_types
                    )

        dependencies: list[set[int]] = [set() for _ in actions]
        dependents: list[set[int]] = [set() for _ in actions]
        for index, action in enumerate(actions):
            seen_references: set[str] = set()
            for reference in _action_references(action):
                if not isinstance(reference, str) or not re.fullmatch(
                    LABEL_PATTERN, reference
                ):
                    continue
                if reference in seen_references:
                    continue
                seen_references.add(reference)
                producer = output_to_index.get(reference)
                if producer is None:
                    issues.append(
                        _issue(
                            "undefined_reference",
                            index,
                            action,
                            f"Reference '{reference}' is never produced by this construction.",
                        )
                    )
                    continue
                dependencies[index].add(producer)
                dependents[producer].add(index)

        for style in dsl.render_hints.styles:
            if style.label not in output_to_index:
                issues.append(
                    _issue(
                        "undefined_style_reference",
                        None,
                        None,
                        f"Style target '{style.label}' is never produced by this construction.",
                    )
                )
        interaction = dsl.render_hints.interaction
        if interaction is not None:
            for target in [
                *interaction.movable_points,
                *interaction.animated_objects,
            ]:
                if target not in output_to_index:
                    issues.append(
                        _issue(
                            "undefined_interaction_reference",
                            None,
                            None,
                            f"Interaction target '{target}' is never produced by this construction.",
                        )
                    )

        sorted_indices = self._topological_order(dependencies, dependents)
        if len(sorted_indices) != len(actions):
            cycle_indices = [
                index for index, incoming in enumerate(dependencies) if incoming
            ]
            issues.append(
                _issue(
                    "dependency_cycle",
                    cycle_indices[0] if cycle_indices else None,
                    actions[cycle_indices[0]] if cycle_indices else None,
                    "The construction contains a cyclic dependency involving actions "
                    + ", ".join(str(index) for index in cycle_indices)
                    + ".",
                )
            )
            sorted_indices = list(range(len(actions)))

        # Resolve same-as-input command outputs after dependency sorting so a
        # transformed point remains a point (and likewise for other objects).
        resolved_object_types: dict[str, GeoGebraObjectType] = {}
        for index in sorted_indices:
            action = actions[index]
            output = action.output_label
            if output is not None:
                resolved_object_types[output] = self._declared_output_type(
                    action, resolved_object_types
                )
        object_types = resolved_object_types

        for index in sorted_indices:
            action = actions[index]
            if action.action is GeometryActionType.EXECUTE_COMMAND:
                self._validate_generic_action(
                    action,
                    index,
                    dsl.environment,
                    object_types,
                    allowed,
                    issues,
                )
            self._validate_dimensions(action, index, dsl.environment, issues)

        self._validate_answer_only_point_coordinates(
            actions,
            normalized_problem_text=normalized_problem_text,
            solved_answer_text=solved_answer_text,
            issues=issues,
        )

        result = ValidationResult(
            actions=tuple(actions[index] for index in sorted_indices),
            object_types=object_types,
            issues=tuple(issues),
        )
        error_counts = Counter(
            issue.code
            for issue in result.issues
            if issue.severity is ValidationSeverity.error
        )
        warning_counts = Counter(
            issue.code
            for issue in result.issues
            if issue.severity is ValidationSeverity.warning
        )
        log = logger.warning if error_counts else logger.info
        log(
            "GeoGebra DSL validation outcome environment=%s action_count=%s passed=%s "
            "error_codes=%s warning_codes=%s",
            dsl.environment.value,
            len(actions),
            result.passed,
            json.dumps(dict(sorted(error_counts.items())), sort_keys=True),
            json.dumps(dict(sorted(warning_counts.items())), sort_keys=True),
        )
        return result

    def _validate_answer_only_point_coordinates(
        self,
        actions: list[GeometryAction],
        *,
        normalized_problem_text: str | None,
        solved_answer_text: str | None,
        issues: list[GeoGebraValidationIssue],
    ) -> None:
        """Reject the narrow anti-pattern of plotting answer tuples as point data."""

        if not normalized_problem_text or not solved_answer_text or len(actions) < 2:
            return
        if any(action.action is not GeometryActionType.CREATE_POINT for action in actions):
            return
        if any(action.coordinates is None for action in actions):
            return

        problem_pairs = _numeric_pairs(normalized_problem_text)
        answer_pairs = _numeric_pairs(solved_answer_text)
        action_pairs: list[tuple[Decimal, Decimal]] = []
        for action in actions:
            coordinates = action.coordinates
            if coordinates is None:
                return
            x = _canonical_number(coordinates[0])
            y = _canonical_number(coordinates[1])
            if x is None or y is None:
                return
            action_pairs.append((x, y))

        if not action_pairs or any(pair not in answer_pairs for pair in action_pairs):
            return
        if any(pair in problem_pairs for pair in action_pairs):
            return

        for index, action in enumerate(actions):
            issues.append(
                _issue(
                    "answer_only_point_coordinates",
                    index,
                    action,
                    "CREATE_POINT coordinates reproduce numeric pairs from the solved "
                    "answer, but the problem statement gives no matching coordinates.",
                )
            )

    def _validate_high_level_action(
        self,
        action: GeometryAction,
        index: int,
        issues: list[GeoGebraValidationIssue],
    ) -> None:
        message: str | None = None
        if action.action is GeometryActionType.CREATE_LINE:
            if len(action.points or action.through or []) != 2:
                message = "CREATE_LINE requires exactly two point references."
        elif action.action is GeometryActionType.CREATE_CIRCLE:
            has_center_radius = action.center is not None and action.radius is not None
            through_count = len(action.through or [])
            if not has_center_radius and through_count not in {2, 3}:
                message = "CREATE_CIRCLE requires center/radius or two or three points."
        elif action.action is GeometryActionType.CREATE_POLYGON:
            if len(action.points) < 3:
                message = "CREATE_POLYGON requires at least three points."
        elif action.action is GeometryActionType.INTERSECT:
            objects = action.metadata.get("objects")
            if not isinstance(objects, list) or len(objects) != 2:
                message = "INTERSECT requires exactly two references in metadata.objects."
        elif action.action is GeometryActionType.MIDPOINT:
            if len(action.points) != 2:
                message = "MIDPOINT requires exactly two point references."
        elif action.action in {
            GeometryActionType.PERPENDICULAR,
            GeometryActionType.PARALLEL,
        }:
            through_point = action.metadata.get("through_point")
            if action.line is None or not isinstance(through_point, str):
                message = (
                    f"{action.action.value} requires line and metadata.through_point references."
                )
        elif action.action is GeometryActionType.ANGLE_BISECTOR:
            if len(action.points) != 3:
                message = "ANGLE_BISECTOR requires exactly three point references."
        elif action.action is GeometryActionType.CREATE_FUNCTION:
            if not action.equation:
                message = "CREATE_FUNCTION requires a validated equation expression."
        elif action.action is GeometryActionType.DEFINE_OBJECT:
            if action.output is None or action.object_type is None or action.value is None:
                message = "DEFINE_OBJECT requires output, object_type, and value."

        if message:
            issues.append(_issue("invalid_action_shape", index, action, message))

        for reference in _action_references(action):
            if not isinstance(reference, str) or not re.fullmatch(LABEL_PATTERN, reference):
                issues.append(
                    _issue(
                        "invalid_reference_label",
                        index,
                        action,
                        f"Reference {reference!r} is not a valid GeoGebra label.",
                    )
                )

    def _declared_output_type(
        self,
        action: GeometryAction,
        object_types: dict[str, GeoGebraObjectType] | None = None,
    ) -> GeoGebraObjectType:
        if action.action is GeometryActionType.DEFINE_OBJECT:
            if action.object_type is None:
                return GeoGebraObjectType.UNKNOWN
            return _DEFINITION_OUTPUT_TYPES[action.object_type]
        if action.action is not GeometryActionType.EXECUTE_COMMAND:
            return _HIGH_LEVEL_OUTPUT_TYPES.get(
                action.action, GeoGebraObjectType.UNKNOWN
            )
        if not action.command:
            return GeoGebraObjectType.UNKNOWN
        output_type = self.registry.output_type(action.command)
        if (
            output_type is GeoGebraObjectType.UNKNOWN
            and self.registry.output_type_strategy(action.command)
            == "same_as_first_argument"
        ):
            first = action.arguments[0] if action.arguments else None
            if isinstance(first, ReferenceArgument):
                return (object_types or {}).get(
                    first.value, GeoGebraObjectType.UNKNOWN
                )
            if first is not None:
                return _argument_type(first, {})
        return output_type

    def _topological_order(
        self, dependencies: list[set[int]], dependents: list[set[int]]
    ) -> list[int]:
        remaining = [set(items) for items in dependencies]
        ready = [index for index, incoming in enumerate(remaining) if not incoming]
        heapq.heapify(ready)
        result: list[int] = []
        while ready:
            index = heapq.heappop(ready)
            result.append(index)
            for dependent in sorted(dependents[index]):
                remaining[dependent].discard(index)
                if not remaining[dependent]:
                    heapq.heappush(ready, dependent)
        return result

    def _validate_generic_action(
        self,
        action: GeometryAction,
        index: int,
        environment: VisualizationEnvironment,
        object_types: dict[str, GeoGebraObjectType],
        allowed: set[str] | None,
        issues: list[GeoGebraValidationIssue],
    ) -> None:
        command_name = action.command or ""
        definition = self.registry.lookup(command_name)
        if definition is None:
            issues.append(
                _issue(
                    "unknown_command",
                    index,
                    action,
                    f"Command '{command_name}' is not present in the local GeoGebra registry.",
                )
            )
            return

        action.command = definition.name
        if definition.unsafe_reason:
            issues.append(
                _issue(
                    "unsafe_command",
                    index,
                    action,
                    f"Command '{definition.name}' is disabled because {definition.unsafe_reason}.",
                )
            )
            return
        if (
            allowed is not None
            and definition.name.casefold() not in allowed
        ):
            issues.append(
                _issue(
                    "command_not_retrieved",
                    index,
                    action,
                    f"Command '{definition.name}' was not in the bounded command set supplied to the model.",
                )
            )
            return
        environment_overloads = [
            overload
            for overload in definition.overloads
            if environment in overload.capabilities
        ]
        if not environment_overloads:
            issues.append(
                _issue(
                    "incompatible_environment",
                    index,
                    action,
                    f"Command '{definition.name}' is not enabled for environment '{environment.value}'.",
                )
            )
            return
        eligible_overloads = [
            overload
            for overload in environment_overloads
            if overload.is_runtime_eligible_in(environment)
        ]
        if not eligible_overloads:
            issues.append(
                _issue(
                    "blocked_command",
                    index,
                    action,
                    f"Command '{definition.name}' has no safely executable overload "
                    f"for environment '{environment.value}'.",
                )
            )
            return

        count_overloads = [
            overload
            for overload in eligible_overloads
            if _count_matches(overload, len(action.arguments))
        ]
        if not count_overloads:
            expected = ", ".join(
                overload.signature.original for overload in eligible_overloads[:12]
            )
            issues.append(
                _issue(
                    "argument_count_mismatch",
                    index,
                    action,
                    f"No '{definition.name}' overload accepts {len(action.arguments)} arguments. "
                    f"Available signatures: {expected}",
                )
            )
            return

        matching_overloads = [
            overload
            for overload in count_overloads
            if _types_match(overload, action.arguments, object_types)
        ]
        if not matching_overloads:
            actual = ", ".join(
                _argument_type(argument, object_types).value
                for argument in action.arguments
            )
            issues.append(
                _issue(
                    "argument_type_mismatch",
                    index,
                    action,
                    f"Argument types ({actual}) do not match a known '{definition.name}' overload.",
                )
            )
        elif not any(
            overload.support_status is SupportStatus.supported
            for overload in matching_overloads
        ):
            issues.append(
                _issue(
                    "experimental_command",
                    index,
                    action,
                    f"Command '{definition.name}' uses a safe catalog overload without "
                    "a checked-in browser acceptance record; runtime rejection will "
                    "trigger construction rollback.",
                    severity=ValidationSeverity.warning,
                )
            )

        if action.output is None:
            issues.append(
                _issue(
                    "untracked_output",
                    index,
                    action,
                    "This command has no output label; GeoGebra may auto-label its result, "
                    "but later DSL actions cannot reference it.",
                    severity=ValidationSeverity.warning,
                )
            )

    def _validate_dimensions(
        self,
        action: GeometryAction,
        index: int,
        environment: VisualizationEnvironment,
        issues: list[GeoGebraValidationIssue],
    ) -> None:
        if action.action is GeometryActionType.DEFINE_OBJECT:
            arguments = [action.value] if action.value is not None else []
        elif action.action is GeometryActionType.EXECUTE_COMMAND:
            arguments = action.arguments
        else:
            return
        for argument in self._walk_arguments(arguments):
            if not isinstance(argument, (PointArgument, VectorArgument)):
                continue
            if (
                environment is VisualizationEnvironment.graphics_3d
                and argument.z is None
            ):
                issues.append(
                    _issue(
                        "missing_3d_coordinate",
                        index,
                        action,
                        f"A {argument.kind} literal in graphics_3d must include z.",
                    )
                )
            if (
                environment is not VisualizationEnvironment.graphics_3d
                and argument.z is not None
            ):
                issues.append(
                    _issue(
                        "unexpected_3d_coordinate",
                        index,
                        action,
                        f"A {argument.kind} literal with z requires graphics_3d.",
                    )
                )

    def _walk_arguments(
        self, arguments: Iterable[GeoGebraArgument]
    ) -> Iterable[GeoGebraArgument]:
        for argument in arguments:
            yield argument
            if isinstance(argument, ListArgument):
                yield from self._walk_arguments(argument.items)
