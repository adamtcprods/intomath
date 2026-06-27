from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.schemas.geometry_dsl import GeometryAction, GeometryActionType, GeometryDSL


@dataclass
class TranslationResult:
    commands: list[str]
    command_string: str
    validation_passed: bool
    issues: list[str]


class GeoGebraTranslator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.valid_commands = self._load_catalog(self.settings.geogebra_catalog_path)
        self.fallback_commands = {
            "Line",
            "Circle",
            "Polygon",
            "Intersect",
            "Midpoint",
            "PerpendicularLine",
            "ParallelLine",
            "AngleBisector",
        }
        self.default_coordinates = {
            "A": (-4.0, 1.5),
            "B": (4.0, 1.5),
            "C": (0.0, 5.5),
            "D": (1.5, 3.0),
            "E": (-1.5, 3.0),
            "O": (0.0, 0.0),
        }

    def translate(self, dsl: GeometryDSL) -> TranslationResult:
        commands: list[str] = []
        issues: list[str] = []
        created_labels: set[str] = set()

        for action in dsl.actions:
            emitted = self._emit_action(action, created_labels, issues)
            if emitted:
                commands.extend(emitted)

        self._validate_command_names(commands, issues)
        return TranslationResult(
            commands=commands,
            command_string="; ".join(commands),
            validation_passed=len(issues) == 0,
            issues=issues,
        )

    def _emit_action(
        self, action: GeometryAction, created_labels: set[str], issues: list[str]
    ) -> list[str]:
        if action.action is GeometryActionType.CREATE_POINT:
            label = action.label or f"P{len(created_labels) + 1}"
            coordinates = action.coordinates or self.default_coordinates.get(
                label, (float(len(created_labels)), 0.0)
            )
            created_labels.add(label)
            return [f"{label} = ({coordinates[0]}, {coordinates[1]})"]

        if action.action is GeometryActionType.CREATE_LINE:
            points = action.points or action.through or []
            label_prefix = f"{action.label} = " if action.label else ""
            if len(points) >= 2:
                if any(point not in created_labels for point in points[:2]):
                    issues.append(f"Line references undefined points: {points[:2]}")
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Line({points[0]}, {points[1]})"]
            issues.append("CREATE_LINE requires two points.")
            return []

        if action.action is GeometryActionType.CREATE_CIRCLE:
            label_prefix = f"{action.label} = " if action.label else ""
            if action.center and action.radius is not None:
                if action.center not in created_labels:
                    issues.append(
                        f"Circle references undefined center: {action.center}"
                    )
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Circle({action.center}, {action.radius})"]
            points = action.through or []
            if len(points) >= 2:
                if any(point not in created_labels for point in points[:2]):
                    issues.append(f"Circle references undefined points: {points[:2]}")
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Circle({points[0]}, {points[1]})"]
            issues.append("CREATE_CIRCLE requires center/radius or two points.")
            return []

        if action.action is GeometryActionType.CREATE_POLYGON:
            points = action.points or []
            label_prefix = f"{action.label} = " if action.label else ""
            if len(points) >= 3:
                if any(point not in created_labels for point in points):
                    issues.append(f"Polygon references undefined points: {points}")
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Polygon({', '.join(points)})"]
            issues.append("CREATE_POLYGON requires at least three points.")
            return []

        if action.action is GeometryActionType.INTERSECT:
            objects = action.metadata.get("objects", [])
            label_prefix = f"{action.label} = " if action.label else ""
            if len(objects) >= 2:
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Intersect({objects[0]}, {objects[1]})"]
            issues.append("INTERSECT requires two objects in metadata.objects.")
            return []

        if action.action is GeometryActionType.MIDPOINT:
            label_prefix = f"{action.label} = " if action.label else ""
            points = action.points or []
            if len(points) >= 2:
                if any(point not in created_labels for point in points[:2]):
                    issues.append(f"Midpoint references undefined points: {points[:2]}")
                if action.label:
                    created_labels.add(action.label)
                return [f"{label_prefix}Midpoint({points[0]}, {points[1]})"]
            issues.append("MIDPOINT requires two points.")
            return []

        if action.action is GeometryActionType.PERPENDICULAR:
            label_prefix = f"{action.label} = " if action.label else ""
            through_point = action.metadata.get("through_point")
            reference_line = action.line or action.metadata.get("reference_line")
            if through_point and reference_line:
                if through_point not in created_labels:
                    issues.append(
                        f"Perpendicular line references undefined point: {through_point}"
                    )
                if reference_line not in created_labels:
                    issues.append(
                        f"Perpendicular line references undefined line: {reference_line}"
                    )
                if action.label:
                    created_labels.add(action.label)
                return [
                    f"{label_prefix}PerpendicularLine({through_point}, {reference_line})"
                ]
            issues.append(
                "PERPENDICULAR requires metadata.through_point and a line reference."
            )
            return []

        if action.action is GeometryActionType.PARALLEL:
            label_prefix = f"{action.label} = " if action.label else ""
            through_point = action.metadata.get("through_point")
            reference_line = action.line or action.metadata.get("reference_line")
            if through_point and reference_line:
                if through_point not in created_labels:
                    issues.append(
                        f"Parallel line references undefined point: {through_point}"
                    )
                if reference_line not in created_labels:
                    issues.append(
                        f"Parallel line references undefined line: {reference_line}"
                    )
                if action.label:
                    created_labels.add(action.label)
                return [
                    f"{label_prefix}ParallelLine({through_point}, {reference_line})"
                ]
            issues.append(
                "PARALLEL requires metadata.through_point and a line reference."
            )
            return []

        if action.action is GeometryActionType.ANGLE_BISECTOR:
            label_prefix = f"{action.label} = " if action.label else ""
            points = action.points or []
            if len(points) >= 3:
                if any(point not in created_labels for point in points[:3]):
                    issues.append(
                        f"Angle bisector references undefined points: {points[:3]}"
                    )
                if action.label:
                    created_labels.add(action.label)
                return [
                    f"{label_prefix}AngleBisector({points[0]}, {points[1]}, {points[2]})"
                ]
            issues.append("ANGLE_BISECTOR requires three points.")
            return []

        if action.action is GeometryActionType.CREATE_FUNCTION:
            equation = action.equation or "x"
            label = action.label or "f"
            return [f"{label}(x) = {equation}"]

        issues.append(f"Unsupported action: {action.action}")
        return []

    def _validate_command_names(self, commands: list[str], issues: list[str]) -> None:
        valid_commands = self.valid_commands or self.fallback_commands
        for command in commands:
            if "=" in command:
                expression = command.split("=", maxsplit=1)[1].strip()
            else:
                expression = command.strip()
            if "(" not in expression:
                continue
            command_name = expression.split("(", maxsplit=1)[0].strip()
            if (
                command_name
                and command_name not in valid_commands
                and not command_name.endswith("(x)")
            ):
                issues.append(
                    f"Command '{command_name}' is not in the GeoGebra command catalog."
                )

    def _load_catalog(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        return {
            entry.get("command_name", "")
            for entry in payload
            if isinstance(entry, dict) and entry.get("command_name")
        }
