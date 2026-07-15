from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.integrations.llama_client import LlamaClient
from app.integrations.openrouter_client import OpenRouterClient
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryAction, GeometryActionType, GeometryDSL


LOCAL_GEOMETRY_EXTRACTION_PROMPT = """
Convert the math problem into a minimal 2D GeoGebra construction plan.
Return exactly one JSON object and nothing else. Do not solve or prove the problem.
Do not invent coordinates, lengths, labels, or relationships that the problem does not give.
Create every referenced object before it is used and preserve labels exactly.

Allowed actions:
CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT,
MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION.

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

LOCAL_GEOMETRY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 200},
        "dsl": {
            "type": "object",
            "properties": {
                "version": {"type": "string", "const": "1.0"},
                "space": {"type": "string", "const": "euclidean_2d"},
                "actions": {
                    "type": "array",
                    "maxItems": 40,
                    "items": {
                        "oneOf": [
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
                                {"through": {**_POINT_LABELS, "minItems": 2, "maxItems": 2}},
                            ),
                            _action_schema(
                                "CREATE_POLYGON",
                                ["points"],
                                {"points": {**_POINT_LABELS, "minItems": 3}},
                            ),
                            _action_schema(
                                "INTERSECT",
                                ["metadata"],
                                {"metadata": _OBJECT_PAIR_METADATA},
                            ),
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
                    },
                },
                "render_hints": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            "required": ["version", "space", "actions", "render_hints"],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "dsl"],
    "additionalProperties": False,
}

_MAX_LOCAL_ACTIONS = 40
_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_SAFE_FUNCTION_EXPRESSION = re.compile(r"^[0-9A-Za-z_+\-*/^()., ]{1,200}$")


@dataclass
class GeometryExtractionResult:
    dsl: GeometryDSL
    summary: str | None
    warnings: list[str]


class GeometryExtractor:
    def __init__(
        self,
        client: OpenRouterClient,
        llama_client: LlamaClient | None = None,
        settings: Any | None = None,
    ) -> None:
        self.client = client
        self.llama_client = llama_client or LlamaClient()
        self.settings = settings or get_settings()

    async def extract(
        self, text: str, problem_type: ProblemType, parser_model: str
    ) -> GeometryExtractionResult:
        # Visualization should not disappear just because the model router
        # mislabeled an otherwise recognizable geometry prompt.
        extraction_type = problem_type
        if problem_type not in {ProblemType.geometry, ProblemType.algebra}:
            if self._looks_like_geometry_problem(text):
                extraction_type = ProblemType.geometry
            else:
                return GeometryExtractionResult(
                    dsl=GeometryDSL(), summary=None, warnings=[]
                )

        problem_type = extraction_type

        if parser_model.startswith("local:"):
            if problem_type is ProblemType.algebra:
                deterministic = self._extract_heuristically(text, problem_type)
                if deterministic.dsl.actions:
                    return deterministic
            if self._local_geometry_parser_enabled() and len(text.strip()) <= 4_000:
                try:
                    return await self._extract_with_local_llama(text)
                except Exception:
                    pass
            return self._extract_heuristically(text, problem_type)

        if self.client.enabled and problem_type is ProblemType.geometry:
            try:
                return await self._extract_with_llm(text, parser_model)
            except Exception:
                pass

        return self._extract_heuristically(text, problem_type)

    def _local_geometry_parser_enabled(self) -> bool:
        return bool(
            getattr(self.settings, "local_llama_geometry_extraction_enabled", False)
            and getattr(self.llama_client, "enabled", False)
        )

    async def _extract_with_local_llama(self, text: str) -> GeometryExtractionResult:
        payload = await self.llama_client.generate_json(
            prompt=f"{LOCAL_GEOMETRY_EXTRACTION_PROMPT}\n\nProblem:\n{text.strip()}",
            max_tokens=int(
                getattr(self.settings, "local_llama_geometry_max_tokens", 1_200)
            ),
            timeout_seconds=float(
                getattr(self.settings, "local_llama_geometry_timeout_seconds", 8.0)
            ),
            json_schema=LOCAL_GEOMETRY_RESPONSE_SCHEMA,
        )

        dsl = GeometryDSL.model_validate(payload.get("dsl", {}))
        dsl = self._sanitize_local_dsl(text, dsl)


        issues = self._validate_local_dsl(dsl)
        issues.extend(self._validate_intent_alignment(text, dsl))
        if issues:
            raise ValueError("Invalid local geometry DSL: " + "; ".join(issues))

        return GeometryExtractionResult(
            dsl=dsl,
            summary=self._coerce_optional_string(payload.get("summary")),
            warnings=[],
        )

    async def _extract_with_llm(
        self, text: str, parser_model: str
    ) -> GeometryExtractionResult:
        payload = await self.client.complete_json(
            model=parser_model,
            system_prompt=(
                "Extract only visualization intents for a math problem. Output JSON with keys: "
                "summary, dsl. The dsl must have version, space, actions, render_hints. "
                "Supported actions: CREATE_POINT, CREATE_LINE, CREATE_CIRCLE, CREATE_POLYGON, INTERSECT, "
                "MIDPOINT, PERPENDICULAR, PARALLEL, ANGLE_BISECTOR, CREATE_FUNCTION."
            ),
            user_prompt=text,
            temperature=0.1,
        )
        dsl = GeometryDSL.model_validate(payload.get("dsl", {}))
        return GeometryExtractionResult(
            dsl=dsl,
            summary=str(payload.get("summary", "")).strip() or None,
            warnings=[],
        )

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
        ]



    def _coerce_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text[:200] or None

    def _extract_heuristically(
        self, text: str, problem_type: ProblemType
    ) -> GeometryExtractionResult:
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
            dsl=GeometryDSL(actions=actions),
            summary=summary,
            warnings=warnings,
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
