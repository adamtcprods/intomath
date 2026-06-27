from __future__ import annotations

import re
from dataclasses import dataclass

from app.integrations.openrouter_client import OpenRouterClient
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryAction, GeometryActionType, GeometryDSL


@dataclass
class GeometryExtractionResult:
    dsl: GeometryDSL
    summary: str | None
    warnings: list[str]


class GeometryExtractor:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    async def extract(
        self, text: str, problem_type: ProblemType, parser_model: str
    ) -> GeometryExtractionResult:
        if problem_type not in {
            ProblemType.geometry,
            ProblemType.coordinate_geometry,
            ProblemType.functions,
            ProblemType.algebra,
        }:
            return GeometryExtractionResult(
                dsl=GeometryDSL(), summary=None, warnings=[]
            )

        if (
            self.client.enabled
            and not parser_model.startswith("local:")
            and problem_type in {ProblemType.geometry, ProblemType.coordinate_geometry}
        ):
            try:
                return await self._extract_with_llm(text, parser_model)
            except Exception:
                pass

        return self._extract_heuristically(text, problem_type)

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
        if not has_circle and any(
            phrase in lowered for phrase in ["đường tròn", "duong tron", "circle"]
        ):
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

        if not actions and problem_type in {
            ProblemType.geometry,
            ProblemType.coordinate_geometry,
        }:
            warnings.append(
                "No deterministic geometry pattern was recognized, so no visualization was generated."
            )

        if not actions and problem_type is ProblemType.functions:
            warnings.append("No graphable expression was detected in the prompt.")

        return GeometryExtractionResult(
            dsl=GeometryDSL(actions=actions),
            summary=summary,
            warnings=warnings,
        )

    def _clean_equation(self, expression: str) -> str:
        return expression.replace("²", "^2").replace("−", "-").strip(" .,:;?\n\t")
