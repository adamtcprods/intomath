import asyncio
from typing import Any, cast

from app.integrations.openrouter_client import OpenRouterClient
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryActionType
from app.services.geometry_extractor import GeometryExtractor


class DisabledClient:
    enabled = False


class FailingEnabledClient:
    enabled = True

    async def complete_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM extraction should not run for local parser routes")


class ReturningEnabledClient:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "summary": "LLM visualization",
            "dsl": {
                "version": "1.0",
                "space": "euclidean_2d",
                "actions": [{"action": "CREATE_POINT", "label": "P"}],
                "render_hints": {},
            },
        }


VIETNAMESE_GEOMETRY_PROOF = r"""
Cho tam giác (ABC) ((AB < AC)) nội tiếp đường tròn ((O;R)) có đường kính (BC).
Trên cung nhỏ (AC) lấy điểm (D). Đường thẳng (BD) cắt (AC) tại (E).
Từ (E) kẻ (EF \perp BC) tại (F).
""".strip()


def test_heuristic_extractor_handles_vietnamese_triangle_and_circle_generically() -> (
    None
):
    result = GeometryExtractor(
        cast(OpenRouterClient, DisabledClient())
    )._extract_heuristically(
        VIETNAMESE_GEOMETRY_PROOF,
        ProblemType.geometry,
    )

    action_types = [action.action for action in result.dsl.actions]
    points = {
        action.label: action
        for action in result.dsl.actions
        if action.action is GeometryActionType.CREATE_POINT
    }
    circle = next(
        action
        for action in result.dsl.actions
        if action.action is GeometryActionType.CREATE_CIRCLE
    )

    assert GeometryActionType.CREATE_POLYGON in action_types
    assert GeometryActionType.CREATE_CIRCLE in action_types
    assert set(points) >= {"A", "B", "C", "O"}
    assert points["B"].coordinates is None
    assert circle.through == ["O", "B"]
    assert result.summary == "Triangle construction"
    assert result.warnings == []


def test_diameter_arc_pattern_uses_enabled_llm_parser_for_remote_routes() -> None:
    client = ReturningEnabledClient()

    result = asyncio.run(
        GeometryExtractor(cast(OpenRouterClient, client)).extract(
            VIETNAMESE_GEOMETRY_PROOF,
            ProblemType.geometry,
            "remote-parser",
        )
    )

    assert client.calls == 1
    assert result.summary == "LLM visualization"
    assert [action.label for action in result.dsl.actions] == ["P"]
    assert result.warnings == []


def test_local_parser_route_uses_heuristic_even_with_enabled_client() -> None:
    result = asyncio.run(
        GeometryExtractor(cast(OpenRouterClient, FailingEnabledClient())).extract(
            "Construct triangle ABC",
            ProblemType.geometry,
            "local:heuristic-parser",
        )
    )

    action_types = [action.action for action in result.dsl.actions]
    assert GeometryActionType.CREATE_POLYGON in action_types
