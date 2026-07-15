import asyncio
from typing import Any, cast

from app.integrations.llama_client import LlamaClient
from app.integrations.openrouter_client import OpenRouterClient
from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryAction, GeometryActionType, GeometryDSL
from app.services.geometry_extractor import GeometryExtractor


class DisabledClient:
    enabled = False


class FailingEnabledClient:
    enabled = True

    async def complete_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("OpenRouter extraction should not run for local parser routes")


class FakeLlamaClient:
    enabled = True
    model = "local:test-geometry-parser"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    async def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.last_kwargs = kwargs
        return self.payload


class DisabledLlamaClient(FakeLlamaClient):
    enabled = False


class GeometryParserSettings:
    local_llama_geometry_extraction_enabled = True
    local_llama_geometry_max_tokens = 900
    local_llama_geometry_timeout_seconds = 6.0


class DisabledGeometryParserSettings(GeometryParserSettings):
    local_llama_geometry_extraction_enabled = False


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


def test_geometry_visualization_survives_incorrect_general_classification() -> None:
    result = asyncio.run(
        GeometryExtractor(cast(OpenRouterClient, DisabledClient())).extract(
            VIETNAMESE_GEOMETRY_PROOF,
            ProblemType.general,
            "local:structural-fallback",
        )
    )

    action_types = {action.action for action in result.dsl.actions}
    assert GeometryActionType.CREATE_POLYGON in action_types
    assert GeometryActionType.CREATE_CIRCLE in action_types
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


def test_local_parser_route_uses_llama_geometry_extractor() -> None:
    llama = FakeLlamaClient(
        {
            "visualizable": True,
            "confidence": 0.93,
            "summary": "Triangle ABC with midpoint M",
            "dsl": {
                "version": "1.0",
                "space": "euclidean_2d",
                "actions": [
                    {"action": "CREATE_POINT", "label": "A"},
                    {"action": "CREATE_POINT", "label": "B"},
                    {"action": "CREATE_POINT", "label": "C"},
                    {
                        "action": "CREATE_POLYGON",
                        "label": "polyABC",
                        "points": ["A", "B", "C"],
                    },
                    {
                        "action": "MIDPOINT",
                        "label": "M",
                        "points": ["A", "B"],
                    },
                ],
                "render_hints": {},
            },
        }
    )

    result = asyncio.run(
        GeometryExtractor(
            cast(OpenRouterClient, FailingEnabledClient()),
            cast(LlamaClient, llama),
            GeometryParserSettings(),
        ).extract(
            "Make ABC a triangle and put M halfway between A and B.",
            ProblemType.geometry,
            "local:llama-geometry-parser",
        )
    )

    assert llama.calls == 1
    assert llama.last_kwargs["max_tokens"] == 900
    assert llama.last_kwargs["timeout_seconds"] == 6.0
    assert llama.last_kwargs["json_schema"]["required"] == ["summary", "dsl"]
    assert result.summary == "Triangle ABC with midpoint M"
    assert [action.label for action in result.dsl.actions] == [
        "A",
        "B",
        "C",
        "polyABC",
        "M",
    ]
    assert result.warnings == []


def test_invalid_local_llama_dsl_falls_back_to_deterministic_extraction() -> None:
    llama = FakeLlamaClient(
        {
            "visualizable": True,
            "confidence": 0.98,
            "dsl": {
                "version": "1.0",
                "space": "euclidean_2d",
                "actions": [
                    {
                        "action": "CREATE_LINE",
                        "label": "badLine",
                        "points": ["A", "B"],
                    }
                ],
                "render_hints": {},
            },
        }
    )

    result = asyncio.run(
        GeometryExtractor(
            cast(OpenRouterClient, FailingEnabledClient()),
            cast(LlamaClient, llama),
            GeometryParserSettings(),
        ).extract(
            "Construct triangle ABC",
            ProblemType.geometry,
            "local:llama-geometry-parser",
        )
    )

    assert llama.calls == 1
    assert GeometryActionType.CREATE_POLYGON in {
        action.action for action in result.dsl.actions
    }


def test_semantically_mismatched_local_dsl_uses_deterministic_fallback() -> None:
    llama = FakeLlamaClient(
        {
            "visualizable": True,
            "confidence": 0.95,
            "summary": "Circle",
            "dsl": {
                "version": "1.0",
                "space": "euclidean_2d",
                "actions": [{"action": "CREATE_POINT", "label": "O"}],
                "render_hints": {},
            },
        }
    )

    result = asyncio.run(
        GeometryExtractor(
            cast(OpenRouterClient, FailingEnabledClient()),
            cast(LlamaClient, llama),
            GeometryParserSettings(),
        ).extract(
            "Draw a circle with center O at (0, 0) and radius 5",
            ProblemType.geometry,
            "local:llama-geometry-parser",
        )
    )

    assert llama.calls == 1
    assert GeometryActionType.CREATE_CIRCLE in {
        action.action for action in result.dsl.actions
    }


def test_local_dsl_sanitizer_removes_invented_coordinates_and_adds_triangle() -> None:
    extractor = GeometryExtractor(cast(OpenRouterClient, DisabledClient()))
    dsl = GeometryDSL(
        actions=[
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="A",
                coordinates=(0, 0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="B",
                coordinates=(2, 0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="C",
                coordinates=(1, 1.7),
            ),
            GeometryAction(
                action=GeometryActionType.MIDPOINT,
                label="M",
                points=["A", "B"],
            ),
        ]
    )

    sanitized = extractor._sanitize_local_dsl(
        "Make ABC a triangle and put M halfway between A and B.", dsl
    )

    point_actions = [
        action
        for action in sanitized.actions
        if action.action is GeometryActionType.CREATE_POINT
    ]
    assert all(action.coordinates is None for action in point_actions)
    polygon = next(
        action
        for action in sanitized.actions
        if action.action is GeometryActionType.CREATE_POLYGON
    )
    assert polygon.points == ["A", "B", "C"]


def test_local_dsl_sanitizer_repairs_explicit_circle_center_and_label() -> None:
    extractor = GeometryExtractor(cast(OpenRouterClient, DisabledClient()))
    dsl = GeometryDSL(
        actions=[
            GeometryAction(
                action=GeometryActionType.CREATE_CIRCLE,
                label="O",
                center="0,0",
                radius=5,
            )
        ]
    )

    sanitized = extractor._sanitize_local_dsl(
        "Draw a circle with center O at (0, 0) and radius 5.", dsl
    )

    assert sanitized.actions[0].action is GeometryActionType.CREATE_POINT
    assert sanitized.actions[0].label == "O"
    assert sanitized.actions[0].coordinates == (0.0, 0.0)
    circle = sanitized.actions[1]
    assert circle.action is GeometryActionType.CREATE_CIRCLE
    assert circle.label == "c"
    assert circle.center == "O"


def test_disabled_local_geometry_parser_uses_deterministic_fallback() -> None:
    llama = DisabledLlamaClient({})

    result = asyncio.run(
        GeometryExtractor(
            cast(OpenRouterClient, FailingEnabledClient()),
            cast(LlamaClient, llama),
            DisabledGeometryParserSettings(),
        ).extract(
            "Construct triangle ABC",
            ProblemType.geometry,
            "local:llama-geometry-parser",
        )
    )

    assert llama.calls == 0
    assert GeometryActionType.CREATE_POLYGON in {
        action.action for action in result.dsl.actions
    }
