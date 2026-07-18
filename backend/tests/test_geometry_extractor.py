import asyncio

from app.schemas.common import ProblemType
from app.schemas.geometry_dsl import GeometryActionType, VisualizationEnvironment
from app.services.geometry_extractor import GeometryExtractor


def _valid_triangle_dsl() -> dict:
    return {
        "version": "1.1",
        "space": "euclidean_2d",
        "environment": "geometry_2d",
        "actions": [
            {"action": "CREATE_POINT", "label": "A"},
            {"action": "CREATE_POINT", "label": "B"},
            {"action": "CREATE_POINT", "label": "C"},
            {
                "action": "CREATE_POLYGON",
                "label": "polyABC",
                "points": ["A", "B", "C"],
            },
        ],
        "render_hints": {},
    }


def test_geometry_payload_normalizes_model_envelope_and_long_summary() -> None:
    extractor = GeometryExtractor()
    payload = {
        "geometry_visualization": {
            "summary": "A faithful triangle construction. " * 10,
            "dsl": {**_valid_triangle_dsl(), "model_note": "ignored metadata"},
            "confidence": 0.95,
        },
        "reasoning": "ignored envelope metadata",
    }

    dsl, summary = extractor._parse_geometry_payload(
        payload,
        environment=VisualizationEnvironment.geometry_2d,
        source="test",
        allowed_command_names=set(),
    )

    assert len(summary) == 200
    assert summary.endswith("...")
    assert [action.label for action in dsl.actions] == ["A", "B", "C", "polyABC"]


def test_triangle_heuristic_does_not_parse_the_word_with_as_w_i_t() -> None:
    extractor = GeometryExtractor()
    result = extractor._extract_heuristically(
        "A triangle with an incircle is given.",
        ProblemType.geometry,
    )

    assert result.dsl.actions == []
    assert result.dsl.version == "1.1"
    assert result.dsl.environment is VisualizationEnvironment.geometry_2d


def test_triangle_heuristic_recognizes_latex_wrapped_uppercase_labels() -> None:
    extractor = GeometryExtractor()
    result = extractor._extract_heuristically(
        r"Let \(ABC\) be a triangle with \(AB < AC < BC\). "
        r"The incircle of triangle \(ABC\) has center \(I\).",
        ProblemType.geometry,
    )

    assert [action.label for action in result.dsl.actions] == [
        "A",
        "B",
        "C",
        "poly1",
    ]
    assert result.dsl.actions[-1].action is GeometryActionType.CREATE_POLYGON


def test_geometry_heuristic_does_not_parse_incenter_and_as_center_a() -> None:
    extractor = GeometryExtractor()
    result = extractor._extract_heuristically(
        r"Let \(ABC\) be a triangle. Let the incenter and incircle of triangle "
        r"\(ABC\) be \(I\) and \(\omega\), respectively.",
        ProblemType.geometry,
    )

    assert [action.label for action in result.dsl.actions] == [
        "A",
        "B",
        "C",
        "poly1",
    ]


def test_valid_remote_geometry_is_kept_when_intent_alignment_is_incomplete() -> None:
    class CompletionClient:
        async def complete_json(self, **_: object) -> dict:
            return {
                "summary": "Triangle edges from the remote construction",
                "dsl": {
                    "version": "1.1",
                    "space": "euclidean_2d",
                    "environment": "geometry_2d",
                    "actions": [
                        {"action": "CREATE_POINT", "label": "A"},
                        {"action": "CREATE_POINT", "label": "B"},
                        {"action": "CREATE_POINT", "label": "C"},
                        {
                            "action": "CREATE_LINE",
                            "label": "AB",
                            "points": ["A", "B"],
                        },
                        {
                            "action": "CREATE_LINE",
                            "label": "BC",
                            "points": ["B", "C"],
                        },
                        {
                            "action": "CREATE_LINE",
                            "label": "CA",
                            "points": ["C", "A"],
                        },
                    ],
                    "render_hints": {},
                },
            }

    extractor = GeometryExtractor()
    result = asyncio.run(
        extractor._extract_with_llm(
            "Construct triangle ABC and a line parallel to AB.",
            "test-model",
            VisualizationEnvironment.geometry_2d,
            (),
            completion_client=CompletionClient(),
            provider_name="test",
        )
    )

    assert len(result.dsl.actions) == 6
    assert result.warnings
    assert "may omit some stated relationships" in result.warnings[0]
