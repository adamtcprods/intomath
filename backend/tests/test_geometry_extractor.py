import asyncio
from types import SimpleNamespace

from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.geogebra_translator import GeoGebraTranslator
from app.services.geometry_extractor import (
    GeometryExtractor,
    _ordered_geometry_fallback_models,
    geometry_response_schema,
)


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


def test_no_model_backed_parser_returns_an_empty_plan() -> None:
    extractor = GeometryExtractor(
        settings=SimpleNamespace(local_llama_geometry_extraction_enabled=False),
        nvidia_client=SimpleNamespace(enabled=False),
    )

    result = asyncio.run(
        extractor.extract(
            "Graph y = x^2 - 4x + 3.",
            "local:llama-geometry-parser",
            environment=VisualizationEnvironment.graphing,
        )
    )

    assert result.dsl.actions == []
    assert result.dsl.environment is VisualizationEnvironment.graphing
    assert "model" in result.warnings[0].lower()


def test_local_prompt_grounding_is_validation_only() -> None:
    extractor = GeometryExtractor()
    dsl, _ = extractor._parse_geometry_payload(
        {
            "summary": "Point",
            "dsl": {
                "version": "1.1",
                "space": "euclidean_2d",
                "environment": "geometry_2d",
                "actions": [
                    {"action": "CREATE_POINT", "label": "A", "coordinates": [4, 5]}
                ],
                "render_hints": {},
            },
        },
        environment=VisualizationEnvironment.geometry_2d,
        source="test",
        allowed_command_names=set(),
    )
    before = dsl.model_dump()

    issues = extractor._validate_local_prompt_grounding("Create point A.", dsl)

    assert issues == [
        "Point 'A' has coordinates that are not explicitly stated in the prompt."
    ]
    assert dsl.model_dump() == before


def test_model_semantic_retrieval_and_planning_use_generic_3d_command() -> None:
    class CubePlanningClient:
        enabled = True
        available = True
        model = "test-geometry-model"

        def __init__(self) -> None:
            self.request: dict[str, object] = {}

        async def generate_json(self, **kwargs: object) -> dict:
            self.request = kwargs
            return {
                "summary": "Interactive 3D solid",
                "dsl": {
                    "version": "1.1",
                    "space": "euclidean_3d",
                    "environment": "graphics_3d",
                    "actions": [
                        {
                            "action": "EXECUTE_COMMAND",
                            "output": "solid1",
                            "command": "Cube",
                            "arguments": [
                                {"kind": "point", "x": 0, "y": 0, "z": 0},
                                {"kind": "point", "x": 2, "y": 0, "z": 0},
                            ],
                        }
                    ],
                    "render_hints": {},
                },
            }

    client = CubePlanningClient()
    settings = SimpleNamespace(
        local_llama_geometry_extraction_enabled=True,
        local_llama_geometry_max_tokens=1_200,
        local_llama_geometry_timeout_seconds=8.0,
    )
    extractor = GeometryExtractor(
        llama_client=client,
        settings=settings,
        nvidia_client=SimpleNamespace(enabled=False),
    )
    result = asyncio.run(
        extractor.extract(
            "Visualize a regular hexahedron in three dimensions.",
            "local:llama-geometry-parser",
            environment=VisualizationEnvironment.graphics_3d,
            semantic_query_terms=("Cube", "solid", "three dimensional"),
        )
    )
    translation = GeoGebraTranslator().translate(
        result.dsl, allowed_command_names=result.allowed_commands
    )

    assert "Selected environment: graphics_3d" in client.request["prompt"]
    action_variants = client.request["json_schema"]["properties"]["dsl"][
        "properties"
    ]["actions"]["items"]["oneOf"]
    assert {
        variant["properties"]["action"]["const"] for variant in action_variants
    } == {"EXECUTE_COMMAND"}
    assert min(
        variant["properties"]["arguments"]["minItems"]
        for variant in action_variants
    ) == 2
    assert "Cube" in result.allowed_commands
    assert result.dsl.space == "euclidean_3d"
    assert result.dsl.actions[0].command == "Cube"
    assert translation.validation_passed is True
    assert translation.commands == ["solid1 = Cube((0, 0, 0), (2, 0, 0))"]


def test_response_schema_exposes_define_object_without_generic_commands() -> None:
    schema = geometry_response_schema([])
    action_variants = schema["properties"]["dsl"]["properties"]["actions"][
        "items"
    ]["oneOf"]

    definition_variants = [
        variant
        for variant in action_variants
        if variant["properties"]["action"].get("const") == "DEFINE_OBJECT"
    ]
    generic_variants = [
        variant
        for variant in action_variants
        if variant["properties"]["action"].get("const") == "EXECUTE_COMMAND"
    ]

    assert len(definition_variants) == 10
    assert generic_variants == []


def test_strict_payload_accepts_nested_matrix_arguments() -> None:
    extractor = GeometryExtractor()
    dsl, _ = extractor._parse_geometry_payload(
        {
            "summary": "Matrix determinant",
            "dsl": {
                "version": "1.1",
                "space": "euclidean_2d",
                "environment": "graphing",
                "actions": [
                    {
                        "action": "EXECUTE_COMMAND",
                        "output": "det",
                        "command": "Determinant",
                        "arguments": [
                            {
                                "kind": "list",
                                "items": [
                                    {
                                        "kind": "list",
                                        "items": [
                                            {"kind": "number", "value": 1},
                                            {"kind": "number", "value": 2},
                                        ],
                                    },
                                    {
                                        "kind": "list",
                                        "items": [
                                            {"kind": "number", "value": 3},
                                            {"kind": "number", "value": 4},
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
                "render_hints": {},
            },
        },
        environment=VisualizationEnvironment.graphing,
        source="test",
        allowed_command_names={"Determinant"},
    )

    assert dsl.actions[0].command == "Determinant"


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
    assert "PARALLEL" in result.warnings[0]
    assert "CREATE_POLYGON" not in result.warnings[0]
    assert "DSL has no" not in result.warnings[0]


def test_geometry_fallback_retries_routed_model_before_alternate() -> None:
    assert _ordered_geometry_fallback_models("openai/gpt-oss-20b") == (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    )
    assert _ordered_geometry_fallback_models("openai/gpt-oss-120b") == (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    )
