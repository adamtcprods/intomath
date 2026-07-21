import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.geometry_dsl import (
    GeoGebraArgument,
    GeometryDSL,
    ValidationSeverity,
)
from app.services.geogebra_command_registry import GeoGebraObjectType
from app.services.geogebra_translator import GeoGebraTranslator


def _dsl(actions: list[dict], **extra: object) -> GeometryDSL:
    return GeometryDSL.model_validate(
        {"version": "1.1", "actions": actions, **extra}
    )


def test_geometry_dsl_defaults_to_version_1_1() -> None:
    assert GeometryDSL().version == "1.1"


def test_geometry_dsl_rejects_version_1_0() -> None:
    with pytest.raises(ValidationError):
        GeometryDSL.model_validate({"version": "1.0", "actions": []})


def test_define_object_function_is_validated_and_translated() -> None:
    dsl = _dsl(
        [
            {
                "action": "DEFINE_OBJECT",
                "output": "f",
                "object_type": "function",
                "value": {"kind": "equation", "value": "f(x)=x^2"},
            }
        ],
        environment="graphing",
    )

    result = GeoGebraTranslator().translate(dsl)

    assert result.validation_passed is True
    assert result.commands == ["f(x) = x^2"]


def test_define_object_values_are_deterministic_and_reference_aware() -> None:
    dsl = _dsl(
        [
            {
                "action": "DEFINE_OBJECT",
                "output": "values",
                "object_type": "list",
                "value": {
                    "kind": "list",
                    "items": [
                        {"kind": "reference", "value": "n"},
                        {"kind": "number", "value": 2},
                    ],
                },
            },
            {
                "action": "DEFINE_OBJECT",
                "output": "n",
                "object_type": "number",
                "value": {"kind": "number", "value": 3},
            },
            {
                "action": "DEFINE_OBJECT",
                "output": "unitCircle",
                "object_type": "equation",
                "value": {
                    "kind": "equation",
                    "value": "x^2+y^2=1",
                },
            },
        ]
    )

    result = GeoGebraTranslator().translate(dsl)

    assert result.validation_passed is True
    assert result.commands == [
        "n = 3",
        "values = {n, 2}",
        "unitCircle: x^2+y^2 = 1",
    ]


@pytest.mark.parametrize(
    "action",
    [
        {
            "action": "DEFINE_OBJECT",
            "output": "f",
            "object_type": "function",
            "value": {"kind": "equation", "value": "g(x) = x^2"},
        },
        {
            "action": "DEFINE_OBJECT",
            "output": "f",
            "object_type": "function",
            "value": {"kind": "equation", "value": "f(x) = x; Delete(A)"},
        },
        {
            "action": "DEFINE_OBJECT",
            "output": "n",
            "object_type": "number",
            "value": {"kind": "expression", "value": "2"},
        },
        {
            "action": "DEFINE_OBJECT",
            "output": "n",
            "label": "n",
            "object_type": "number",
            "value": {"kind": "number", "value": 2},
        },
        {
            "action": "DEFINE_OBJECT",
            "output": "n",
            "object_type": "number",
            "value": {"kind": "number", "value": 2},
            "points": ["A"],
        },
    ],
)
def test_define_object_rejects_mismatched_or_unsafe_shapes(action: dict) -> None:
    with pytest.raises(ValidationError):
        _dsl([action])


def test_generic_tangent_is_dependency_sorted_and_translated() -> None:
    dsl = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "output": "t",
                "command": "Tangent",
                "arguments": [
                    {"kind": "reference", "value": "B"},
                    {"kind": "reference", "value": "c"},
                ],
            },
            {"action": "CREATE_CIRCLE", "label": "c", "center": "A", "radius": 4},
            {"action": "CREATE_POINT", "label": "A", "coordinates": [0, 0]},
            {"action": "CREATE_POINT", "label": "B", "coordinates": [4, 0]},
        ]
    )

    result = GeoGebraTranslator().translate(
        dsl, allowed_command_names={"Tangent"}
    )

    assert result.validation_passed is True
    assert result.commands == [
        "A = (0.0, 0.0)",
        "c = Circle(A, 4.0)",
        "B = (4.0, 0.0)",
        "t = Tangent(B, c)",
    ]


def test_generic_rotate_formats_degree_angle() -> None:
    dsl = _dsl(
        [
            {"action": "CREATE_POINT", "label": "A", "coordinates": [1, 0]},
            {"action": "CREATE_POINT", "label": "O", "coordinates": [0, 0]},
            {
                "action": "EXECUTE_COMMAND",
                "output": "rotatedA",
                "command": "Rotate",
                "arguments": [
                    {"kind": "reference", "value": "A"},
                    {"kind": "angle", "value": 90, "unit": "degree"},
                    {"kind": "reference", "value": "O"},
                ],
            },
        ]
    )

    translator = GeoGebraTranslator()
    result = translator.translate(dsl, allowed_command_names={"Rotate"})
    validation = translator.validator.validate(
        dsl, allowed_command_names={"Rotate"}
    )

    assert result.commands[-1] == "rotatedA = Rotate(A, 90°, O)"
    assert validation.object_types["rotatedA"] is GeoGebraObjectType.POINT


def test_every_argument_kind_has_deterministic_serialization() -> None:
    translator = GeoGebraTranslator()
    adapter = TypeAdapter(GeoGebraArgument)
    cases = [
        ({"kind": "reference", "value": "A"}, "A"),
        ({"kind": "number", "value": 2.5}, "2.5"),
        ({"kind": "angle", "value": 1.5, "unit": "radian"}, "(1.5 rad)"),
        ({"kind": "point", "x": 1, "y": 2}, "(1, 2)"),
        ({"kind": "point", "x": 1, "y": 2, "z": 3}, "(1, 2, 3)"),
        ({"kind": "vector", "x": 1, "y": -2}, "Vector((1, -2))"),
        ({"kind": "text", "value": 'say "hi"'}, '"say \\"hi\\""'),
        ({"kind": "boolean", "value": True}, "true"),
        ({"kind": "expression", "value": "sin(x) + 2"}, "sin(x) + 2"),
        ({"kind": "equation", "value": "x^2 = 4"}, "(x^2 = 4)"),
        (
            {
                "kind": "list",
                "items": [
                    {"kind": "number", "value": 1},
                    {"kind": "point", "x": 2, "y": 3},
                ],
            },
            "{1, (2, 3)}",
        ),
        (
            {
                "kind": "interval",
                "lower": 0,
                "upper": 1,
                "lower_inclusive": False,
                "upper_inclusive": True,
            },
            "(0 < x ≤ 1)",
        ),
    ]

    for payload, expected in cases:
        assert translator.serialize_argument(adapter.validate_python(payload)) == expected


def test_undefined_reference_duplicate_label_and_cycle_are_rejected() -> None:
    undefined = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "output": "m",
                "command": "Midpoint",
                "arguments": [
                    {"kind": "reference", "value": "A"},
                    {"kind": "reference", "value": "missing"},
                ],
            },
            {"action": "CREATE_POINT", "label": "A"},
        ]
    )
    duplicate = _dsl(
        [
            {"action": "CREATE_POINT", "label": "A"},
            {"action": "CREATE_POINT", "label": "A"},
        ]
    )
    cycle = _dsl(
        [
            {"action": "CREATE_POINT", "label": "C"},
            {"action": "MIDPOINT", "label": "A", "points": ["B", "C"]},
            {"action": "MIDPOINT", "label": "B", "points": ["A", "C"]},
        ]
    )

    results = [GeoGebraTranslator().translate(item) for item in (undefined, duplicate, cycle)]
    codes = [{issue.code for issue in result.issues} for result in results]
    assert "undefined_reference" in codes[0]
    assert "duplicate_label" in codes[1]
    assert "dependency_cycle" in codes[2]
    assert all(result.commands == [] for result in results)


def test_validation_logs_group_error_counts_by_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dsl = _dsl(
        [
            {
                "action": "CREATE_LINE",
                "label": "l",
                "points": ["A", "B"],
            }
        ]
    )

    with caplog.at_level("WARNING"):
        result = GeoGebraTranslator().translate(dsl)

    assert result.validation_passed is False
    assert any(
        'error_codes={"undefined_reference": 2}' in record.getMessage()
        for record in caplog.records
    )


def test_answer_only_point_coordinates_are_rejected_with_problem_context() -> None:
    dsl = _dsl(
        [
            {"action": "CREATE_POINT", "label": "P1", "coordinates": [1, 1]},
            {"action": "CREATE_POINT", "label": "P2", "coordinates": [2, 2]},
            {"action": "CREATE_POINT", "label": "P3", "coordinates": [3, 3]},
            {"action": "CREATE_POINT", "label": "P4", "coordinates": [1, 2]},
            {"action": "CREATE_POINT", "label": "P5", "coordinates": [2, 1]},
        ],
        environment="graphing",
    )

    result = GeoGebraTranslator().translate(
        dsl,
        normalized_problem_text=(
            "Determine all positive integers a and b satisfying the gcd condition."
        ),
        solved_answer_text="The pairs are (1,1), (2,2), (3,3), (1,2), and (2,1).",
    )

    assert result.validation_passed is False
    assert result.commands == []
    assert {issue.code for issue in result.issues} == {
        "answer_only_point_coordinates"
    }


def test_stated_point_coordinates_are_not_rejected_as_answer_only() -> None:
    dsl = _dsl(
        [
            {"action": "CREATE_POINT", "label": "A", "coordinates": [1, 1]},
            {"action": "CREATE_POINT", "label": "B", "coordinates": [2, 2]},
        ]
    )

    result = GeoGebraTranslator().translate(
        dsl,
        normalized_problem_text="Plot point A at (1,1) and point B at (2,2).",
        solved_answer_text="The plotted points are (1,1) and (2,2).",
    )

    assert result.validation_passed is True
    assert result.commands


def test_argument_count_type_environment_and_retrieval_are_enforced() -> None:
    base = [{"action": "CREATE_POINT", "label": "A"}]
    wrong_count = _dsl(
        base
        + [
            {
                "action": "EXECUTE_COMMAND",
                "output": "t",
                "command": "Tangent",
                "arguments": [{"kind": "reference", "value": "A"}],
            }
        ]
    )
    wrong_type = _dsl(
        base
        + [
            {
                "action": "EXECUTE_COMMAND",
                "output": "t",
                "command": "Tangent",
                "arguments": [
                    {"kind": "number", "value": 2},
                    {"kind": "number", "value": 3},
                ],
            }
        ]
    )
    cas_only = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "output": "s",
                "command": "CSolve",
                "arguments": [{"kind": "equation", "value": "x^2 = 1"}],
            }
        ]
    )
    not_retrieved = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "output": "d",
                "command": "Derivative",
                "arguments": [{"kind": "expression", "value": "x^2"}],
            }
        ],
        environment="graphing",
    )

    translator = GeoGebraTranslator()
    assert {issue.code for issue in translator.translate(wrong_count).issues} >= {
        "argument_count_mismatch"
    }
    assert {issue.code for issue in translator.translate(wrong_type).issues} >= {
        "argument_type_mismatch"
    }
    assert {issue.code for issue in translator.translate(cas_only).issues} >= {
        "incompatible_environment"
    }
    assert {issue.code for issue in translator.translate(
        not_retrieved, allowed_command_names=set()
    ).issues} >= {"command_not_retrieved"}


def test_retrieved_experimental_command_translates_with_runtime_warning() -> None:
    dsl = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "output": "d",
                "command": "Derivative",
                "arguments": [{"kind": "expression", "value": "x^2"}],
            }
        ],
        environment="graphing",
    )

    result = GeoGebraTranslator().translate(
        dsl, allowed_command_names={"Derivative"}
    )

    assert result.validation_passed is True
    assert result.commands == ["d = Derivative(x^2)"]
    assert any(
        issue.code == "experimental_command"
        and issue.severity is ValidationSeverity.warning
        for issue in result.issues
    )


def test_retrieval_allowlist_cannot_override_permanent_denylist() -> None:
    dsl = _dsl(
        [
            {
                "action": "EXECUTE_COMMAND",
                "command": "Execute",
                "arguments": [
                    {"kind": "text", "value": "Line(A, B)"}
                ],
            }
        ]
    )

    result = GeoGebraTranslator().translate(
        dsl, allowed_command_names={"Execute"}
    )

    assert result.validation_passed is False
    assert result.commands == []
    assert {issue.code for issue in result.issues} == {"unsafe_command"}


def test_nested_lists_represent_matrix_arguments() -> None:
    dsl = _dsl(
        [
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
        environment="graphing",
    )

    result = GeoGebraTranslator().translate(
        dsl, allowed_command_names={"Determinant"}
    )

    assert result.validation_passed is True
    assert result.commands == ["det = Determinant({{1, 2}, {3, 4}})"]


def test_no_output_is_allowed_but_warned_and_remains_unreferenceable() -> None:
    dsl = _dsl(
        [
            {"action": "CREATE_POINT", "label": "A"},
            {"action": "CREATE_POINT", "label": "B"},
            {
                "action": "EXECUTE_COMMAND",
                "command": "Line",
                "arguments": [
                    {"kind": "reference", "value": "A"},
                    {"kind": "reference", "value": "B"},
                ],
            },
        ]
    )

    result = GeoGebraTranslator().translate(dsl)

    assert result.validation_passed is True
    assert result.commands[-1] == "Line(A, B)"
    assert result.issues[-1].code == "untracked_output"
    assert result.issues[-1].severity is ValidationSeverity.warning


@pytest.mark.parametrize(
    "argument",
    [
        {"kind": "text", "value": "safe\nDelete(A)"},
        {"kind": "text", "value": "javascript:alert(1)"},
        {"kind": "expression", "value": "Execute(A)"},
        {"kind": "expression", "value": "x; Delete(A)"},
        {"kind": "number", "value": float("nan")},
    ],
)
def test_argument_injection_and_non_finite_numbers_fail_schema(argument: dict) -> None:
    with pytest.raises(ValidationError):
        _dsl(
            [
                {
                    "action": "EXECUTE_COMMAND",
                    "output": "x",
                    "command": "Text",
                    "arguments": [argument],
                }
            ]
        )


def test_excessive_command_count_and_list_depth_fail_schema() -> None:
    with pytest.raises(ValidationError):
        _dsl([{"action": "CREATE_POINT", "label": f"P{i}"} for i in range(41)])

    nested: dict = {"kind": "number", "value": 1}
    for _ in range(4):
        nested = {"kind": "list", "items": [nested]}
    with pytest.raises(ValidationError):
        _dsl(
            [
                {
                    "action": "EXECUTE_COMMAND",
                    "output": "x",
                    "command": "Element",
                    "arguments": [nested, {"kind": "number", "value": 1}],
                }
            ]
        )
