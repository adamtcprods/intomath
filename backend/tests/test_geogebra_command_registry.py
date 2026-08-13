from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.geogebra_command_registry import (
    GeoGebraCommandRegistry,
    GeoGebraObjectType,
    evaluate_overload_support,
    parse_signature,
)
from app.services.geogebra_support_policy import SupportStatus


def test_registry_loads_and_groups_duplicate_overloads() -> None:
    registry = GeoGebraCommandRegistry()

    assert len(registry) >= 500
    assert registry.overload_count >= 1_050
    circle = registry.lookup("circle")
    assert circle is not None
    assert circle.name == "Circle"
    assert len(circle.overloads) >= 7
    assert all(overload.source for overload in circle.overloads)
    assert "geometry_2d" in circle.families


def test_registry_can_query_generated_command_families() -> None:
    registry = GeoGebraCommandRegistry()

    geometry_names = {definition.name for definition in registry.by_family("geometry_2d")}
    three_d_names = {
        definition.name
        for definition in registry.by_family("graphics_3d", include_unsafe=True)
    }
    scripting_names = {
        definition.name
        for definition in registry.by_family("scripting", include_unsafe=True)
    }
    logic_names = {definition.name for definition in registry.by_family("logic")}

    assert "Circle" in geometry_names
    assert "Cube" in three_d_names
    assert "Execute" in scripting_names
    assert "If" in logic_names


def test_signature_parser_handles_variadic_and_preserves_uncertainty() -> None:
    signature = parse_signature("ANOVA( <List>, <List>, ...)", "ANOVA")
    uncertain = parse_signature("not a formal signature", "Mystery")

    assert signature.min_arguments == 2
    assert signature.max_arguments is None
    assert signature.variadic is True
    assert signature.expected_types[0] == frozenset({GeoGebraObjectType.LIST})
    assert uncertain.original == "not a formal signature"
    assert uncertain.normalization_certain is False


def test_registry_maps_special_capabilities_and_unsafe_commands() -> None:
    registry = GeoGebraCommandRegistry()

    csolve = registry.lookup("CSolve")
    cone = registry.lookup("Cone")
    execute = registry.lookup("Execute")
    assert csolve is not None
    assert VisualizationEnvironment.cas in csolve.capabilities
    assert VisualizationEnvironment.graphing in csolve.capabilities
    assert cone is not None
    assert cone.capabilities == frozenset({VisualizationEnvironment.graphics_3d})
    assert execute is not None
    assert execute.unsafe_reason is not None
    assert registry.lookup("DefinitelyNotACommand") is None


def test_retrieval_is_bounded_relevant_and_excludes_unsafe_noise() -> None:
    registry = GeoGebraCommandRegistry()
    results = registry.search(
        "Draw a circle through A, B, C, construct the tangent at A, and reflect it across BC.",
        VisualizationEnvironment.geometry_2d,
        limit=8,
    )
    names = [result.name for result in results]

    assert {"Circle", "Tangent", "Reflect"}.issubset(names)
    assert "ANOVA" not in names
    assert "Execute" not in names
    assert "Sphere" not in names
    assert len(names) <= 8


def test_every_safe_catalog_command_has_a_runtime_eligible_environment() -> None:
    registry = GeoGebraCommandRegistry()
    safe_definitions = [
        definition
        for definition in registry._definitions.values()
        if definition.unsafe_reason is None
    ]

    runtime_eligibility = registry.metadata["runtime_eligibility"]
    assert runtime_eligibility["eligible"] == {
        "command_names": len(safe_definitions),
        "overloads": sum(len(definition.overloads) for definition in safe_definitions),
    }
    assert all(
        any(
            definition.runtime_eligible_overloads(environment)
            for environment in VisualizationEnvironment
        )
        for definition in safe_definitions
    )


def test_retrieval_reaches_each_specialized_environment() -> None:
    registry = GeoGebraCommandRegistry()
    cases = [
        ("Derivative", VisualizationEnvironment.graphing),
        ("Histogram", VisualizationEnvironment.statistics),
        ("BinomialDist", VisualizationEnvironment.probability),
        ("Sphere", VisualizationEnvironment.graphics_3d),
        ("Solve", VisualizationEnvironment.cas),
        ("Cell", VisualizationEnvironment.spreadsheet),
    ]

    for command_name, environment in cases:
        names = {
            result.name
            for result in registry.search(
                f"Use the {command_name} command for this construction.",
                environment,
                limit=10,
            )
        }
        assert command_name in names


def test_capability_query_does_not_mix_cas_into_geometry() -> None:
    registry = GeoGebraCommandRegistry()
    geometry_names = {
        definition.name
        for definition in registry.by_capability(VisualizationEnvironment.geometry_2d)
    }

    assert "Circle" in geometry_names
    assert "CSolve" not in geometry_names


def test_support_status_is_overload_specific_and_derived_from_requirements() -> None:
    registry = GeoGebraCommandRegistry()
    tangent = registry.lookup("Tangent")
    line = registry.lookup("Line")
    derivative = registry.lookup("Derivative")
    execute = registry.lookup("Execute")

    assert tangent is not None
    assert line is not None
    assert derivative is not None
    assert execute is not None

    tangent_by_signature = {
        overload.signature.original: overload for overload in tangent.overloads
    }
    accepted_tangent = tangent_by_signature["Tangent( <Point>, <Conic> )"]
    assert accepted_tangent.support_status is SupportStatus.supported
    assert accepted_tangent.support_requirements == ()
    assert accepted_tangent.runtime_accepted_environments == frozenset(
        {VisualizationEnvironment.geometry_2d}
    )
    assert any(
        overload.support_status is SupportStatus.experimental
        and "runtime_acceptance_test" in overload.support_requirements
        for overload in tangent.overloads
        if overload is not accepted_tangent
    )

    accepted_line = next(
        overload
        for overload in line.overloads
        if overload.signature.original == "Line( <Point>, <Point> )"
    )
    assert accepted_line.support_status is SupportStatus.supported
    assert derivative.support_status is SupportStatus.experimental
    assert all(
        overload.support_status is SupportStatus.blocked
        and overload.support_requirements == ("permanently_blocked",)
        for overload in execute.overloads
    )


def test_every_catalog_overload_has_one_support_status() -> None:
    registry = GeoGebraCommandRegistry()
    overloads = [
        overload
        for definition in registry.by_capability(
            VisualizationEnvironment.geometry_2d, include_unsafe=True
        )
        for overload in definition.overloads
    ]

    assert overloads
    assert all(overload.support_status in SupportStatus for overload in overloads)


def test_every_enablement_prerequisite_can_hold_an_overload_experimental() -> None:
    cases = [
        (
            {
                "command_name": "Line",
                "syntax": "not a signature",
                "categories": ["geometry"],
                "families": ["geometry_2d"],
                "is_cas": False,
                "runtime_environment_values": ["geometry_2d"],
                "runtime_output_type_value": "Line",
            },
            "safely_normalized_signature",
        ),
        (
            {
                "command_name": "Mystery",
                "syntax": "Mystery( <Point> )",
                "categories": ["geometry"],
                "families": ["geometry_2d"],
                "is_cas": False,
                "runtime_environment_values": ["geometry_2d"],
            },
            "known_output_type",
        ),
        (
            {
                "command_name": "Line",
                "syntax": "Line( <Point>, <Point> )",
                "categories": ["geometry"],
                "families": ["geometry_2d"],
                "is_cas": False,
                "runtime_environment_values": ["cas"],
                "runtime_output_type_value": "Line",
            },
            "correct_environment_metadata",
        ),
        (
            {
                "command_name": "Line",
                "syntax": "Line( <Point>, <Point> )",
                "categories": ["geometry"],
                "families": ["geometry_2d"],
                "is_cas": False,
                "runtime_environment_values": [],
                "runtime_output_type_value": "Line",
            },
            "runtime_acceptance_test",
        ),
    ]

    for kwargs, missing_requirement in cases:
        evaluation = evaluate_overload_support(**kwargs)
        assert evaluation.status is SupportStatus.experimental
        assert missing_requirement in evaluation.requirements

    generic_argument = evaluate_overload_support(
        command_name="Line",
        syntax="Line( <Mystery Token> )",
        categories=["geometry"],
        families=["geometry_2d"],
        is_cas=False,
        runtime_environment_values=["geometry_2d"],
        runtime_output_type_value="Line",
    )
    assert generic_argument.status is SupportStatus.supported
    assert "representable_typed_arguments" not in generic_argument.requirements
