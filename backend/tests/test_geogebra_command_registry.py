from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.geogebra_command_registry import (
    GeoGebraCommandRegistry,
    GeoGebraObjectType,
    parse_signature,
)


def test_registry_loads_and_groups_duplicate_overloads() -> None:
    registry = GeoGebraCommandRegistry()

    assert len(registry) >= 500
    assert registry.overload_count >= 1_050
    circle = registry.lookup("circle")
    assert circle is not None
    assert circle.name == "Circle"
    assert len(circle.overloads) >= 7
    assert all(overload.source for overload in circle.overloads)


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
    assert csolve.capabilities == frozenset({VisualizationEnvironment.cas})
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


def test_capability_query_does_not_mix_cas_into_geometry() -> None:
    registry = GeoGebraCommandRegistry()
    geometry_names = {
        definition.name
        for definition in registry.by_capability(VisualizationEnvironment.geometry_2d)
    }

    assert "Circle" in geometry_names
    assert "CSolve" not in geometry_names
