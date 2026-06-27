from app.schemas.geometry_dsl import GeometryAction, GeometryActionType, GeometryDSL
from app.services.geogebra_translator import GeoGebraTranslator


def test_translator_builds_basic_circle_commands() -> None:
    translator = GeoGebraTranslator()
    dsl = GeometryDSL(
        actions=[
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="O",
                coordinates=(0, 0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_CIRCLE,
                label="c",
                center="O",
                radius=5,
            ),
        ]
    )

    result = translator.translate(dsl)

    assert result.commands == ["O = (0.0, 0.0)", "c = Circle(O, 5.0)"]
    assert result.validation_passed is True


def test_translator_reuses_intersection_points_in_construction() -> None:
    translator = GeoGebraTranslator()
    dsl = GeometryDSL(
        actions=[
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="O",
                coordinates=(0.0, 0.0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="B",
                coordinates=(-2.5, 0.0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="C",
                coordinates=(2.5, 0.0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="A",
                coordinates=(-1.5, 2.0),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POINT,
                label="D",
                coordinates=(0.0, 2.5),
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_CIRCLE,
                label="c",
                center="O",
                radius=2.5,
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_POLYGON,
                label="polyABC",
                points=["A", "B", "C"],
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_LINE,
                label="lBC",
                points=["B", "C"],
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_LINE,
                label="lBD",
                points=["B", "D"],
            ),
            GeometryAction(
                action=GeometryActionType.CREATE_LINE,
                label="lAC",
                points=["A", "C"],
            ),
            GeometryAction(
                action=GeometryActionType.INTERSECT,
                label="E",
                metadata={"objects": ["lBD", "lAC"]},
            ),
            GeometryAction(
                action=GeometryActionType.PERPENDICULAR,
                label="lEF",
                line="lBC",
                metadata={"through_point": "E"},
            ),
            GeometryAction(
                action=GeometryActionType.INTERSECT,
                label="F",
                metadata={"objects": ["lEF", "lBC"]},
            ),
            GeometryAction(
                action=GeometryActionType.MIDPOINT,
                label="M",
                points=["E", "C"],
            ),
        ]
    )

    result = translator.translate(dsl)

    assert result.commands == [
        "O = (0.0, 0.0)",
        "B = (-2.5, 0.0)",
        "C = (2.5, 0.0)",
        "A = (-1.5, 2.0)",
        "D = (0.0, 2.5)",
        "c = Circle(O, 2.5)",
        "polyABC = Polygon(A, B, C)",
        "lBC = Line(B, C)",
        "lBD = Line(B, D)",
        "lAC = Line(A, C)",
        "E = Intersect(lBD, lAC)",
        "lEF = PerpendicularLine(E, lBC)",
        "F = Intersect(lEF, lBC)",
        "M = Midpoint(E, C)",
    ]
    assert result.validation_passed is True
    assert result.issues == []
