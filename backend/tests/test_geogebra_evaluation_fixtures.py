import json
from pathlib import Path

from app.schemas.geometry_dsl import GeometryActionType, GeometryDSL
from app.services.geogebra_command_registry import GeoGebraCommandRegistry
from app.services.geogebra_support_policy import load_runtime_acceptance
from scripts.evaluate_geogebra_fixtures import FIXTURES, evaluate


def test_evaluation_fixture_families_and_offline_metrics() -> None:
    report = evaluate()
    families = {case["family"] for case in report["cases"]}

    assert {
        "core_geometry",
        "transformations",
        "measurements",
        "algebra_graphing",
        "calculus",
        "lists_sequences",
        "statistics",
        "probability",
        "loci_advanced_geometry",
        "graphics_3d",
        "cas",
    } == families
    assert report["metrics"]["offline_reviewed_cases"] == 4
    assert report["metrics"]["dsl_schema_valid"] == 4
    assert report["metrics"]["registry_and_dependency_valid"] == 4
    assert report["metrics"]["validation_failure_cases"] == 0
    assert report["metrics"]["validation_failure_rate"] == 0.0
    assert report["metrics"]["validation_failure_counts_by_code"] == {}
    current_retrieval = report["metrics"]["retrieval"]["current"]
    candidate_retrieval = report["metrics"]["retrieval"]["candidate_16"]
    assert current_retrieval["expected_recall"] == 1.0
    assert candidate_retrieval["expected_recall"] >= current_retrieval["expected_recall"]
    assert candidate_retrieval["expected_precision"] >= current_retrieval["expected_precision"]
    assert report["metrics"]["catalog_runtime_eligibility"] == {
        "eligible": {"command_names": 434, "overloads": 933},
        "blocked": {"command_names": 68, "overloads": 119},
    }
    assert report["metrics"]["runtime_acceptance"] == "not_run_without_browser_applet"


def test_runtime_acceptance_records_link_exactly_to_reviewed_fixtures() -> None:
    fixtures = {
        case["id"]: case
        for case in json.loads(Path(FIXTURES).read_text(encoding="utf-8"))
    }
    acceptance = load_runtime_acceptance()
    registry = GeoGebraCommandRegistry()

    assert acceptance
    for record in acceptance.values():
        case = fixtures[record.fixture_id]
        assert case["status"] == "supported"
        assert record.command_name in case["allowed_commands"]
        dsl = GeometryDSL.model_validate(case["dsl"])
        assert dsl.environment.value in record.environments
        assert any(
            action.action is GeometryActionType.EXECUTE_COMMAND
            and action.command == record.command_name
            for action in dsl.actions
        )
        definition = registry.lookup(record.command_name)
        assert definition is not None
        overload = next(
            item
            for item in definition.overloads
            if item.signature.original == record.signature
        )
        assert overload.is_supported_in(dsl.environment)
