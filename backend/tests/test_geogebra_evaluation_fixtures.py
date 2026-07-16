from scripts.evaluate_geogebra_fixtures import evaluate


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
    assert report["metrics"]["offline_reviewed_cases"] == 3
    assert report["metrics"]["dsl_schema_valid"] == 3
    assert report["metrics"]["registry_and_dependency_valid"] == 3
    assert report["metrics"]["validation_failure_cases"] == 0
    assert report["metrics"]["validation_failure_rate"] == 0.0
    assert report["metrics"]["validation_failure_counts_by_code"] == {}
    current_retrieval = report["metrics"]["retrieval"]["current"]
    candidate_retrieval = report["metrics"]["retrieval"]["candidate_16"]
    assert current_retrieval["expected_recall"] == 1.0
    assert candidate_retrieval["expected_recall"] >= current_retrieval["expected_recall"]
    assert candidate_retrieval["expected_precision"] >= current_retrieval["expected_precision"]
    assert report["metrics"]["runtime_acceptance"] == "not_run_without_browser_applet"
