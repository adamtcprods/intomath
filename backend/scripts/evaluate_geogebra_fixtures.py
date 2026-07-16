#!/usr/bin/env python3
"""Run offline schema/registry/dependency checks for reviewed GeoGebra fixtures."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.geometry_dsl import GeometryDSL, ValidationSeverity  # noqa: E402
from app.services.geogebra_command_registry import (  # noqa: E402
    ROLLED_OUT_GENERIC_COMMANDS,
    GeoGebraCommandRegistry,
)
from app.services.geogebra_translator import GeoGebraTranslator  # noqa: E402
from app.services.geometry_extractor import GEOMETRY_RETRIEVAL_LIMIT  # noqa: E402


FIXTURES = BACKEND_DIR / "tests/fixtures/geogebra_evaluation_cases.json"


def evaluate() -> dict[str, Any]:
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))
    translator = GeoGebraTranslator()
    registry = GeoGebraCommandRegistry()
    results: list[dict[str, Any]] = []
    validation_issue_counts: Counter[str] = Counter()
    schema_issue_counts: Counter[str] = Counter()
    retrieval_expected_total = 0
    retrieval_metrics = {
        "current": {"hits": 0, "retrieved": 0},
        "candidate_16": {"hits": 0, "retrieved": 0},
    }
    for case in cases:
        dsl: GeometryDSL | None = None
        result: dict[str, Any] = {
            "id": case["id"],
            "family": case["family"],
            "rollout_status": case["status"],
            "dsl_schema_valid": None,
            "registry_and_dependency_valid": None,
            "runtime_acceptance": "not_run",
            "mathematical_correctness": "manual_review_required",
        }
        if "dsl" in case:
            try:
                dsl = GeometryDSL.model_validate(case["dsl"])
                result["dsl_schema_valid"] = True
                translation = translator.translate(
                    dsl,
                    allowed_command_names=case.get("allowed_commands", []),
                )
                result["registry_and_dependency_valid"] = translation.validation_passed
                result["validation_issue_codes"] = [
                    issue.code for issue in translation.issues
                ]
                validation_issue_counts.update(
                    issue.code
                    for issue in translation.issues
                    if issue.severity is ValidationSeverity.error
                )
                result["commands"] = translation.commands
            except ValidationError as exc:
                result["dsl_schema_valid"] = False
                result["registry_and_dependency_valid"] = False
                result["schema_error_count"] = len(exc.errors())
                schema_codes = [
                    f"schema_{str(item.get('type', 'invalid')).replace('.', '_')}"
                    for item in exc.errors(include_url=False)
                ]
                schema_issue_counts.update(schema_codes)
                result["schema_issue_codes"] = schema_codes

        expected_commands = set(case.get("allowed_commands", []))
        if expected_commands and result["dsl_schema_valid"] is True:
            assert dsl is not None
            environment = dsl.environment
            retrieval_expected_total += len(expected_commands)
            for label, limit in (
                ("current", GEOMETRY_RETRIEVAL_LIMIT),
                ("candidate_16", 16),
            ):
                retrieved_names = [
                    command.name
                    for command in registry.search(
                        case["prompt"], environment, limit=limit
                    )
                    if command.name in ROLLED_OUT_GENERIC_COMMANDS
                ]
                hits = len(expected_commands.intersection(retrieved_names))
                retrieval_metrics[label]["hits"] += hits
                retrieval_metrics[label]["retrieved"] += len(retrieved_names)
                result[f"retrieved_commands_{label}"] = retrieved_names
        results.append(result)

    reviewed = [item for item in results if item["dsl_schema_valid"] is not None]
    validation_failures = sum(
        item["registry_and_dependency_valid"] is False for item in reviewed
    )
    retrieval_report: dict[str, dict[str, float | int]] = {}
    for label, counts in retrieval_metrics.items():
        retrieved_count = counts["retrieved"]
        retrieval_report[label] = {
            "limit": GEOMETRY_RETRIEVAL_LIMIT if label == "current" else 16,
            "expected_commands": retrieval_expected_total,
            "expected_hits": counts["hits"],
            "expected_recall": (
                counts["hits"] / retrieval_expected_total
                if retrieval_expected_total
                else 1.0
            ),
            "expected_precision": (
                counts["hits"] / retrieved_count if retrieved_count else 1.0
            ),
            "retrieved_commands": retrieved_count,
        }
    return {
        "cases": results,
        "metrics": {
            "total_cases": len(results),
            "offline_reviewed_cases": len(reviewed),
            "dsl_schema_valid": sum(item["dsl_schema_valid"] is True for item in reviewed),
            "registry_and_dependency_valid": sum(
                item["registry_and_dependency_valid"] is True for item in reviewed
            ),
            "validation_failure_cases": validation_failures,
            "validation_failure_rate": (
                validation_failures / len(reviewed) if reviewed else 0.0
            ),
            "validation_failure_counts_by_code": dict(
                sorted((validation_issue_counts + schema_issue_counts).items())
            ),
            "retrieval": retrieval_report,
            "runtime_acceptance": "not_run_without_browser_applet",
            "mathematical_correctness": "not_inferred_from_runtime_acceptance",
        },
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
