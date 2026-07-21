import hashlib
import json
from pathlib import Path

from scripts.generate_geogebra_catalog import (
    command_families_for_categories,
    generate_catalog,
    parse_manual,
)


FIXTURE = Path(__file__).parent / "fixtures/geogebra_manual"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "https://github.com/geogebra/manual"


def test_parse_manual_extracts_overloads_categories_and_sources() -> None:
    entries, page_count = parse_manual(
        FIXTURE, repository=REPOSITORY, commit=COMMIT
    )

    assert page_count == 3
    assert [(entry["command_name"], entry["syntax"]) for entry in entries] == [
        ("Circle", "Circle( <Point>, <Point>, <Point> )"),
        ("Circle", "Circle( <Point>, <Radius Number> )"),
        ("If", "If( <Condition>, <Then> )"),
        ("Tangent", "Tangent( <Point>, <Conic> )"),
    ]
    assert entries[0]["all_categories"] == ["geometry"]
    assert entries[0]["families"] == ["geometry_2d"]
    assert entries[0]["support_status"] == "experimental"
    assert entries[0]["support_requirements"] == [
        "known_output_type",
        "runtime_acceptance_test",
    ]
    assert entries[0]["runtime_eligible"] is True
    assert entries[0]["capabilities"] == ["geometry_2d"]
    assert entries[0]["output_type"] == "Unknown"
    assert entries[0]["output_type_strategy"] is None
    tangent = next(entry for entry in entries if entry["command_name"] == "Tangent")
    assert tangent["support_status"] == "supported"
    assert tangent["support_requirements"] == []
    assert tangent["runtime_eligible"] is True
    assert tangent["runtime_accepted_environments"] == ["geometry_2d"]
    assert tangent["output_type"] == "Line"
    assert entries[0]["source"] == {
        "repository": REPOSITORY,
        "path": "en/modules/ROOT/pages/commands/Circle.adoc",
        "commit": COMMIT,
    }


def test_command_families_group_categories_and_preserve_overlap() -> None:
    assert command_families_for_categories(
        ["3d", "conic", "geometry", "transformation"]
    ) == ["geometry_2d", "transformations", "graphics_3d"]
    assert command_families_for_categories(
        ["algebra", "functions_and_calculus", "cas"]
    ) == ["graphing_calculus", "cas"]
    assert command_families_for_categories(["chart", "statistics"]) == [
        "statistics"
    ]
    assert command_families_for_categories(["scripting"]) == ["scripting"]
    assert command_families_for_categories(["logic"]) == ["logic"]
    assert command_families_for_categories(["future_manual_category"]) == ["other"]


def test_generate_catalog_is_valid_and_content_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_metadata = generate_catalog(
        FIXTURE,
        repository=REPOSITORY,
        commit=COMMIT,
        output=first,
    )
    second_metadata = generate_catalog(
        FIXTURE,
        repository=REPOSITORY,
        commit=COMMIT,
        output=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["catalog_sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["metadata"]["command_names"] == 3
    assert payload["metadata"]["overloads"] == 4
    assert payload["metadata"]["schema_version"] == "1.4"
    assert payload["metadata"]["families"] == {
        "geometry_2d": {"command_names": 2, "overloads": 3},
        "logic": {"command_names": 1, "overloads": 1},
    }
    assert payload["metadata"]["support_statuses"] == {
        "supported": {"command_names": 1, "overloads": 1},
        "experimental": {"command_names": 2, "overloads": 3},
    }
    assert payload["metadata"]["runtime_eligibility"] == {
        "eligible": {"command_names": 3, "overloads": 4},
        "blocked": {"command_names": 0, "overloads": 0},
    }
    assert "generation_timestamp" not in payload["metadata"]
    assert first.with_suffix(".json.metadata.json").exists()
