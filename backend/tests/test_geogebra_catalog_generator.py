import hashlib
import json
from pathlib import Path

from scripts.generate_geogebra_catalog import generate_catalog, parse_manual


FIXTURE = Path(__file__).parent / "fixtures/geogebra_manual"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "https://github.com/geogebra/manual"


def test_parse_manual_extracts_overloads_categories_and_sources() -> None:
    entries, page_count = parse_manual(
        FIXTURE, repository=REPOSITORY, commit=COMMIT
    )

    assert page_count == 2
    assert [(entry["command_name"], entry["syntax"]) for entry in entries] == [
        ("Circle", "Circle( <Point>, <Point>, <Point> )"),
        ("Circle", "Circle( <Point>, <Radius Number> )"),
        ("Tangent", "Tangent( <Point>, <Conic> )"),
    ]
    assert entries[0]["all_categories"] == ["geometry"]
    assert entries[0]["source"] == {
        "repository": REPOSITORY,
        "path": "en/modules/ROOT/pages/commands/Circle.adoc",
        "commit": COMMIT,
    }


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
    assert payload["metadata"]["command_names"] == 2
    assert payload["metadata"]["overloads"] == 3
    assert "generation_timestamp" not in payload["metadata"]
    assert first.with_suffix(".json.metadata.json").exists()
