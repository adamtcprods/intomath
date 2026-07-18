from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.visualization_artifact import VisualizationArtifact
from app.schemas.geometry_dsl import GeometryDSL
from scripts.migrate_geometry_dsl_1_1 import migrate_artifacts, upgrade_legacy_dsl


def test_upgrade_legacy_dsl_adds_the_complete_1_1_envelope() -> None:
    upgraded, changed = upgrade_legacy_dsl(
        {
            "version": "1.0",
            "actions": [{"action": "CREATE_POINT", "label": "A"}],
        }
    )

    assert changed is True
    assert upgraded["version"] == "1.1"
    assert upgraded["space"] == "euclidean_2d"
    assert upgraded["environment"] == "geometry_2d"
    assert upgraded["actions"][0]["action"] == "CREATE_POINT"
    assert upgraded["actions"][0]["label"] == "A"
    assert "render_hints" in upgraded
    assert GeometryDSL.model_validate(upgraded).version == "1.1"


def test_upgrade_legacy_dsl_uses_3d_space_for_3d_artifacts() -> None:
    upgraded, changed = upgrade_legacy_dsl(
        {"version": "1.0", "environment": "graphics_3d", "actions": []}
    )

    assert changed is True
    assert upgraded["space"] == "euclidean_3d"


def test_upgrade_legacy_dsl_rejects_an_inconsistent_legacy_envelope() -> None:
    with pytest.raises(ValidationError):
        upgrade_legacy_dsl(
            {
                "version": "1.0",
                "space": "euclidean_2d",
                "environment": "graphics_3d",
                "actions": [],
            }
        )


def test_migrate_artifacts_updates_dsl_and_preserves_commands(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'artifacts.db'}"
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            VisualizationArtifact(
                attempt_id="attempt-1",
                kind="geogebra",
                dsl_json={
                    "version": "1.0",
                    "actions": [{"action": "CREATE_POINT", "label": "A"}],
                },
                commands_json=["A = (0, 0)"],
            )
        )
        session.add(
            VisualizationArtifact(
                attempt_id="attempt-2",
                kind="none",
                dsl_json=None,
                commands_json=None,
            )
        )
        session.commit()

    scanned, upgraded, failures = migrate_artifacts(database_url, apply=True)

    assert (scanned, upgraded, failures) == (1, 1, [])
    with Session(engine) as session:
        artifact = session.scalar(
            select(VisualizationArtifact).where(
                VisualizationArtifact.attempt_id == "attempt-1"
            )
        )
        assert artifact is not None
        assert isinstance(artifact.dsl_json, dict)
        assert artifact.dsl_json["version"] == "1.1"
        assert artifact.commands_json == ["A = (0, 0)"]
