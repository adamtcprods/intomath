"""Upgrade persisted visualization artifacts from Geometry DSL 1.0 to 1.1.

The command is a dry run unless ``--apply`` is provided. Run it from the
backend directory so the default SQLite URL resolves to the application DB.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.visualization_artifact import VisualizationArtifact
from app.schemas.geometry_dsl import GeometryDSL

LEGACY_VERSIONS = {None, "1.0"}


def upgrade_legacy_dsl(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return a validated DSL 1.1 payload and whether an upgrade was needed."""
    version = payload.get("version")
    if version == "1.1":
        return deepcopy(payload), False
    if version not in LEGACY_VERSIONS:
        raise ValueError(f"Unsupported Geometry DSL version: {version!r}")

    upgraded = deepcopy(payload)
    environment = upgraded.setdefault("environment", "geometry_2d")
    upgraded["version"] = "1.1"
    upgraded.setdefault(
        "space", "euclidean_3d" if environment == "graphics_3d" else "euclidean_2d"
    )
    upgraded.setdefault("actions", [])
    upgraded.setdefault("render_hints", {})

    normalized = GeometryDSL.model_validate(upgraded).model_dump(mode="json")
    return normalized, True


def migrate_artifacts(
    database_url: str,
    *,
    apply: bool,
) -> tuple[int, int, list[dict[str, str]]]:
    """Inspect or migrate legacy records and return scanned, upgraded, failures."""
    engine = create_engine(database_url, future=True)
    if not inspect(engine).has_table(VisualizationArtifact.__tablename__):
        raise RuntimeError(
            "The database does not contain the visualization_artifacts table."
        )

    scanned = 0
    upgraded_count = 0
    failures: list[dict[str, str]] = []

    with Session(engine) as session:
        artifacts = session.scalars(
            select(VisualizationArtifact).where(
                VisualizationArtifact.dsl_json.is_not(None)
            )
        )
        for artifact in artifacts:
            payload = artifact.dsl_json
            if payload is None:
                continue
            if not isinstance(payload, dict):
                failures.append(
                    {"artifact_id": artifact.id, "error": "dsl_json is not an object"}
                )
                continue
            if payload.get("version") not in LEGACY_VERSIONS:
                continue

            scanned += 1
            try:
                upgraded, changed = upgrade_legacy_dsl(payload)
            except (TypeError, ValueError) as exc:
                failures.append({"artifact_id": artifact.id, "error": str(exc)})
                continue

            if changed:
                upgraded_count += 1
                if apply:
                    artifact.dsl_json = upgraded

        if apply:
            session.commit()
        else:
            session.rollback()

    return scanned, upgraded_count, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade persisted Geometry DSL 1.0 artifacts to DSL 1.1."
    )
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help="SQLAlchemy database URL; defaults to DATABASE_URL/application settings.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit valid upgrades. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--failure-report",
        type=Path,
        help="Optional JSON file for malformed artifact IDs and validation errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        scanned, upgraded, failures = migrate_artifacts(
            args.database_url,
            apply=args.apply,
        )
    except RuntimeError as exc:
        print(f"Geometry DSL 1.1 migration could not start: {exc}")
        return 2

    mode = "applied" if args.apply else "dry-run"
    print(
        f"Geometry DSL 1.1 migration ({mode}): "
        f"legacy={scanned} valid={upgraded} failures={len(failures)}"
    )
    for failure in failures:
        print(f"- {failure['artifact_id']}: {failure['error']}")

    if args.failure_report and failures:
        args.failure_report.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Failure report written to {args.failure_report}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
