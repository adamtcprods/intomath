from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_migration_upgrade_from_empty_database(tmp_path: Path) -> None:
    database_path = tmp_path / "migration-test.db"
    backend_dir = Path(__file__).resolve().parents[1]
    config = Config(str(backend_dir / "alembic.ini"))
    config.set_main_option("script_location", str(backend_dir / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")

    command.upgrade(config, "head")
    command.check(config)

    engine = create_engine(f"sqlite:///{database_path}")
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == {
        "alembic_version",
        "problem_attempts",
        "solver_runs",
        "visualization_artifacts",
    }
    solver_columns = {column["name"] for column in inspector.get_columns("solver_runs")}
    assert {
        "request_id",
        "request_started_at",
        "solve_completed_at",
        "solve_duration_ms",
        "solver_duration_ms",
        "visualization_duration_ms",
        "persistence_duration_ms",
        "total_duration_ms",
    } <= solver_columns
    engine.dispose()
