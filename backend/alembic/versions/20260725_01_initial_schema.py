"""Create the initial IntoMath result schema.

Revision ID: 20260725_01
Revises:
Create Date: 2026-07-25
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260725_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "problem_attempts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("input_type", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_problem_attempts_created_at",
        "problem_attempts",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "solver_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("attempt_id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("parser_model", sa.String(length=128), nullable=False),
        sa.Column("solver_model", sa.String(length=128), nullable=False),
        sa.Column("vision_model", sa.String(length=128), nullable=True),
        sa.Column("problem_type", sa.String(length=64), nullable=False),
        sa.Column("difficulty", sa.String(length=32), nullable=False),
        sa.Column("route_reason", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("solve_completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("solve_duration_ms", sa.Float(), nullable=False),
        sa.Column("solver_duration_ms", sa.Float(), nullable=False),
        sa.Column("visualization_duration_ms", sa.Float(), nullable=False),
        sa.Column("persistence_duration_ms", sa.Float(), nullable=False),
        sa.Column("total_duration_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["attempt_id"], ["problem_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_solver_runs_attempt_id",
        "solver_runs",
        ["attempt_id"],
        unique=False,
    )
    op.create_index(
        "ix_solver_runs_created_at",
        "solver_runs",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_solver_runs_request_id",
        "solver_runs",
        ["request_id"],
        unique=True,
    )

    op.create_table(
        "visualization_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("attempt_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("dsl_json", sa.JSON(), nullable=True),
        sa.Column("commands_json", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["attempt_id"], ["problem_attempts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_visualization_artifacts_attempt_id",
        "visualization_artifacts",
        ["attempt_id"],
        unique=False,
    )
    op.create_index(
        "ix_visualization_artifacts_created_at",
        "visualization_artifacts",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_visualization_artifacts_created_at",
        table_name="visualization_artifacts",
    )
    op.drop_index(
        "ix_visualization_artifacts_attempt_id",
        table_name="visualization_artifacts",
    )
    op.drop_table("visualization_artifacts")

    op.drop_index("ix_solver_runs_request_id", table_name="solver_runs")
    op.drop_index("ix_solver_runs_created_at", table_name="solver_runs")
    op.drop_index("ix_solver_runs_attempt_id", table_name="solver_runs")
    op.drop_table("solver_runs")

    op.drop_index(
        "ix_problem_attempts_created_at",
        table_name="problem_attempts",
    )
    op.drop_table("problem_attempts")
