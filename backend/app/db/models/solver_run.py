import uuid
from datetime import UTC, datetime

from app.db.base import Base
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column


class SolverRun(Base):
    __tablename__ = "solver_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("problem_attempts.id"), nullable=False, index=True
    )
    parser_model: Mapped[str] = mapped_column(String(128), nullable=False)
    solver_model: Mapped[str] = mapped_column(String(128), nullable=False)
    vision_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    problem_type: Mapped[str] = mapped_column(String(64), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(32), nullable=False)
    route_reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_solver_runs_created_at", "created_at"),
    )
