import uuid
from datetime import datetime

from app.db.base import Base
from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column


class VisualizationArtifact(Base):
    __tablename__ = "visualization_artifacts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("problem_attempts.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    dsl_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    commands_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
