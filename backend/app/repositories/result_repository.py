"""Best-effort persistence for completed solver responses."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy.orm import Session

from app.db.models.problem_attempt import ProblemAttempt
from app.db.models.solver_run import SolverRun
from app.db.models.visualization_artifact import VisualizationArtifact
from app.schemas.solve import SolveRequest, SolveResponse

logger = logging.getLogger(__name__)


class SessionFactory(Protocol):
    def __call__(self) -> Session: ...


@dataclass(frozen=True)
class SolveTimings:
    """Caller-observed timings captured before persistence starts."""

    request_started_at: datetime
    solve_completed_at: datetime
    solve_duration_ms: float
    solver_duration_ms: float
    visualization_duration_ms: float


class ResultRepository:
    """Persist one solve result without blocking the async event loop.

    A fresh synchronous SQLAlchemy session is constructed, used, rolled back when
    necessary, and closed entirely inside the worker thread.
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def save(
        self,
        request: SolveRequest,
        raw_text: str,
        normalized_text: str,
        response: SolveResponse,
        timings: SolveTimings,
    ) -> bool:
        try:
            await asyncio.to_thread(
                self._save_sync,
                request,
                raw_text,
                normalized_text,
                response,
                timings,
            )
        except Exception:
            logger.exception(
                "Failed to persist solve result request_id=%s",
                response.request_id,
            )
            return False
        return True

    def _save_sync(
        self,
        request: SolveRequest,
        raw_text: str,
        normalized_text: str,
        response: SolveResponse,
        timings: SolveTimings,
    ) -> None:
        persistence_started = time.monotonic()
        session = self._session_factory()
        try:
            attempt_id = str(uuid.uuid4())
            session.add(
                ProblemAttempt(
                    id=attempt_id,
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    input_type="image" if request.input.image_base64 else "text",
                    language=request.input.language,
                )
            )

            run = SolverRun(
                attempt_id=attempt_id,
                request_id=response.request_id,
                parser_model=response.routing.parser_model,
                solver_model=response.routing.solver_model,
                vision_model=response.routing.vision_model,
                problem_type=response.problem_type,
                difficulty=response.difficulty,
                route_reason=response.routing.reason,
                confidence=response.confidence,
                cached=response.cached,
                status=response.status,
                request_started_at=timings.request_started_at,
                solve_completed_at=timings.solve_completed_at,
                solve_duration_ms=max(0.0, timings.solve_duration_ms),
                solver_duration_ms=max(0.0, timings.solver_duration_ms),
                visualization_duration_ms=max(
                    0.0, timings.visualization_duration_ms
                ),
                persistence_duration_ms=0.0,
                total_duration_ms=max(0.0, timings.solve_duration_ms),
            )
            session.add(run)
            session.add(
                VisualizationArtifact(
                    attempt_id=attempt_id,
                    kind=response.visualization.kind,
                    dsl_json=response.visualization.dsl.model_dump(mode="json")
                    if response.visualization.dsl
                    else None,
                    commands_json=response.visualization.geogebra.commands
                    if response.visualization.geogebra
                    else None,
                    summary=response.visualization.summary,
                )
            )

            # Flush all writes before capturing persistence time. The following
            # commit is intentionally part of the same transaction; any failure
            # rolls every row back together.
            session.flush()
            persistence_duration_ms = (
                time.monotonic() - persistence_started
            ) * 1000.0
            run.persistence_duration_ms = persistence_duration_ms
            run.total_duration_ms = (
                max(0.0, timings.solve_duration_ms) + persistence_duration_ms
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


__all__ = ["ResultRepository", "SolveTimings"]
