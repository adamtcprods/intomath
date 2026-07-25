from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models.problem_attempt import ProblemAttempt
from app.db.models.solver_run import SolverRun
from app.db.models.visualization_artifact import VisualizationArtifact
from app.repositories.result_repository import ResultRepository, SolveTimings
from app.schemas.solve import SolveRequest, SolveResponse


def _database() -> tuple[object, sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _request() -> SolveRequest:
    return SolveRequest.model_validate(
        {
            "input": {"text": "What is 2 + 2?", "language": "en"},
            "options": {"include_visualization": False},
        }
    )


def _response(*, request_id: str = "request-correlation-id") -> SolveResponse:
    return SolveResponse.model_validate(
        {
            "request_id": request_id,
            "status": "ok",
            "problem_type": "arithmetic",
            "difficulty": "easy",
            "answer": {"text": "4", "latex": "4"},
            "steps": [
                {
                    "index": 1,
                    "title": "Add",
                    "explanation": "Add the values.",
                    "latex": ["2+2=4"],
                }
            ],
            "visualization": {"kind": "none"},
            "confidence": 1.0,
            "routing": {
                "parser_model": "test-parser",
                "solver_model": "test-solver",
                "vision_model": None,
                "reason": "test route",
            },
            "cached": False,
            "warnings": [],
        }
    )


def _timings() -> SolveTimings:
    now = datetime.now(UTC)
    return SolveTimings(
        request_started_at=now,
        solve_completed_at=now,
        solve_duration_ms=125.0,
        solver_duration_ms=80.0,
        visualization_duration_ms=15.0,
    )


def test_successful_persistence_and_request_id_correlation() -> None:
    engine, session_factory = _database()
    repository = ResultRepository(session_factory)

    saved = asyncio.run(
        repository.save(
            _request(),
            "What is 2 + 2?",
            "What is 2 + 2?",
            _response(),
            _timings(),
        )
    )

    assert saved is True
    with session_factory() as session:
        attempt = session.scalar(select(ProblemAttempt))
        run = session.scalar(select(SolverRun))
        artifact = session.scalar(select(VisualizationArtifact))

        assert attempt is not None
        assert run is not None
        assert artifact is not None
        assert run.attempt_id == attempt.id == artifact.attempt_id
        assert run.request_id == "request-correlation-id"
        assert run.solve_duration_ms == 125.0
        assert run.solver_duration_ms == 80.0
        assert run.visualization_duration_ms == 15.0
        assert run.persistence_duration_ms >= 0.0
        assert run.total_duration_ms >= run.solve_duration_ms

    engine.dispose()


def test_persistence_failure_rolls_back_and_logs_request_id(caplog: object) -> None:
    engine, _ = _database()

    class FailingCommitSession(Session):
        def commit(self) -> None:
            self.flush()
            raise RuntimeError("simulated commit failure")

    failing_factory = sessionmaker(
        bind=engine,
        class_=FailingCommitSession,
        expire_on_commit=False,
    )
    repository = ResultRepository(failing_factory)

    with caplog.at_level(logging.ERROR):
        saved = asyncio.run(
            repository.save(
                _request(),
                "What is 2 + 2?",
                "What is 2 + 2?",
                _response(request_id="rollback-request-id"),
                _timings(),
            )
        )

    assert saved is False
    assert "request_id=rollback-request-id" in caplog.text
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(ProblemAttempt)) == 0
        assert session.scalar(select(func.count()).select_from(SolverRun)) == 0
        assert (
            session.scalar(select(func.count()).select_from(VisualizationArtifact))
            == 0
        )

    engine.dispose()


def test_slow_persistence_does_not_block_event_loop() -> None:
    engine, session_factory = _database()

    class SlowResultRepository(ResultRepository):
        def _save_sync(self, *args: object, **kwargs: object) -> None:
            time.sleep(0.2)
            super()._save_sync(*args, **kwargs)

    repository = SlowResultRepository(session_factory)

    async def run() -> tuple[bool, float]:
        started = time.monotonic()
        save_task = asyncio.create_task(
            repository.save(
                _request(),
                "What is 2 + 2?",
                "What is 2 + 2?",
                _response(request_id="slow-persistence-id"),
                _timings(),
            )
        )
        await asyncio.sleep(0.02)
        scheduling_delay = time.monotonic() - started
        return await save_task, scheduling_delay

    saved, scheduling_delay = asyncio.run(run())

    assert saved is True
    assert scheduling_delay < 0.1
    engine.dispose()
