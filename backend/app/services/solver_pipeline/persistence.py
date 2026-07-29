"""Database persistence for completed solver responses."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db.models.problem_attempt import ProblemAttempt
from app.db.models.solver_run import SolverRun
from app.db.models.visualization_artifact import VisualizationArtifact
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.model_router import RoutingDecision


logger = logging.getLogger(__name__)


def persist_solve_result(
    db: Session,
    request: SolveRequest,
    raw_text: str,
    normalized_text: str,
    routing: RoutingDecision,
    response: SolveResponse,
) -> None:
    try:
        attempt = ProblemAttempt(
            raw_text=raw_text,
            normalized_text=normalized_text,
            input_type="image" if request.input.image_base64 else "text",
            language=request.input.language,
        )
        db.add(attempt)
        db.flush()

        run = SolverRun(
            attempt_id=attempt.id,
            parser_model=routing.parser_model,
            solver_model=routing.solver_model,
            vision_model=routing.vision_model,
            problem_type=routing.problem_type.value,
            difficulty=routing.difficulty.value,
            route_reason=routing.reason,
            confidence=response.confidence,
            cached=response.cached,
            status=response.status,
        )
        db.add(run)

        artifact = VisualizationArtifact(
            attempt_id=attempt.id,
            kind=response.visualization.kind,
            dsl_json=response.visualization.dsl.model_dump(mode="json")
            if response.visualization.dsl
            else None,
            commands_json=response.visualization.geogebra.commands
            if response.visualization.geogebra
            else None,
            summary=response.visualization.summary,
        )
        db.add(artifact)
        db.commit()
    except Exception:
        logger.exception("Failed to persist solve result")
        db.rollback()


__all__ = ["persist_solve_result"]
