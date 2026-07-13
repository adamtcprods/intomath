import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.solver_service import SolverService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/solve", response_model=SolveResponse, status_code=status.HTTP_200_OK)
async def solve_problem(
    request: SolveRequest, db: Session = Depends(get_db)
) -> SolveResponse:
    """Solve a math problem and return a structured step-by-step response."""
    input_type = "image" if request.input.image_base64 else "text"
    logger.info(
        "Solve endpoint received request input_type=%s text_chars=%s include_visualization=%s",
        input_type,
        len(request.input.text.strip()),
        request.options.include_visualization,
    )
    try:
        service = SolverService(db)
        response = await service.solve(request)
        logger.info(
            "Solve endpoint completed request_id=%s status=%s cached=%s",
            response.request_id,
            response.status,
            response.cached,
        )
        return response
    except Exception as exc:
        logger.exception("Unexpected error in solve endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while solving the problem. Please try again.",
        ) from exc
