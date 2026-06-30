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
    try:
        service = SolverService(db)
        return await service.solve(request)
    except Exception as exc:
        logger.exception("Unexpected error in solve endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while solving the problem. Please try again.",
        ) from exc
