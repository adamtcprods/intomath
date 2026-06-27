from app.db.session import get_db
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.solver_service import SolverService
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/solve", response_model=SolveResponse)
async def solve_problem(
    request: SolveRequest, db: Session = Depends(get_db)
) -> SolveResponse:
    service = SolverService(db)
    return await service.solve(request)
