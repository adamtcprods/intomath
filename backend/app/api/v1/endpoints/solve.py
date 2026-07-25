import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_solver_service
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.input_validation import SolveInputError
from app.services.solver_service import SolverService
from app.services.solver_pipeline.errors import SolveRequestTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/solve", response_model=SolveResponse, status_code=status.HTTP_200_OK)
async def solve_problem(
    request: SolveRequest,
    service: SolverService = Depends(get_solver_service),
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
        response = await service.solve(request)
        logger.info(
            "Solve endpoint completed request_id=%s status=%s cached=%s",
            response.request_id,
            response.status,
            response.cached,
        )
        return response
    except SolveRequestTimeoutError as exc:
        logger.warning(
            "Solve endpoint returning timeout request_id=%s stage=%s",
            exc.request_id,
            exc.stage,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The solve request exceeded its time limit. Please try again.",
        ) from None
    except SolveInputError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from None
    except Exception as exc:
        logger.exception("Unexpected error in solve endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while solving the problem. Please try again.",
        ) from exc
