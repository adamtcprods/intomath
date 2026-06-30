import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

_API_VERSION = "2.0.0"


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Return service health including a lightweight DB connectivity probe."""
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Health check: database probe failed", exc_info=True)
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
        version=_API_VERSION,
    )
