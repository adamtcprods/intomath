import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter()

_API_VERSION = "2.0.0"


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health including a lightweight DB connectivity probe."""
    try:
        await asyncio.to_thread(_probe_database)
    except Exception:
        logger.warning("Health check: database probe failed", exc_info=True)
        db_status = "error"
    else:
        db_status = "ok"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
        version=_API_VERSION,
    )


def _probe_database() -> None:
    # Keep the synchronous session's full lifecycle inside one worker thread.
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
