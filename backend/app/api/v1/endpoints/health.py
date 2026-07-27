import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import text

from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter()

_API_VERSION = "2.0.0"


class SemanticRouterHealth(BaseModel):
    status: str
    model_loaded: bool
    model_path: str | None
    model_version: str | None
    load_time_ms: float | None
    approximate_memory_mb: float | None


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str
    semantic_router: SemanticRouterHealth


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return database and optional process-local semantic-router health."""
    try:
        await asyncio.to_thread(_probe_database)
    except Exception:
        logger.warning("Health check: database probe failed", exc_info=True)
        db_status = "error"
    else:
        db_status = "ok"

    semantic_health, semantic_degraded = _semantic_router_health(request)
    return HealthResponse(
        status=(
            "ok"
            if db_status == "ok" and not semantic_degraded
            else "degraded"
        ),
        db=db_status,
        version=_API_VERSION,
        semantic_router=semantic_health,
    )


def _semantic_router_health(
    request: Request,
) -> tuple[SemanticRouterHealth, bool]:
    clients = getattr(request.app.state, "model_clients", None)
    semantic_router = getattr(clients, "semantic_router", None)
    if semantic_router is None:
        return (
            SemanticRouterHealth(
                status="disabled",
                model_loaded=False,
                model_path=None,
                model_version=None,
                load_time_ms=None,
                approximate_memory_mb=None,
            ),
            False,
        )

    status = semantic_router.status()
    return (
        SemanticRouterHealth(
            status=status.state,
            model_loaded=status.state == "ready",
            model_path=status.model_path or status.model_source,
            model_version=status.model_version,
            load_time_ms=status.load_time_ms,
            approximate_memory_mb=status.approximate_memory_mb,
        ),
        status.enabled and status.state == "unavailable",
    )


def _probe_database() -> None:
    # Keep the synchronous session's full lifecycle inside one worker thread.
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
