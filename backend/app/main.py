import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_router
from app.core.config import get_settings
from app.db import models as _models  # noqa: F401 -- Register SQLAlchemy models.
from app.db.base import Base
from app.db.session import engine
from app.dependencies import create_shared_model_clients

settings = get_settings()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: create all tables if they don't exist.
    Base.metadata.create_all(bind=engine)
    model_clients = create_shared_model_clients(settings)
    app.state.model_clients = model_clients
    try:
        if settings.local_llama_enabled:
            available = await model_clients.llama.probe_health(
                timeout_seconds=settings.local_llama_startup_probe_timeout_seconds
            )
            if not available:
                logger.warning(
                    "Local llama.cpp is configured but unavailable at startup base_url=%s; "
                    "local model stages will be skipped during the connectivity cooldown",
                    settings.local_solver_llama_base_url,
                )
        yield
    finally:
        try:
            await model_clients.aclose()
        finally:
            del app.state.model_clients


app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    version="2.0.0",
    description="IntoMath backend for structured math solving and validated visualization.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "IntoMath API is running"}


app.include_router(api_router, prefix="/api/v1")
