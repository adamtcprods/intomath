from app.api.v1.router import router as api_router
from app.core.config import get_settings
from app.db.base import Base
from app.db.models.problem_attempt import ProblemAttempt  # noqa: F401
from app.db.models.solver_run import SolverRun  # noqa: F401
from app.db.models.visualization_artifact import VisualizationArtifact  # noqa: F401
from app.db.session import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    version="2.0.0",
    description="IntoMath 2.0 backend for structured math solving and deterministic visualization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "IntoMath 2.0 API is running"}


app.include_router(api_router, prefix="/api/v1")
