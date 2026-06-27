from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.solve import router as solve_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(solve_router, tags=["solve"])
