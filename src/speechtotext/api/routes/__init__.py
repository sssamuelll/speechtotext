from fastapi import APIRouter

from speechtotext.api.routes.health import router as health_router
from speechtotext.api.routes.pronunciation import router as pronunciation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(pronunciation_router)

__all__ = ["api_router"]
