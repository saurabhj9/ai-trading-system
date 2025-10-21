"""
FastAPI application for the AI Trading System.
"""
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.monitoring import (
    router as monitoring_router,
    increment_errors,
    increment_requests,
    record_request_duration,
)
from src.api.routes.signals import router as signals_router
from src.api.routes.status import router as status_router
from src.config.settings import settings
from src.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="AI Trading System API",
    description="API for accessing trade signals and system status",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Trading System API")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    increment_requests()

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        record_request_duration(duration)
        return response
    except Exception as e:
        increment_errors()
        raise e

# Include routers
app.include_router(signals_router, prefix="/api/v1", tags=["signals"])
app.include_router(status_router, prefix="/api/v1", tags=["status"])
app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["monitoring"])

@app.get("/")
async def root():
    return {"message": "AI Trading System API", "version": "1.0.0"}
