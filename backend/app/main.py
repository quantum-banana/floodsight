from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "FloodSight API starting",
            extra={"environment": settings.environment, "version": __version__},
        )
        yield
        logger.info("FloodSight API stopped")

    application = FastAPI(
        title="FloodSight API",
        summary="Flood-response decision-intelligence service",
        description=(
            "Phase 0 API foundation. No machine-learning inference or rescue analytics "
            "are configured in this phase."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept", "Content-Type"],
    )
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
