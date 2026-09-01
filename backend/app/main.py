import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.inference.pipeline import InferencePipeline
from app.services.inference_coordinator import InferenceCoordinator
from app.services.ingestion_sessions import IngestionSessionManager


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    inference_pipeline = InferencePipeline(settings)
    inference_coordinator = InferenceCoordinator(inference_pipeline)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "FloodSight API starting",
            extra={"environment": settings.environment, "version": __version__},
        )
        model_loading = asyncio.create_task(asyncio.to_thread(inference_pipeline.initialize))
        yield
        if not model_loading.done():
            model_loading.cancel()
        logger.info("FloodSight API stopped")

    application = FastAPI(
        title="FloodSight API",
        summary="Flood-response decision-intelligence service",
        description=(
            "FloodSight application integration: model adapters, scene fusion, explainable "
            "rescue-zone priority, relative accessibility, and ordered live updates."
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
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Accept", "Content-Type"],
    )
    register_exception_handlers(application)
    application.state.inference_pipeline = inference_pipeline
    application.state.inference_coordinator = inference_coordinator
    application.state.ingestion_manager = IngestionSessionManager(
        settings=settings,
        on_remove=inference_coordinator.close,
    )
    application.include_router(api_router)
    return application


app = create_app()
