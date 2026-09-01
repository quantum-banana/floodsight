from app.schemas.model_status import (
    InferenceState,
    ModelOperationalMode,
    ModelState,
    ModelStatus,
    ModelStatusResponse,
)


def get_model_status() -> ModelStatusResponse:
    unconfigured = ModelStatus(
        status=ModelState.NOT_CONFIGURED,
        model=None,
        mode=ModelOperationalMode.UNAVAILABLE,
        message="No application inference pipeline is attached.",
    )
    return ModelStatusResponse(
        segmentation=unconfigured,
        detection=unconfigured.model_copy(deep=True),
        inference_state=InferenceState.MODEL_UNAVAILABLE,
    )
