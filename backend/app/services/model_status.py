from app.schemas.model_status import ModelState, ModelStatus, ModelStatusResponse


def get_model_status() -> ModelStatusResponse:
    unconfigured = ModelStatus(status=ModelState.NOT_CONFIGURED, model=None)
    return ModelStatusResponse(
        segmentation=unconfigured,
        detection=unconfigured.model_copy(deep=True),
    )
