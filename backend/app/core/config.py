from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOODSIGHT_",
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    demo_stream_interval_ms: int = Field(default=1_400, ge=100, le=60_000)
    ingest_capture_fps: float = Field(default=4.0, ge=1, le=10)
    ingest_jpeg_quality: float = Field(default=0.75, ge=0.5, le=0.95)
    ingest_max_frame_bytes: int = Field(default=2_000_000, ge=50_000, le=10_000_000)
    ingest_max_sessions: int = Field(default=24, ge=1, le=256)
    ingest_session_ttl_seconds: int = Field(default=900, ge=30, le=86_400)
    ingest_dark_luminance_threshold: float = Field(default=35.0, ge=0, le=255)
    ingest_bright_luminance_threshold: float = Field(default=220.0, ge=0, le=255)
    ingest_blur_variance_threshold: float = Field(default=60.0, ge=0)
    model_registry_path: str = "configs/models/registry.json"
    segmentation_checkpoint: str | None = None
    detection_checkpoint: str | None = None
    detection_fallback_checkpoint: str | None = None
    inference_device: str = "auto"
    inference_precision: str = "auto"
    inference_resolution: int = Field(default=768, ge=256, le=2048)
    inference_frame_stride: int = Field(default=1, ge=1, le=120)
    segmentation_cadence: int = Field(default=1, ge=1, le=120)
    detection_cadence: int = Field(default=1, ge=1, le=120)
    detection_confidence_threshold: float = Field(default=0.25, ge=0.01, le=1)
    detection_iou_threshold: float = Field(default=0.7, ge=0.1, le=1)
    temporal_window_ms: int = Field(default=1_500, ge=500, le=5_000)
    temporal_track_ttl_ms: int = Field(default=2_000, ge=500, le=10_000)
    urgent_person_confidence: float = Field(default=0.85, ge=0.5, le=1)
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalized

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, values: list[str]) -> list[str]:
        for origin in values:
            if not origin.startswith(("http://", "https://")):
                raise ValueError("CORS origins must be absolute HTTP(S) origins")
        return values

    @field_validator("ingest_bright_luminance_threshold")
    @classmethod
    def validate_brightness_thresholds(cls, value: float, info: object) -> float:
        dark = getattr(info, "data", {}).get("ingest_dark_luminance_threshold", 35.0)
        if value <= dark:
            raise ValueError("bright luminance threshold must exceed the dark threshold")
        return value

    @field_validator("inference_precision")
    @classmethod
    def validate_inference_precision(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("inference precision must be auto, float32, float16, or bfloat16")
        return normalized


@lru_cache
def get_settings() -> Settings:
    return Settings()
