from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "IntoMath 2.0 API"
    app_env: str = "development"
    app_debug: bool = True
    remote_model_attempt_timeout_seconds: float = Field(default=25.0, gt=0, le=120)
    nvidia_large_model_attempt_timeout_seconds: float = Field(
        default=50.0, gt=0, le=120
    )
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_direct_enabled: bool = True
    deepseek_ocr_model_id: str = "deepseek-ai/deepseek-ocr-2"
    local_solver_first: bool = True
    local_llama_enabled: bool = True
    local_solver_llama_detection_enabled: bool = True
    local_solver_llama_trivia_enabled: bool = True
    local_llama_geometry_extraction_enabled: bool = True
    local_solver_llama_base_url: str = "http://localhost:8080"
    local_solver_llama_model: str = "unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL"
    local_solver_llama_timeout_seconds: float = 20.0
    local_llama_startup_probe_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    local_llama_unavailable_cooldown_seconds: float = Field(default=60.0, gt=0, le=600)
    local_llama_geometry_timeout_seconds: float = 30.0
    local_llama_geometry_max_tokens: int = 1_200
    database_url: str = "sqlite:///./intomath.db"
    cors_origins: str = Field(default="http://localhost:3000")

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def geogebra_catalog_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "geogebra_commands.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
