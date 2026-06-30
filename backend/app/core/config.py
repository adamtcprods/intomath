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
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = "IntoMath 2.0"
    openrouter_site_url: str = "http://localhost:3000"
    deepseek_ocr_model_id: str = "deepseek-ai/deepseek-ocr-2"
    local_solver_first: bool = True
    local_solver_llama_detection_enabled: bool = True
    local_solver_llama_trivia_enabled: bool = True
    local_solver_llama_base_url: str = "http://localhost:8080"
    local_solver_llama_model: str = "hf.co/unsloth/LiquidAI/LFM2.5-350M-GGUF"
    local_solver_llama_timeout_seconds: float = 4.0
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
