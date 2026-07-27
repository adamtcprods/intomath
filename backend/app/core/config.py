from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "IntoMath API"
    app_env: str = "development"
    app_debug: bool = True
    solve_request_timeout_seconds: float = Field(default=70.0, gt=0, le=300)
    max_solve_text_length: int = Field(default=20_000, ge=100, le=1_000_000)
    max_image_base64_length: int = Field(
        default=14_000_000, ge=1_024, le=100_000_000
    )
    max_decoded_image_bytes: int = Field(
        default=10_485_760, ge=1_024, le=50_000_000
    )
    max_image_width: int = Field(default=8_192, ge=1, le=32_768)
    max_image_height: int = Field(default=8_192, ge=1, le=32_768)
    response_cache_ttl_seconds: float = Field(default=900.0, gt=0, le=86_400)
    response_cache_max_size: int = Field(default=500, ge=1, le=100_000)
    ocr_cache_ttl_seconds: float = Field(default=3_600.0, gt=0, le=604_800)
    ocr_cache_max_size: int = Field(default=256, ge=1, le=10_000)
    remote_model_attempt_timeout_seconds: float = Field(default=25.0, gt=0, le=120)
    nvidia_large_model_attempt_timeout_seconds: float = Field(
        default=50.0, gt=0, le=120
    )
    structured_solution_max_tokens: int = Field(default=4_500, ge=500, le=10_000)
    geometry_extraction_max_tokens: int = Field(default=1_200, ge=200, le=4_000)
    geometry_repair_max_tokens: int = Field(default=800, ge=100, le=3_000)
    missing_step_repair_max_tokens: int = Field(default=2_000, ge=200, le=5_000)
    content_repair_max_tokens: int = Field(default=2_500, ge=200, le=6_000)
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
    local_router_llama_model: str = "unsloth/LFM2.5-8B-A1B-GGUF:Q4_K_XL"
    local_router_llama_timeout_seconds: float = Field(default=30.0, gt=0, le=60)
    local_router_llama_max_tokens: int = Field(default=300, ge=64, le=1_000)
    local_solver_llama_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    local_llama_startup_probe_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    local_llama_unavailable_cooldown_seconds: float = Field(default=60.0, gt=0, le=600)
    local_llama_geometry_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    local_llama_geometry_max_tokens: int = Field(default=1_200, ge=200, le=4_000)
    semantic_router_enabled: bool = True
    semantic_router_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    semantic_router_model_path: str = ""
    semantic_router_artifact_path: str = ""
    semantic_router_device: str = "cpu"
    semantic_router_max_text_chars: int = Field(default=4_000, ge=100, le=100_000)
    semantic_router_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    semantic_router_min_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    semantic_router_min_raw_similarity: float = Field(default=0.20, ge=-1.0, le=1.0)
    semantic_router_term_min_similarity: float = Field(default=0.55, ge=-1.0, le=1.0)
    semantic_router_fallback_to_llm: bool = True
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

    @property
    def semantic_router_data_path(self) -> Path:
        return BACKEND_DIR / "data" / "semantic_router"


@lru_cache
def get_settings() -> Settings:
    return Settings()
