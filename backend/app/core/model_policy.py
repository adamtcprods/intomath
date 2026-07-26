"""Provider-neutral model names, endpoint order, timeouts, and failure labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

EASY_MODEL = "openai/gpt-oss-20b"
HARD_MODEL = "openai/gpt-oss-120b"

NVIDIA_GPT_OSS_120B_MODEL = HARD_MODEL
NVIDIA_GPT_OSS_20B_MODEL = EASY_MODEL
NVIDIA_DIRECT_FALLBACK_MODELS = (
    NVIDIA_GPT_OSS_120B_MODEL,
    NVIDIA_GPT_OSS_20B_MODEL,
)
NVIDIA_GPT_OSS_MODELS = frozenset(NVIDIA_DIRECT_FALLBACK_MODELS)
NVIDIA_LARGE_MODELS = frozenset({NVIDIA_GPT_OSS_120B_MODEL})
NVIDIA_DIRECT_ROUTING_PREFIX = "nvidia-direct:"

LOCAL_DETERMINISTIC_SOLVER_MODEL = "local:deterministic-solver"
LOCAL_SAFE_FALLBACK_MODEL = "local:safe-fallback"
LOCAL_LLAMA_GEOMETRY_PARSER_MODEL = "local:llama-geometry-parser"
LOCAL_LLAMA_TRIVIA_MODEL = "local:llama-trivia"
VISION_MODEL = "local:deepseek-ai/deepseek-ocr-2"


class SolveRoute(str, Enum):
    deterministic = "deterministic"
    local_trivia = "local_trivia"
    remote = "remote"


class ModelFailureCategory(str, Enum):
    rate_limited = "rate_limited"
    provider_unavailable = "structured_output_provider_unavailable"
    invalid_json = "raw_parse_failure"
    invalid_schema = "schema_validation_failure"
    timeout = "timeout"
    request_failure = "request_failure"


@dataclass(frozen=True)
class StructuredModelEndpoint:
    provider: str
    model: str
    routing_model: str


def structured_model_endpoints(
    preferred_model: str,
) -> tuple[StructuredModelEndpoint, ...]:
    """Return the bounded, de-duplicated endpoint order for a structured solve."""
    if preferred_model not in {EASY_MODEL, HARD_MODEL}:
        raise ValueError(f"Unsupported structured solver model: {preferred_model}")
    alternate_model = EASY_MODEL if preferred_model == HARD_MODEL else HARD_MODEL
    ordered_models = (preferred_model, alternate_model, *NVIDIA_DIRECT_FALLBACK_MODELS)
    return tuple(
        StructuredModelEndpoint(
            provider="nvidia_direct",
            model=native_model,
            routing_model=f"{NVIDIA_DIRECT_ROUTING_PREFIX}{native_model}",
        )
        for native_model in dict.fromkeys(ordered_models)
    )


def remote_model_timeout_seconds(
    settings: Any,
    *,
    provider: str,
    model: str,
) -> float:
    """Select the per-attempt timeout without depending on routing services."""
    if provider == "nvidia_direct" and model in NVIDIA_LARGE_MODELS:
        return float(
            getattr(settings, "nvidia_large_model_attempt_timeout_seconds", 50.0)
        )
    return float(getattr(settings, "remote_model_attempt_timeout_seconds", 25.0))


__all__ = [
    "EASY_MODEL",
    "HARD_MODEL",
    "LOCAL_DETERMINISTIC_SOLVER_MODEL",
    "LOCAL_LLAMA_GEOMETRY_PARSER_MODEL",
    "LOCAL_LLAMA_TRIVIA_MODEL",
    "LOCAL_SAFE_FALLBACK_MODEL",
    "ModelFailureCategory",
    "NVIDIA_DIRECT_FALLBACK_MODELS",
    "NVIDIA_DIRECT_ROUTING_PREFIX",
    "NVIDIA_GPT_OSS_120B_MODEL",
    "NVIDIA_GPT_OSS_20B_MODEL",
    "NVIDIA_GPT_OSS_MODELS",
    "SolveRoute",
    "StructuredModelEndpoint",
    "VISION_MODEL",
    "remote_model_timeout_seconds",
    "structured_model_endpoints",
]
