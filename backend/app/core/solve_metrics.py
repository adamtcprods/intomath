"""Small request-scoped timing and model-attempt metrics abstraction."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator

_STAGES = (
    "ocr",
    "routing",
    "solver",
    "visualization",
    "persistence",
)


@dataclass
class SolveMetrics:
    request_id: str
    started_at: float = field(default_factory=time.monotonic)
    durations_ms: dict[str, float] = field(
        default_factory=lambda: {stage: 0.0 for stage in _STAGES}
    )
    model_call_count: int = 0
    cache_status: str = "not_checked"
    single_flight_role: str | None = None
    outcome: str = "started"
    attempt_counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    routing_source: str | None = None
    routing_abstention_reason: str | None = None
    routing_confidence: float | None = None
    routing_margin: float | None = None
    semantic_inference_count: int = 0
    semantic_routing_latency_ms: float = 0.0
    llm_routing_fallback_count: int = 0

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.durations_ms[stage] = self.durations_ms.get(stage, 0.0) + (
                time.monotonic() - started
            ) * 1000.0

    def record_model_attempt(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
    ) -> None:
        self.model_call_count += 1
        self.attempt_counts[(provider, model, operation)] += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "event": "solve_metrics",
            "request_id": self.request_id,
            "outcome": self.outcome,
            "total_duration_ms": round(
                (time.monotonic() - self.started_at) * 1000.0, 3
            ),
            **{
                f"{stage}_duration_ms": round(self.durations_ms[stage], 3)
                for stage in _STAGES
            },
            "model_call_count": self.model_call_count,
            "cache_status": self.cache_status,
            "single_flight_role": self.single_flight_role,
            "routing_source": self.routing_source,
            "routing_abstention_reason": self.routing_abstention_reason,
            "routing_confidence": self.routing_confidence,
            "routing_margin": self.routing_margin,
            "semantic_inference_count": self.semantic_inference_count,
            "semantic_routing_latency_ms": round(
                self.semantic_routing_latency_ms, 3
            ),
            "llm_routing_fallback_count": self.llm_routing_fallback_count,
            "attempt_counts": [
                {
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "count": count,
                }
                for (provider, model, operation), count in sorted(
                    self.attempt_counts.items()
                )
            ],
        }


_CURRENT_METRICS: ContextVar[SolveMetrics | None] = ContextVar(
    "intomath_solve_metrics",
    default=None,
)


def bind_solve_metrics(metrics: SolveMetrics) -> Token[SolveMetrics | None]:
    return _CURRENT_METRICS.set(metrics)


def reset_solve_metrics(token: Token[SolveMetrics | None]) -> None:
    _CURRENT_METRICS.reset(token)


def current_solve_metrics() -> SolveMetrics | None:
    return _CURRENT_METRICS.get()


def record_model_attempt(
    *,
    provider: str,
    model: str,
    operation: str,
) -> None:
    metrics = current_solve_metrics()
    if metrics is not None:
        metrics.record_model_attempt(
            provider=provider,
            model=model,
            operation=operation,
        )


def emit_solve_metrics(metrics: SolveMetrics, logger: logging.Logger) -> None:
    logger.info(
        "%s",
        json.dumps(metrics.as_dict(), separators=(",", ":"), sort_keys=True),
    )


__all__ = [
    "SolveMetrics",
    "bind_solve_metrics",
    "current_solve_metrics",
    "emit_solve_metrics",
    "record_model_attempt",
    "reset_solve_metrics",
]
