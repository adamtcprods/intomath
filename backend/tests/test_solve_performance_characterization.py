from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from types import SimpleNamespace

from PIL import Image

from app.core.model_policy import EASY_MODEL, SolveRoute
from app.core.solve_metrics import record_model_attempt
from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.schemas.solve import SolveAnswer, SolveRequest, SolveStep
from app.services.cache import AsyncSingleFlight, TTLCache
from app.services.exact_solver import try_solve_exact
from app.services.model_router import RoutingDecision
from app.services.ocr_service import OCRResult
from app.services.solver_pipeline.orchestration import solve_request
from app.services.solver_pipeline.response_builder import StructuredSolveDraft


class _NoOCR:
    async def extract_problem_text(self, *_: object) -> None:
        return None


class _VisualOCR:
    async def extract_problem_text(self, *_: object) -> OCRResult:
        record_model_attempt(
            provider="local",
            model="test-ocr",
            operation="ocr",
        )
        return OCRResult(
            raw_text="Graph y = x^2 - 4x + 3",
            cleaned_text="Graph y = x^2 - 4x + 3",
            confidence=1.0,
        )


class _ScriptedRouter:
    def __init__(
        self,
        *,
        environment: VisualizationEnvironment | None = None,
        count_model_call: bool = True,
    ) -> None:
        self.environment = environment
        self.count_model_call = count_model_call

    async def route_async(self, text: str, *, has_image: bool) -> RoutingDecision:
        if self.count_model_call:
            record_model_attempt(
                provider="llama.cpp",
                model="test-router",
                operation="local_unified_routing",
            )
        return RoutingDecision(
            problem_type=(
                ProblemType.geometry
                if self.environment is not None
                else ProblemType.calculus
            ),
            difficulty=Difficulty.medium,
            parser_model=EASY_MODEL,
            solver_model=EASY_MODEL,
            vision_model="test-vision" if has_image else None,
            visualization_environment=self.environment,
            reason="scripted characterization route",
            solve_route=SolveRoute.remote,
            normalized_prompt=text,
        )


class _VisualizationExtractor:
    settings = SimpleNamespace(app_debug=False)

    async def extract(self, *_: object, **__: object) -> None:
        record_model_attempt(
            provider="nvidia_direct",
            model=EASY_MODEL,
            operation="geometry_extraction",
        )
        raise RuntimeError("characterization stops after the measured model call")


class _CharacterizationService:
    def __init__(
        self,
        *,
        router: _ScriptedRouter | None = None,
        solve_delay_seconds: float = 0.0,
    ) -> None:
        self.settings = SimpleNamespace(
            solve_request_timeout_seconds=2.0,
            max_solve_text_length=20_000,
            max_image_base64_length=14_000_000,
            max_decoded_image_bytes=10_485_760,
            max_image_width=8_192,
            max_image_height=8_192,
        )
        self.result_repository = None
        self.ocr_service = _NoOCR()
        self.router = router or _ScriptedRouter()
        self.local_solver_selector = SimpleNamespace()
        self.geometry_extractor = _VisualizationExtractor()
        self.solve_delay_seconds = solve_delay_seconds

    @staticmethod
    def try_solve_exact(text: str):
        return try_solve_exact(text)

    @staticmethod
    def _build_cache_key(normalized_text: str, request: SolveRequest) -> str:
        return f"{normalized_text}:{request.options.include_visualization}"

    async def _solve_structured(self, **_: object) -> StructuredSolveDraft:
        if self.solve_delay_seconds:
            await asyncio.sleep(self.solve_delay_seconds)
        record_model_attempt(
            provider="nvidia_direct",
            model=EASY_MODEL,
            operation="structured_math_solution",
        )
        return StructuredSolveDraft(
            answer=SolveAnswer(text="Characterized solution.", latex=None),
            steps=[
                SolveStep(
                    index=1,
                    title="Solve",
                    explanation="Use the selected structured solver.",
                )
            ],
            confidence=0.8,
            warnings=[],
            parts=[],
            solver_model=f"nvidia-direct:{EASY_MODEL}",
        )


def _request(
    text: str,
    *,
    include_visualization: bool = False,
) -> SolveRequest:
    return SolveRequest.model_validate(
        {
            "input": {"text": text},
            "options": {"include_visualization": include_visualization},
        }
    )


def _visual_image_request() -> SolveRequest:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buffer, format="PNG")
    return SolveRequest.model_validate(
        {
            "input": {
                "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "image_mime_type": "image/png",
            },
            "options": {"include_visualization": True},
        }
    )


def _metrics(caplog) -> list[dict[str, object]]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith('{"attempt_counts"')
        and '"event":"solve_metrics"' in record.message
    ]


def test_deterministic_arithmetic_has_zero_model_calls(caplog) -> None:
    service = _CharacterizationService()
    with caplog.at_level(logging.INFO):
        response = asyncio.run(
            solve_request(
                service,  # type: ignore[arg-type]
                _request("12 * (3 + 4) - 5"),
                TTLCache(ttl_seconds=60, max_size=10),
                AsyncSingleFlight(),
            )
        )

    [measurement] = _metrics(caplog)
    assert response.answer.text == "The value is 79."
    assert {
        "total_duration_ms",
        "ocr_duration_ms",
        "routing_duration_ms",
        "solver_duration_ms",
        "visualization_duration_ms",
        "persistence_duration_ms",
        "model_call_count",
        "cache_status",
        "attempt_counts",
    } <= measurement.keys()
    assert measurement["model_call_count"] == 0
    assert measurement["cache_status"] == "miss"
    assert measurement["total_duration_ms"] >= 0
    assert measurement["solver_duration_ms"] >= 0


def test_remote_nonvisual_normally_uses_one_or_two_model_calls(caplog) -> None:
    service = _CharacterizationService()
    with caplog.at_level(logging.INFO):
        asyncio.run(
            solve_request(
                service,  # type: ignore[arg-type]
                _request("Find the indefinite integral of sin(x^2)."),
                TTLCache(ttl_seconds=60, max_size=10),
                AsyncSingleFlight(),
            )
        )

    [measurement] = _metrics(caplog)
    assert 1 <= measurement["model_call_count"] <= 2
    assert measurement["model_call_count"] == 2
    assert measurement["routing_duration_ms"] >= 0


def test_visual_path_normally_uses_two_model_calls(caplog) -> None:
    service = _CharacterizationService()
    service.ocr_service = _VisualOCR()
    with caplog.at_level(logging.INFO):
        asyncio.run(
            solve_request(
                service,  # type: ignore[arg-type]
                _visual_image_request(),
                TTLCache(ttl_seconds=60, max_size=10),
                AsyncSingleFlight(),
            )
        )

    [measurement] = _metrics(caplog)
    assert measurement["model_call_count"] == 2
    operations = {
        attempt["operation"] for attempt in measurement["attempt_counts"]
    }
    assert operations == {"ocr", "geometry_extraction"}
    assert measurement["visualization_duration_ms"] >= 0


def test_cache_hit_has_zero_model_calls(caplog) -> None:
    service = _CharacterizationService()
    cache = TTLCache(ttl_seconds=60, max_size=10)
    request = _request("Find the indefinite integral of sin(x^2).")
    with caplog.at_level(logging.INFO):
        asyncio.run(
            solve_request(
                service,  # type: ignore[arg-type]
                request,
                cache,
                AsyncSingleFlight(),
            )
        )
        cached = asyncio.run(
            solve_request(
                service,  # type: ignore[arg-type]
                request,
                cache,
                AsyncSingleFlight(),
            )
        )

    measurements = _metrics(caplog)
    assert cached.cached is True
    assert measurements[-1]["cache_status"] == "hit"
    assert measurements[-1]["model_call_count"] == 0


def test_concurrent_identical_requests_share_attempts(caplog) -> None:
    service = _CharacterizationService(solve_delay_seconds=0.02)
    cache = TTLCache(ttl_seconds=60, max_size=10)
    single_flight = AsyncSingleFlight()
    request = _request("Find the indefinite integral of sin(x^2).")

    async def run() -> None:
        await asyncio.gather(
            solve_request(
                service,  # type: ignore[arg-type]
                request,
                cache,
                single_flight,
            ),
            solve_request(
                service,  # type: ignore[arg-type]
                request,
                cache,
                single_flight,
            ),
        )

    with caplog.at_level(logging.INFO):
        asyncio.run(run())

    measurements = _metrics(caplog)
    assert len(measurements) == 2
    assert {item["single_flight_role"] for item in measurements} == {
        "leader",
        "waiter",
    }
    assert sum(item["model_call_count"] for item in measurements) == 2
    leader = next(
        item for item in measurements if item["single_flight_role"] == "leader"
    )
    assert all(attempt["count"] == 1 for attempt in leader["attempt_counts"])
