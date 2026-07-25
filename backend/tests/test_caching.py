from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models.solver_run import SolverRun
from app.repositories.result_repository import ResultRepository
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolveRequest, SolveStep
from app.services.cache import AsyncSingleFlight, TTLCache
from app.services.local_solver_types import LocalSolveResult
from app.services.model_router import RoutingDecision
from app.services.ocr_service import OCRResult, OCRService
from app.services.solver_pipeline.orchestration import solve_request


class _NoOCR:
    async def extract_problem_text(self, *_: object) -> None:
        return None


class _Router:
    async def route_async(self, *_: object, **__: object) -> RoutingDecision:
        return RoutingDecision(
            problem_type=ProblemType.arithmetic,
            difficulty=Difficulty.easy,
            parser_model="test-parser",
            solver_model="test-solver",
            vision_model=None,
            visualization_environment=None,
            reason="test route",
        )


class _CountingSelector:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.fail = False
        self.cancelled = False

    async def solve_if_supported(self, text: str, *_: object) -> LocalSolveResult:
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise RuntimeError("shared solve failed")
        return LocalSolveResult(
            answer=SolveAnswer(text="4", latex="4"),
            steps=[
                SolveStep(
                    index=1,
                    title="Add",
                    explanation="Add the two values.",
                    latex=["2+2=4"],
                )
            ],
            confidence=1.0,
            warnings=[],
            normalized_text=text,
            problem_type=ProblemType.arithmetic,
            reason="matched a test expression",
        )


class _SolveService:
    def __init__(
        self,
        selector: _CountingSelector,
        *,
        result_repository: ResultRepository | None = None,
    ) -> None:
        self.settings = SimpleNamespace(
            solve_request_timeout_seconds=2.0,
            max_solve_text_length=20_000,
            max_image_base64_length=14_000_000,
            max_decoded_image_bytes=10_485_760,
            max_image_width=8_192,
            max_image_height=8_192,
        )
        self.result_repository = result_repository
        self.ocr_service = _NoOCR()
        self.router = _Router()
        self.local_solver_selector = selector

    def _build_cache_key(self, *_: object) -> str:
        return "identical-response-key"

    def _with_local_solver_routing(
        self,
        routing: RoutingDecision,
        *_: object,
        **__: object,
    ) -> RoutingDecision:
        return routing

    def _without_backend_config_warnings(self, warnings: list[str]) -> list[str]:
        return warnings


def _request() -> SolveRequest:
    return SolveRequest.model_validate(
        {
            "input": {"text": "What is 2 + 2?"},
            "options": {"include_visualization": False},
        }
    )


def test_ocr_cache_hit_avoids_repeated_ocr() -> None:
    class LocalOCR:
        model_id = "test-ocr-v1"

        def __init__(self) -> None:
            self.calls = 0

        async def extract_text(self, **_: Any) -> str:
            self.calls += 1
            return "2 + 2"

    async def run() -> tuple[OCRResult | None, OCRResult | None, int]:
        local_ocr = LocalOCR()
        service = OCRService(
            local_ocr=local_ocr,
            cache=TTLCache(ttl_seconds=60, max_size=2),
        )
        first = await service.extract_problem_text(b"decoded-image", "image/png")
        second = await service.extract_problem_text(b"decoded-image", "image/png")
        return first, second, local_ocr.calls

    first, second, calls = asyncio.run(run())

    assert calls == 1
    assert first is not None and first.cached is False
    assert second is not None and second.cached is True
    assert second.cleaned_text == "2 + 2"


def test_concurrent_identical_requests_share_one_solve_computation() -> None:
    async def run() -> tuple[object, object, int]:
        selector = _CountingSelector()
        selector.release.clear()
        service = _SolveService(selector)
        cache = TTLCache(ttl_seconds=60, max_size=10)
        single_flight = AsyncSingleFlight()

        leader = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await selector.entered.wait()
        waiter = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await asyncio.sleep(0)
        selector.release.set()
        return await leader, await waiter, selector.calls

    leader_response, waiter_response, calls = asyncio.run(run())

    assert calls == 1
    assert leader_response is not waiter_response
    assert leader_response.request_id != waiter_response.request_id
    assert leader_response.cached is False
    assert waiter_response.cached is False


def test_cancelling_one_waiter_does_not_cancel_shared_computation() -> None:
    async def run() -> tuple[int, bool, str]:
        selector = _CountingSelector()
        selector.release.clear()
        service = _SolveService(selector)
        cache = TTLCache(ttl_seconds=60, max_size=10)
        single_flight = AsyncSingleFlight()

        leader = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await selector.entered.wait()
        waiter = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        selector.release.set()
        response = await leader
        return selector.calls, selector.cancelled, response.status

    calls, computation_cancelled, status = asyncio.run(run())

    assert calls == 1
    assert computation_cancelled is False
    assert status == "ok"


def test_shared_computation_failure_is_removed_from_registry() -> None:
    async def run() -> tuple[list[object], int, int, str]:
        selector = _CountingSelector()
        selector.fail = True
        selector.release.clear()
        service = _SolveService(selector)
        cache = TTLCache(ttl_seconds=60, max_size=10)
        single_flight = AsyncSingleFlight()

        leader = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await selector.entered.wait()
        waiter = asyncio.create_task(
            solve_request(service, _request(), cache, single_flight)
        )
        await asyncio.sleep(0)
        selector.release.set()
        results = await asyncio.gather(
            leader,
            waiter,
            return_exceptions=True,
        )
        count_after_failure = await single_flight.in_flight_count()
        selector.fail = False
        response = await solve_request(service, _request(), cache, single_flight)
        return results, count_after_failure, selector.calls, response.status

    results, count_after_failure, calls, status = asyncio.run(run())

    assert all(isinstance(result, RuntimeError) for result in results)
    assert count_after_failure == 0
    assert calls == 2
    assert status == "ok"


def test_ttl_cache_expiration() -> None:
    async def run() -> str | None:
        cache = TTLCache[str](ttl_seconds=0.01, max_size=2)
        cache.set("key", "value")
        await asyncio.sleep(0.02)
        return cache.get("key")

    assert asyncio.run(run()) is None


def test_ttl_cache_maximum_size_evicts_least_recently_used() -> None:
    cache = TTLCache[str](ttl_seconds=60, max_size=2)
    cache.set("first", "1")
    cache.set("second", "2")
    assert cache.get("first") == "1"

    cache.set("third", "3")

    assert cache.get("first") == "1"
    assert cache.get("second") is None
    assert cache.get("third") == "3"


def test_cache_hit_replaces_request_id_and_returns_independent_response() -> None:
    async def run() -> tuple[object, object]:
        selector = _CountingSelector()
        service = _SolveService(selector)
        cache = TTLCache(ttl_seconds=60, max_size=10)
        single_flight = AsyncSingleFlight()
        first = await solve_request(service, _request(), cache, single_flight)
        second = await solve_request(service, _request(), cache, single_flight)
        return first, second

    first, second = asyncio.run(run())

    assert first.request_id != second.request_id
    assert first is not second
    assert first.cached is False
    assert second.cached is True


def test_cache_hits_are_persisted_as_cached_runs() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        selector = _CountingSelector()
        service = _SolveService(
            selector,
            result_repository=ResultRepository(session_factory),
        )
        cache = TTLCache(ttl_seconds=60, max_size=10)
        single_flight = AsyncSingleFlight()

        async def run() -> None:
            await solve_request(service, _request(), cache, single_flight)
            await solve_request(service, _request(), cache, single_flight)

        asyncio.run(run())
        with session_factory() as session:
            runs = list(
                session.scalars(select(SolverRun).order_by(SolverRun.created_at))
            )

        assert len(runs) == 2
        assert [run.cached for run in runs] == [False, True]
        assert runs[0].request_id != runs[1].request_id
        assert selector.calls == 1
    finally:
        engine.dispose()
