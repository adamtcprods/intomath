import asyncio
import time
from types import SimpleNamespace
from typing import Any

from app.integrations.errors import (
    IntegrationFailureCategory,
    IntegrationRequestError,
)
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.geometry_extractor import (
    GEOMETRY_FAILURE_ACTIONS,
    GeometryAttemptOutcome,
    GeometryExtractor,
    GeometryFailureAction,
    GeometryFailureCategory,
    GeometryOperation,
    GeometryProvider,
)
from app.services.model_router import EASY_MODEL, HARD_MODEL


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        solve_request_timeout_seconds=5.0,
        local_llama_geometry_extraction_enabled=True,
        local_llama_geometry_timeout_seconds=0.5,
        local_llama_geometry_max_tokens=1_200,
        remote_model_attempt_timeout_seconds=0.5,
        nvidia_large_model_attempt_timeout_seconds=0.5,
        geometry_extraction_max_tokens=1_200,
        geometry_repair_max_tokens=800,
    )


def _triangle_payload() -> dict[str, Any]:
    return {
        "summary": "Triangle ABC",
        "dsl": {
            "version": "1.1",
            "space": "euclidean_2d",
            "environment": "geometry_2d",
            "actions": [
                {"action": "CREATE_POINT", "label": "A"},
                {"action": "CREATE_POINT", "label": "B"},
                {"action": "CREATE_POINT", "label": "C"},
                {
                    "action": "CREATE_POLYGON",
                    "label": "polyABC",
                    "points": ["A", "B", "C"],
                },
            ],
            "render_hints": {},
        },
    }


def _invalid_line_payload() -> dict[str, Any]:
    return {
        "summary": "Line AB",
        "dsl": {
            "version": "1.1",
            "space": "euclidean_2d",
            "environment": "geometry_2d",
            "actions": [
                {"action": "CREATE_POINT", "label": "A"},
                {"action": "CREATE_POINT", "label": "B"},
                {
                    "action": "CREATE_LINE",
                    "label": "AB",
                    "points": ["A", "missing"],
                },
            ],
            "render_hints": {},
        },
    }


class FakeLocalClient:
    model = "test-local-geometry"

    def __init__(
        self,
        response: dict[str, Any] | Exception,
        *,
        enabled: bool = True,
        available: bool = True,
    ) -> None:
        self.response = response
        self.enabled = enabled
        self.available = available
        self.calls = 0

    async def generate_json(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeNvidiaClient:
    enabled = True

    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        key = (str(kwargs["model"]), str(kwargs["operation"]))
        self.calls.append(key)
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response


def _remote_error(
    model: str,
    category: IntegrationFailureCategory,
    *,
    status_code: int | None = None,
) -> IntegrationRequestError:
    return IntegrationRequestError(
        f"{category.value} failure",
        provider="NVIDIA",
        model=model,
        operation="geometry_extraction",
        status_code=status_code,
        failure_category=category,
    )


def _extract(
    local: FakeLocalClient,
    remote: FakeNvidiaClient,
    *,
    parser_model: str = EASY_MODEL,
    request_deadline: float | None = None,
):
    extractor = GeometryExtractor(
        llama_client=local,
        settings=_settings(),
        nvidia_client=remote,
    )
    return asyncio.run(
        extractor.extract(
            "Construct triangle ABC.",
            parser_model,
            environment=VisualizationEnvironment.geometry_2d,
            request_deadline=request_deadline,
        )
    )


def test_local_success_makes_no_remote_call() -> None:
    local = FakeLocalClient(_triangle_payload())
    remote = FakeNvidiaClient({})

    result = _extract(local, remote)

    assert result.dsl.actions
    assert local.calls == 1
    assert remote.calls == []
    assert [attempt.provider for attempt in result.attempts] == [
        GeometryProvider.local_llama
    ]


def test_invalid_local_schema_uses_one_remote_fallback() -> None:
    local = FakeLocalClient({})
    remote = FakeNvidiaClient(
        {(EASY_MODEL, GeometryOperation.extraction.value): _triangle_payload()}
    )

    result = _extract(local, remote)

    assert result.dsl.actions
    assert local.calls == 1
    assert remote.calls == [(EASY_MODEL, GeometryOperation.extraction.value)]
    assert result.attempts[0].failure_category is GeometryFailureCategory.invalid_schema


def test_http_429_uses_alternate_without_retrying_same_model() -> None:
    local = FakeLocalClient({}, enabled=False)
    remote = FakeNvidiaClient(
        {
            (
                EASY_MODEL,
                GeometryOperation.extraction.value,
            ): _remote_error(
                EASY_MODEL,
                IntegrationFailureCategory.rate_limit,
                status_code=429,
            ),
            (HARD_MODEL, GeometryOperation.extraction.value): _triangle_payload(),
        }
    )

    result = _extract(local, remote)

    assert result.dsl.actions
    assert remote.calls == [
        (EASY_MODEL, GeometryOperation.extraction.value),
        (HARD_MODEL, GeometryOperation.extraction.value),
    ]
    assert result.attempts[1].failure_category is GeometryFailureCategory.rate_limit


def test_timeout_uses_alternate_without_retrying_same_model() -> None:
    local = FakeLocalClient({}, enabled=False)
    remote = FakeNvidiaClient(
        {
            (
                EASY_MODEL,
                GeometryOperation.extraction.value,
            ): _remote_error(EASY_MODEL, IntegrationFailureCategory.timeout),
            (HARD_MODEL, GeometryOperation.extraction.value): _triangle_payload(),
        }
    )

    result = _extract(local, remote)

    assert result.dsl.actions
    assert remote.calls == [
        (EASY_MODEL, GeometryOperation.extraction.value),
        (HARD_MODEL, GeometryOperation.extraction.value),
    ]


def test_invalid_dsl_triggers_at_most_one_action_scoped_repair() -> None:
    local = FakeLocalClient({}, enabled=False)
    remote = FakeNvidiaClient(
        {
            (EASY_MODEL, GeometryOperation.extraction.value): _invalid_line_payload(),
            (
                EASY_MODEL,
                GeometryOperation.repair.value,
            ): {
                "replacements": [
                    {
                        "action_index": 2,
                        "action": {
                            "action": "CREATE_LINE",
                            "label": "AB",
                            "points": ["A", "B"],
                        },
                    }
                ]
            },
        }
    )
    extractor = GeometryExtractor(
        llama_client=local,
        settings=_settings(),
        nvidia_client=remote,
    )

    result = asyncio.run(
        extractor.extract(
            "Construct line AB.",
            EASY_MODEL,
            environment=VisualizationEnvironment.geometry_2d,
        )
    )

    assert result.dsl.actions
    assert remote.calls == [
        (EASY_MODEL, GeometryOperation.extraction.value),
        (EASY_MODEL, GeometryOperation.repair.value),
    ]
    assert sum(
        attempt.operation is GeometryOperation.repair for attempt in result.attempts
    ) == 1


def test_failed_repair_uses_one_alternate_source() -> None:
    local = FakeLocalClient({}, enabled=False)
    remote = FakeNvidiaClient(
        {
            (EASY_MODEL, GeometryOperation.extraction.value): _invalid_line_payload(),
            (EASY_MODEL, GeometryOperation.repair.value): {},
            (HARD_MODEL, GeometryOperation.extraction.value): _triangle_payload(),
        }
    )

    result = _extract(local, remote)

    assert result.dsl.actions
    assert remote.calls == [
        (EASY_MODEL, GeometryOperation.extraction.value),
        (EASY_MODEL, GeometryOperation.repair.value),
        (HARD_MODEL, GeometryOperation.extraction.value),
    ]


def test_duplicate_provider_model_operation_attempts_are_impossible() -> None:
    local = FakeLocalClient({}, enabled=False)
    remote = FakeNvidiaClient(
        {
            (
                EASY_MODEL,
                GeometryOperation.extraction.value,
            ): _remote_error(EASY_MODEL, IntegrationFailureCategory.connectivity),
            (
                HARD_MODEL,
                GeometryOperation.extraction.value,
            ): _remote_error(HARD_MODEL, IntegrationFailureCategory.connectivity),
        }
    )

    result = _extract(local, remote)
    keys = [
        (attempt.provider, attempt.model, attempt.operation)
        for attempt in result.attempts
    ]

    assert len(keys) == len(set(keys))
    assert all(
        attempt.outcome is not GeometryAttemptOutcome.succeeded
        for attempt in result.attempts
    )


def test_exhausted_deadline_stops_before_remote_call() -> None:
    local = FakeLocalClient(_triangle_payload())
    remote = FakeNvidiaClient({})

    result = _extract(
        local,
        remote,
        request_deadline=time.monotonic() - 0.01,
    )

    assert result.dsl.actions == []
    assert local.calls == 0
    assert remote.calls == []
    assert result.attempts[0].failure_category is GeometryFailureCategory.deadline_exceeded


def test_failure_action_matrix_is_explicit_for_required_transitions() -> None:
    assert GEOMETRY_FAILURE_ACTIONS[
        (GeometryProvider.local_llama, GeometryFailureCategory.local_unavailable)
    ] is GeometryFailureAction.preferred_remote
    assert GEOMETRY_FAILURE_ACTIONS[
        (GeometryProvider.local_llama, GeometryFailureCategory.invalid_schema)
    ] is GeometryFailureAction.preferred_remote
    for category in (
        GeometryFailureCategory.rate_limit,
        GeometryFailureCategory.timeout,
        GeometryFailureCategory.connectivity,
        GeometryFailureCategory.invalid_json,
        GeometryFailureCategory.invalid_schema,
    ):
        assert GEOMETRY_FAILURE_ACTIONS[
            (GeometryProvider.nvidia, category)
        ] is GeometryFailureAction.alternate_remote
    assert GEOMETRY_FAILURE_ACTIONS[
        (GeometryProvider.nvidia, GeometryFailureCategory.invalid_dsl)
    ] is GeometryFailureAction.repair_once
