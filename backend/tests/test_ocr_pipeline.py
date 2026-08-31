import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import solve as solve_endpoint
from app.schemas.solve import SolveRequest
from app.services.cache import TTLCache
from app.services.model_router import ModelRouter
from app.services.ocr_service import (
    OCRInputError,
    OCRResult,
    OCRUnavailableError,
)
from app.services.solver_pipeline import orchestration
from app.services.solver_pipeline.orchestration import solve_request
from app.services.solver_service import SolverService


class DisabledLlamaClient:
    enabled = False
    available = False
    model = "disabled"


class DisabledNvidiaClient:
    enabled = False


class RouterMustNotRun:
    async def route_async(self, *_: object, **__: object) -> None:
        raise AssertionError("routing must not run after an image-only OCR failure")


def _image_only_request() -> SolveRequest:
    return SolveRequest.model_validate(
        {
            "input": {
                "image_base64": "aW1hZ2U=",
                "image_mime_type": "image/png",
            },
            "options": {"include_visualization": False},
        }
    )


def test_image_only_ocr_exception_stops_before_routing() -> None:
    class FailingOCR:
        async def extract_problem_text(self, *_: object) -> None:
            raise OCRUnavailableError("OCR is temporarily unavailable.")

    service = SimpleNamespace(ocr_service=FailingOCR(), router=RouterMustNotRun())

    with pytest.raises(OCRUnavailableError, match="OCR is temporarily unavailable"):
        asyncio.run(solve_request(service, _image_only_request(), TTLCache()))


def test_image_only_blank_ocr_result_stops_before_routing() -> None:
    class BlankOCR:
        async def extract_problem_text(self, *_: object) -> OCRResult:
            return OCRResult(raw_text="   ", cleaned_text="", confidence=0.0)

    service = SimpleNamespace(ocr_service=BlankOCR(), router=RouterMustNotRun())

    with pytest.raises(OCRInputError, match="could not extract usable text"):
        asyncio.run(solve_request(service, _image_only_request(), TTLCache()))


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (OCRInputError("OCR could not extract usable text."), 422),
        (OCRUnavailableError("OCR is temporarily unavailable."), 503),
    ],
)
def test_solve_endpoint_returns_ocr_specific_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    class FailingSolverService:
        def __init__(self, _: object) -> None:
            pass

        async def solve(self, _: SolveRequest) -> None:
            raise error

    monkeypatch.setattr(solve_endpoint, "SolverService", FailingSolverService)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(solve_endpoint.solve_problem(_image_only_request(), db=object()))

    assert exc_info.value.status_code == expected_status
    assert "OCR" in exc_info.value.detail


def test_image_and_text_requests_do_not_share_cached_ocr_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingImageOCR:
        async def extract_problem_text(
            self, image_base64: str | None, _: str | None
        ) -> None:
            if image_base64:
                raise OCRUnavailableError("OCR is temporarily unavailable.")
            return None

    monkeypatch.setattr(orchestration, "persist_solve_result", lambda *_: None)
    llama_client = DisabledLlamaClient()
    service = SolverService(
        db=SimpleNamespace(),
        nvidia_client=DisabledNvidiaClient(),
        llama_client=llama_client,
        router=ModelRouter(llama_client),
        ocr_service=FailingImageOCR(),
    )
    cache = TTLCache(ttl_seconds=60)
    image_request = SolveRequest.model_validate(
        {
            "input": {
                "text": "2+2",
                "image_base64": "aW1hZ2U=",
                "image_mime_type": "image/png",
            },
            "options": {"include_visualization": False},
        }
    )
    text_request = SolveRequest.model_validate(
        {
            "input": {"text": "2+2"},
            "options": {"include_visualization": False},
        }
    )

    image_response = asyncio.run(solve_request(service, image_request, cache))
    text_response = asyncio.run(solve_request(service, text_request, cache))

    assert image_response.routing.vision_model is not None
    assert any("OCR" in warning for warning in image_response.warnings)
    assert text_response.cached is False
    assert text_response.routing.vision_model is None
    assert not any("OCR" in warning for warning in text_response.warnings)
    assert "visual input" not in text_response.routing.reason
