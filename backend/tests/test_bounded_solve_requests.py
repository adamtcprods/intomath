from __future__ import annotations

import base64
import asyncio
import io
import logging
import threading
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.v1.endpoints.solve import router as solve_router
from app.core.config import Settings
from app.core.model_policy import StructuredModelEndpoint
from app.dependencies import get_solver_service
from app.schemas.geometry_dsl import GeoGebraValidationIssue, VisualizationEnvironment
from app.services.geometry_extractor import GeometryExtractor
from app.services.solver_service import SolverService
from app.services.solver_pipeline.content_quality import structured_content_issues
from app.services.solver_pipeline.content_repair import repair_structured_content
from app.services.solver_pipeline.strict_models import validate_structured_payload


def _service_with_settings(**overrides: Any) -> SolverService:
    service = SolverService.__new__(SolverService)
    values = {
        "solve_request_timeout_seconds": 1.0,
        "max_solve_text_length": 20_000,
        "max_image_base64_length": 14_000_000,
        "max_decoded_image_bytes": 10_485_760,
        "max_image_width": 8_192,
        "max_image_height": 8_192,
    }
    values.update(overrides)
    service.settings = SimpleNamespace(**values)
    return service


def _client_for_service(service: SolverService) -> TestClient:
    app = FastAPI()
    app.include_router(solve_router, prefix="/api/v1")
    app.dependency_overrides[get_solver_service] = lambda: service
    return TestClient(app)


def _png_base64(width: int = 1, height: int = 1) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_request(payload: str) -> dict[str, Any]:
    return {
        "input": {
            "image_base64": payload,
            "image_mime_type": "image/png",
        }
    }


def test_total_request_timeout_returns_controlled_504_and_logs_stage(
    caplog: Any,
) -> None:
    cancelled = threading.Event()

    class BlockingOCR:
        async def extract_problem_text(self, *_: object) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    service = _service_with_settings(solve_request_timeout_seconds=0.02)
    service.ocr_service = BlockingOCR()

    with caplog.at_level(logging.ERROR):
        with _client_for_service(service) as client:
            response = client.post(
                "/api/v1/solve",
                json={"input": {"text": "Solve 2 + 2."}},
            )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "The solve request exceeded its time limit. Please try again."
    }
    assert cancelled.is_set()
    assert "stage=ocr" in caplog.text
    assert "request_id=" in caplog.text
    assert "CancelledError" not in response.text


def test_oversized_text_returns_413() -> None:
    service = _service_with_settings(max_solve_text_length=5)
    with _client_for_service(service) as client:
        response = client.post(
            "/api/v1/solve",
            json={"input": {"text": "123456"}},
        )
    assert response.status_code == 413


def test_oversized_base64_input_returns_413() -> None:
    service = _service_with_settings(max_image_base64_length=12)
    with _client_for_service(service) as client:
        response = client.post(
            "/api/v1/solve",
            json=_image_request(_png_base64()),
        )
    assert response.status_code == 413


def test_invalid_base64_returns_422() -> None:
    service = _service_with_settings()
    with _client_for_service(service) as client:
        response = client.post(
            "/api/v1/solve",
            json=_image_request("not-valid-base64!"),
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid Base64 image data."


def test_oversized_decoded_image_returns_413() -> None:
    service = _service_with_settings(max_decoded_image_bytes=16)
    with _client_for_service(service) as client:
        response = client.post(
            "/api/v1/solve",
            json=_image_request(_png_base64()),
        )
    assert response.status_code == 413


def test_excessive_image_dimensions_return_413() -> None:
    service = _service_with_settings(max_image_width=1, max_image_height=1)
    with _client_for_service(service) as client:
        response = client.post(
            "/api/v1/solve",
            json=_image_request(_png_base64(width=2, height=1)),
        )
    assert response.status_code == 413


def test_image_mime_mismatch_returns_422() -> None:
    service = _service_with_settings()
    request = _image_request(_png_base64())
    request["input"]["image_mime_type"] = "image/jpeg"
    with _client_for_service(service) as client:
        response = client.post("/api/v1/solve", json=request)
    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]


def test_default_operation_token_budgets_are_bounded() -> None:
    settings = Settings(_env_file=None)

    assert settings.local_router_llama_max_tokens == 300
    assert settings.local_llama_geometry_max_tokens == 1_200
    assert settings.geometry_extraction_max_tokens == 1_200
    assert settings.geometry_repair_max_tokens == 800
    assert settings.missing_step_repair_max_tokens == 2_000
    assert settings.content_repair_max_tokens == 2_500
    assert settings.structured_solution_max_tokens == 4_500


def test_content_repair_supplies_its_operation_token_budget() -> None:
    class CompletionClient:
        request: dict[str, Any]

        async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
            self.request = kwargs
            return repaired

    original = validate_structured_payload(
        {
            "answer": {"text": "x = 2", "latex": "x=2"},
            "steps": [
                {
                    "index": 1,
                    "title": "Solve",
                    "explanation": "Wait, maybe x = 2? Let me check.",
                    "latex": ["x=2"],
                }
            ],
            "parts": [],
            "confidence": 0.8,
            "warnings": [],
        },
        request_id="budget-test",
        provider="test",
        model="test",
    )
    repaired = {
        **original,
        "steps": [
            {
                **original["steps"][0],
                "explanation": "Solving the equation gives x = 2.",
            }
        ],
    }
    client = CompletionClient()
    result = asyncio.run(
        repair_structured_content(
            original,
            structured_content_issues(original),
            completion_client=client,
            candidate=StructuredModelEndpoint("test", "test", "test"),
            timeout_seconds=1.0,
            request_id="budget-test",
        )
    )

    assert result["steps"][0]["explanation"] == "Solving the equation gives x = 2."
    assert client.request["operation"] == "structured_math_solution_repair"
    assert client.request["max_tokens"] == 2_500


def test_geometry_operations_supply_separate_token_budgets() -> None:
    class CompletionClient:
        requests: list[dict[str, Any]]

        def __init__(self) -> None:
            self.requests = []

        async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
            self.requests.append(kwargs)
            if kwargs["operation"] == "geometry_repair":
                return {
                    "replacements": [
                        {
                            "action_index": 0,
                            "action": {"action": "CREATE_POINT", "label": "A"},
                        }
                    ]
                }
            return {
                "summary": "Point A",
                "dsl": {
                    "version": "1.1",
                    "space": "euclidean_2d",
                    "environment": "geometry_2d",
                    "actions": [{"action": "CREATE_POINT", "label": "A"}],
                    "render_hints": {},
                },
            }

    settings = SimpleNamespace(
        geometry_extraction_max_tokens=1_200,
        geometry_repair_max_tokens=800,
        remote_model_attempt_timeout_seconds=1.0,
        nvidia_large_model_attempt_timeout_seconds=1.0,
    )
    extractor = GeometryExtractor(
        llama_client=SimpleNamespace(),
        settings=settings,
        nvidia_client=SimpleNamespace(enabled=False),
    )
    client = CompletionClient()
    asyncio.run(
        extractor._extract_with_llm(
            "Create point A.",
            "openai/gpt-oss-20b",
            VisualizationEnvironment.geometry_2d,
            (),
            completion_client=client,
            provider_name="test",
        )
    )
    dsl, _ = extractor._parse_geometry_payload(
        {
            "summary": "Invalid line",
            "dsl": {
                "version": "1.1",
                "space": "euclidean_2d",
                "environment": "geometry_2d",
                "actions": [
                    {
                        "action": "CREATE_LINE",
                        "label": "lineAB",
                        "points": ["A", "B"],
                    }
                ],
                "render_hints": {},
            },
        },
        environment=VisualizationEnvironment.geometry_2d,
        source="test",
        allowed_command_names=set(),
    )
    asyncio.run(
        extractor._repair_remote_dsl(
            dsl,
            (
                GeoGebraValidationIssue(
                    code="unknown_reference",
                    action_index=0,
                    message="A and B are missing.",
                ),
            ),
            parser_model="openai/gpt-oss-20b",
            environment=VisualizationEnvironment.geometry_2d,
            retrieved=(),
            completion_client=client,
            provider_name="test",
        )
    )

    assert {
        request["operation"]: request["max_tokens"] for request in client.requests
    } == {
        "geometry_extraction": 1_200,
        "geometry_repair": 800,
    }
