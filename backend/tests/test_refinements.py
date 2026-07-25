import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from app.schemas.solve import ProblemInput, SolveRequest, SolveResponse
from app.services.cache import TTLCache
from app.services.solver_service import SolverService
from app.services.solver_pipeline.orchestration import (
    _cached_response_is_usable,
    solve_request,
)
from fastapi.testclient import TestClient
from app.main import app

def test_problem_input_validation() -> None:
    # 1. Reject empty/blank text with no image
    with pytest.raises(ValidationError) as exc_info:
        ProblemInput(text="   ")
    assert "At least one of 'text' or 'image_base64' must be provided." in str(exc_info.value)

    # 2. Allow blank text if image is provided
    pi_with_img = ProblemInput(text="  ", image_base64="some_base64_data", image_mime_type="image/png")
    assert pi_with_img.image_mime_type == "image/png"

    # 3. Reject unsupported image_mime_type
    with pytest.raises(ValidationError) as exc_info:
        ProblemInput(text="Solve this", image_base64="some_base64_data", image_mime_type="image/bmp")
    assert "Unsupported image_mime_type" in str(exc_info.value)


def test_ttl_cache_refinements() -> None:
    # 1. Enforce max_size with eviction
    cache: TTLCache[str] = TTLCache(ttl_seconds=100, max_size=2)
    cache.set("key1", "val1")
    cache.set("key2", "val2")
    assert cache.get("key1") == "val1"
    assert cache.get("key2") == "val2"

    cache.set("key3", "val3")
    # key1 should have been evicted as it was the oldest
    assert cache.get("key1") is None
    assert cache.get("key2") == "val2"
    assert cache.get("key3") == "val3"

    # 2. test delete()
    cache.delete("key2")
    assert cache.get("key2") is None
    assert cache.get("key3") == "val3"

    # 3. test clear()
    cache.clear()
    assert cache.get("key3") is None


def test_cache_key_includes_language() -> None:
    service = SolverService()
    try:
        req_en = SolveRequest.model_validate({
            "input": {"text": "Solve 2+2", "language": "en"},
            "options": {"include_visualization": False}
        })
        req_vi = SolveRequest.model_validate({
            "input": {"text": "Solve 2+2", "language": "vi"},
            "options": {"include_visualization": False}
        })

        key_en = service._build_cache_key("Solve 2+2", req_en)
        key_vi = service._build_cache_key("Solve 2+2", req_vi)

        # Check they do not collide
        assert key_en != key_vi
    finally:
        asyncio.run(service.aclose())


def test_cache_key_includes_catalog_and_visualization_versions() -> None:
    service = SolverService.__new__(SolverService)
    service.translator = SimpleNamespace(
        registry=SimpleNamespace(
            metadata={
                "schema_version": "1.3",
                "generator_version": "1.3.0",
                "upstream_commit": "first",
            }
        )
    )
    request = SolveRequest.model_validate(
        {"input": {"text": "Visualize a 3D cube!"}}
    )

    first = service._build_cache_key(request.input.text, request)
    service.translator.registry.metadata["upstream_commit"] = "second"
    second = service._build_cache_key(request.input.text, request)

    assert first != second


def test_valid_cache_hit_returns_before_model_routing() -> None:
    class OCR:
        async def extract_problem_text(self, *_: object) -> None:
            return None

    class Router:
        async def route_async(self, *_: object, **__: object) -> None:
            raise AssertionError("routing must not run before a valid cache hit")

    service = SimpleNamespace(
        ocr_service=OCR(),
        router=Router(),
        _build_cache_key=lambda *_: "cube-cache-key",
    )
    request = SolveRequest.model_validate(
        {"input": {"text": "Visualize a 3D cube!"}}
    )
    cache: TTLCache[SolveResponse] = TTLCache(ttl_seconds=60)
    cache.set(
        "cube-cache-key",
        SolveResponse.model_validate(
            {
                "request_id": "cached-request",
                "problem_type": "geometry",
                "difficulty": "easy",
                "answer": {"text": "A cube."},
                "steps": [],
                "visualization": {"kind": "geogebra"},
                "confidence": 0.8,
                "routing": {
                    "parser_model": "cached",
                    "solver_model": "cached",
                    "visualization_environment": "graphics_3d",
                    "reason": "cached",
                },
            }
        ),
    )

    response = asyncio.run(solve_request(service, request, cache))

    assert response.cached is True
    assert response.request_id != "cached-request"


def test_missing_visualization_cache_entry_is_not_reusable() -> None:
    request = SolveRequest.model_validate(
        {"input": {"text": "Visualize a 3D cube!"}}
    )
    response = SolveResponse.model_validate(
        {
            "request_id": "missing-visualization",
            "problem_type": "algebra",
            "difficulty": "easy",
            "answer": {"text": "A cube."},
            "steps": [],
            "visualization": {"kind": "none"},
            "confidence": 0.8,
            "routing": {
                "parser_model": "fallback",
                "solver_model": "fallback",
                "visualization_environment": "graphics_3d",
                "reason": "fallback",
            },
        }
    )

    assert _cached_response_is_usable(request, request.input.text, response) is False


def test_unclassified_missing_visualization_cache_entry_is_not_reusable() -> None:
    request = SolveRequest.model_validate(
        {"input": {"text": "Visualize a tetrahedron!"}}
    )
    response = SolveResponse.model_validate(
        {
            "request_id": "unclassified-visualization",
            "problem_type": "general",
            "difficulty": "medium",
            "answer": {"text": "A tetrahedron."},
            "steps": [],
            "visualization": {"kind": "none"},
            "confidence": 0.8,
            "routing": {
                "parser_model": "fallback",
                "solver_model": "fallback",
                "visualization_environment": None,
                "reason": "local router unavailable; subject and visualization environment left unclassified",
            },
        }
    )

    assert _cached_response_is_usable(request, request.input.text, response) is False


def test_geometry_without_visualization_cache_entry_is_not_reusable() -> None:
    request = SolveRequest.model_validate(
        {"input": {"text": "Prove a geometry theorem."}}
    )
    response = SolveResponse.model_validate(
        {
            "request_id": "geometry-without-visualization",
            "problem_type": "geometry",
            "difficulty": "hard",
            "answer": {"text": "Proof."},
            "steps": [],
            "visualization": {"kind": "none"},
            "confidence": 0.8,
            "routing": {
                "parser_model": "router",
                "solver_model": "solver",
                "visualization_environment": None,
                "reason": "classified by local router",
            },
        }
    )

    assert _cached_response_is_usable(request, request.input.text, response) is False


def test_classified_nonvisual_cache_entry_remains_reusable() -> None:
    request = SolveRequest.model_validate(
        {"input": {"text": "What is 2 + 2?"}}
    )
    response = SolveResponse.model_validate(
        {
            "request_id": "classified-nonvisual",
            "problem_type": "arithmetic",
            "difficulty": "easy",
            "answer": {"text": "4"},
            "steps": [],
            "visualization": {"kind": "none"},
            "confidence": 1,
            "routing": {
                "parser_model": "router",
                "solver_model": "solver",
                "visualization_environment": None,
                "reason": "classified as arithmetic; no interactive view is useful",
            },
        }
    )

    assert _cached_response_is_usable(request, request.input.text, response) is True


def test_health_endpoint_success() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["db"] in ("ok", "error")
    assert data["version"] == "2.0.0"


def test_solve_endpoint_validation_error() -> None:
    with TestClient(app) as client:
        # Send empty body
        response = client.post("/api/v1/solve", json={"input": {"text": "   "}})
    assert response.status_code == 422
    assert "At least one of" in response.text
