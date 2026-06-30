import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.schemas.solve import ProblemInput, SolveRequest
from app.services.cache import TTLCache
from app.services.solver_service import SolverService
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
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        service = SolverService(db)
        
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
        db.close()


def test_health_endpoint_success() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["db"] in ("ok", "error")
    assert data["version"] == "2.0.0"


def test_solve_endpoint_validation_error() -> None:
    client = TestClient(app)
    # Send empty body
    response = client.post("/api/v1/solve", json={"input": {"text": "   "}})
    assert response.status_code == 422
    assert "At least one of" in response.text
