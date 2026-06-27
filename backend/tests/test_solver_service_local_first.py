import asyncio
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models.problem_attempt import ProblemAttempt  # noqa: F401
from app.db.models.solver_run import SolverRun  # noqa: F401
from app.db.models.visualization_artifact import VisualizationArtifact  # noqa: F401
from app.schemas.solve import SolveRequest
from app.services.model_router import HARD_MODEL, LOCAL_DETERMINISTIC_SOLVER_MODEL
from app.services.solver_service import SolverService


class FailingModelClient:
    enabled = True

    async def complete_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "OpenRouter should not be called for local-supported prompts"
        )


class ProofModelClient:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "answer": {
                "text": "The geometry proof requires model-backed reasoning.",
                "latex": None,
            },
            "steps": [
                {
                    "index": 1,
                    "title": "Use model-backed proof reasoning",
                    "explanation": "The local deterministic solver does not contain hardcoded theorem proofs.",
                }
            ],
            "parts": [
                {
                    "label": "a",
                    "answer": {"text": "Model proof for part a.", "latex": None},
                    "steps": _model_proof_steps("a"),
                },
                {
                    "label": "b",
                    "answer": {"text": "Model proof for part b.", "latex": None},
                    "steps": _model_proof_steps("b"),
                },
                {
                    "label": "c",
                    "answer": {"text": "Model proof for part c.", "latex": None},
                    "steps": _model_proof_steps("c"),
                },
            ],
            "confidence": 0.81,
            "warnings": [],
        }


def _model_proof_steps(label: str) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "title": f"Model step {label}{index}",
            "explanation": f"Detailed model-backed proof step {index} for part {label}.",
        }
        for index in range(1, 4)
    ]


def test_solver_service_routes_supported_prompt_to_local_solver_first() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        service = SolverService(db)
        service.client = FailingModelClient()  # type: ignore[assignment]

        response = asyncio.run(
            service.solve(
                SolveRequest.model_validate(
                    {
                        "input": {"text": "Solve 2x + 5 = 3x - 1"},
                        "options": {"include_visualization": False},
                    }
                )
            )
        )
    finally:
        db.close()

    assert response.answer.latex == "x = 6"
    assert response.routing.solver_model == LOCAL_DETERMINISTIC_SOLVER_MODEL
    assert "deterministic local solver used" in response.routing.reason
    assert response.warnings == []


def test_solver_service_routes_geometry_proof_to_model_instead_of_local_bypass() -> (
    None
):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    client = ProofModelClient()
    try:
        service = SolverService(db)
        service.client = client  # type: ignore[assignment]

        response = asyncio.run(
            service.solve(
                SolveRequest.model_validate(
                    {
                        "input": {
                            "text": (
                                "Cho tam giác ABC nội tiếp đường tròn O có đường "
                                "kính BC. Trên cung nhỏ AC lấy điểm D. Đường "
                                "thẳng BD cắt AC tại E. Từ E kẻ EF vuông góc BC "
                                "tại F.\n\n"
                                "a) Chứng minh tứ giác BAEF nội tiếp một đường "
                                "tròn.\n"
                                "b) Chứng minh DB là phân giác của góc ADF.\n"
                                "c) Gọi M là trung điểm của EC. Chứng minh "
                                "DM * CA = CF * CO."
                            )
                        },
                        "options": {"include_visualization": False},
                    }
                )
            )
        )
    finally:
        db.close()

    assert client.calls == 1
    assert response.routing.solver_model == HARD_MODEL
    assert response.routing.solver_model != LOCAL_DETERMINISTIC_SOLVER_MODEL
    assert "deterministic local solver used" not in response.routing.reason
    assert [part.label for part in response.parts] == ["a", "b", "c"]
    assert [part.answer.text for part in response.parts] == [
        "Model proof for part a.",
        "Model proof for part b.",
        "Model proof for part c.",
    ]
    assert all(len(part.steps) == 3 for part in response.parts)
