import asyncio
from types import SimpleNamespace

from app.core.model_policy import HARD_MODEL, NVIDIA_GPT_OSS_20B_MODEL
from app.schemas.common import Difficulty, ProblemType
from app.services.fallback_solver import FallbackSolver
from app.services.solver_service import SolverService


def test_twenty_b_model_recovers_solution_after_large_model_timeout() -> None:
    class CompletionClient:
        enabled = True

        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        async def complete_json(self, **kwargs: object) -> dict:
            self.requests.append(kwargs)
            operation = kwargs.get("operation")
            model = kwargs.get("model")
            if operation == "structured_math_solution" and model == HARD_MODEL:
                raise TimeoutError("The large model timed out.")
            if operation == "structured_math_solution":
                return {
                    "answer": {
                        "text": "The two angles are supplementary.",
                        "latex": "\\angle KIL + \\angle YPX = 180^\\circ",
                    },
                    "confidence": 0.8,
                    "warnings": [],
                }
            if operation == "structured_math_steps_repair":
                return {
                    "steps": [
                        {
                            "index": index,
                            "title": f"Geometry step {index}",
                            "explanation": f"Apply the required relation in stage {index}.",
                            "latex": [],
                        }
                        for index in range(1, 4)
                    ]
                }
            raise AssertionError(f"Unexpected operation: {operation}")

    service = SolverService.__new__(SolverService)
    completion_client = CompletionClient()
    service.nvidia_client = completion_client
    service.settings = SimpleNamespace(
        remote_model_attempt_timeout_seconds=25.0,
        nvidia_large_model_attempt_timeout_seconds=50.0,
        structured_solution_max_tokens=4_500,
        missing_step_repair_max_tokens=2_000,
        content_repair_max_tokens=2_500,
    )
    service.fallback_solver = FallbackSolver()

    draft = asyncio.run(
        service._solve_structured(
            text="Prove that the two angles are supplementary.",
            problem_type=ProblemType.geometry,
            difficulty=Difficulty.hard,
            model=HARD_MODEL,
            subquestions=[],
            request_id="recovery-test",
        )
    )

    assert draft.answer.text == "The two angles are supplementary."
    assert len(draft.steps) == 3
    assert draft.solver_model.endswith(NVIDIA_GPT_OSS_20B_MODEL)
    assert any("preferred solver was unavailable" in item for item in draft.warnings)
    assert {
        request["operation"]: request["max_tokens"]
        for request in completion_client.requests
    } == {
        "structured_math_solution": 4_500,
        "structured_math_steps_repair": 2_000,
    }
