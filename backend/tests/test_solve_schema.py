import pytest

from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolvePart, SolveStep
from app.services.fallback_solver import FallbackSolver
from app.services.solver_service import SolverService


def test_solve_step_coerces_model_string_shape_mistakes() -> None:
    step = SolveStep.model_validate(
        {
            "index": 1,
            "title": ["Use Thales' theorem"],
            "explanation": ["Because BC is a diameter, angle BAC is right."],
            "why_it_happens": ["An angle subtending a diameter is 90 degrees."],
            "common_mistakes": "Forgetting to cite the diameter condition.",
            "alternative_approaches": "Use directed angles.",
            "hints": "Look for right angles.",
            "exam_tip": ["State Thales' theorem explicitly."],
            "latex": "\\angle BAC = 90^\\circ",
        }
    )

    assert step.title == "Use Thales' theorem"
    assert step.explanation == "Because BC is a diameter, angle BAC is right."
    assert step.why_it_happens == "An angle subtending a diameter is 90 degrees."
    assert step.common_mistakes == ["Forgetting to cite the diameter condition."]
    assert step.alternative_approaches == ["Use directed angles."]
    assert step.hints == ["Look for right angles."]
    assert step.exam_tip == "State Thales' theorem explicitly."
    assert step.latex == ["\\angle BAC = 90^\\circ"]


def test_solve_answer_coerces_latex_list_to_string() -> None:
    answer = SolveAnswer.model_validate(
        {"text": ["Part a proved", "Part b proved"], "latex": ["x=1", "y=2"]}
    )

    assert answer.text == "Part a proved\nPart b proved"
    assert answer.latex == "x=1\ny=2"


def test_solve_schema_strips_latex_delimiters() -> None:
    answer = SolveAnswer.model_validate({"text": "x = 6", "latex": "$x = 6$"})
    step = SolveStep.model_validate(
        {
            "index": 1,
            "title": "Solve",
            "explanation": "Move constants.",
            "latex": ["$$2x=12$$", "\\(x=6\\)", "\\[x=6\\]"],
        }
    )

    assert answer.latex == "x = 6"
    assert step.latex == ["2x=12", "x=6", "x=6"]


def test_solve_part_coerces_label_and_question() -> None:
    part = SolvePart.model_validate(
        {
            "label": ["1"],
            "question": ["Prove the cyclic quadrilateral."],
            "answer": {"text": "Proved."},
            "steps": [],
        }
    )

    assert part.label == "1"
    assert part.question == "Prove the cyclic quadrilateral."


def test_solver_service_detects_and_normalizes_subquestions() -> None:
    service = SolverService.__new__(SolverService)

    subquestions = service._detect_subquestions(
        "Given a triangle.\n1) First proof.\nii) Second proof.\n(c) Third proof."
    )

    assert [subquestion.label for subquestion in subquestions] == ["a", "b", "c"]
    assert [subquestion.question for subquestion in subquestions] == [
        "First proof.",
        "Second proof.",
        "Third proof.",
    ]


def test_solver_service_normalizes_model_parts_to_expected_subquestions() -> None:
    service = SolverService.__new__(SolverService)
    subquestions = service._detect_subquestions("1) First task.\n2) Second task.")

    draft = service._draft_from_payload(
        {
            "answer": {"text": "Both parts are solved."},
            "steps": [
                {
                    "index": 1,
                    "title": "Overview",
                    "explanation": "Solve parts in order.",
                }
            ],
            "parts": [
                {
                    "label": "1",
                    "answer": {"text": "First solved."},
                    "steps": [
                        {
                            "index": 3,
                            "title": "First",
                            "explanation": "Do the first task.",
                        }
                    ],
                },
                {
                    "label": "ii",
                    "answer": {"text": "Second solved."},
                    "steps": [
                        {
                            "index": 7,
                            "title": "Second",
                            "explanation": "Do the second task.",
                        }
                    ],
                },
            ],
            "confidence": 0.8,
            "warnings": [],
        },
        expected_subquestions=subquestions,
    )

    assert [part.label for part in draft.parts] == ["a", "b"]
    assert [part.question for part in draft.parts] == ["First task.", "Second task."]
    assert [part.steps[0].index for part in draft.parts] == [1, 1]


def test_solver_service_rejects_too_concise_proof_parts() -> None:
    service = SolverService.__new__(SolverService)
    subquestions = service._detect_subquestions("1) Prove first.\n2) Prove second.")

    with pytest.raises(ValueError, match="too few steps"):
        service._draft_from_payload(
            {
                "answer": {"text": "Both parts are solved."},
                "steps": [
                    {
                        "index": 1,
                        "title": "Overview",
                        "explanation": "Solve parts in order.",
                    }
                ],
                "parts": [
                    {
                        "answer": {"text": "First solved."},
                        "steps": [
                            {
                                "index": 1,
                                "title": "First",
                                "explanation": "Do the proof in one step.",
                            }
                        ],
                    },
                    {
                        "answer": {"text": "Second solved."},
                        "steps": [
                            {
                                "index": 1,
                                "title": "Second",
                                "explanation": "Do the proof in one step.",
                            }
                        ],
                    },
                ],
                "confidence": 0.8,
                "warnings": [],
            },
            expected_subquestions=subquestions,
        )


def test_solver_service_fallback_builds_steps_for_each_subquestion() -> None:
    service = SolverService.__new__(SolverService)
    service.fallback_solver = FallbackSolver()
    subquestions = service._detect_subquestions("1) 2x + 5 = 17\n2) 3x - 6 = 0")

    draft = service._fallback_draft_for_subquestions(
        text="1) 2x + 5 = 17\n2) 3x - 6 = 0",
        problem_type=ProblemType.algebra,
        difficulty=Difficulty.easy,
        subquestions=subquestions,
    )

    assert [part.label for part in draft.parts] == ["a", "b"]
    assert all(part.steps for part in draft.parts)
    assert [part.steps[0].index for part in draft.parts] == [1, 1]
