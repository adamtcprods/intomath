from app.schemas.common import Difficulty, ProblemType
from types import SimpleNamespace

from app.services.model_router import (
    EASY_MODEL,
    HARD_MODEL,
    ModelRouter,
    remote_model_timeout_seconds,
    structured_model_endpoints,
)

VIETNAMESE_GEOMETRY_PROOF = r"""
Cho tam giác (ABC) ((AB < AC)) nội tiếp đường tròn ((O;R)) có đường kính (BC).
Trên cung nhỏ (AC) lấy điểm (D). Đường thẳng (BD) cắt (AC) tại (E).
Từ (E) kẻ (EF \perp BC) tại (F).

Chứng minh tứ giác (BAEF) nội tiếp một đường tròn.
""".strip()


def test_router_classifies_vietnamese_geometry_proof_as_hard() -> None:
    routing = ModelRouter().route(VIETNAMESE_GEOMETRY_PROOF, has_image=False)

    assert routing.problem_type is ProblemType.geometry
    assert routing.difficulty is Difficulty.hard
    assert routing.solver_model == HARD_MODEL


def test_router_classifies_gcd_integer_problem_as_number_theory() -> None:
    problem = (
        "Determine all pairs (a, b) of positive integers for which there exist "
        "positive integers g and N such that gcd(a^n+b, b^n+a) = g holds for "
        "all integers n ≥ N."
    )

    routing = ModelRouter().route(problem, has_image=False)

    assert routing.problem_type is ProblemType.number_theory
    assert routing.difficulty is Difficulty.hard
    assert routing.solver_model == HARD_MODEL


def test_structured_endpoints_use_only_the_gpt_oss_models() -> None:
    endpoints = structured_model_endpoints(HARD_MODEL)

    assert [endpoint.model for endpoint in endpoints] == [HARD_MODEL, EASY_MODEL]


def test_only_gpt_oss_120b_receives_the_large_model_timeout() -> None:
    settings = SimpleNamespace(
        remote_model_attempt_timeout_seconds=25.0,
        nvidia_large_model_attempt_timeout_seconds=50.0,
    )

    assert (
        remote_model_timeout_seconds(
            settings, provider="nvidia_direct", model=HARD_MODEL
        )
        == 50.0
    )
    assert (
        remote_model_timeout_seconds(
            settings, provider="nvidia_direct", model=EASY_MODEL
        )
        == 25.0
    )
