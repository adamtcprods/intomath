from app.schemas.common import Difficulty, ProblemType
from app.services.model_router import HARD_MODEL, ModelRouter

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
