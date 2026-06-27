from app.schemas.common import Difficulty, ProblemType
from app.services.fallback_solver import FallbackSolver

VIETNAMESE_DIAMETER_ARC_PROOF = r"""
Cho tam giác (ABC) ((AB < AC)) nội tiếp đường tròn ((O;R)) có đường kính (BC).
Trên cung nhỏ (AC) lấy điểm (D). Đường thẳng (BD) cắt (AC) tại (E).
Từ (E) kẻ (EF \perp BC) tại (F).

a) Chứng minh tứ giác (BAEF) nội tiếp một đường tròn.
b) Chứng minh (DB) là phân giác của góc (ADF).
c) Gọi (M) là trung điểm của (EC). Chứng minh (DM \cdot CA = CF \cdot CO).
""".strip()


def test_fallback_solver_handles_linear_equation_with_variables_on_both_sides() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Solve 2x + 5 = 3x - 1",
        ProblemType.algebra,
        Difficulty.easy,
    )

    assert answer.text == "The solution is x = 6."
    assert answer.latex == "x = 6"
    assert confidence == 0.92
    assert warnings == []
    assert steps[0].latex == ["2x + 5 = 3x - 1", "-x = -6"]


def test_fallback_solver_handles_parenthesized_linear_equation() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Solve 2(x + 3) = 14",
        ProblemType.algebra,
        Difficulty.easy,
    )

    assert answer.latex == "x = 4"
    assert confidence == 0.92
    assert warnings == []
    assert steps[0].latex == ["2x + 6 = 14", "2x = 8"]


def test_fallback_solver_handles_linear_equation_with_division() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "x/2 + 3 = 7",
        ProblemType.algebra,
        Difficulty.easy,
    )

    assert answer.latex == "x = 8"
    assert confidence == 0.92
    assert warnings == []
    assert steps[0].latex == ["0.5x + 3 = 7", "0.5x = 4"]


def test_fallback_solver_handles_implicit_multiplication_in_arithmetic() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Calculate 2(3 + 4)",
        ProblemType.arithmetic,
        Difficulty.easy,
    )

    assert answer.text == "The value is 14."
    assert answer.latex == "14"
    assert confidence == 0.95
    assert warnings == []
    assert steps[0].title == "Evaluate the expression"


def test_fallback_solver_rejects_diameter_arc_geometry_proof_family() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        VIETNAMESE_DIAMETER_ARC_PROOF,
        ProblemType.geometry,
        Difficulty.hard,
    )

    assert confidence == 0.35
    assert any(
        "outside the local deterministic solver" in warning for warning in warnings
    )
    assert "DM * CA = CF * CO" not in answer.text
    assert steps[0].title == "Use a supported local pattern or enable the model backend"


def test_fallback_solver_limits_recursive_depth_in_arithmetic() -> None:
    # Construct a deeply nested arithmetic expression
    deep_expression = "1 + " * 40 + "1"
    answer, steps, confidence, warnings = FallbackSolver().solve(
        deep_expression,
        ProblemType.arithmetic,
        Difficulty.easy,
    )
    assert confidence == 0.35
    assert any("outside the local deterministic solver" in warning for warning in warnings)


def test_fallback_solver_prevents_exponent_overflow() -> None:
    # Huge exponent
    huge_exp = "2^9999999"
    answer, steps, confidence, warnings = FallbackSolver().solve(
        huge_exp,
        ProblemType.arithmetic,
        Difficulty.easy,
    )
    assert confidence == 0.35
    assert any("outside the local deterministic solver" in warning for warning in warnings)


def test_fallback_solver_limits_value_magnitude() -> None:
    # Magnitude greater than 1e50
    huge_val = "10^60"
    answer, steps, confidence, warnings = FallbackSolver().solve(
        huge_val,
        ProblemType.arithmetic,
        Difficulty.easy,
    )
    assert confidence == 0.35
    assert any("outside the local deterministic solver" in warning for warning in warnings)


def test_fallback_solver_limits_linear_recursion_depth() -> None:
    # Deeply nested linear equation
    deep_eq = "x" + " + 0" * 40 + " = 5"
    answer, steps, confidence, warnings = FallbackSolver().solve(
        deep_eq,
        ProblemType.algebra,
        Difficulty.easy,
    )
    assert confidence == 0.35
    assert any("outside the local deterministic solver" in warning for warning in warnings)


def test_fallback_solver_handles_basic_inequality() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Solve 2x + 3 < 7",
        ProblemType.algebra,
        Difficulty.easy,
    )
    assert confidence == 0.92
    assert answer.text == "The solution is x < 2."
    assert answer.latex == "x < 2"
    assert steps[0].latex == ["2x + 3 < 7", "2x < 4"]
    assert steps[1].latex == ["x < 2"]


def test_fallback_solver_flips_inequality_on_negative_division() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Solve -2x + 3 < 7",
        ProblemType.algebra,
        Difficulty.easy,
    )
    assert confidence == 0.92
    assert answer.text == "The solution is x > -2."
    assert answer.latex == "x > -2"
    assert steps[0].latex == ["-2x + 3 < 7", "-2x < 4"]
    assert "flip the inequality sign" in steps[1].explanation
    assert steps[1].latex == ["x > -2"]


def test_fallback_solver_flips_inequality_on_negative_division_equal_op() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Solve -x >= 5",
        ProblemType.algebra,
        Difficulty.easy,
    )
    assert confidence == 0.92
    assert answer.text == "The solution is x <= -5."
    assert answer.latex == "x <= -5"
    assert steps[0].latex == ["-x >= 5", "-x >= 5"]
    assert "flip the inequality sign" in steps[1].explanation
    assert steps[1].latex == ["x <= -5"]


def test_fallback_solver_handles_factored_quadratic() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Graph y = (x - 1)*(x - 2)",
        ProblemType.functions,
        Difficulty.easy,
    )
    assert confidence == 0.86
    # y = x^2 - 3x + 2 -> vertex at (-b/2a) = 3/2 = 1.5, y = 1.5^2 - 4.5 + 2 = -0.25
    assert "The vertex is at (1.5, -0.25)." in answer.text


def test_fallback_solver_handles_fractional_quadratic() -> None:
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Graph y = x^2 / 2 - 2",
        ProblemType.functions,
        Difficulty.easy,
    )
    assert confidence == 0.86
    # y = 0.5x^2 - 2 -> vertex at 0, -2
    assert "The vertex is at (0, -2)." in answer.text


def test_fallback_solver_rejects_invalid_math_syntax_fast() -> None:
    # Unbalanced parentheses
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Calculate 2 * (3 + 4",
        ProblemType.arithmetic,
        Difficulty.easy,
    )
    assert confidence == 0.35

    # Operator start/end
    answer, steps, confidence, warnings = FallbackSolver().solve(
        "Calculate *2 + 3",
        ProblemType.arithmetic,
        Difficulty.easy,
    )
    assert confidence == 0.35



