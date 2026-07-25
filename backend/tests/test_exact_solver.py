import pytest

from app.schemas.common import Difficulty, ProblemType
from app.schemas.geometry_dsl import VisualizationEnvironment
from app.services.exact_solver import try_solve_exact


def test_exact_numeric_arithmetic_derives_route_without_a_model() -> None:
    result = try_solve_exact("12 * (3 + 4) - 5")

    assert result is not None
    assert result.problem_type is ProblemType.arithmetic
    assert result.difficulty is Difficulty.easy
    assert result.solve_route == "deterministic"
    assert result.normalized_prompt == "12 * (3 + 4) - 5"
    assert result.visualization_environment is None
    assert result.local_result.answer.latex == "79"


def test_exact_linear_equation_and_inequality_are_supported() -> None:
    equation = try_solve_exact("Solve 2(x + 3) = 14")
    inequality = try_solve_exact("-2x + 3 < 7")

    assert equation is not None
    assert equation.local_result.answer.latex == "x = 4"
    assert equation.problem_type is ProblemType.algebra
    assert inequality is not None
    assert inequality.local_result.answer.latex == "x > -2"


def test_exact_quadratic_graph_derives_graphing_environment() -> None:
    result = try_solve_exact("Graph f(x) = (x - 1) * (x - 2)")

    assert result is not None
    assert result.problem_type is ProblemType.algebra
    assert result.difficulty is Difficulty.medium
    assert result.visualization_environment is VisualizationEnvironment.graphing
    assert result.visualization_search_terms == ("Function", "Vertex", "Root")
    assert result.normalized_prompt == "Graph y = (x - 1) * (x - 2)"


@pytest.mark.parametrize(
    "prompt",
    [
        "Graph y = 2x + 4",  # linear graphs are not in the exact grammar
        "Factor x^2 - 5x + 6",
        "Solve x + y = 3 and x - y = 1",
        "Solve x + 1 = 2 or x + 3 = 4",
        "A taxi costs 3 + 2*5 dollars. What is the fare?",
        "Prove that x + 1 = 1 + x",
        "Construct triangle ABC with AB = 3.",
        "2 + 2\n3 + 3",
        "Calculate 2 + 2 and explain why",
        "What is 2 + 2 or 3 + 3?",
        "2 * pi",
    ],
)
def test_exact_solver_rejects_unsupported_or_ambiguous_text(prompt: str) -> None:
    assert try_solve_exact(prompt) is None


@pytest.mark.parametrize(
    "prompt",
    [
        "__import__('os').system('echo unsafe')",
        "(lambda: 1)()",
        "open('/etc/passwd').read()",
        "2 + getattr(object, '__class__')",
        "[1, 2, 3][0] + 1",
        "2 ** 9999999",
        "10^60",
        "1 / 0",
        "x.__class__ = 1",
        "x + 1 = __import__('os').system('id')",
    ],
)
def test_exact_solver_rejects_malicious_or_unsafe_expressions(prompt: str) -> None:
    assert try_solve_exact(prompt) is None
