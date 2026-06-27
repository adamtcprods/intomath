from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import cast

from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveAnswer, SolveStep


@dataclass(frozen=True)
class LinearEquation:
    raw: str
    left_a: float
    left_b: float
    right_a: float
    right_b: float
    net_a: float
    net_c: float
    solution: float
    op: str = "="
    solution_op: str = "="


@dataclass(frozen=True)
class LinearExpression:
    coefficient: float
    constant: float


@dataclass(frozen=True)
class QuadraticFunction:
    expression: str
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class QuadraticExpression:
    a: float
    b: float
    c: float


class FallbackSolver:
    def solve(
        self, text: str, problem_type: ProblemType, difficulty: Difficulty
    ) -> tuple[SolveAnswer, list[SolveStep], float, list[str]]:
        _ = (problem_type, difficulty)
        text = text.strip()
        warnings: list[str] = []

        linear = self._try_linear_equation(text)
        if linear is not None:
            return (*self._build_linear_response(linear), 0.92, warnings)

        quadratic = self._try_quadratic_graph(text)
        if quadratic is not None:
            answer_text, steps, latex_payload = self._build_quadratic_response(
                quadratic
            )
            return (
                SolveAnswer(
                    text=answer_text, latex=latex_payload[0] if latex_payload else None
                ),
                steps,
                0.86,
                warnings,
            )

        geometry = self._try_geometry_construction(text)
        if geometry is not None:
            answer, steps = geometry
            return answer, steps, 0.72, warnings

        arithmetic = self._try_arithmetic(text)
        if arithmetic is not None:
            value = arithmetic
            value_text = self._format_number(value)
            return (
                SolveAnswer(text=f"The value is {value_text}.", latex=value_text),
                [
                    SolveStep(
                        index=1,
                        title="Evaluate the expression",
                        explanation="Compute the arithmetic expression using the standard order of operations.",
                        why_it_happens="Parentheses, exponents, multiplication/division, then addition/subtraction determine the result.",
                        common_mistakes=["Evaluating addition before multiplication."],
                        hints=["Work inside parentheses first."],
                        exam_tip="Rewrite the expression neatly before simplifying.",
                        latex=[value_text],
                    )
                ],
                0.95,
                warnings,
            )

        warnings.append(
            "This prompt is outside the local deterministic solver. Configure OPENROUTER_API_KEY for full model-backed solving and OCR."
        )
        answer = SolveAnswer(
            text="I could not produce a complete solution with the local solver for this prompt. Try a typed arithmetic expression, a linear equation like 2(x + 3) = 14 or x/2 + 3 = 7, a quadratic graph like y = x^2 - 4x + 3, or configure the model backend for broader coverage.",
            latex=None,
        )
        steps = [
            SolveStep(
                index=1,
                title="Use a supported local pattern or enable the model backend",
                explanation="The request was received and classified, but it did not match one of the deterministic local solving patterns.",
                why_it_happens="Without an OpenRouter API key, IntoMath avoids inventing an answer for unsupported prompts.",
                common_mistakes=[
                    "Trusting a guessed answer when the solver has not matched a known pattern."
                ],
                hints=[
                    "For local solving, type a supported equation plainly, for example: 2(x + 3) = 14."
                ],
                exam_tip="Prefer a clear problem statement over a long prompt when using deterministic tools.",
            )
        ]
        return answer, steps, 0.35, warnings

    def _try_arithmetic(self, text: str) -> float | int | None:
        expression = self._extract_arithmetic_expression(text)
        if expression is None or not self._is_likely_valid_math_syntax(expression):
            return None
        cleaned = self._prepare_math_ast_expression(expression)
        try:
            tree = ast.parse(cleaned, mode="eval")
            value = self._safe_eval(tree.body)
        except Exception:
            return None
        return int(value) if float(value).is_integer() else round(float(value), 6)

    def _safe_eval(self, node: ast.AST, depth: int = 0) -> float:
        if depth > 30:
            raise ValueError("Expression too deep")

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = float(node.value)
            if abs(val) > 1e50:
                raise ValueError("Constant value too large")
            return val

        if isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, depth + 1)
            right = self._safe_eval(node.right, depth + 1)
            if abs(left) > 1e50 or abs(right) > 1e50:
                raise ValueError("Operand too large")

            if isinstance(node.op, ast.Add):
                res = left + right
            elif isinstance(node.op, ast.Sub):
                res = left - right
            elif isinstance(node.op, ast.Mult):
                res = left * right
            elif isinstance(node.op, ast.Div):
                if math.isclose(right, 0.0):
                    raise ZeroDivisionError("Division by zero")
                res = left / right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > 1000:
                    raise ValueError("Exponent too large")
                if abs(left) > 1e50 and right > 1:
                    raise ValueError("Base too large")
                try:
                    res = math.pow(left, right)
                except (OverflowError, ZeroDivisionError, ValueError):
                    raise ValueError("Invalid power operation or overflow")
            else:
                raise ValueError("Unsupported operation")

            if abs(res) > 1e50:
                raise ValueError("Result overflow")
            return res

        if isinstance(node, ast.UnaryOp):
            value = self._safe_eval(node.operand, depth + 1)
            if abs(value) > 1e50:
                raise ValueError("Operand too large")
            if isinstance(node.op, ast.USub):
                res = -value
            elif isinstance(node.op, ast.UAdd):
                res = value
            else:
                raise ValueError("Unsupported unary operation")

            if abs(res) > 1e50:
                raise ValueError("Result overflow")
            return res

        raise ValueError("Unsupported arithmetic expression")

    def _try_linear_equation(self, text: str) -> LinearEquation | None:
        equation = self._extract_linear_equation(text)
        if not equation:
            return None

        # Find the comparison operator
        op = None
        for candidate_op in ["<=", ">=", "<", ">", "="]:
            if candidate_op in equation:
                op = candidate_op
                break
        if op is None or equation.count(op) != 1:
            return None

        left_raw, right_raw = equation.split(op, 1)
        left = self._parse_linear_side(left_raw)
        right = self._parse_linear_side(right_raw)
        if left is None or right is None:
            return None

        left_a, left_b = left
        right_a, right_b = right
        net_a = left_a - right_a
        net_c = right_b - left_b
        if math.isclose(net_a, 0.0):
            return None

        # Determine solution operator
        solution_op = op
        if net_a < 0:
            if op == "<":
                solution_op = ">"
            elif op == ">":
                solution_op = "<"
            elif op == "<=":
                solution_op = ">="
            elif op == ">=":
                solution_op = "<="

        return LinearEquation(
            raw=equation,
            left_a=left_a,
            left_b=left_b,
            right_a=right_a,
            right_b=right_b,
            net_a=net_a,
            net_c=net_c,
            solution=net_c / net_a,
            op=op,
            solution_op=solution_op,
        )

    def _build_linear_response(
        self, linear: LinearEquation
    ) -> tuple[SolveAnswer, list[SolveStep]]:
        original_latex = (
            f"{self._format_linear_expression(linear.left_a, linear.left_b)} {linear.op} "
            f"{self._format_linear_expression(linear.right_a, linear.right_b)}"
        )
        net_a_text = self._format_number(linear.net_a)
        net_c_text = self._format_number(linear.net_c)
        solution_text = self._format_number(linear.solution)
        collected_latex = f"{self._format_linear_term(linear.net_a)} {linear.op} {net_c_text}"

        flip_notice = ""
        if linear.op != "=" and linear.net_a < 0:
            flip_notice = " Since we are dividing by a negative number, we must flip the inequality sign."

        steps = [
            SolveStep(
                index=1,
                title="Collect like terms",
                explanation=(
                    "Move all x-terms to the left and constants to the right. "
                    f"The inequality becomes {collected_latex}." if linear.op != "=" else
                    f"The equation becomes {collected_latex}."
                ),
                why_it_happens="Equivalent operations preserve the relation while simplifying the expression.",
                common_mistakes=[
                    "Changing the sign incorrectly when moving a term across the comparison operator."
                ] if linear.op != "=" else [
                    "Changing the sign incorrectly when moving a term across the equals sign."
                ],
                hints=["Keep x-terms and number terms in separate columns."],
                latex=[original_latex, collected_latex],
            ),
            SolveStep(
                index=2,
                title="Solve for x",
                explanation=f"Divide both sides by {net_a_text} to get x by itself.{flip_notice}",
                why_it_happens="Division by the coefficient reverses the multiplication attached to x.",
                common_mistakes=[
                    "Dividing only one side.",
                    "Forgetting to flip the inequality sign when dividing by a negative number."
                ] if linear.op != "=" else ["Dividing only one side of the equation."],
                hints=[
                    "Substitute a test value back into the original inequality to check the solution range."
                ] if linear.op != "=" else [
                    "Substitute the value back into the original equation to check it."
                ],
                exam_tip="Always remember to flip the inequality direction when multiplying or dividing by a negative number." if linear.op != "=" else "A quick substitution check catches most sign errors.",
                latex=[f"x {linear.solution_op} {solution_text}"],
            ),
        ]
        answer = SolveAnswer(
            text=f"The solution is x {linear.solution_op} {solution_text}.", latex=f"x {linear.solution_op} {solution_text}"
        )
        return answer, steps

    def _try_quadratic_graph(self, text: str) -> QuadraticFunction | None:
        expression = self._extract_function_expression(text)
        if not expression:
            return None
        coefficients = self._parse_quadratic_coefficients(expression)
        if coefficients is None:
            return None
        a, b, c = coefficients
        if math.isclose(a, 0.0):
            return None
        return QuadraticFunction(expression=expression, a=a, b=b, c=c)

    def _build_quadratic_response(
        self, quadratic: QuadraticFunction
    ) -> tuple[str, list[SolveStep], list[str]]:
        a, b, c = quadratic.a, quadratic.b, quadratic.c
        vertex_x = -b / (2 * a)
        vertex_y = a * vertex_x**2 + b * vertex_x + c
        discriminant = b**2 - 4 * a * c
        a_text = self._format_number(a)
        b_text = self._format_number(b)
        c_text = self._format_number(c)
        vertex_x_text = self._format_number(vertex_x)
        vertex_y_text = self._format_number(vertex_y)
        latex_payload = [f"x_v = -\\frac{{{b_text}}}{{2({a_text})}} = {vertex_x_text}"]
        roots_text = (
            "The graph does not cross the x-axis because the discriminant is negative."
        )
        roots_latex: list[str] = []
        if math.isclose(discriminant, 0.0):
            root = -b / (2 * a)
            root_text = self._format_number(root)
            roots_text = f"The graph touches the x-axis once at x = {root_text}."
            roots_latex = [f"x = {root_text}"]
        elif discriminant > 0:
            root_one = (-b - math.sqrt(discriminant)) / (2 * a)
            root_two = (-b + math.sqrt(discriminant)) / (2 * a)
            root_one_text = self._format_number(root_one)
            root_two_text = self._format_number(root_two)
            roots_text = f"The graph crosses the x-axis at x = {root_one_text} and x = {root_two_text}."
            roots_latex = [
                f"x = \\frac{{-{b_text} \\pm \\sqrt{{{self._format_number(discriminant)}}}}}{{2({a_text})}}"
            ]
        latex_payload.extend(roots_latex)
        answer_text = (
            f"The vertex is at ({vertex_x_text}, {vertex_y_text}). {roots_text} "
            f"Because a = {a_text}, the parabola opens {'upward' if a > 0 else 'downward'}."
        )
        steps = [
            SolveStep(
                index=1,
                title="Identify the graph type",
                explanation=f"The function y = {quadratic.expression} is quadratic, so its graph is a parabola.",
                why_it_happens="A nonzero x² term creates a parabola rather than a line.",
                common_mistakes=["Mixing up the vertex with an x-intercept."],
                hints=["Set y = 0 to find x-intercepts."],
            ),
            SolveStep(
                index=2,
                title="Find the vertex",
                explanation="Use x = -b/(2a), then substitute that x-value back into the function.",
                why_it_happens="The vertex lies on the axis of symmetry of a quadratic function.",
                common_mistakes=["Using -b/a instead of -b/(2a)."],
                hints=[
                    "Once you have the axis of symmetry, plug it into the equation."
                ],
                exam_tip="The sign of a tells you whether the vertex is a maximum or minimum.",
                latex=[
                    latex_payload[0],
                    f"y_v = {a_text}({vertex_x_text})^2 + {b_text}({vertex_x_text}) + {c_text} = {vertex_y_text}",
                ],
            ),
            SolveStep(
                index=3,
                title="Locate the x-intercepts",
                explanation=roots_text,
                why_it_happens="x-intercepts occur exactly where the function value is zero.",
                common_mistakes=[
                    "Forgetting to set the entire function equal to zero before solving."
                ],
                hints=["Factor if possible, or use the quadratic formula."],
                latex=roots_latex,
            ),
        ]
        return answer_text, steps, latex_payload

    def _try_geometry_construction(
        self, text: str
    ) -> tuple[SolveAnswer, list[SolveStep]] | None:
        lowered = text.lower()
        perpendicular_match = re.search(
            r"perpendicular\s+bisector\s+of\s+([A-Z])([A-Z])",
            text,
            flags=re.IGNORECASE,
        )
        if perpendicular_match:
            p1, p2 = (
                perpendicular_match.group(1).upper(),
                perpendicular_match.group(2).upper(),
            )
            answer = SolveAnswer(
                text=f"Construct the midpoint M of {p1}{p2}, then draw the line through M perpendicular to {p1}{p2}. That line is the perpendicular bisector of {p1}{p2}.",
                latex=None,
            )
            steps = [
                SolveStep(
                    index=1,
                    title=f"Mark the midpoint of {p1}{p2}",
                    explanation=f"Create point M so that {p1}M and M{p2} have the same length.",
                    why_it_happens="A perpendicular bisector must pass through the midpoint of the segment it bisects.",
                    hints=[
                        "Look for equal tick marks or equal distance from the endpoints."
                    ],
                ),
                SolveStep(
                    index=2,
                    title="Draw the perpendicular line",
                    explanation=f"Draw a line through M that meets {p1}{p2} at a right angle.",
                    why_it_happens="Every point on a perpendicular bisector is the same distance from the two endpoints.",
                    common_mistakes=[
                        "Drawing a perpendicular line that does not pass through the midpoint."
                    ],
                    exam_tip="Use both conditions: midpoint and 90° angle.",
                ),
            ]
            return answer, steps

        circle_match = re.search(
            r"circle\s+(?:with|has)\s+center\s+([A-Z])(?:\s+at\s*\(([-\d.]+),\s*([-\d.]+)\))?(?:\s+and)?\s+radius\s*([-\d.]+)",
            text,
            flags=re.IGNORECASE,
        )
        point_match = re.search(
            r"point\s+([A-Z])\s+(?:at|=)\s*\(([-\d.]+),\s*([-\d.]+)\)",
            text,
            flags=re.IGNORECASE,
        )
        if circle_match:
            center = circle_match.group(1).upper()
            radius = float(circle_match.group(4))
            radius_text = self._format_number(radius)
            answer_text = (
                f"Draw a circle centered at {center} with radius {radius_text}."
            )
            steps = [
                SolveStep(
                    index=1,
                    title="Plot the center",
                    explanation=f"Place center {center} first; every point on the circle will be {radius_text} units from it.",
                    why_it_happens="A circle is the set of all points at a fixed distance from its center.",
                ),
                SolveStep(
                    index=2,
                    title="Set the radius",
                    explanation=f"Use radius {radius_text} to draw the circle around {center}.",
                    hints=[
                        "The radius is measured from the center to the circle, not across the circle."
                    ],
                ),
            ]
            if point_match:
                point = point_match.group(1).upper()
                x_coord = float(point_match.group(2))
                y_coord = float(point_match.group(3))
                distance = math.sqrt(x_coord**2 + y_coord**2)
                distance_text = self._format_number(distance)
                on_circle = math.isclose(distance, radius)
                answer_text += (
                    f" Point {point} = ({self._format_number(x_coord)}, {self._format_number(y_coord)}) is "
                    f"{'on' if on_circle else 'not on'} the circle if {center} is at the origin, because its distance from {center} is {distance_text}."
                )
                steps.append(
                    SolveStep(
                        index=3,
                        title=f"Plot point {point}",
                        explanation=f"Point {point} has coordinates ({self._format_number(x_coord)}, {self._format_number(y_coord)}). Its distance from the origin is {distance_text}.",
                        why_it_happens="Use the distance formula to compare the point's distance from the center with the radius.",
                        common_mistakes=[
                            "Comparing x or y alone to the radius instead of using distance."
                        ],
                        latex=[
                            f"d = \\sqrt{{{self._format_number(x_coord)}^2 + {self._format_number(y_coord)}^2}} = {distance_text}"
                        ],
                    )
                )
            return SolveAnswer(text=answer_text, latex=None), steps

        triangle_match = re.search(
            r"triangle\s+([A-Z])([A-Z])([A-Z])", text, flags=re.IGNORECASE
        )
        if triangle_match and any(
            keyword in lowered for keyword in ["construct", "draw", "plot"]
        ):
            labels = "".join(group.upper() for group in triangle_match.groups())
            answer = SolveAnswer(
                text=f"Construct triangle {labels} by plotting its three vertices and connecting them in order.",
                latex=None,
            )
            steps = [
                SolveStep(
                    index=1,
                    title="Place the vertices",
                    explanation=f"Create points {', '.join(labels)} in the diagram.",
                    why_it_happens="A triangle is determined by three non-collinear points.",
                ),
                SolveStep(
                    index=2,
                    title="Connect the sides",
                    explanation=f"Draw segments {labels[0]}{labels[1]}, {labels[1]}{labels[2]}, and {labels[2]}{labels[0]}.",
                    common_mistakes=[
                        "Forgetting to close the triangle with the final side."
                    ],
                ),
            ]
            return answer, steps

        return None

    def _extract_arithmetic_expression(self, text: str) -> str | None:
        normalized = self._normalize_math_expression(text)
        if re.search(r"[A-Za-z=]", normalized):
            raw_candidates = cast(list[str], re.findall(r"[-+*/^().\d\s]+", normalized))
            candidates: list[str] = [candidate.strip() for candidate in raw_candidates]
            candidates = [
                candidate
                for candidate in candidates
                if re.search(r"\d", candidate) and re.search(r"[-+*/^]", candidate)
            ]
            if not candidates:
                return None
            return max(candidates, key=len)
        if not re.search(r"\d", normalized) or not re.search(r"[-+*/^]", normalized):
            return None
        return normalized

    def _extract_linear_equation(self, text: str) -> str | None:
        normalized = self._normalize_math_expression(text)
        raw_candidates = cast(
            list[str],
            re.findall(
                r"[0-9xX+\-*/^().\s]+(?:<=|>=|<=|>=|<|>|=)[0-9xX+\-*/^().\s]+",
                normalized,
            ),
        )
        candidates = [
            candidate.strip(" .,:;?\n\t")
            for candidate in raw_candidates
            if "x" in candidate.lower()
        ]
        return max(candidates, key=len) if candidates else None

    def _parse_linear_side(self, side: str) -> tuple[float, float] | None:
        if not self._is_likely_valid_math_syntax(side):
            return None
        prepared = self._prepare_math_ast_expression(side).replace("X", "x")
        try:
            tree = ast.parse(prepared, mode="eval")
            expression = self._linear_from_ast(tree.body)
        except Exception:
            return None
        return expression.coefficient, expression.constant

    def _linear_from_ast(self, node: ast.AST, depth: int = 0) -> LinearExpression:
        if depth > 30:
            raise ValueError("Expression too deep")

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = float(node.value)
            if abs(val) > 1e50:
                raise ValueError("Constant value too large")
            return LinearExpression(0.0, val)

        if isinstance(node, ast.Name):
            if node.id != "x":
                raise ValueError("Unsupported variable")
            return LinearExpression(1.0, 0.0)

        if isinstance(node, ast.UnaryOp):
            value = self._linear_from_ast(node.operand, depth + 1)
            if abs(value.coefficient) > 1e50 or abs(value.constant) > 1e50:
                raise ValueError("Operand too large")
            if isinstance(node.op, ast.USub):
                coeff = -value.coefficient
                const = -value.constant
            elif isinstance(node.op, ast.UAdd):
                coeff = value.coefficient
                const = value.constant
            else:
                raise ValueError("Unsupported unary operation")

            if abs(coeff) > 1e50 or abs(const) > 1e50:
                raise ValueError("Result overflow")
            return LinearExpression(coeff, const)

        if isinstance(node, ast.BinOp):
            left = self._linear_from_ast(node.left, depth + 1)
            right = self._linear_from_ast(node.right, depth + 1)
            if (
                abs(left.coefficient) > 1e50
                or abs(left.constant) > 1e50
                or abs(right.coefficient) > 1e50
                or abs(right.constant) > 1e50
            ):
                raise ValueError("Operand too large")

            if isinstance(node.op, ast.Add):
                coeff = left.coefficient + right.coefficient
                const = left.constant + right.constant
            elif isinstance(node.op, ast.Sub):
                coeff = left.coefficient - right.coefficient
                const = left.constant - right.constant
            elif isinstance(node.op, ast.Mult):
                if math.isclose(left.coefficient, 0.0):
                    coeff = right.coefficient * left.constant
                    const = right.constant * left.constant
                elif math.isclose(right.coefficient, 0.0):
                    coeff = left.coefficient * right.constant
                    const = left.constant * right.constant
                else:
                    raise ValueError("Nonlinear multiplication")
            elif isinstance(node.op, ast.Div):
                if not math.isclose(right.coefficient, 0.0):
                    raise ValueError("Division by an expression containing x")
                if math.isclose(right.constant, 0.0):
                    raise ZeroDivisionError("Division by zero")
                coeff = left.coefficient / right.constant
                const = left.constant / right.constant
            elif isinstance(node.op, ast.Pow):
                if math.isclose(left.coefficient, 0.0) and math.isclose(
                    right.coefficient, 0.0
                ):
                    if abs(right.constant) > 1000:
                        raise ValueError("Exponent too large")
                    if abs(left.constant) > 1e50 and right.constant > 1:
                        raise ValueError("Base too large")
                    try:
                        val = math.pow(left.constant, right.constant)
                    except (OverflowError, ZeroDivisionError, ValueError):
                        raise ValueError("Invalid power operation or overflow")
                    coeff = 0.0
                    const = val
                elif (
                    math.isclose(left.coefficient, 1.0)
                    and math.isclose(left.constant, 0.0)
                    and math.isclose(right.coefficient, 0.0)
                    and math.isclose(right.constant, 1.0)
                ):
                    return left
                elif math.isclose(right.coefficient, 0.0) and math.isclose(
                    right.constant, 0.0
                ):
                    coeff = 0.0
                    const = 1.0
                else:
                    raise ValueError("Nonlinear power")
            else:
                raise ValueError("Unsupported operation")

            if abs(coeff) > 1e50 or abs(const) > 1e50:
                raise ValueError("Result overflow")
            return LinearExpression(coeff, const)

        raise ValueError("Unsupported linear expression")

    def _extract_function_expression(self, text: str) -> str | None:
        normalized = self._normalize_math_expression(text)
        match = re.search(
            r"(?:y|f\s*\(\s*x\s*\))\s*=\s*([0-9xX+\-*/^().\s]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        expression = match.group(1).strip(" .,:;?\n\t")
        return expression or None

    def _parse_quadratic_coefficients(
        self, expression: str
    ) -> tuple[float, float, float] | None:
        if not self._is_likely_valid_math_syntax(expression):
            return None
        prepared = self._prepare_math_ast_expression(expression).replace("X", "x")
        try:
            tree = ast.parse(prepared, mode="eval")
            quad = self._quadratic_from_ast(tree.body)
            return quad.a, quad.b, quad.c
        except Exception:
            return None

    def _normalize_math_expression(self, text: str) -> str:
        return (
            text.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("×", "*")
            .replace("²", "^2")
        )

    def _prepare_math_ast_expression(self, text: str) -> str:
        normalized = self._normalize_math_expression(text)
        normalized = normalized.replace("^", "**")
        normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
        normalized = re.sub(r"(?<=[0-9.)xX])\s*(?=[xX(])", "*", normalized)
        normalized = re.sub(r"(?<=\))\s*(?=\d)", "*", normalized)
        return normalized.strip()

    def _normalize_text_for_matching(self, text: str) -> str:
        text = text.replace("Đ", "D").replace("đ", "d")
        without_accents = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", without_accents.lower())

    def _format_linear_expression(self, coefficient: float, constant: float) -> str:
        terms: list[str] = []
        if not math.isclose(coefficient, 0.0):
            terms.append(self._format_linear_term(coefficient))
        if not math.isclose(constant, 0.0) or not terms:
            constant_text = self._format_number(abs(constant))
            if not terms:
                terms.append(self._format_number(constant))
            elif constant > 0:
                terms.append(f"+ {constant_text}")
            else:
                terms.append(f"- {constant_text}")
        return " ".join(terms)

    def _format_linear_term(self, coefficient: float) -> str:
        if math.isclose(coefficient, 1.0):
            return "x"
        if math.isclose(coefficient, -1.0):
            return "-x"
        return f"{self._format_number(coefficient)}x"

    def _format_number(self, value: float | int) -> str:
        numeric = float(value)
        if math.isclose(numeric, round(numeric), abs_tol=1e-9):
            return str(int(round(numeric)))
        return f"{numeric:.6f}".rstrip("0").rstrip(".")

    def _is_likely_valid_math_syntax(self, expression: str) -> bool:
        if expression.count("(") != expression.count(")"):
            return False
        stripped = "".join(expression.split())
        if not stripped:
            return False
        if stripped[0] in {"*", "/", "^", "."}:
            return False
        if stripped[-1] in {"+", "-", "*", "/", "^", "."}:
            return False
        return True

    def _quadratic_from_ast(self, node: ast.AST, depth: int = 0) -> QuadraticExpression:
        if depth > 30:
            raise ValueError("Expression too deep")

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = float(node.value)
            if abs(val) > 1e50:
                raise ValueError("Constant value too large")
            return QuadraticExpression(0.0, 0.0, val)

        if isinstance(node, ast.Name):
            if node.id != "x":
                raise ValueError("Unsupported variable")
            return QuadraticExpression(0.0, 1.0, 0.0)

        if isinstance(node, ast.UnaryOp):
            value = self._quadratic_from_ast(node.operand, depth + 1)
            if abs(value.a) > 1e50 or abs(value.b) > 1e50 or abs(value.c) > 1e50:
                raise ValueError("Operand too large")
            if isinstance(node.op, ast.USub):
                a, b, c = -value.a, -value.b, -value.c
            elif isinstance(node.op, ast.UAdd):
                a, b, c = value.a, value.b, value.c
            else:
                raise ValueError("Unsupported unary operation")

            if abs(a) > 1e50 or abs(b) > 1e50 or abs(c) > 1e50:
                raise ValueError("Result overflow")
            return QuadraticExpression(a, b, c)

        if isinstance(node, ast.BinOp):
            left = self._quadratic_from_ast(node.left, depth + 1)
            right = self._quadratic_from_ast(node.right, depth + 1)
            if (
                abs(left.a) > 1e50 or abs(left.b) > 1e50 or abs(left.c) > 1e50 or
                abs(right.a) > 1e50 or abs(right.b) > 1e50 or abs(right.c) > 1e50
            ):
                raise ValueError("Operand too large")

            if isinstance(node.op, ast.Add):
                a = left.a + right.a
                b = left.b + right.b
                c = left.c + right.c
            elif isinstance(node.op, ast.Sub):
                a = left.a - right.a
                b = left.b - right.b
                c = left.c - right.c
            elif isinstance(node.op, ast.Mult):
                x4_coeff = left.a * right.a
                x3_coeff = left.a * right.b + left.b * right.a
                if not math.isclose(x4_coeff, 0.0) or not math.isclose(x3_coeff, 0.0):
                    raise ValueError("Degree exceeds quadratic")
                a = left.a * right.c + left.b * right.b + left.c * right.a
                b = left.b * right.c + left.c * right.b
                c = left.c * right.c
            elif isinstance(node.op, ast.Div):
                if not math.isclose(right.a, 0.0) or not math.isclose(right.b, 0.0):
                    raise ValueError("Division by non-constant")
                if math.isclose(right.c, 0.0):
                    raise ZeroDivisionError("Division by zero")
                a = left.a / right.c
                b = left.b / right.c
                c = left.c / right.c
            elif isinstance(node.op, ast.Pow):
                if not math.isclose(right.a, 0.0) or not math.isclose(right.b, 0.0):
                    raise ValueError("Exponent must be constant")
                exponent = right.c
                if not exponent.is_integer() or exponent < 0 or exponent > 2:
                    raise ValueError("Unsupported exponent for quadratic")
                exp_int = int(exponent)
                if exp_int == 0:
                    a, b, c = 0.0, 0.0, 1.0
                elif exp_int == 1:
                    a, b, c = left.a, left.b, left.c
                elif exp_int == 2:
                    if not math.isclose(left.a, 0.0):
                        raise ValueError("Squaring a quadratic exceeds degree 2")
                    a = left.b ** 2
                    b = 2 * left.b * left.c
                    c = left.c ** 2
            else:
                raise ValueError("Unsupported operation")

            if abs(a) > 1e50 or abs(b) > 1e50 or abs(c) > 1e50:
                raise ValueError("Result overflow")
            return QuadraticExpression(a, b, c)

        raise ValueError("Unsupported quadratic expression")
