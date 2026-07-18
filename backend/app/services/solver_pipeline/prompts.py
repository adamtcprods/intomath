"""Prompts and JSON schemas used by the structured solver pipeline."""

from __future__ import annotations

from typing import Any


STRUCTURED_SOLVE_SYSTEM_PROMPT = """
You are IntoMath's structured math solver.
Return only the final JSON object required by the API response_format schema.
Do not copy or describe the schema. Do not use placeholders like "..." or "string".

Rules:
- Solve the math problem completely and put the actual solution in the JSON fields.
- If the problem has subquestions, return one `parts` item for every subquestion.
- Label subquestions by order as lowercase letters: a, b, c, d, ... Ignore original labels such as 1, 2 or i, ii.
- Every `parts[].steps` must be a real step-by-step guide for that specific subquestion; never leave it empty.
- For proof-style subquestions, split the reasoning into at least 3 small steps, usually 4-7; do not put the whole proof in one step.
- For geometry proofs, include the concrete angle, length, cyclicity, similarity, or algebra transformations used between steps.
- If there are no subquestions, use `parts: []` and put the guide in top-level `steps`.
- Top-level `answer` should summarize all requested work. For multiple subquestions, top-level `steps` may summarize the overall flow.
- Keep every string under 220 characters.
- Arrays must contain strings only; use [] when there are no items.
- Put ordinary language only in `text`, `explanation`, and `why_it_happens`; never put prose or a complete sentence in a `latex` field.
- Use `latex` only for a standalone mathematical expression. Do not include `$`, `$$`, `\\(`, `\\)`, `\\[`, `\\]`, or Markdown fences.
- If a step's `explanation` or `why_it_happens` contains mathematical notation or names an expression such as a^n, gcd(a,b), sqrt(x), an integral, equation, or inequality, populate that step's `latex` array with every matching expression as valid KaTeX.
- Set step `latex` to [] only when neither step field contains a mathematical expression. Do not repeat prose as LaTeX.
- Present only final, organized reasoning in `explanation` and `why_it_happens`: no trial-and-error, chronological backtracking, self-questioning, hedge language, or uncertainty about discarded attempts.
- Present genuine case analysis as a pre-organized enumeration of cases and outcomes, not as a log of cases tried and abandoned.
- Scratch example: "Maybe a=1 works. Wait, no; perhaps try a=2?" Clean equivalent: "Case a=1 fails the divisibility condition; case a=2 satisfies it."
- Scratch example: "I think this gives x=3, but actually I may have changed the sign." Clean equivalent: "Preserving the sign gives x=-3."
- `answer` and every part `answer` must be objects, never strings.
- `title`, `explanation`, `why_it_happens`, and `exam_tip` must be strings or null, never arrays.
""".strip()

STRUCTURED_CONTENT_REPAIR_SYSTEM_PROMPT = """
You are IntoMath's bounded structured-solution editor.
Return the complete solution JSON object using the required schema.
Preserve every answer, conclusion, step order, and unflagged field exactly.
Rewrite only the flagged `explanation` or `why_it_happens` fields as concise,
finished reasoning with no trial-and-error, backtracking, hedging, or self-questioning.
If a rewritten field contains math notation, populate that same step's `latex` array
with matching standalone KaTeX expressions and no delimiters or prose.
""".strip()

SOLVE_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "latex": {"type": ["string", "null"]},
    },
    "required": ["text", "latex"],
    "additionalProperties": False,
}

SOLVE_STEP_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "title": {"type": "string"},
        "explanation": {"type": "string"},
        "why_it_happens": {"type": ["string", "null"]},
        "common_mistakes": {"type": "array", "items": {"type": "string"}},
        "alternative_approaches": {"type": "array", "items": {"type": "string"}},
        "hints": {"type": "array", "items": {"type": "string"}},
        "exam_tip": {"type": ["string", "null"]},
        "latex": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "index",
        "title",
        "explanation",
        "why_it_happens",
        "common_mistakes",
        "alternative_approaches",
        "hints",
        "exam_tip",
        "latex",
    ],
    "additionalProperties": False,
}

SOLVE_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": SOLVE_ANSWER_JSON_SCHEMA,
        "steps": {"type": "array", "items": SOLVE_STEP_JSON_SCHEMA},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "question": {"type": "string"},
                    "answer": SOLVE_ANSWER_JSON_SCHEMA,
                    "steps": {"type": "array", "items": SOLVE_STEP_JSON_SCHEMA},
                },
                "required": ["label", "question", "answer", "steps"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "steps", "parts", "confidence", "warnings"],
    "additionalProperties": False,
}

SOLVE_STEPS_REPAIR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 3,
            "maxItems": 7,
            "items": SOLVE_STEP_JSON_SCHEMA,
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


__all__ = [
    "SOLVE_ANSWER_JSON_SCHEMA",
    "SOLVE_RESPONSE_JSON_SCHEMA",
    "SOLVE_STEPS_REPAIR_JSON_SCHEMA",
    "SOLVE_STEP_JSON_SCHEMA",
    "STRUCTURED_CONTENT_REPAIR_SYSTEM_PROMPT",
    "STRUCTURED_SOLVE_SYSTEM_PROMPT",
]
