from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.problem_attempt import ProblemAttempt
from app.db.models.solver_run import SolverRun
from app.db.models.visualization_artifact import VisualizationArtifact
from app.integrations.openrouter_client import OpenRouterClient
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import (
    GeoGebraPayload,
    RoutingPayload,
    SolveAnswer,
    SolvePart,
    SolveRequest,
    SolveResponse,
    SolveStep,
    VisualizationPayload,
)
from app.services.cache import TTLCache
from app.services.fallback_solver import FallbackSolver
from app.services.geogebra_translator import GeoGebraTranslator
from app.services.geometry_extractor import GeometryExtractor
from app.services.local_solver_selector import (
    LOCAL_SOLVER_MIN_CONFIDENCE,
    UNSUPPORTED_LOCAL_SOLVER_MARKER,
    LocalSolverSelector,
)
from app.services.local_solver_types import LocalSolveResult
from app.services.model_router import (
    JSON_FALLBACK_MODEL,
    JSON_SECONDARY_FALLBACK_MODEL,
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_HEURISTIC_PARSER_MODEL,
    LOCAL_LLAMA_TRIVIA_MODEL,
    ModelRouter,
    RoutingDecision,
)
from app.services.ocr_service import OCRService

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
- `answer` and every part `answer` must be objects, never strings.
- `title`, `explanation`, `why_it_happens`, and `exam_tip` must be strings or null, never arrays.
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


@dataclass(frozen=True)
class DetectedSubquestion:
    label: str
    question: str


@dataclass
class StructuredSolveDraft:
    answer: SolveAnswer
    steps: list[SolveStep]
    confidence: float
    warnings: list[str]
    parts: list[SolvePart] = field(default_factory=list)


_RESPONSE_CACHE: TTLCache[SolveResponse] = TTLCache(ttl_seconds=900)


class SolverService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.client = OpenRouterClient()
        self.router = ModelRouter()
        self.ocr_service = OCRService(self.client)
        self.geometry_extractor = GeometryExtractor(self.client)
        self.translator = GeoGebraTranslator()
        self.fallback_solver = FallbackSolver()
        self.local_solver_selector = LocalSolverSelector(self.fallback_solver)

    async def solve(self, request: SolveRequest) -> SolveResponse:
        request_id = str(uuid.uuid4())
        warnings: list[str] = []
        raw_text = request.input.text.strip()
        normalized_text = raw_text

        ocr_result = await self.ocr_service.extract_problem_text(
            request.input.image_base64,
            request.input.image_mime_type,
        )
        if ocr_result:
            if ocr_result.warning:
                warnings.append(ocr_result.warning)
            if ocr_result.cleaned_text:
                normalized_text = (
                    f"{raw_text}\n\n{ocr_result.cleaned_text}".strip()
                    if raw_text
                    else ocr_result.cleaned_text
                )

        detected_subquestions = self._detect_subquestions(normalized_text)
        routing = self.router.route(
            normalized_text, has_image=bool(request.input.image_base64)
        )
        cache_key = self._build_cache_key(normalized_text, request)
        cached_response = _RESPONSE_CACHE.get(cache_key)
        if cached_response is not None:
            return cached_response.model_copy(
                update={"request_id": request_id, "cached": True}
            )

        solve_text = normalized_text
        local_result = None
        local_subquestion_draft = None
        local_subquestion_reason = None
        if len(detected_subquestions) <= 1:
            local_result = await self.local_solver_selector.solve_if_supported(
                normalized_text,
                routing.problem_type,
                routing.difficulty,
            )
        else:
            candidate_local_subquestion_draft = self._fallback_draft_for_subquestions(
                text=normalized_text,
                problem_type=routing.problem_type,
                difficulty=routing.difficulty,
                subquestions=detected_subquestions,
            )
            if self._is_supported_local_draft(candidate_local_subquestion_draft):
                local_subquestion_draft = candidate_local_subquestion_draft
                local_subquestion_reason = (
                    "matched deterministic local solver patterns for every subquestion"
                )
            elif not self.client.enabled:
                candidate_local_subquestion_draft.warnings.append(
                    "Multiple subquestions were detected, but the model backend is not configured. "
                    "Each subquestion was routed through the local fallback solver."
                )
                local_subquestion_draft = candidate_local_subquestion_draft

        if local_result is not None:
            draft = StructuredSolveDraft(
                answer=local_result.answer,
                steps=local_result.steps,
                confidence=local_result.confidence,
                warnings=local_result.warnings,
                parts=[],
            )
            solve_text = local_result.normalized_text
            routing = self._with_local_solver_routing(
                routing, local_result, original_text=normalized_text
            )
        elif local_subquestion_draft is not None:
            draft = local_subquestion_draft
            if local_subquestion_reason:
                routing = self._with_local_subquestion_routing(
                    routing, reason=local_subquestion_reason
                )
        else:
            draft = await self._solve_structured(
                text=normalized_text,
                problem_type=routing.problem_type,
                difficulty=routing.difficulty,
                model=routing.solver_model,
                subquestions=detected_subquestions,
            )
        warnings.extend(draft.warnings)

        visualization = VisualizationPayload(
            kind="none", summary=None, dsl=None, geogebra=None
        )
        if request.options.include_visualization:
            extraction = await self.geometry_extractor.extract(
                solve_text, routing.problem_type, routing.parser_model
            )
            warnings.extend(extraction.warnings)
            if extraction.dsl.actions:
                translation = self.translator.translate(extraction.dsl)
                warnings.extend(translation.issues)
                kind = (
                    "graph"
                    if routing.problem_type
                    in {
                        ProblemType.functions,
                        ProblemType.algebra,
                        ProblemType.calculus,
                    }
                    else "geogebra"
                )
                visualization = VisualizationPayload(
                    kind=kind,
                    summary=extraction.summary,
                    dsl=extraction.dsl,
                    geogebra=GeoGebraPayload(
                        commands=translation.commands,
                        command_string=translation.command_string,
                        validation_passed=translation.validation_passed,
                        issues=translation.issues,
                    ),
                )

        response = SolveResponse(
            request_id=request_id,
            status="ok",
            problem_type=routing.problem_type.value,
            difficulty=routing.difficulty.value,
            answer=draft.answer,
            steps=draft.steps,
            parts=draft.parts,
            visualization=visualization,
            confidence=max(0.0, min(1.0, draft.confidence)),
            routing=RoutingPayload(
                parser_model=routing.parser_model,
                solver_model=routing.solver_model,
                vision_model=routing.vision_model,
                reason=routing.reason,
            ),
            cached=False,
            warnings=warnings,
        )

        self._persist(request, raw_text, normalized_text, routing, response)
        _RESPONSE_CACHE.set(cache_key, response)
        return response

    def _with_local_solver_routing(
        self,
        routing: RoutingDecision,
        local_result: LocalSolveResult,
        *,
        original_text: str,
    ) -> RoutingDecision:
        is_trivia = local_result.reason == "answered a math trivia or concept question"
        solver_model = LOCAL_LLAMA_TRIVIA_MODEL if is_trivia else LOCAL_DETERMINISTIC_SOLVER_MODEL

        reason_parts = [
            routing.reason,
            f"local llama.cpp trivia solver used because it {local_result.reason}"
            if is_trivia
            else f"deterministic local solver used because it {local_result.reason}",
        ]
        if local_result.detector_model:
            reason_parts.append(
                f"local llama.cpp model {local_result.detector_model} produced the answer"
                if is_trivia
                else f"local llama.cpp detector {local_result.detector_model} selected a supported canonical form"
            )
        if not is_trivia and local_result.normalized_text.strip() != original_text.strip():
            reason_parts.append("prompt was normalized before deterministic solving")

        return RoutingDecision(
            problem_type=local_result.problem_type or routing.problem_type,
            difficulty=routing.difficulty,
            parser_model=LOCAL_HEURISTIC_PARSER_MODEL,
            solver_model=solver_model,
            vision_model=routing.vision_model,
            reason="; ".join(reason_parts),
        )

    def _with_local_subquestion_routing(
        self, routing: RoutingDecision, *, reason: str
    ) -> RoutingDecision:
        return RoutingDecision(
            problem_type=routing.problem_type,
            difficulty=routing.difficulty,
            parser_model=LOCAL_HEURISTIC_PARSER_MODEL,
            solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
            vision_model=routing.vision_model,
            reason=(
                f"{routing.reason}; deterministic local solver used because it {reason}"
            ),
        )

    def _is_supported_local_draft(self, draft: StructuredSolveDraft) -> bool:
        if draft.confidence < LOCAL_SOLVER_MIN_CONFIDENCE:
            return False
        return not any(
            UNSUPPORTED_LOCAL_SOLVER_MARKER in warning for warning in draft.warnings
        )

    def _without_backend_config_warnings(self, warnings: list[str]) -> list[str]:
        return [
            warning
            for warning in warnings
            if "Configure OPENROUTER_API_KEY" not in warning
        ]

    def _fallback_draft_for_subquestions(
        self,
        *,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
        subquestions: list[DetectedSubquestion],
        warning: str | None = None,
        suppress_config_warnings: bool = False,
    ) -> StructuredSolveDraft:
        answer, steps, confidence, warnings = self.fallback_solver.solve(
            text, problem_type, difficulty
        )
        warnings = list(warnings)
        if suppress_config_warnings:
            warnings = self._without_backend_config_warnings(warnings)
        if warning:
            warnings.append(warning)

        if len(subquestions) <= 1:
            return StructuredSolveDraft(
                answer=answer,
                steps=steps,
                confidence=confidence,
                warnings=warnings,
                parts=[],
            )

        parts: list[SolvePart] = []
        confidences = [confidence]
        for subquestion in subquestions:
            subquestion_text = self._subquestion_with_context(
                text, subquestion.question
            )
            part_answer, part_steps, part_confidence, part_warnings = (
                self.fallback_solver.solve(subquestion_text, problem_type, difficulty)
            )
            confidences.append(part_confidence)
            if suppress_config_warnings:
                part_warnings = self._without_backend_config_warnings(part_warnings)
            warnings.extend(
                f"Question {subquestion.label}: {part_warning}"
                for part_warning in part_warnings
            )
            if not part_steps:
                part_steps = [
                    SolveStep(
                        index=1,
                        title=f"Review question {subquestion.label}",
                        explanation=(
                            "This subquestion was detected, but the local fallback solver "
                            "could not produce detailed steps for it."
                        ),
                        why_it_happens=(
                            "Model-backed solving is required for unsupported proof and "
                            "construction formats."
                        ),
                        hints=[
                            (
                                "The configured model backend failed. Try again or use a faster JSON-stable model."
                                if suppress_config_warnings
                                else "Configure OPENROUTER_API_KEY for full multi-question solving."
                            )
                        ],
                    )
                ]
            parts.append(
                SolvePart(
                    label=subquestion.label,
                    question=subquestion.question,
                    answer=part_answer,
                    steps=self._reindex_steps(part_steps),
                )
            )

        labels = ", ".join(part.label for part in parts)
        return StructuredSolveDraft(
            answer=SolveAnswer(
                text=(
                    f"This problem has {len(parts)} subquestions: {labels}. "
                    "Each subquestion has its own answer and step-by-step guide."
                ),
                latex=None,
            ),
            steps=[
                SolveStep(
                    index=1,
                    title="Use the per-question guides",
                    explanation=(
                        f"The prompt was split into {labels}. Select a question "
                        "to view its dedicated steps."
                    ),
                    why_it_happens=(
                        "Multi-part prompts need separate step sequences so each "
                        "requested proof or computation is traceable."
                    ),
                )
            ],
            confidence=min(confidences),
            warnings=warnings,
            parts=parts,
        )

    async def _solve_structured(
        self,
        *,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
        model: str,
        subquestions: list[DetectedSubquestion],
    ) -> StructuredSolveDraft:
        if self.client.enabled:
            user_prompt = (
                f"Problem type: {problem_type.value}\n"
                f"Difficulty: {difficulty.value}\n"
                f"{self._format_subquestion_hint(subquestions)}\n"
                f"Problem:\n{text}"
            )
            model_errors: list[str] = []
            for candidate_model in self._model_candidates(model):
                try:
                    payload = await self.client.complete_json(
                        model=candidate_model,
                        system_prompt=STRUCTURED_SOLVE_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        json_schema=SOLVE_RESPONSE_JSON_SCHEMA,
                        schema_name="structured_math_solution",
                    )
                    draft = self._draft_from_payload(
                        payload, expected_subquestions=subquestions
                    )
                    if candidate_model != model:
                        draft.warnings.append(
                            f"Primary model {model} failed; solved with fallback model {candidate_model}."
                        )
                    return draft
                except Exception as exc:
                    model_errors.append(f"{candidate_model}: {exc}")

            (
                fallback_answer,
                fallback_steps,
                fallback_confidence,
                fallback_warnings,
            ) = self.fallback_solver.solve(text, problem_type, difficulty)
            fallback_warnings = [
                warning
                for warning in fallback_warnings
                if "Configure OPENROUTER_API_KEY" not in warning
            ]
            fallback_warnings.append(
                "Model-backed solving failed, so the local deterministic solver was used instead: "
                + "; ".join(model_errors)
            )
            if len(subquestions) > 1:
                return self._fallback_draft_for_subquestions(
                    text=text,
                    problem_type=problem_type,
                    difficulty=difficulty,
                    subquestions=subquestions,
                    warning=fallback_warnings[-1],
                    suppress_config_warnings=True,
                )
            return StructuredSolveDraft(
                answer=fallback_answer,
                steps=fallback_steps,
                confidence=fallback_confidence,
                warnings=fallback_warnings,
                parts=[],
            )

        if len(subquestions) > 1:
            return self._fallback_draft_for_subquestions(
                text=text,
                problem_type=problem_type,
                difficulty=difficulty,
                subquestions=subquestions,
                warning=(
                    "Multiple subquestions were detected, but the model backend is not configured. "
                    "Each subquestion was routed through the local fallback solver."
                ),
            )

        answer, steps, confidence, warnings = self.fallback_solver.solve(
            text, problem_type, difficulty
        )
        return StructuredSolveDraft(
            answer=answer,
            steps=steps,
            confidence=confidence,
            warnings=warnings,
            parts=[],
        )

    def _model_candidates(self, preferred_model: str) -> list[str]:
        candidates = [
            preferred_model,
            JSON_SECONDARY_FALLBACK_MODEL,
            JSON_FALLBACK_MODEL,
        ]
        return list(dict.fromkeys(candidates))

    def _draft_from_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_subquestions: list[DetectedSubquestion] | None = None,
    ) -> StructuredSolveDraft:
        expected_subquestions = expected_subquestions or []
        answer = self._answer_from_raw(payload.get("answer", {}))
        steps = self._steps_from_raw(payload.get("steps", []))
        parts = self._parts_from_payload(
            payload.get("parts", []),
            expected_subquestions=expected_subquestions,
            top_level_answer=answer,
            top_level_steps=steps,
        )

        if expected_subquestions and len(expected_subquestions) > 1 and not parts:
            raise ValueError("Model returned no per-question parts")
        if expected_subquestions and len(expected_subquestions) > 1:
            self._validate_subquestion_step_depth(parts, expected_subquestions)
        if not steps and parts:
            steps = self._reindex_steps(parts[0].steps)
        if not steps:
            raise ValueError("Model returned no steps")

        confidence = float(payload.get("confidence", 0.0) or 0.0)
        warnings = self._coerce_string_list(payload.get("warnings", []))
        return StructuredSolveDraft(
            answer=answer,
            steps=steps,
            confidence=confidence,
            warnings=warnings,
            parts=parts,
        )

    def _validate_subquestion_step_depth(
        self, parts: list[SolvePart], expected_subquestions: list[DetectedSubquestion]
    ) -> None:
        for index, subquestion in enumerate(expected_subquestions):
            if index >= len(parts):
                raise ValueError(
                    f"Model returned no part for question {subquestion.label}"
                )
            part = parts[index]
            if (
                self._is_proof_like_question(subquestion.question)
                and len(part.steps) < 3
            ):
                raise ValueError(
                    f"Model returned too few steps for proof question {subquestion.label}"
                )

    def _is_proof_like_question(self, question: str) -> bool:
        fallback_solver = getattr(self, "fallback_solver", None) or FallbackSolver()
        normalized = fallback_solver._normalize_text_for_matching(question)
        proof_markers = {
            "prove",
            "proof",
            "show that",
            "justify",
            "chung minh",
            "cyclic",
            "noi tiep",
            "angle bisector",
            "bisects",
            "phan giac",
        }
        return any(marker in normalized for marker in proof_markers)

    def _answer_from_raw(self, raw_answer: Any) -> SolveAnswer:
        if isinstance(raw_answer, str):
            raw_answer = {"text": raw_answer}
        if not isinstance(raw_answer, dict):
            raw_answer = {"text": str(raw_answer) if raw_answer is not None else ""}
        if "text" not in raw_answer or raw_answer.get("text") is None:
            raw_answer = {**raw_answer, "text": ""}
        return SolveAnswer.model_validate(raw_answer)

    def _steps_from_raw(self, raw_steps: Any) -> list[SolveStep]:
        if raw_steps is None:
            return []
        if isinstance(raw_steps, dict):
            raw_steps = [raw_steps]
        if not isinstance(raw_steps, list):
            return []
        return [SolveStep.model_validate(step) for step in raw_steps]

    def _parts_from_payload(
        self,
        raw_parts: Any,
        *,
        expected_subquestions: list[DetectedSubquestion],
        top_level_answer: SolveAnswer,
        top_level_steps: list[SolveStep],
    ) -> list[SolvePart]:
        if raw_parts is None:
            raw_parts = []
        if isinstance(raw_parts, dict):
            raw_parts = [raw_parts]
        if not isinstance(raw_parts, list):
            raw_parts = []

        if expected_subquestions and len(expected_subquestions) > 1:
            return [
                self._part_from_raw(
                    raw_parts[index] if index < len(raw_parts) else {},
                    index=index,
                    expected_subquestion=subquestion,
                    top_level_answer=top_level_answer,
                    top_level_steps=top_level_steps,
                    require_steps=True,
                )
                for index, subquestion in enumerate(expected_subquestions)
            ]

        parts: list[SolvePart] = []
        for index, raw_part in enumerate(raw_parts):
            if not isinstance(raw_part, dict):
                continue
            part = self._part_from_raw(
                raw_part,
                index=index,
                expected_subquestion=None,
                top_level_answer=top_level_answer,
                top_level_steps=top_level_steps,
                require_steps=False,
            )
            if part.steps:
                parts.append(part)
        return parts

    def _part_from_raw(
        self,
        raw_part: Any,
        *,
        index: int,
        expected_subquestion: DetectedSubquestion | None,
        top_level_answer: SolveAnswer,
        top_level_steps: list[SolveStep],
        require_steps: bool,
    ) -> SolvePart:
        raw_part = raw_part if isinstance(raw_part, dict) else {}
        label = self._alpha_label(index)
        question = expected_subquestion.question if expected_subquestion else ""
        if not question:
            question = str(raw_part.get("question") or raw_part.get("prompt") or "")

        answer = self._answer_from_raw(raw_part.get("answer", {}))
        if not answer.text:
            answer = SolveAnswer(
                text=top_level_answer.text
                or f"See the step-by-step guide for question {label}.",
                latex=top_level_answer.latex,
            )

        steps = self._steps_from_raw(raw_part.get("steps", []))
        if not steps:
            steps = self._fallback_steps_for_part(top_level_steps, label, index)
        if require_steps and not steps:
            raise ValueError(f"Model returned no steps for question {label}")

        return SolvePart(
            label=label,
            question=question,
            answer=answer,
            steps=self._reindex_steps(steps),
        )

    def _fallback_steps_for_part(
        self, steps: list[SolveStep], label: str, index: int
    ) -> list[SolveStep]:
        if not steps:
            return []

        label_pattern = re.compile(
            rf"(?:\b(?:part|question)\s*\(?{re.escape(label)}\)?\b|\({re.escape(label)}\)|\b{re.escape(label)}[\).:])",
            flags=re.IGNORECASE,
        )
        matching_steps = [
            step
            for step in steps
            if label_pattern.search(f"{step.title} {step.explanation}")
        ]
        if matching_steps:
            return self._reindex_steps(matching_steps)
        if index < len(steps):
            return self._reindex_steps([steps[index]])
        return []

    def _reindex_steps(self, steps: list[SolveStep]) -> list[SolveStep]:
        return [
            step.model_copy(update={"index": index + 1})
            for index, step in enumerate(steps)
        ]

    def _detect_subquestions(self, text: str) -> list[DetectedSubquestion]:
        marker_pattern = re.compile(
            r"^[ \t]*(?:\(([A-Za-z]|\d{1,2}|[ivxlcdm]+)\)|([A-Za-z]|\d{1,2}|[ivxlcdm]+)[\).:])\s+",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        matches = list(marker_pattern.finditer(text))
        if len(matches) <= 1:
            return []

        subquestions: list[DetectedSubquestion] = []
        for index, match in enumerate(matches):
            body_start = match.end()
            body_end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(text)
            )
            question = text[body_start:body_end].strip()
            if question:
                subquestions.append(
                    DetectedSubquestion(
                        label=self._alpha_label(index), question=question
                    )
                )
        return subquestions if len(subquestions) > 1 else []

    def _format_subquestion_hint(self, subquestions: list[DetectedSubquestion]) -> str:
        if len(subquestions) <= 1:
            return "Detected subquestions: none."
        formatted = "\n".join(
            f"{subquestion.label}) {subquestion.question}"
            for subquestion in subquestions
        )
        return (
            "Detected subquestions (use these normalized labels and solve each separately):\n"
            f"{formatted}"
        )

    def _subquestion_with_context(self, full_text: str, question: str) -> str:
        position = full_text.find(question)
        if position <= 0:
            return question
        marker_pattern = re.compile(
            r"^[ \t]*(?:\(([A-Za-z]|\d{1,2}|[ivxlcdm]+)\)|([A-Za-z]|\d{1,2}|[ivxlcdm]+)[\).:])\s+",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        first_marker = marker_pattern.search(full_text)
        context_end = (
            first_marker.start()
            if first_marker is not None and first_marker.start() < position
            else position
        )
        context = re.sub(
            r"^[ \t]*(?:\([A-Za-z\d]+\)|[A-Za-z\d]+[\).:])\s*$",
            "",
            full_text[:context_end].strip(),
            flags=re.MULTILINE,
        ).strip()
        return f"{context}\n{question}".strip() if context else question

    def _alpha_label(self, index: int) -> str:
        label = ""
        cursor = index
        while cursor >= 0:
            label = f"{chr(97 + (cursor % 26))}{label}"
            cursor = cursor // 26 - 1
        return label

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    def _build_cache_key(self, normalized_text: str, request: SolveRequest) -> str:
        payload = {
            "text": normalized_text,
            "options": request.options.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _persist(
        self,
        request: SolveRequest,
        raw_text: str,
        normalized_text: str,
        routing: RoutingDecision,
        response: SolveResponse,
    ) -> None:
        try:
            attempt = ProblemAttempt(
                raw_text=raw_text,
                normalized_text=normalized_text,
                input_type="image" if request.input.image_base64 else "text",
                language=request.input.language,
            )
            self.db.add(attempt)
            self.db.flush()

            run = SolverRun(
                attempt_id=attempt.id,
                parser_model=routing.parser_model,
                solver_model=routing.solver_model,
                vision_model=routing.vision_model,
                problem_type=routing.problem_type.value,
                difficulty=routing.difficulty.value,
                route_reason=routing.reason,
                confidence=response.confidence,
                cached=response.cached,
                status=response.status,
            )
            self.db.add(run)

            artifact = VisualizationArtifact(
                attempt_id=attempt.id,
                kind=response.visualization.kind,
                dsl_json=response.visualization.dsl.model_dump(mode="json")
                if response.visualization.dsl
                else None,
                commands_json=response.visualization.geogebra.commands
                if response.visualization.geogebra
                else None,
                summary=response.visualization.summary,
            )
            self.db.add(artifact)
            self.db.commit()
        except Exception:
            self.db.rollback()
