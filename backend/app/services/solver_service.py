from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.problem_attempt import ProblemAttempt
from app.db.models.solver_run import SolverRun
from app.db.models.visualization_artifact import VisualizationArtifact
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.integrations.errors import exception_diagnostics
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
from app.services.geometry_extractor import (
    GeometryExtractor,
    classify_visualization_capability,
)
from app.services.local_solver_selector import (
    LOCAL_SOLVER_MIN_CONFIDENCE,
    UNSUPPORTED_LOCAL_SOLVER_MARKER,
    LocalSolverSelector,
)
from app.services.local_solver_types import LocalSolveResult
from app.services.model_router import (
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    LOCAL_LLAMA_TRIVIA_MODEL,
    ModelRouter,
    RoutingDecision,
    StructuredModelEndpoint,
    remote_model_timeout_seconds,
    structured_model_endpoints,
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


class _StrictStructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    latex: str | None


class _StrictStructuredStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    title: str
    explanation: str
    why_it_happens: str | None
    common_mistakes: list[str]
    alternative_approaches: list[str]
    hints: list[str]
    exam_tip: str | None
    latex: list[str]


class _StrictStructuredPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    question: str
    answer: _StrictStructuredAnswer
    steps: list[_StrictStructuredStep]


class _StrictStructuredSolvePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: _StrictStructuredAnswer
    steps: list[_StrictStructuredStep]
    parts: list[_StrictStructuredPart]
    confidence: float
    warnings: list[str]


class StructuredPayloadValidationError(ValueError):
    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class StructuredContentIssue:
    code: str
    path: str
    step_index: int
    part_index: int | None
    fields: tuple[str, ...]
    message: str


_MATH_NOTATION_PATTERN = re.compile(
    r"\^|_[A-Za-z0-9{]|\\(?:int|sqrt|frac|sum|prod|gcd|le|ge|neq)\b|"
    r"\b(?:gcd|lcm|sqrt)\s*\(|(?:<=|>=)|[≤≥≠≈∞∫√]|"
    r"(?:[A-Za-z0-9})\]])\s*(?:=|<|>|\+|\*|/)\s*(?:[A-Za-z0-9({\[])"
)
_STRONG_SCRATCH_PATTERN = re.compile(
    r"\b(?:wait|hold on|scratch that|never mind|i was wrong|not sure|"
    r"let me (?:try|check|rethink|reconsider)|is that right|does that work|hmm+)\b",
    flags=re.IGNORECASE,
)
_HEDGE_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|probably|i think|i guess|seems? like|might be|could be)\b",
    flags=re.IGNORECASE,
)


def contains_clear_math_notation(value: str | None) -> bool:
    return bool(value and _MATH_NOTATION_PATTERN.search(value))


def looks_like_scratch_work(value: str | None) -> bool:
    if not value:
        return False
    if _STRONG_SCRATCH_PATTERN.search(value):
        return True
    if len(_HEDGE_PATTERN.findall(value)) >= 2:
        return True
    if "?" in value and re.search(
        r"\b(?:i|we|let me|should i|what if)\b", value, flags=re.IGNORECASE
    ):
        return True

    assignments: dict[str, set[str]] = {}
    for variable, conclusion in re.findall(
        r"\b([A-Za-z])\s*=\s*([^,;.?!]+)", value
    ):
        assignments.setdefault(variable.casefold(), set()).add(conclusion.strip())
    has_contradictory_transition = bool(
        re.search(r"\b(?:but|actually|however|no[,;:]?)\b", value, re.IGNORECASE)
    )
    return has_contradictory_transition and any(
        len(conclusions) > 1 for conclusions in assignments.values()
    )


def structured_content_issues(payload: dict[str, Any]) -> list[StructuredContentIssue]:
    locations: list[tuple[str, int | None, int, dict[str, Any]]] = []
    for step_index, step in enumerate(payload.get("steps", [])):
        if isinstance(step, dict):
            locations.append((f"steps[{step_index}]", None, step_index, step))
    for part_index, part in enumerate(payload.get("parts", [])):
        if not isinstance(part, dict):
            continue
        for step_index, step in enumerate(part.get("steps", [])):
            if isinstance(step, dict):
                locations.append(
                    (
                        f"parts[{part_index}].steps[{step_index}]",
                        part_index,
                        step_index,
                        step,
                    )
                )

    issues: list[StructuredContentIssue] = []
    for path, part_index, step_index, step in locations:
        explanation = str(step.get("explanation") or "")
        why_it_happens = str(step.get("why_it_happens") or "")
        latex = step.get("latex")
        latex_empty = not (
            isinstance(latex, list)
            and any(isinstance(item, str) and item.strip() for item in latex)
        )
        if latex_empty and (
            contains_clear_math_notation(explanation)
            or contains_clear_math_notation(why_it_happens)
        ):
            issues.append(
                StructuredContentIssue(
                    code="missing_latex_for_math",
                    path=path,
                    step_index=step_index,
                    part_index=part_index,
                    fields=("latex",),
                    message=(
                        f"{path} contains mathematical notation but its latex array is empty."
                    ),
                )
            )

        scratch_fields = tuple(
            field_name
            for field_name, value in (
                ("explanation", explanation),
                ("why_it_happens", why_it_happens),
            )
            if looks_like_scratch_work(value)
        )
        if scratch_fields:
            issues.append(
                StructuredContentIssue(
                    code="scratch_work_style",
                    path=path,
                    step_index=step_index,
                    part_index=part_index,
                    fields=scratch_fields,
                    message=(
                        f"{path} contains scratch-work markers and may not read as a finished explanation."
                    ),
                )
            )
    return issues


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
    solver_model: str | None = None


_RESPONSE_CACHE: TTLCache[SolveResponse] = TTLCache(ttl_seconds=900)
logger = logging.getLogger(__name__)


class SolverService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.nvidia_client = NvidiaClient()
        self.llama_client = LlamaClient()
        self.router = ModelRouter(self.llama_client)
        self.ocr_service = OCRService()
        self.geometry_extractor = GeometryExtractor(
            self.llama_client,
            nvidia_client=self.nvidia_client,
        )
        self.translator = GeoGebraTranslator()
        self.fallback_solver = FallbackSolver()
        self.local_solver_selector = LocalSolverSelector(
            self.fallback_solver, self.llama_client
        )

    async def solve(self, request: SolveRequest) -> SolveResponse:
        request_id = str(uuid.uuid4())
        warnings: list[str] = []
        raw_text = request.input.text.strip()
        normalized_text = raw_text
        input_type = "image" if request.input.image_base64 else "text"
        logger.info(
            "Solve request started request_id=%s input_type=%s text_chars=%s include_visualization=%s",
            request_id,
            input_type,
            len(raw_text),
            request.options.include_visualization,
        )

        ocr_result = await self.ocr_service.extract_problem_text(
            request.input.image_base64,
            request.input.image_mime_type,
        )
        if ocr_result:
            logger.info(
                "OCR extraction completed request_id=%s cleaned_text_chars=%s has_warning=%s",
                request_id,
                len(ocr_result.cleaned_text or ""),
                bool(ocr_result.warning),
            )
            if ocr_result.warning:
                logger.warning(
                    "OCR extraction warning request_id=%s warning=%s",
                    request_id,
                    ocr_result.warning,
                )
                warnings.append(ocr_result.warning)
            if ocr_result.cleaned_text:
                normalized_text = (
                    f"{raw_text}\n\n{ocr_result.cleaned_text}".strip()
                    if raw_text
                    else ocr_result.cleaned_text
                )

        detected_subquestions = self._detect_subquestions(normalized_text)
        routing = await self.router.route_async(
            normalized_text, has_image=bool(request.input.image_base64)
        )
        logger.info(
            "Solve routing request_id=%s problem_type=%s difficulty=%s parser_model=%s solver_model=%s vision_model=%s",
            request_id,
            routing.problem_type.value,
            routing.difficulty.value,
            routing.parser_model,
            routing.solver_model,
            routing.vision_model,
        )
        cache_key = self._build_cache_key(normalized_text, request)
        cached_response = _RESPONSE_CACHE.get(cache_key)
        if cached_response is not None:
            logger.info(
                "Solve cache hit request_id=%s cache_key_prefix=%s",
                request_id,
                cache_key[:12],
            )
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
            elif not self.nvidia_client.enabled:
                logger.warning(
                    "Multi-question solve used local fallback because all model clients are disabled request_id=%s subquestions=%s",
                    request_id,
                    len(detected_subquestions),
                )
                local_subquestion_draft = candidate_local_subquestion_draft
                local_subquestion_reason = "was the only available solver"

        if local_result is not None:
            logger.info(
                "Using local solver result request_id=%s confidence=%.3f",
                request_id,
                local_result.confidence,
            )
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
            logger.info(
                "Using local subquestion draft request_id=%s subquestions=%s",
                request_id,
                len(detected_subquestions),
            )
            draft = local_subquestion_draft
            if local_subquestion_reason:
                routing = self._with_local_subquestion_routing(
                    routing, reason=local_subquestion_reason
                )
        else:
            logger.info(
                "Using structured model solve request_id=%s model=%s subquestions=%s",
                request_id,
                routing.solver_model,
                len(detected_subquestions),
            )
            draft = await self._solve_structured(
                text=normalized_text,
                problem_type=routing.problem_type,
                difficulty=routing.difficulty,
                model=routing.solver_model,
                subquestions=detected_subquestions,
                request_id=request_id,
            )
            if draft.solver_model:
                routing = self._with_structured_solver_routing(
                    routing, solver_model=draft.solver_model
                )
        warnings.extend(draft.warnings)

        visualization = VisualizationPayload(
            kind="none", summary=None, dsl=None, geogebra=None
        )
        visualization_classification = classify_visualization_capability(solve_text)
        solved_answer_text = "\n".join(
            value
            for value in [
                draft.answer.text,
                draft.answer.latex,
                *[
                    value
                    for part in draft.parts
                    for value in (part.answer.text, part.answer.latex)
                ],
            ]
            if value
        )
        if (
            request.options.include_visualization
            and visualization_classification.visualizable
        ):
            visualization_stage = "extraction"
            try:
                logger.info(
                    "Visualization extraction started request_id=%s problem_type=%s parser_model=%s",
                    request_id,
                    routing.problem_type.value,
                    routing.parser_model,
                )
                extraction = await self.geometry_extractor.extract(
                    solve_text,
                    routing.problem_type,
                    routing.parser_model,
                    request_id=request_id,
                )
                if extraction.warnings:
                    logger.warning(
                        "Visualization extraction completed with warnings request_id=%s warning_count=%s",
                        request_id,
                        len(extraction.warnings),
                    )
                warnings.extend(extraction.warnings)
                if extraction.dsl.actions:
                    visualization_stage = "translation"
                    translation = self.translator.translate(
                        extraction.dsl,
                        allowed_command_names=extraction.allowed_commands,
                        normalized_problem_text=solve_text,
                        solved_answer_text=solved_answer_text,
                    )
                    if translation.issues:
                        logger.warning(
                            "Visualization translation completed with issues request_id=%s issue_count=%s",
                            request_id,
                            len(translation.issues),
                        )
                    if not translation.validation_passed:
                        warnings.append(
                            "The model-generated visualization plan failed validation, "
                            "so no shape could be constructed."
                        )
                    warnings.extend(translation.issue_messages)
                    kind = (
                        "graph"
                        if extraction.dsl.environment.value == "graphing"
                        else "geogebra"
                    )
                    if not translation.commands:
                        kind = "none"
                    retrieved_debug = []
                    if getattr(self.geometry_extractor.settings, "app_debug", False):
                        retrieved_debug = [
                            {
                                "name": command.name,
                                "score": command.score,
                                "signatures": list(command.signatures),
                            }
                            for command in extraction.retrieved_commands
                        ]
                    visualization = VisualizationPayload(
                        kind=kind,
                        summary=extraction.summary,
                        dsl=extraction.dsl,
                        geogebra=GeoGebraPayload(
                            commands=translation.commands,
                            command_string=translation.command_string,
                            validation_passed=translation.validation_passed,
                            issues=translation.issue_messages,
                            validation_issues=translation.issues,
                            environment=extraction.dsl.environment,
                            retrieved_commands=retrieved_debug,
                        ),
                    )
            except Exception as exc:
                diagnostics = exception_diagnostics(exc)
                logger.warning(
                    "Visualization generation failed open request_id=%s stage=%s "
                    "error_type=%s error_message=%s status_code=%s response_body=%s",
                    request_id,
                    visualization_stage,
                    diagnostics.error_type,
                    diagnostics.error_message,
                    diagnostics.status_code,
                    diagnostics.response_body,
                )
                warnings.append(
                    "Visualization generation failed unexpectedly, so no shape could be constructed."
                )
        elif request.options.include_visualization:
            logger.info(
                "Visualization skipped before extraction request_id=%s capability=%s reason=%s",
                request_id,
                visualization_classification.capability.value,
                visualization_classification.reason,
            )

        public_warnings = self._without_backend_config_warnings(warnings)
        if len(public_warnings) != len(warnings):
            logger.info(
                "Suppressed backend-only warnings from API response request_id=%s suppressed_count=%s",
                request_id,
                len(warnings) - len(public_warnings),
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
            warnings=public_warnings,
        )

        if warnings:
            logger.warning(
                "Solve completed with warnings request_id=%s warning_count=%s",
                request_id,
                len(warnings),
            )
        logger.info(
            "Solve completed request_id=%s confidence=%.3f steps=%s parts=%s visualization=%s cached=%s",
            request_id,
            response.confidence,
            len(response.steps),
            len(response.parts),
            response.visualization.kind,
            response.cached,
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
        solver_model = (
            LOCAL_LLAMA_TRIVIA_MODEL if is_trivia else LOCAL_DETERMINISTIC_SOLVER_MODEL
        )

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
        if (
            not is_trivia
            and local_result.normalized_text.strip() != original_text.strip()
        ):
            reason_parts.append("prompt was normalized before deterministic solving")

        return RoutingDecision(
            problem_type=local_result.problem_type or routing.problem_type,
            difficulty=routing.difficulty,
            parser_model=LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
            solver_model=solver_model,
            vision_model=routing.vision_model,
            reason="; ".join(reason_parts),
        )

    def _with_structured_solver_routing(
        self, routing: RoutingDecision, *, solver_model: str
    ) -> RoutingDecision:
        if solver_model == routing.solver_model:
            return routing

        return RoutingDecision(
            problem_type=routing.problem_type,
            difficulty=routing.difficulty,
            parser_model=routing.parser_model,
            solver_model=solver_model,
            vision_model=routing.vision_model,
            reason=f"{routing.reason}; result produced by {solver_model}",
        )

    def _with_local_subquestion_routing(
        self, routing: RoutingDecision, *, reason: str
    ) -> RoutingDecision:
        return RoutingDecision(
            problem_type=routing.problem_type,
            difficulty=routing.difficulty,
            parser_model=LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
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
        backend_only_markers = (
            "Configure NVIDIA_API_KEY",
            "NVIDIA_API_KEY is not configured",
            "Model-backed solving failed",
            "model backend is not configured",
            "model client is disabled",
            "JSON repair attempt failed",
            "returned invalid JSON",
        )
        return [
            warning
            for warning in warnings
            if not any(marker in warning for marker in backend_only_markers)
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
                            "Some proof and construction prompts need more structure before "
                            "a reliable step-by-step solution can be produced."
                        ),
                        hints=[
                            "Try again with a clearer problem statement or type this subquestion separately."
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
        request_id: str | None = None,
    ) -> StructuredSolveDraft:
        if self.nvidia_client.enabled:
            user_prompt = (
                f"Problem type: {problem_type.value}\n"
                f"Difficulty: {difficulty.value}\n"
                f"{self._format_subquestion_hint(subquestions)}\n"
                f"Problem:\n{text}"
            )
            model_failures: list[tuple[str, str, str, str, int | None]] = []
            candidates = self._model_candidates(model)
            for attempt_number, candidate in enumerate(candidates, start=1):
                candidate_client: Any = self.nvidia_client
                attempt_timeout = remote_model_timeout_seconds(
                    self.settings,
                    provider=candidate.provider,
                    model=candidate.model,
                )
                logger.info(
                    "Structured solve model attempt started request_id=%s provider=%s "
                    "model=%s operation=structured_math_solution attempt=%s/%s "
                    "problem_type=%s difficulty=%s subquestions=%s timeout_seconds=%.1f",
                    request_id,
                    candidate.provider,
                    candidate.model,
                    attempt_number,
                    len(candidates),
                    problem_type.value,
                    difficulty.value,
                    len(subquestions),
                    attempt_timeout,
                )
                try:
                    payload = await candidate_client.complete_json(
                        model=candidate.model,
                        system_prompt=STRUCTURED_SOLVE_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        json_schema=SOLVE_RESPONSE_JSON_SCHEMA,
                        schema_name="structured_math_solution",
                        require_parameters=False,
                        allow_schema_downgrade=False,
                        repair_invalid_json=False,
                        timeout_seconds=attempt_timeout,
                        operation="structured_math_solution",
                        trace_id=request_id,
                    )
                    try:
                        validated_payload = _StrictStructuredSolvePayload.model_validate(
                            payload
                        ).model_dump(mode="python")
                    except ValidationError as exc:
                        issue_codes = Counter(
                            str(item.get("type", "invalid")).replace(".", "_")
                            for item in exc.errors(include_url=False)
                        )
                        bare_step = (
                            isinstance(payload, dict)
                            and {"index", "title", "latex"}.issubset(payload)
                            and "answer" not in payload
                            and "steps" not in payload
                        )
                        failure_shape = (
                            "bare_step_object" if bare_step else "schema_validation_failure"
                        )
                        logger.warning(
                            "Structured solve schema validation failed request_id=%s "
                            "provider=%s model=%s operation=structured_math_solution "
                            "failure_shape=%s issue_count=%s issue_codes=%s",
                            request_id,
                            candidate.provider,
                            candidate.model,
                            failure_shape,
                            len(exc.errors(include_url=False)),
                            json.dumps(dict(sorted(issue_codes.items())), sort_keys=True),
                        )
                        raise StructuredPayloadValidationError(
                            "Structured solve response was a bare step object instead of the "
                            "required solution object."
                            if bare_step
                            else "Structured solve response did not match the required schema.",
                            failure_code=failure_shape,
                        ) from exc
                    initial_content_issues = self._log_structured_step_quality(
                        validated_payload,
                        request_id=request_id,
                        provider=candidate.provider,
                        model=candidate.model,
                        stage="initial",
                    )
                    if any(
                        issue.code == "scratch_work_style"
                        for issue in initial_content_issues
                    ):
                        validated_payload = await self._repair_structured_content(
                            validated_payload,
                            initial_content_issues,
                            completion_client=candidate_client,
                            candidate=candidate,
                            timeout_seconds=attempt_timeout,
                            request_id=request_id,
                        )
                    final_content_issues = self._log_structured_step_quality(
                        validated_payload,
                        request_id=request_id,
                        provider=candidate.provider,
                        model=candidate.model,
                        stage="final",
                    )
                    self._append_content_quality_warnings(
                        validated_payload, final_content_issues
                    )
                    draft = self._draft_from_payload(
                        validated_payload, expected_subquestions=subquestions
                    )
                    if candidate.routing_model != model:
                        logger.warning(
                            "Structured solve succeeded with fallback model request_id=%s "
                            "preferred_model=%s fallback_provider=%s fallback_model=%s",
                            request_id,
                            model,
                            candidate.provider,
                            candidate.model,
                        )
                        draft.warnings.append(
                            "The preferred solver was unavailable; this result was produced by another solver."
                        )
                    draft.solver_model = candidate.routing_model
                    logger.info(
                        "Structured solve model attempt succeeded request_id=%s provider=%s "
                        "model=%s routing_model=%s confidence=%.3f steps=%s parts=%s warnings=%s",
                        request_id,
                        candidate.provider,
                        candidate.model,
                        candidate.routing_model,
                        draft.confidence,
                        len(draft.steps),
                        len(draft.parts),
                        len(draft.warnings),
                    )
                    return draft
                except Exception as exc:
                    diagnostics = exception_diagnostics(exc)
                    failure_code = self._structured_failure_code(exc)
                    model_failures.append(
                        (
                            candidate.provider,
                            candidate.model,
                            failure_code,
                            diagnostics.error_type,
                            diagnostics.status_code,
                        )
                    )
                    logger.warning(
                        "Structured solve model attempt failed request_id=%s provider=%s "
                        "model=%s operation=structured_math_solution attempt=%s/%s "
                        "problem_type=%s difficulty=%s failure_code=%s error_type=%s "
                        "error_message=%s status_code=%s response_body=%s",
                        request_id,
                        candidate.provider,
                        candidate.model,
                        attempt_number,
                        len(candidates),
                        problem_type.value,
                        difficulty.value,
                        failure_code,
                        diagnostics.error_type,
                        diagnostics.error_message,
                        diagnostics.status_code,
                        diagnostics.response_body,
                    )

            (
                fallback_answer,
                fallback_steps,
                fallback_confidence,
                fallback_warnings,
            ) = self.fallback_solver.solve(text, problem_type, difficulty)
            fallback_warnings = self._without_backend_config_warnings(fallback_warnings)
            logger.error(
                "All structured solve model attempts failed; using local fallback problem_type=%s difficulty=%s failures=%s",
                problem_type.value,
                difficulty.value,
                model_failures,
            )
            fallback_notice = (
                "The preferred solver was unavailable; this result was produced by the local solver."
            )
            if len(subquestions) > 1:
                fallback_draft = self._fallback_draft_for_subquestions(
                    text=text,
                    problem_type=problem_type,
                    difficulty=difficulty,
                    subquestions=subquestions,
                    warning=fallback_notice,
                    suppress_config_warnings=True,
                )
                fallback_draft.solver_model = LOCAL_DETERMINISTIC_SOLVER_MODEL
                return fallback_draft
            fallback_warnings.append(fallback_notice)
            return StructuredSolveDraft(
                answer=fallback_answer,
                steps=fallback_steps,
                confidence=fallback_confidence,
                warnings=fallback_warnings,
                parts=[],
                solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
            )

        if len(subquestions) > 1:
            logger.warning(
                "Multi-question solve used local fallback because model client is disabled subquestions=%s problem_type=%s difficulty=%s",
                len(subquestions),
                problem_type.value,
                difficulty.value,
            )
            fallback_draft = self._fallback_draft_for_subquestions(
                text=text,
                problem_type=problem_type,
                difficulty=difficulty,
                subquestions=subquestions,
                suppress_config_warnings=True,
            )
            fallback_draft.solver_model = LOCAL_DETERMINISTIC_SOLVER_MODEL
            return fallback_draft

        answer, steps, confidence, warnings = self.fallback_solver.solve(
            text, problem_type, difficulty
        )
        warnings = self._without_backend_config_warnings(warnings)
        return StructuredSolveDraft(
            answer=answer,
            steps=steps,
            confidence=confidence,
            warnings=warnings,
            parts=[],
            solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
        )

    def _model_candidates(
        self, preferred_model: str
    ) -> list[StructuredModelEndpoint]:
        return list(structured_model_endpoints(preferred_model))

    def _log_structured_step_quality(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None,
        provider: str,
        model: str,
        stage: str,
    ) -> list[StructuredContentIssue]:
        issues = structured_content_issues(payload)
        issue_codes_by_path: dict[str, list[str]] = {}
        for issue in issues:
            issue_codes_by_path.setdefault(issue.path, []).append(issue.code)

        step_locations: list[tuple[str, dict[str, Any]]] = []
        for step_index, step in enumerate(payload.get("steps", [])):
            if isinstance(step, dict):
                step_locations.append((f"steps[{step_index}]", step))
        for part_index, part in enumerate(payload.get("parts", [])):
            if not isinstance(part, dict):
                continue
            for step_index, step in enumerate(part.get("steps", [])):
                if isinstance(step, dict):
                    step_locations.append(
                        (f"parts[{part_index}].steps[{step_index}]", step)
                    )

        for path, step in step_locations:
            latex = step.get("latex")
            latex_empty = not (
                isinstance(latex, list)
                and any(isinstance(item, str) and item.strip() for item in latex)
            )
            logger.info(
                "Structured solve step content quality request_id=%s provider=%s "
                "model=%s stage=%s step_path=%s latex_empty=%s issue_codes=%s",
                request_id,
                provider,
                model,
                stage,
                path,
                latex_empty,
                json.dumps(sorted(issue_codes_by_path.get(path, []))),
            )
        return issues

    async def _repair_structured_content(
        self,
        payload: dict[str, Any],
        issues: list[StructuredContentIssue],
        *,
        completion_client: Any,
        candidate: StructuredModelEndpoint,
        timeout_seconds: float,
        request_id: str | None,
    ) -> dict[str, Any]:
        scratch_issues = [
            issue for issue in issues if issue.code == "scratch_work_style"
        ]
        if not scratch_issues:
            return payload

        issue_summary = [
            {"path": issue.path, "fields": list(issue.fields)}
            for issue in scratch_issues
        ]
        logger.warning(
            "Structured solve content repair started request_id=%s provider=%s "
            "model=%s operation=structured_math_solution_repair flagged_steps=%s",
            request_id,
            candidate.provider,
            candidate.model,
            len(scratch_issues),
        )
        try:
            repair_payload = await completion_client.complete_json(
                model=candidate.model,
                system_prompt=STRUCTURED_CONTENT_REPAIR_SYSTEM_PROMPT,
                user_prompt=(
                    "Flagged fields:\n"
                    + json.dumps(issue_summary, ensure_ascii=False, separators=(",", ":"))
                    + "\n\nOriginal solution JSON:\n"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                ),
                temperature=0.1,
                json_schema=SOLVE_RESPONSE_JSON_SCHEMA,
                schema_name="structured_math_solution_repair",
                require_parameters=False,
                allow_schema_downgrade=False,
                repair_invalid_json=False,
                timeout_seconds=timeout_seconds,
                operation="structured_math_solution_repair",
                trace_id=request_id,
            )
            repaired = _StrictStructuredSolvePayload.model_validate(
                repair_payload
            ).model_dump(mode="python")
            merged = self._merge_structured_content_repair(
                payload, repaired, scratch_issues
            )
            remaining = sum(
                issue.code == "scratch_work_style"
                for issue in structured_content_issues(merged)
            )
            logger.info(
                "Structured solve content repair completed request_id=%s provider=%s "
                "model=%s operation=structured_math_solution_repair remaining_scratch_issues=%s",
                request_id,
                candidate.provider,
                candidate.model,
                remaining,
            )
            return merged
        except Exception as exc:
            diagnostics = exception_diagnostics(exc)
            logger.warning(
                "Structured solve content repair failed request_id=%s provider=%s "
                "model=%s operation=structured_math_solution_repair error_type=%s "
                "error_message=%s status_code=%s response_body=%s",
                request_id,
                candidate.provider,
                candidate.model,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            return payload

    def _merge_structured_content_repair(
        self,
        original: dict[str, Any],
        repaired: dict[str, Any],
        issues: list[StructuredContentIssue],
    ) -> dict[str, Any]:
        merged = deepcopy(original)
        for issue in issues:
            original_step = self._structured_step_at(merged, issue)
            repaired_step = self._structured_step_at(repaired, issue)
            if original_step is None or repaired_step is None:
                raise ValueError(f"Structured content repair omitted {issue.path}")
            for field_name in issue.fields:
                original_step[field_name] = repaired_step[field_name]
            repaired_latex = repaired_step.get("latex")
            if isinstance(repaired_latex, list) and any(
                isinstance(item, str) and item.strip() for item in repaired_latex
            ):
                original_step["latex"] = repaired_latex
        return _StrictStructuredSolvePayload.model_validate(merged).model_dump(
            mode="python"
        )

    def _structured_step_at(
        self, payload: dict[str, Any], issue: StructuredContentIssue
    ) -> dict[str, Any] | None:
        steps: Any
        if issue.part_index is None:
            steps = payload.get("steps", [])
        else:
            parts = payload.get("parts", [])
            if not isinstance(parts, list) or issue.part_index >= len(parts):
                return None
            part = parts[issue.part_index]
            if not isinstance(part, dict):
                return None
            steps = part.get("steps", [])
        if not isinstance(steps, list) or issue.step_index >= len(steps):
            return None
        step = steps[issue.step_index]
        return step if isinstance(step, dict) else None

    def _append_content_quality_warnings(
        self,
        payload: dict[str, Any],
        issues: list[StructuredContentIssue],
    ) -> None:
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            payload["warnings"] = warnings
        for issue in issues:
            if issue.code == "missing_latex_for_math":
                warning = (
                    f"{issue.path} contains math notation but has no LaTeX formula "
                    "for math rendering."
                )
            elif issue.code == "scratch_work_style":
                warning = (
                    f"{issue.path} may contain unedited scratch work after one bounded "
                    "cleanup attempt."
                )
            else:
                continue
            if warning not in warnings:
                warnings.append(warning)

    def _structured_failure_code(self, error: Exception) -> str:
        explicit_failure_code = getattr(error, "failure_code", None)
        if isinstance(explicit_failure_code, str):
            return explicit_failure_code
        diagnostics = exception_diagnostics(error)
        message = diagnostics.error_message.casefold()
        if diagnostics.status_code == 429 or "rate limit" in message:
            return "rate_limited"
        if diagnostics.status_code == 404 and (
            "no endpoint" in message or "requested parameters" in message
        ):
            return "structured_output_provider_unavailable"
        if "invalid json" in message:
            return "raw_parse_failure"
        if isinstance(error, ValueError):
            return "schema_validation_failure"
        if "timed out" in message:
            return "timeout"
        return "request_failure"

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
            "response_version": 2,
            "text": normalized_text,
            "language": request.input.language,
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
