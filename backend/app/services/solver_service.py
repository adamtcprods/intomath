"""Dependency-injected orchestration facade for the IntoMath solver pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.integrations.errors import exception_diagnostics
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.integrations.protocols import StructuredCompletionClient
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.cache import TTLCache
from app.services.fallback_solver import FallbackSolver
from app.services.geogebra_translator import GeoGebraTranslator
from app.services.geometry_extractor import GeometryExtractor
from app.services.local_solver_selector import LocalSolverSelector
from app.services.local_solver_types import LocalSolveResult
from app.services.model_router import (
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    ModelRouter,
    RoutingDecision,
    StructuredModelEndpoint,
    remote_model_timeout_seconds,
    structured_model_endpoints,
)
from app.services.ocr_service import OCRService
from app.services.solver_pipeline.content_repair import (
    append_content_quality_warnings,
    log_structured_step_quality,
    repair_missing_structured_steps,
    repair_structured_content,
)
from app.services.solver_pipeline.orchestration import solve_request
from app.services.solver_pipeline.prompts import (
    SOLVE_RESPONSE_JSON_SCHEMA,
    STRUCTURED_SOLVE_SYSTEM_PROMPT,
)
from app.services.solver_pipeline.response_builder import (
    StructuredSolveDraft,
    draft_from_payload,
    fallback_draft_for_subquestions,
)
from app.services.solver_pipeline.routing import (
    is_supported_local_draft,
    with_local_solver_routing,
    with_local_subquestion_routing,
    with_structured_solver_routing,
    without_backend_config_warnings,
)
from app.services.solver_pipeline.strict_models import (
    validate_structured_payload,
)
from app.services.solver_pipeline.subquestion import (
    DetectedSubquestion,
    detect_subquestions,
    format_subquestion_hint,
)

# Re-exports for backward compatibility
from app.services.solver_pipeline.content_quality import (  # noqa: F401
    StructuredContentIssue,
    contains_clear_math_notation,
    looks_like_scratch_work,
    structured_content_issues,
)


_RESPONSE_CACHE: TTLCache[SolveResponse] = TTLCache(ttl_seconds=900, max_size=500)
logger = logging.getLogger(__name__)
_CACHE_RESPONSE_VERSION = 5
_VISUALIZATION_PIPELINE_VERSION = "geogebra-catalog-1.4-semantic-retrieval-v4-dimensions"


class SolverService:
    def __init__(
        self,
        db: Session,
        *,
        nvidia_client: StructuredCompletionClient | None = None,
        llama_client: LlamaClient | None = None,
        router: ModelRouter | None = None,
        ocr_service: OCRService | None = None,
        geometry_extractor: GeometryExtractor | None = None,
        translator: GeoGebraTranslator | None = None,
        fallback_solver: FallbackSolver | None = None,
        local_solver_selector: LocalSolverSelector | None = None,
        settings: Any | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self._owns_nvidia_client = nvidia_client is None
        self._owns_llama_client = llama_client is None
        self.nvidia_client = (
            nvidia_client
            if nvidia_client is not None
            else NvidiaClient(self.settings)
        )
        self.llama_client = (
            llama_client if llama_client is not None else LlamaClient(self.settings)
        )
        self.router = router or ModelRouter(self.llama_client)
        self.ocr_service = ocr_service or OCRService()
        self.geometry_extractor = geometry_extractor or GeometryExtractor(
            self.llama_client, nvidia_client=self.nvidia_client
        )
        self.translator = translator or GeoGebraTranslator()
        self.fallback_solver = fallback_solver or FallbackSolver()
        self.local_solver_selector = local_solver_selector or LocalSolverSelector(
            self.fallback_solver, self.llama_client
        )

    async def aclose(self) -> None:
        """Close only model clients constructed by this service.

        Request-scoped services receive lifespan clients and therefore cannot close
        the shared pools through this method.
        """
        try:
            if self._owns_nvidia_client:
                await self.nvidia_client.aclose()  # type: ignore[attr-defined]
        finally:
            if self._owns_llama_client:
                await self.llama_client.aclose()

    async def solve(self, request: SolveRequest) -> SolveResponse:
        return await solve_request(self, request, _RESPONSE_CACHE)

    def _with_local_solver_routing(
        self,
        routing: RoutingDecision,
        local_result: LocalSolveResult,
        *,
        original_text: str,
    ) -> RoutingDecision:
        return with_local_solver_routing(
            routing, local_result, original_text=original_text
        )

    def _with_structured_solver_routing(
        self, routing: RoutingDecision, *, solver_model: str
    ) -> RoutingDecision:
        return with_structured_solver_routing(routing, solver_model=solver_model)

    def _with_local_subquestion_routing(
        self, routing: RoutingDecision, *, reason: str
    ) -> RoutingDecision:
        return with_local_subquestion_routing(routing, reason=reason)

    def _is_supported_local_draft(self, draft: StructuredSolveDraft) -> bool:
        return is_supported_local_draft(draft)

    def _without_backend_config_warnings(self, warnings: list[str]) -> list[str]:
        return without_backend_config_warnings(warnings)

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
        return fallback_draft_for_subquestions(
            fallback_solver=self.fallback_solver,
            text=text,
            problem_type=problem_type,
            difficulty=difficulty,
            subquestions=subquestions,
            warning=warning,
            suppress_config_warnings=suppress_config_warnings,
            warning_filter=self._without_backend_config_warnings,
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
                f"{format_subquestion_hint(subquestions)}\nProblem:\n{text}"
            )
            model_failures: list[tuple[str, str, str, str, int | None]] = []
            candidates = self._model_candidates(model)
            for attempt_number, candidate in enumerate(candidates, start=1):
                candidate_client: StructuredCompletionClient = self.nvidia_client
                attempt_timeout = remote_model_timeout_seconds(
                    self.settings, provider=candidate.provider, model=candidate.model
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
                    validated_payload = validate_structured_payload(
                        payload,
                        request_id=request_id,
                        provider=candidate.provider,
                        model=candidate.model,
                    )
                    validated_payload = await repair_missing_structured_steps(
                        validated_payload,
                        problem_text=text,
                        completion_client=candidate_client,
                        candidate=candidate,
                        timeout_seconds=attempt_timeout,
                        request_id=request_id,
                    )
                    initial_issues = log_structured_step_quality(
                        validated_payload,
                        request_id=request_id,
                        provider=candidate.provider,
                        model=candidate.model,
                        stage="initial",
                    )
                    if any(i.code == "scratch_work_style" for i in initial_issues):
                        validated_payload = await repair_structured_content(
                            validated_payload,
                            initial_issues,
                            completion_client=candidate_client,
                            candidate=candidate,
                            timeout_seconds=attempt_timeout,
                            request_id=request_id,
                        )
                    final_issues = log_structured_step_quality(
                        validated_payload,
                        request_id=request_id,
                        provider=candidate.provider,
                        model=candidate.model,
                        stage="final",
                    )
                    append_content_quality_warnings(validated_payload, final_issues)
                    draft = draft_from_payload(
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

            answer, steps, confidence, warnings = self.fallback_solver.solve(
                text, problem_type, difficulty
            )
            warnings = self._without_backend_config_warnings(warnings)
            logger.error(
                "All structured solve model attempts failed; using local fallback problem_type=%s difficulty=%s failures=%s",
                problem_type.value,
                difficulty.value,
                model_failures,
            )
            fallback_notice = "The preferred solver was unavailable; this result was produced by the local solver."
            if len(subquestions) > 1:
                draft = self._fallback_draft_for_subquestions(
                    text=text,
                    problem_type=problem_type,
                    difficulty=difficulty,
                    subquestions=subquestions,
                    warning=fallback_notice,
                    suppress_config_warnings=True,
                )
                draft.solver_model = LOCAL_DETERMINISTIC_SOLVER_MODEL
                return draft
            warnings.append(fallback_notice)
            return StructuredSolveDraft(
                answer,
                steps,
                confidence,
                warnings,
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
            draft = self._fallback_draft_for_subquestions(
                text=text,
                problem_type=problem_type,
                difficulty=difficulty,
                subquestions=subquestions,
                suppress_config_warnings=True,
            )
            draft.solver_model = LOCAL_DETERMINISTIC_SOLVER_MODEL
            return draft

        answer, steps, confidence, warnings = self.fallback_solver.solve(
            text, problem_type, difficulty
        )
        return StructuredSolveDraft(
            answer,
            steps,
            confidence,
            self._without_backend_config_warnings(warnings),
            parts=[],
            solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
        )

    def _model_candidates(
        self, preferred_model: str
    ) -> list[StructuredModelEndpoint]:
        return list(structured_model_endpoints(preferred_model))

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

    def _build_cache_key(self, normalized_text: str, request: SolveRequest) -> str:
        registry_metadata = getattr(
            getattr(self.translator, "registry", None), "metadata", {}
        )
        payload = {
            "response_version": _CACHE_RESPONSE_VERSION,
            "visualization_pipeline_version": _VISUALIZATION_PIPELINE_VERSION,
            "catalog_schema_version": registry_metadata.get("schema_version"),
            "catalog_generator_version": registry_metadata.get("generator_version"),
            "catalog_upstream_commit": registry_metadata.get("upstream_commit"),
            "text": normalized_text,
            "language": request.input.language,
            "options": request.options.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    # Compatibility delegates for callers of the former private helpers.
    def _detect_subquestions(self, text: str) -> list[DetectedSubquestion]:
        return detect_subquestions(text)

    def _draft_from_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_subquestions: list[DetectedSubquestion] | None = None,
    ) -> StructuredSolveDraft:
        return draft_from_payload(
            payload, expected_subquestions=expected_subquestions
        )


__all__ = [
    "DetectedSubquestion",
    "SolverService",
    "StructuredContentIssue",
    "StructuredSolveDraft",
    "contains_clear_math_notation",
    "looks_like_scratch_work",
    "structured_content_issues",
]
