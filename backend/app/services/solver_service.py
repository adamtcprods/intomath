"""Dependency-injected orchestration facade for the IntoMath solver pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.config import get_settings
from app.core.model_policy import (
    LOCAL_SAFE_FALLBACK_MODEL,
    ModelFailureCategory,
    StructuredModelEndpoint,
    remote_model_timeout_seconds,
    structured_model_endpoints,
)
from app.integrations.errors import exception_diagnostics
from app.integrations.llama_client import LlamaClient
from app.integrations.nvidia_client import NvidiaClient
from app.integrations.protocols import StructuredCompletionClient
from app.repositories.result_repository import ResultRepository
from app.schemas.common import Difficulty, ProblemType
from app.schemas.solve import SolveRequest, SolveResponse
from app.services.cache import TTLCache
from app.services.fallback_solver import FallbackSolver
from app.services.exact_solver import ExactSolveResult
from app.services.geogebra_translator import GeoGebraTranslator
from app.services.geometry_extractor import GeometryExtractor
from app.services.local_solver_selector import LocalSolverSelector
from app.services.model_router import ModelRouter
from app.services.ocr_service import OCRService
from app.services.semantic_router import SemanticRouter
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
)
from app.services.solver_pipeline.routing import without_backend_config_warnings
from app.services.solver_pipeline.strict_models import (
    validate_structured_payload,
)
from app.services.solver_pipeline.subquestion import (
    DetectedSubquestion,
    format_subquestion_hint,
)


_settings = get_settings()
_RESPONSE_CACHE: TTLCache[SolveResponse] = TTLCache(
    ttl_seconds=_settings.response_cache_ttl_seconds,
    max_size=_settings.response_cache_max_size,
)
logger = logging.getLogger(__name__)
_CACHE_RESPONSE_VERSION = 8
_VISUALIZATION_PIPELINE_VERSION = "geogebra-catalog-1.4-semantic-retrieval-v4-dimensions"
_SEMANTIC_ROUTING_POLICY_VERSION = "embedding-independent-axes-v1"


class SolverService:
    def __init__(
        self,
        *,
        result_repository: ResultRepository | None = None,
        nvidia_client: StructuredCompletionClient | None = None,
        llama_client: LlamaClient | None = None,
        semantic_router: SemanticRouter | None = None,
        router: ModelRouter | None = None,
        ocr_service: OCRService | None = None,
        geometry_extractor: GeometryExtractor | None = None,
        translator: GeoGebraTranslator | None = None,
        fallback_solver: FallbackSolver | None = None,
        local_solver_selector: LocalSolverSelector | None = None,
        settings: Any | None = None,
    ) -> None:
        self.result_repository = result_repository
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
        self.semantic_router = semantic_router
        self.router = router or ModelRouter(
            self.llama_client,
            semantic_router=self.semantic_router,
            settings=self.settings,
        )
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

    def try_solve_exact(self, text: str) -> ExactSolveResult | None:
        return self.local_solver_selector.try_solve_exact(text)

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
        if not self.nvidia_client.enabled:
            return self._local_fallback_draft()

        user_prompt = self._structured_user_prompt(
            text=text,
            problem_type=problem_type,
            difficulty=difficulty,
            subquestions=subquestions,
        )
        failures: list[tuple[str, str, str, str, int | None]] = []
        candidates = structured_model_endpoints(model)
        for attempt_number, candidate in enumerate(candidates, start=1):
            attempt_timeout = remote_model_timeout_seconds(
                self.settings,
                provider=candidate.provider,
                model=candidate.model,
            )
            self._log_structured_attempt_started(
                candidate=candidate,
                attempt_number=attempt_number,
                attempt_count=len(candidates),
                timeout_seconds=attempt_timeout,
                problem_type=problem_type,
                difficulty=difficulty,
                subquestion_count=len(subquestions),
                request_id=request_id,
            )
            try:
                draft = await self._attempt_structured_candidate(
                    candidate=candidate,
                    user_prompt=user_prompt,
                    problem_text=text,
                    subquestions=subquestions,
                    timeout_seconds=attempt_timeout,
                    request_id=request_id,
                )
            except Exception as exc:
                failures.append(
                    self._structured_failure_record(candidate, exc)
                )
                self._log_structured_attempt_failed(
                    candidate=candidate,
                    error=exc,
                    attempt_number=attempt_number,
                    attempt_count=len(candidates),
                    problem_type=problem_type,
                    difficulty=difficulty,
                    request_id=request_id,
                )
                continue

            return self._finish_structured_success(
                draft,
                candidate=candidate,
                preferred_model=model,
                request_id=request_id,
            )

        logger.error(
            "All structured solve model attempts failed; using local fallback "
            "problem_type=%s difficulty=%s failures=%s",
            problem_type.value,
            difficulty.value,
            failures,
        )
        return self._local_fallback_draft(provider_unavailable=True)

    @staticmethod
    def _structured_user_prompt(
        *,
        text: str,
        problem_type: ProblemType,
        difficulty: Difficulty,
        subquestions: list[DetectedSubquestion],
    ) -> str:
        return (
            f"Problem type: {problem_type.value}\n"
            f"Difficulty: {difficulty.value}\n"
            f"{format_subquestion_hint(subquestions)}\nProblem:\n{text}"
        )

    async def _attempt_structured_candidate(
        self,
        *,
        candidate: StructuredModelEndpoint,
        user_prompt: str,
        problem_text: str,
        subquestions: list[DetectedSubquestion],
        timeout_seconds: float,
        request_id: str | None,
    ) -> StructuredSolveDraft:
        client: StructuredCompletionClient = self.nvidia_client
        payload = await client.complete_json(
            model=candidate.model,
            system_prompt=STRUCTURED_SOLVE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            json_schema=SOLVE_RESPONSE_JSON_SCHEMA,
            schema_name="structured_math_solution",
            max_tokens=int(
                getattr(self.settings, "structured_solution_max_tokens", 4_500)
            ),
            require_parameters=False,
            allow_schema_downgrade=False,
            repair_invalid_json=False,
            timeout_seconds=timeout_seconds,
            operation="structured_math_solution",
            trace_id=request_id,
        )
        validated = validate_structured_payload(
            payload,
            request_id=request_id,
            provider=candidate.provider,
            model=candidate.model,
        )
        validated = await self._repair_structured_payload(
            validated,
            problem_text=problem_text,
            client=client,
            candidate=candidate,
            timeout_seconds=timeout_seconds,
            request_id=request_id,
        )
        return draft_from_payload(validated, expected_subquestions=subquestions)

    async def _repair_structured_payload(
        self,
        payload: dict[str, Any],
        *,
        problem_text: str,
        client: StructuredCompletionClient,
        candidate: StructuredModelEndpoint,
        timeout_seconds: float,
        request_id: str | None,
    ) -> dict[str, Any]:
        repaired = await repair_missing_structured_steps(
            payload,
            problem_text=problem_text,
            completion_client=client,
            candidate=candidate,
            timeout_seconds=timeout_seconds,
            max_tokens=int(
                getattr(self.settings, "missing_step_repair_max_tokens", 2_000)
            ),
            request_id=request_id,
        )
        initial_issues = log_structured_step_quality(
            repaired,
            request_id=request_id,
            provider=candidate.provider,
            model=candidate.model,
            stage="initial",
        )
        if any(issue.code == "scratch_work_style" for issue in initial_issues):
            repaired = await repair_structured_content(
                repaired,
                initial_issues,
                completion_client=client,
                candidate=candidate,
                timeout_seconds=timeout_seconds,
                max_tokens=int(
                    getattr(self.settings, "content_repair_max_tokens", 2_500)
                ),
                request_id=request_id,
            )
        final_issues = log_structured_step_quality(
            repaired,
            request_id=request_id,
            provider=candidate.provider,
            model=candidate.model,
            stage="final",
        )
        append_content_quality_warnings(repaired, final_issues)
        return repaired

    def _finish_structured_success(
        self,
        draft: StructuredSolveDraft,
        *,
        candidate: StructuredModelEndpoint,
        preferred_model: str,
        request_id: str | None,
    ) -> StructuredSolveDraft:
        if candidate.routing_model != preferred_model:
            logger.warning(
                "Structured solve succeeded with fallback model request_id=%s "
                "preferred_model=%s fallback_provider=%s fallback_model=%s",
                request_id,
                preferred_model,
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

    def _local_fallback_draft(
        self,
        *,
        provider_unavailable: bool = False,
    ) -> StructuredSolveDraft:
        answer, steps, confidence, warnings = self.fallback_solver.unsupported_result()
        public_warnings = without_backend_config_warnings(warnings)
        if provider_unavailable:
            public_warnings.append(
                "The preferred solver was unavailable; this result was produced by the local solver."
            )
        return StructuredSolveDraft(
            answer,
            steps,
            confidence,
            public_warnings,
            parts=[],
            solver_model=LOCAL_SAFE_FALLBACK_MODEL,
        )

    @staticmethod
    def _structured_failure_record(
        candidate: StructuredModelEndpoint,
        error: Exception,
    ) -> tuple[str, str, str, str, int | None]:
        diagnostics = exception_diagnostics(error)
        return (
            candidate.provider,
            candidate.model,
            SolverService._structured_failure_code(error),
            diagnostics.error_type,
            diagnostics.status_code,
        )

    @staticmethod
    def _log_structured_attempt_started(
        *,
        candidate: StructuredModelEndpoint,
        attempt_number: int,
        attempt_count: int,
        timeout_seconds: float,
        problem_type: ProblemType,
        difficulty: Difficulty,
        subquestion_count: int,
        request_id: str | None,
    ) -> None:
        logger.info(
            "Structured solve model attempt started request_id=%s provider=%s "
            "model=%s operation=structured_math_solution attempt=%s/%s "
            "problem_type=%s difficulty=%s subquestions=%s timeout_seconds=%.1f",
            request_id,
            candidate.provider,
            candidate.model,
            attempt_number,
            attempt_count,
            problem_type.value,
            difficulty.value,
            subquestion_count,
            timeout_seconds,
        )

    @staticmethod
    def _log_structured_attempt_failed(
        *,
        candidate: StructuredModelEndpoint,
        error: Exception,
        attempt_number: int,
        attempt_count: int,
        problem_type: ProblemType,
        difficulty: Difficulty,
        request_id: str | None,
    ) -> None:
        diagnostics = exception_diagnostics(error)
        logger.warning(
            "Structured solve model attempt failed request_id=%s provider=%s "
            "model=%s operation=structured_math_solution attempt=%s/%s "
            "problem_type=%s difficulty=%s failure_code=%s error_type=%s "
            "error_message=%s status_code=%s response_body=%s",
            request_id,
            candidate.provider,
            candidate.model,
            attempt_number,
            attempt_count,
            problem_type.value,
            difficulty.value,
            SolverService._structured_failure_code(error),
            diagnostics.error_type,
            diagnostics.error_message,
            diagnostics.status_code,
            diagnostics.response_body,
        )

    @staticmethod
    def _structured_failure_code(error: Exception) -> str:
        explicit_failure_code = getattr(error, "failure_code", None)
        if isinstance(explicit_failure_code, str):
            return explicit_failure_code
        diagnostics = exception_diagnostics(error)
        message = diagnostics.error_message.casefold()
        if diagnostics.status_code == 429 or "rate limit" in message:
            return ModelFailureCategory.rate_limited.value
        if diagnostics.status_code == 404 and (
            "no endpoint" in message or "requested parameters" in message
        ):
            return ModelFailureCategory.provider_unavailable.value
        if "invalid json" in message:
            return ModelFailureCategory.invalid_json.value
        if isinstance(error, ValueError):
            return ModelFailureCategory.invalid_schema.value
        if "timed out" in message:
            return ModelFailureCategory.timeout.value
        return ModelFailureCategory.request_failure.value

    def _build_cache_key(self, normalized_text: str, request: SolveRequest) -> str:
        registry_metadata = getattr(
            getattr(self.translator, "registry", None), "metadata", {}
        )
        payload = {
            "response_version": _CACHE_RESPONSE_VERSION,
            "visualization_pipeline_version": _VISUALIZATION_PIPELINE_VERSION,
            "semantic_routing_policy_version": _SEMANTIC_ROUTING_POLICY_VERSION,
            "semantic_router_artifact": (
                getattr(self, "semantic_router").artifact_identity
                if getattr(self, "semantic_router", None) is not None
                else "llm-only"
            ),
            "catalog_schema_version": registry_metadata.get("schema_version"),
            "catalog_generator_version": registry_metadata.get("generator_version"),
            "catalog_upstream_commit": registry_metadata.get("upstream_commit"),
            "text": normalized_text,
            "input_type": "image" if request.input.image_base64 else "text",
            "language": request.input.language,
            "options": request.options.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

__all__ = ["SolverService"]
