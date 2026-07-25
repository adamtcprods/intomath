"""Top-level request orchestration for the solver service."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.integrations.errors import exception_diagnostics
from app.repositories.result_repository import SolveTimings
from app.services.model_router import (
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    RoutingDecision,
    SolveRoute,
    VISION_MODEL,
)
from app.schemas.solve import (
    GeoGebraPayload,
    RoutingPayload,
    SolveRequest,
    SolveResponse,
    VisualizationPayload,
)
from app.services.cache import AsyncSingleFlight, TTLCache
from app.services.input_validation import validate_solve_input

from .errors import SolveRequestTimeoutError
from .response_builder import StructuredSolveDraft
from .subquestion import detect_subquestions

if TYPE_CHECKING:
    from app.services.solver_service import SolverService


logger = logging.getLogger(__name__)


@dataclass
class _SolveExecution:
    request_id: str
    stage: str = "input_validation"
    deadline: float | None = None


@dataclass(frozen=True)
class _PreparedSolveInput:
    normalized_text: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ComputedSolveResponse:
    response: SolveResponse
    solver_duration_ms: float
    visualization_duration_ms: float


@dataclass(frozen=True)
class _SolveOutcome:
    response: SolveResponse
    raw_text: str
    normalized_text: str
    solver_duration_ms: float
    visualization_duration_ms: float


_RESPONSE_SINGLE_FLIGHT = AsyncSingleFlight()


def _cached_response_is_usable(
    request: SolveRequest,
    normalized_text: str,
    response: SolveResponse,
) -> bool:
    _ = normalized_text
    if not request.options.include_visualization:
        return True
    if response.visualization.kind != "none":
        return True
    if response.routing.visualization_environment is not None:
        return False
    if response.problem_type == "geometry":
        # Older router results could contradict themselves by classifying a
        # planar geometry prompt while selecting no visualization environment.
        return False
    return "left unclassified" not in response.routing.reason.casefold()


async def solve_request(
    service: SolverService,
    request: SolveRequest,
    response_cache: TTLCache[SolveResponse],
    single_flight: AsyncSingleFlight[_ComputedSolveResponse] | None = None,
) -> SolveResponse:
    request_id = str(uuid.uuid4())
    request_started_at = datetime.now(UTC)
    request_started = time.monotonic()
    execution = _SolveExecution(request_id=request_id)
    timeout_seconds = float(
        getattr(
            getattr(service, "settings", None),
            "solve_request_timeout_seconds",
            70.0,
        )
    )
    execution.deadline = time.monotonic() + timeout_seconds
    timeout_scope = asyncio.timeout(timeout_seconds)
    try:
        async with timeout_scope:
            outcome = await _execute_solve_request(
                service,
                request,
                response_cache,
                single_flight or _RESPONSE_SINGLE_FLIGHT,
                execution=execution,
            )
    except TimeoutError as exc:
        if not timeout_scope.expired():
            raise
        logger.error(
            "Solve request total timeout request_id=%s stage=%s timeout_seconds=%.1f",
            request_id,
            execution.stage,
            timeout_seconds,
        )
        raise SolveRequestTimeoutError(
            request_id=request_id,
            stage=execution.stage,
        ) from exc

    solve_completed_at = datetime.now(UTC)
    solve_duration_ms = (time.monotonic() - request_started) * 1000.0
    repository = getattr(service, "result_repository", None)
    if repository is not None:
        execution.stage = "persistence"
        await repository.save(
            request,
            outcome.raw_text,
            outcome.normalized_text,
            outcome.response,
            SolveTimings(
                request_started_at=request_started_at,
                solve_completed_at=solve_completed_at,
                solve_duration_ms=solve_duration_ms,
                solver_duration_ms=outcome.solver_duration_ms,
                visualization_duration_ms=outcome.visualization_duration_ms,
            ),
        )
    execution.stage = "completed"
    return outcome.response


async def _execute_solve_request(
    service: SolverService,
    request: SolveRequest,
    response_cache: TTLCache[SolveResponse],
    single_flight: AsyncSingleFlight[_ComputedSolveResponse],
    *,
    execution: _SolveExecution,
) -> _SolveOutcome:
    request_id = execution.request_id
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

    execution.stage = "input_validation"
    validated_input = validate_solve_input(
        request.input, getattr(service, "settings", None)
    )
    execution.stage = "ocr"
    ocr_result = await service.ocr_service.extract_problem_text(
        validated_input.image_bytes,
        validated_input.image_mime_type,
    )
    if validated_input.image_bytes:
        logger.info(
            "OCR cache result request_id=%s cache=ocr result=%s",
            request_id,
            "hit" if bool(getattr(ocr_result, "cached", False)) else "miss",
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

    execution.stage = "cache_lookup"
    cache_key = service._build_cache_key(normalized_text, request)
    cached_response = response_cache.get(cache_key)
    if cached_response is not None:
        if not _cached_response_is_usable(
            request, normalized_text, cached_response
        ):
            logger.info(
                "Solve cache bypassed request_id=%s cache_key_prefix=%s "
                "reason=visualizable_prompt_missing_visualization",
                request_id,
                cache_key[:12],
            )
            response_cache.delete(cache_key)
            logger.info(
                "Solve cache lookup request_id=%s cache=response result=miss "
                "cache_key_prefix=%s reason=unusable",
                request_id,
                cache_key[:12],
            )
        else:
            logger.info(
                "Solve cache lookup request_id=%s cache=response result=hit "
                "cache_key_prefix=%s",
                request_id,
                cache_key[:12],
            )
            response = cached_response.model_copy(
                update={"request_id": request_id, "cached": True},
                deep=True,
            )
            return _SolveOutcome(
                response=response,
                raw_text=raw_text,
                normalized_text=normalized_text,
                solver_duration_ms=0.0,
                visualization_duration_ms=0.0,
            )
    else:
        logger.info(
            "Solve cache lookup request_id=%s cache=response result=miss "
            "cache_key_prefix=%s reason=not_found",
            request_id,
            cache_key[:12],
        )

    prepared = _PreparedSolveInput(
        normalized_text=normalized_text,
        warnings=tuple(warnings),
    )
    execution.stage = "single_flight_wait"
    computed, _is_leader = await single_flight.run(
        cache_key,
        lambda: _compute_solve_response(
            service,
            request,
            response_cache,
            cache_key=cache_key,
            prepared=prepared,
            execution=execution,
        ),
        on_role=lambda leader: logger.info(
            "Solve single-flight request_id=%s cache_key_prefix=%s role=%s",
            request_id,
            cache_key[:12],
            "leader" if leader else "waiter",
        ),
    )
    response = computed.response.model_copy(
        update={"request_id": request_id, "cached": False},
        deep=True,
    )
    return _SolveOutcome(
        response=response,
        raw_text=raw_text,
        normalized_text=normalized_text,
        solver_duration_ms=computed.solver_duration_ms,
        visualization_duration_ms=computed.visualization_duration_ms,
    )


async def _compute_solve_response(
    service: SolverService,
    request: SolveRequest,
    response_cache: TTLCache[SolveResponse],
    *,
    cache_key: str,
    prepared: _PreparedSolveInput,
    execution: _SolveExecution,
) -> _ComputedSolveResponse:
    request_id = execution.request_id
    normalized_text = prepared.normalized_text
    warnings = list(prepared.warnings)
    detected_subquestions = detect_subquestions(normalized_text)
    solver_started = time.monotonic()

    execution.stage = "exact_solver"
    exact_solver = getattr(service, "try_solve_exact", None)
    exact_result = (
        exact_solver(normalized_text) if callable(exact_solver) else None
    )
    if exact_result is not None:
        routing = RoutingDecision(
            problem_type=exact_result.problem_type,
            difficulty=exact_result.difficulty,
            parser_model=LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
            solver_model=LOCAL_DETERMINISTIC_SOLVER_MODEL,
            vision_model=(
                VISION_MODEL if bool(request.input.image_base64) else None
            ),
            visualization_environment=exact_result.visualization_environment,
            reason=(
                f"deterministic local solver used because it {exact_result.reason}"
            ),
            visualization_search_terms=exact_result.visualization_search_terms,
            solve_route=SolveRoute.deterministic,
            normalized_prompt=exact_result.normalized_prompt,
        )
    else:
        execution.stage = "routing"
        routing = await service.router.route_async(
            normalized_text, has_image=bool(request.input.image_base64)
        )
    logger.info(
        "Solve routing request_id=%s problem_type=%s difficulty=%s parser_model=%s "
        "solver_model=%s solve_route=%s vision_model=%s visualization_environment=%s",
        request_id,
        routing.problem_type.value,
        routing.difficulty.value,
        routing.parser_model,
        routing.solver_model,
        routing.solve_route.value,
        routing.vision_model,
        (
            routing.visualization_environment.value
            if routing.visualization_environment is not None
            else "none"
        ),
    )

    solve_text = normalized_text
    local_result = (
        exact_result.local_result if exact_result is not None else None
    )
    selected_solver = getattr(
        service.local_solver_selector,
        "solve_selected_route",
        None,
    )
    if (
        exact_result is None
        and routing.solve_route is not SolveRoute.remote
        and callable(selected_solver)
    ):
        execution.stage = "selected_local_solver"
        local_result = await selected_solver(
            normalized_text,
            routing.problem_type,
            routing.difficulty,
            routing.solve_route,
        )
        if local_result is None:
            rejected_route = routing.solve_route
            rejection_reason = (
                "deterministic parser rejected the original prompt; fell through "
                "safely to remote solving"
                if rejected_route is SolveRoute.deterministic
                else "selected local trivia solver declined; fell through to remote solving"
            )
            routing = service._with_remote_solver_routing(
                routing,
                reason=rejection_reason,
            )
    elif exact_result is None and not callable(selected_solver):
        # Compatibility for injected pre-unification selectors.
        execution.stage = "legacy_local_solver"
        local_result = await service.local_solver_selector.solve_if_supported(
            normalized_text,
            routing.problem_type,
            routing.difficulty,
        )

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
        if exact_result is None:
            routing = service._with_local_solver_routing(
                routing, local_result, original_text=normalized_text
            )
    else:
        execution.stage = "structured_solution"
        logger.info(
            "Using structured model solve request_id=%s model=%s subquestions=%s",
            request_id,
            routing.solver_model,
            len(detected_subquestions),
        )
        draft = await service._solve_structured(
            text=normalized_text,
            problem_type=routing.problem_type,
            difficulty=routing.difficulty,
            model=routing.solver_model,
            subquestions=detected_subquestions,
            request_id=request_id,
        )
        if draft.solver_model:
            routing = service._with_structured_solver_routing(
                routing, solver_model=draft.solver_model
            )
    warnings.extend(draft.warnings)
    solver_duration_ms = (time.monotonic() - solver_started) * 1000.0

    visualization_started = time.monotonic()
    visualization = VisualizationPayload(
        kind="none", summary=None, dsl=None, geogebra=None
    )
    visualization_environment = routing.visualization_environment
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
    if request.options.include_visualization and visualization_environment is not None:
        visualization_stage = "extraction"
        try:
            execution.stage = "visualization_extraction"
            logger.info(
                "Visualization extraction started request_id=%s problem_type=%s parser_model=%s",
                request_id,
                routing.problem_type.value,
                routing.parser_model,
            )
            extraction = await service.geometry_extractor.extract(
                solve_text,
                routing.parser_model,
                environment=visualization_environment,
                semantic_query_terms=routing.visualization_search_terms,
                request_id=request_id,
                request_deadline=execution.deadline,
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
                execution.stage = "visualization_translation"
                translation = service.translator.translate(
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
                if getattr(service.geometry_extractor.settings, "app_debug", False):
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
            "Visualization skipped before extraction request_id=%s "
            "visualization_environment=none reason=model_router_selected_none",
            request_id,
        )
    visualization_duration_ms = (time.monotonic() - visualization_started) * 1000.0

    execution.stage = "response_building"
    public_warnings = service._without_backend_config_warnings(warnings)
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
            visualization_environment=routing.visualization_environment,
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

    execution.stage = "cache_write"
    if not _cached_response_is_usable(request, solve_text, response):
        logger.info(
            "Solve response not cached request_id=%s reason=visualizable_prompt_missing_visualization",
            request_id,
        )
    else:
        response_cache.set(cache_key, response)
    execution.stage = "completed"
    return _ComputedSolveResponse(
        response=response,
        solver_duration_ms=solver_duration_ms,
        visualization_duration_ms=visualization_duration_ms,
    )


__all__ = ["SolveRequestTimeoutError", "solve_request"]
