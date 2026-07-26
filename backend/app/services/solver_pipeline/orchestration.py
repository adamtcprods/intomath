"""Top-level request orchestration for the solver service."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.core.model_policy import (
    LOCAL_DETERMINISTIC_SOLVER_MODEL,
    LOCAL_LLAMA_GEOMETRY_PARSER_MODEL,
    SolveRoute,
    VISION_MODEL,
)
from app.core.solve_metrics import (
    SolveMetrics,
    bind_solve_metrics,
    current_solve_metrics,
    emit_solve_metrics,
    reset_solve_metrics,
)
from app.integrations.errors import exception_diagnostics
from app.repositories.result_repository import SolveTimings
from app.services.model_router import RoutingDecision
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
from .routing import (
    with_local_solver_routing,
    with_remote_solver_routing,
    with_structured_solver_routing,
    without_backend_config_warnings,
)
from .subquestion import detect_subquestions

if TYPE_CHECKING:
    from app.services.geometry_extractor import GeometryExtractionResult
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


def _measure_stage(stage: str) -> AbstractContextManager[None]:
    metrics = current_solve_metrics()
    return metrics.measure(stage) if metrics is not None else nullcontext()


def _cached_response_is_usable(
    request: SolveRequest,
    response: SolveResponse,
) -> bool:
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
    metrics = SolveMetrics(request_id=request_id)
    metrics_token = bind_solve_metrics(metrics)
    request_started_at = datetime.now(UTC)
    execution = _SolveExecution(request_id=request_id)
    try:
        outcome = await _run_solve_with_timeout(
            service,
            request,
            response_cache,
            single_flight or _RESPONSE_SINGLE_FLIGHT,
            execution=execution,
        )
        solve_completed_at = datetime.now(UTC)
        solve_duration_ms = (time.monotonic() - metrics.started_at) * 1000.0
        await _persist_solve_outcome(
            service,
            request,
            outcome,
            request_started_at=request_started_at,
            solve_completed_at=solve_completed_at,
            solve_duration_ms=solve_duration_ms,
            execution=execution,
        )
        execution.stage = "completed"
        metrics.outcome = "ok"
        return outcome.response
    except SolveRequestTimeoutError:
        metrics.outcome = "timeout"
        raise
    except Exception:
        metrics.outcome = "error"
        raise
    finally:
        emit_solve_metrics(metrics, logger)
        reset_solve_metrics(metrics_token)


async def _run_solve_with_timeout(
    service: SolverService,
    request: SolveRequest,
    response_cache: TTLCache[SolveResponse],
    single_flight: AsyncSingleFlight[_ComputedSolveResponse],
    *,
    execution: _SolveExecution,
) -> _SolveOutcome:
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
            return await _execute_solve_request(
                service,
                request,
                response_cache,
                single_flight,
                execution=execution,
            )
    except TimeoutError as exc:
        if not timeout_scope.expired():
            raise
        logger.error(
            "Solve request total timeout request_id=%s stage=%s timeout_seconds=%.1f",
            execution.request_id,
            execution.stage,
            timeout_seconds,
        )
        raise SolveRequestTimeoutError(
            request_id=execution.request_id,
            stage=execution.stage,
        ) from exc


async def _persist_solve_outcome(
    service: SolverService,
    request: SolveRequest,
    outcome: _SolveOutcome,
    *,
    request_started_at: datetime,
    solve_completed_at: datetime,
    solve_duration_ms: float,
    execution: _SolveExecution,
) -> None:
    repository = getattr(service, "result_repository", None)
    if repository is None:
        return
    execution.stage = "persistence"
    metrics = current_solve_metrics()
    timer = metrics.measure("persistence") if metrics is not None else nullcontext()
    with timer:
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
    metrics = current_solve_metrics()
    ocr_timer = metrics.measure("ocr") if metrics is not None else nullcontext()
    with ocr_timer:
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
        if not _cached_response_is_usable(request, cached_response):
            logger.info(
                "Solve cache bypassed request_id=%s cache_key_prefix=%s "
                "reason=visualizable_prompt_missing_visualization",
                request_id,
                cache_key[:12],
            )
            response_cache.delete(cache_key)
            if metrics is not None:
                metrics.cache_status = "bypassed"
            logger.info(
                "Solve cache lookup request_id=%s cache=response result=miss "
                "cache_key_prefix=%s reason=unusable",
                request_id,
                cache_key[:12],
            )
        else:
            if metrics is not None:
                metrics.cache_status = "hit"
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
        if metrics is not None:
            metrics.cache_status = "miss"
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
    computed, is_leader = await single_flight.run(
        cache_key,
        lambda: _compute_solve_response(
            service,
            request,
            response_cache,
            cache_key=cache_key,
            prepared=prepared,
            execution=execution,
        ),
        on_role=lambda leader: _record_single_flight_role(
            request_id=request_id,
            cache_key=cache_key,
            is_leader=leader,
        ),
    )
    if metrics is not None and not is_leader:
        metrics.cache_status = "coalesced"
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


def _record_single_flight_role(
    *,
    request_id: str,
    cache_key: str,
    is_leader: bool,
) -> None:
    role = "leader" if is_leader else "waiter"
    metrics = current_solve_metrics()
    if metrics is not None:
        metrics.single_flight_role = role
    logger.info(
        "Solve single-flight request_id=%s cache_key_prefix=%s role=%s",
        request_id,
        cache_key[:12],
        role,
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
    metrics = current_solve_metrics()
    solver_duration_before = (
        metrics.durations_ms["solver"] if metrics is not None else 0.0
    )

    execution.stage = "exact_solver"
    exact_solver = getattr(service, "try_solve_exact", None)
    with _measure_stage("solver"):
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
        with _measure_stage("routing"):
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
        with _measure_stage("solver"):
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
            routing = with_remote_solver_routing(
                routing,
                reason=rejection_reason,
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
            routing = with_local_solver_routing(
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
        with _measure_stage("solver"):
            draft = await service._solve_structured(
                text=normalized_text,
                problem_type=routing.problem_type,
                difficulty=routing.difficulty,
                model=routing.solver_model,
                subquestions=detected_subquestions,
                request_id=request_id,
            )
        if draft.solver_model:
            routing = with_structured_solver_routing(
                routing, solver_model=draft.solver_model
            )
    warnings.extend(draft.warnings)
    solver_duration_ms = (
        metrics.durations_ms["solver"] - solver_duration_before
        if metrics is not None
        else 0.0
    )

    visualization, visualization_duration_ms = await _build_visualization(
        service,
        request,
        routing=routing,
        draft=draft,
        solve_text=solve_text,
        warnings=warnings,
        execution=execution,
    )

    execution.stage = "response_building"
    public_warnings = without_backend_config_warnings(warnings)
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
    if not _cached_response_is_usable(request, response):
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


async def _build_visualization(
    service: SolverService,
    request: SolveRequest,
    *,
    routing: RoutingDecision,
    draft: StructuredSolveDraft,
    solve_text: str,
    warnings: list[str],
    execution: _SolveExecution,
) -> tuple[VisualizationPayload, float]:
    started = time.monotonic()
    visualization = VisualizationPayload(
        kind="none",
        summary=None,
        dsl=None,
        geogebra=None,
    )
    environment = routing.visualization_environment
    if not request.options.include_visualization:
        return visualization, _finish_visualization_timing(started)
    if environment is None:
        logger.info(
            "Visualization skipped before extraction request_id=%s "
            "visualization_environment=none reason=model_router_selected_none",
            execution.request_id,
        )
        return visualization, _finish_visualization_timing(started)

    stage = "extraction"
    try:
        execution.stage = "visualization_extraction"
        logger.info(
            "Visualization extraction started request_id=%s problem_type=%s parser_model=%s",
            execution.request_id,
            routing.problem_type.value,
            routing.parser_model,
        )
        extraction = await service.geometry_extractor.extract(
            solve_text,
            routing.parser_model,
            environment=environment,
            semantic_query_terms=routing.visualization_search_terms,
            request_id=execution.request_id,
            request_deadline=execution.deadline,
        )
        warnings.extend(extraction.warnings)
        if extraction.warnings:
            logger.warning(
                "Visualization extraction completed with warnings request_id=%s warning_count=%s",
                execution.request_id,
                len(extraction.warnings),
            )
        if extraction.dsl.actions:
            stage = "translation"
            execution.stage = "visualization_translation"
            visualization = _translate_visualization(
                service,
                extraction,
                solve_text=solve_text,
                solved_answer_text=_solved_answer_text(draft),
                warnings=warnings,
                request_id=execution.request_id,
            )
    except Exception as exc:
        diagnostics = exception_diagnostics(exc)
        logger.warning(
            "Visualization generation failed open request_id=%s stage=%s "
            "error_type=%s error_message=%s status_code=%s response_body=%s",
            execution.request_id,
            stage,
            diagnostics.error_type,
            diagnostics.error_message,
            diagnostics.status_code,
            diagnostics.response_body,
        )
        warnings.append(
            "Visualization generation failed unexpectedly, so no shape could be constructed."
        )
    return visualization, _finish_visualization_timing(started)


def _translate_visualization(
    service: SolverService,
    extraction: GeometryExtractionResult,
    *,
    solve_text: str,
    solved_answer_text: str,
    warnings: list[str],
    request_id: str,
) -> VisualizationPayload:
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
        "graph" if extraction.dsl.environment.value == "graphing" else "geogebra"
    )
    if not translation.commands:
        kind = "none"
    return VisualizationPayload(
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
            retrieved_commands=_retrieved_debug(service, extraction),
        ),
    )


def _retrieved_debug(
    service: SolverService,
    extraction: GeometryExtractionResult,
) -> list[dict[str, object]]:
    if not getattr(service.geometry_extractor.settings, "app_debug", False):
        return []
    return [
        {
            "name": command.name,
            "score": command.score,
            "signatures": list(command.signatures),
        }
        for command in extraction.retrieved_commands
    ]


def _solved_answer_text(draft: StructuredSolveDraft) -> str:
    return "\n".join(
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


def _finish_visualization_timing(started: float) -> float:
    duration_ms = (time.monotonic() - started) * 1000.0
    metrics = current_solve_metrics()
    if metrics is not None:
        metrics.durations_ms["visualization"] += duration_ms
    return duration_ms


__all__ = ["SolveRequestTimeoutError", "solve_request"]
