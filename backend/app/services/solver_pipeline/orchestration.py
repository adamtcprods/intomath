"""Top-level request orchestration for the solver service."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from app.integrations.errors import exception_diagnostics
from app.schemas.solve import (
    GeoGebraPayload,
    RoutingPayload,
    SolveRequest,
    SolveResponse,
    VisualizationPayload,
)
from app.services.cache import TTLCache

from .persistence import persist_solve_result
from .response_builder import StructuredSolveDraft
from .subquestion import detect_subquestions

if TYPE_CHECKING:
    from app.services.solver_service import SolverService


logger = logging.getLogger(__name__)


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
) -> SolveResponse:
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

    ocr_result = await service.ocr_service.extract_problem_text(
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

    detected_subquestions = detect_subquestions(normalized_text)
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
        else:
            logger.info(
                "Solve cache hit request_id=%s cache_key_prefix=%s",
                request_id,
                cache_key[:12],
            )
            return cached_response.model_copy(
                update={"request_id": request_id, "cached": True}
            )

    routing = await service.router.route_async(
        normalized_text, has_image=bool(request.input.image_base64)
    )
    logger.info(
        "Solve routing request_id=%s problem_type=%s difficulty=%s parser_model=%s "
        "solver_model=%s vision_model=%s visualization_environment=%s",
        request_id,
        routing.problem_type.value,
        routing.difficulty.value,
        routing.parser_model,
        routing.solver_model,
        routing.vision_model,
        (
            routing.visualization_environment.value
            if routing.visualization_environment is not None
            else "none"
        ),
    )

    solve_text = normalized_text
    local_result = None
    local_subquestion_draft = None
    local_subquestion_reason = None
    if len(detected_subquestions) <= 1:
        local_result = await service.local_solver_selector.solve_if_supported(
            normalized_text,
            routing.problem_type,
            routing.difficulty,
        )
    else:
        candidate_local_subquestion_draft = service._fallback_draft_for_subquestions(
            text=normalized_text,
            problem_type=routing.problem_type,
            difficulty=routing.difficulty,
            subquestions=detected_subquestions,
        )
        if service._is_supported_local_draft(candidate_local_subquestion_draft):
            local_subquestion_draft = candidate_local_subquestion_draft
            local_subquestion_reason = (
                "matched deterministic local solver patterns for every subquestion"
            )
        elif not service.nvidia_client.enabled:
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
        routing = service._with_local_solver_routing(
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
            routing = service._with_local_subquestion_routing(
                routing, reason=local_subquestion_reason
            )
    else:
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

    persist_solve_result(
        service.db, request, raw_text, normalized_text, routing, response
    )
    if not _cached_response_is_usable(request, solve_text, response):
        logger.info(
            "Solve response not cached request_id=%s reason=visualizable_prompt_missing_visualization",
            request_id,
        )
    else:
        response_cache.set(cache_key, response)
    return response


__all__ = ["solve_request"]
