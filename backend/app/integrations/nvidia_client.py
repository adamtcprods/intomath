from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.integrations.errors import (
    IntegrationRequestError,
    compact_log_text,
    exception_diagnostics,
)
from app.services.model_router import (
    NVIDIA_GPT_OSS_MODELS,
    remote_model_timeout_seconds,
)


logger = logging.getLogger(__name__)


class NvidiaClient:
    """Non-streaming NVIDIA NIM client for complete structured proposals.

    NVIDIA's published ChatRequest schemas for the configured Nemotron and gpt-oss
    models omit ``response_format`` and close the request with
    ``additionalProperties: false``. The schema is therefore supplied in the prompt,
    family-specific reasoning is bounded, and callers must run authoritative local
    schema and semantic validation.
    """

    def __init__(
        self,
        settings: Any | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport

    @property
    def enabled(self) -> bool:
        return bool(
            getattr(self.settings, "nvidia_direct_enabled", True)
            and getattr(self.settings, "nvidia_api_key", None)
        )

    async def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        timeout_seconds: float | None = None,
        operation: str = "json_completion",
        trace_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError(
                "NVIDIA direct integration is disabled or NVIDIA_API_KEY is not configured."
            )

        request_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else remote_model_timeout_seconds(
                self.settings, provider="nvidia_direct", model=model
            )
        )
        schema_instruction = ""
        if json_schema is not None:
            schema_instruction = (
                "\n\nRequired JSON schema (deterministically validated after generation):\n"
                + json.dumps(json_schema, ensure_ascii=False, separators=(",", ":"))
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": self._json_only_system_prompt(system_prompt)
                    + schema_instruction,
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": 6_000,
            "stream": False,
        }
        if model in NVIDIA_GPT_OSS_MODELS:
            payload["reasoning_effort"] = "low"
            reasoning_controls = "reasoning_effort=low"
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_budget"] = 64
            reasoning_controls = "enable_thinking=false reasoning_budget=64"
        logger.info(
            "NVIDIA direct request started operation=%s trace_id=%s model=%s "
            "schema_name=%s timeout_seconds=%.1f stream=false %s "
            "response_format_type=None structured_output_enforced=false",
            operation,
            trace_id,
            model,
            schema_name,
            request_timeout,
            reasoning_controls,
        )

        timeout = httpx.Timeout(
            request_timeout,
            connect=min(10.0, request_timeout),
            read=request_timeout,
            write=min(10.0, request_timeout),
            pool=min(10.0, request_timeout),
        )
        async with httpx.AsyncClient(
            timeout=timeout, transport=self.transport
        ) as client:
            try:
                response = await asyncio.wait_for(
                    client.post(
                        f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        json=payload,
                    ),
                    timeout=request_timeout,
                )
            except (TimeoutError, httpx.TimeoutException) as exc:
                error = IntegrationRequestError(
                    f"NVIDIA direct request timed out after {request_timeout:.1f}s "
                    f"for model {model}.",
                    provider="NVIDIA",
                    model=model,
                    operation=operation,
                )
                diagnostics = exception_diagnostics(error)
                logger.warning(
                    "NVIDIA direct request failed operation=%s trace_id=%s model=%s "
                    "error_type=%s error_message=%s status_code=%s response_body=%s",
                    operation,
                    trace_id,
                    model,
                    diagnostics.error_type,
                    diagnostics.error_message,
                    diagnostics.status_code,
                    diagnostics.response_body,
                )
                raise error from exc
            except httpx.HTTPError as exc:
                diagnostics = exception_diagnostics(exc)
                error = IntegrationRequestError(
                    f"NVIDIA direct request failed for model {model}: "
                    f"{diagnostics.error_message}",
                    provider="NVIDIA",
                    model=model,
                    operation=operation,
                    status_code=diagnostics.status_code,
                    response_body=diagnostics.response_body,
                )
                logger.warning(
                    "NVIDIA direct request failed operation=%s trace_id=%s model=%s "
                    "error_type=%s error_message=%s status_code=%s response_body=%s",
                    operation,
                    trace_id,
                    model,
                    diagnostics.error_type,
                    diagnostics.error_message,
                    diagnostics.status_code,
                    diagnostics.response_body,
                )
                raise error from exc

        response_body = compact_log_text(response.text, limit=2_000)
        if response.status_code >= 400:
            logger.warning(
                "NVIDIA direct request failed operation=%s trace_id=%s model=%s "
                "status_code=%s response_body=%s",
                operation,
                trace_id,
                model,
                response.status_code,
                response_body,
            )
            raise IntegrationRequestError(
                f"NVIDIA direct request failed for model {model}: HTTP {response.status_code}.",
                provider="NVIDIA",
                model=model,
                operation=operation,
                status_code=response.status_code,
                response_body=response.text,
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationRequestError(
                f"NVIDIA direct returned a non-JSON HTTP response for model {model}.",
                provider="NVIDIA",
                model=model,
                operation=operation,
                status_code=response.status_code,
                response_body=response.text,
            ) from exc

        text = self._extract_response_text(data, model=model)
        try:
            result = self._loads_json_response(text, model=model)
        except RuntimeError as exc:
            diagnostics = exception_diagnostics(exc)
            logger.warning(
                "NVIDIA direct JSON parse failed operation=%s trace_id=%s model=%s "
                "error_type=%s error_message=%s status_code=%s response_body=%s",
                operation,
                trace_id,
                model,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            raise
        logger.info(
            "NVIDIA direct completion metadata operation=%s trace_id=%s "
            "requested_model=%s response_model=%s provider=NVIDIA response_chars=%s "
            "response_format_round_tripped=false structured_output_enforced=false",
            operation,
            trace_id,
            model,
            data.get("model"),
            len(text),
        )
        return result

    def _extract_response_text(self, data: dict[str, Any], *, model: str) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise RuntimeError(f"NVIDIA direct returned no output text for model {model}.")

    def _loads_json_response(self, text: str, *, model: str) -> dict[str, Any]:
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError(f"NVIDIA direct returned invalid JSON for model {model}.")

    def _json_only_system_prompt(self, system_prompt: str) -> str:
        return (
            f"{system_prompt.strip()}\n\n"
            "Return exactly one JSON object as the entire response. Do not include "
            "reasoning, Markdown fences, raw scripts, or text outside the JSON object."
        )
