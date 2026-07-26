from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.model_policy import (
    NVIDIA_GPT_OSS_MODELS,
    remote_model_timeout_seconds,
)
from app.core.solve_metrics import record_model_attempt
from app.integrations.errors import (
    IntegrationFailureCategory,
    IntegrationRequestError,
    compact_log_text,
    exception_diagnostics,
)


logger = logging.getLogger(__name__)


class NvidiaClient:
    """Non-streaming NVIDIA NIM client for complete structured proposals.

    NVIDIA's published ChatRequest schemas for the configured gpt-oss models omit
    ``response_format`` and close the request with
    ``additionalProperties: false``. The schema is therefore supplied in the prompt,
    family-specific reasoning is bounded, and callers must run authoritative local
    schema and semantic validation.
    """

    def __init__(
        self,
        settings: Any | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if transport is not None and http_client is not None:
            raise ValueError("Pass either transport or http_client, not both.")
        self._owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(transport=transport)

    async def aclose(self) -> None:
        """Close the owned connection pool.

        Injected HTTP clients remain owned by their caller. Application-created
        clients are closed by the FastAPI lifespan.
        """
        if self._owns_http_client and not self.http_client.is_closed:
            await self.http_client.aclose()

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
        max_tokens: int = 500,
        timeout_seconds: float | None = None,
        operation: str = "json_completion",
        trace_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError(
                "NVIDIA direct integration is disabled or NVIDIA_API_KEY is not configured."
            )

        request_timeout = self._request_timeout(
            model=model,
            timeout_seconds=timeout_seconds,
        )
        payload, reasoning_controls = self._build_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_schema=json_schema,
            max_tokens=max_tokens,
            operation=operation,
        )
        logger.info(
            "NVIDIA direct request started operation=%s trace_id=%s model=%s "
            "schema_name=%s max_tokens=%s timeout_seconds=%.1f stream=false %s "
            "response_format_type=None structured_output_enforced=false",
            operation,
            trace_id,
            model,
            schema_name,
            max_tokens,
            request_timeout,
            reasoning_controls,
        )
        record_model_attempt(
            provider="nvidia_direct",
            model=model,
            operation=operation,
        )
        data = await self._send_request(
            payload,
            model=model,
            operation=operation,
            trace_id=trace_id,
            timeout_seconds=request_timeout,
        )
        try:
            text = self._extract_response_text(data, model=model)
            result = self._loads_json_response(
                text,
                model=model,
                json_schema=json_schema,
            )
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
            raise IntegrationRequestError(
                f"NVIDIA direct returned invalid JSON for model {model}.",
                provider="NVIDIA",
                model=model,
                operation=operation,
                failure_category=IntegrationFailureCategory.invalid_response,
            ) from exc
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

    def _request_timeout(
        self,
        *,
        model: str,
        timeout_seconds: float | None,
    ) -> float:
        if timeout_seconds is not None:
            return float(timeout_seconds)
        return remote_model_timeout_seconds(
            self.settings,
            provider="nvidia_direct",
            model=model,
        )

    def _build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        operation: str,
    ) -> tuple[dict[str, Any], str]:
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
            "max_tokens": max_tokens,
            "stream": False,
        }
        if model in NVIDIA_GPT_OSS_MODELS:
            effort = self._reasoning_effort(operation)
            payload["reasoning_effort"] = effort
            return payload, f"reasoning_effort={effort}"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_budget"] = 64
        return payload, "enable_thinking=false reasoning_budget=64"

    async def _send_request(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        operation: str,
        trace_id: str | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            response = await asyncio.wait_for(
                self.http_client.post(
                    f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                    timeout=self._http_timeout(timeout_seconds),
                ),
                timeout=timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            error = IntegrationRequestError(
                f"NVIDIA direct request timed out after {timeout_seconds:.1f}s "
                f"for model {model}.",
                provider="NVIDIA",
                model=model,
                operation=operation,
                failure_category=IntegrationFailureCategory.timeout,
            )
            self._log_request_error(
                error,
                operation=operation,
                trace_id=trace_id,
                model=model,
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
                failure_category=IntegrationFailureCategory.connectivity,
            )
            self._log_request_error(
                error,
                operation=operation,
                trace_id=trace_id,
                model=model,
            )
            raise error from exc
        return self._response_data(
            response,
            model=model,
            operation=operation,
            trace_id=trace_id,
        )

    @staticmethod
    def _http_timeout(timeout_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            timeout_seconds,
            connect=min(10.0, timeout_seconds),
            read=timeout_seconds,
            write=min(10.0, timeout_seconds),
            pool=min(10.0, timeout_seconds),
        )

    @staticmethod
    def _response_data(
        response: httpx.Response,
        *,
        model: str,
        operation: str,
        trace_id: str | None,
    ) -> dict[str, Any]:
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
                failure_category=(
                    IntegrationFailureCategory.rate_limit
                    if response.status_code == 429
                    else IntegrationFailureCategory.http_error
                ),
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
                failure_category=IntegrationFailureCategory.invalid_response,
            ) from exc
        if not isinstance(data, dict):
            raise IntegrationRequestError(
                f"NVIDIA direct returned an invalid response shape for model {model}.",
                provider="NVIDIA",
                model=model,
                operation=operation,
                failure_category=IntegrationFailureCategory.invalid_response,
            )
        return data

    @staticmethod
    def _log_request_error(
        error: Exception,
        *,
        operation: str,
        trace_id: str | None,
        model: str,
    ) -> None:
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

    def _extract_response_text(self, data: dict[str, Any], *, model: str) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise RuntimeError(f"NVIDIA direct returned no output text for model {model}.")

    def _loads_json_response(
        self,
        text: str,
        *,
        model: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        decoder = json.JSONDecoder()

        required_keys = tuple(
            key
            for key in (json_schema or {}).get("required", [])
            if isinstance(key, str)
        )
        candidates: list[dict[str, Any]] = []
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

        reconstructed = self._reconstruct_required_object(
            stripped,
            required_keys=required_keys,
            decoder=decoder,
        )
        if reconstructed:
            candidates.append(reconstructed)

        if candidates:
            return max(
                enumerate(candidates),
                key=lambda item: (
                    sum(key in item[1] for key in required_keys),
                    len(item[1]),
                    -item[0],
                ),
            )[1]
        raise RuntimeError(f"NVIDIA direct returned invalid JSON for model {model}.")

    def _reconstruct_required_object(
        self,
        text: str,
        *,
        required_keys: tuple[str, ...],
        decoder: json.JSONDecoder,
    ) -> dict[str, Any]:
        """Recover a schema-shaped object from a response containing JSON fragments."""

        if not required_keys:
            return {}
        reconstructed: dict[str, Any] = {}
        for key in required_keys:
            key_pattern = re.compile(rf"{re.escape(json.dumps(key))}\s*:")
            for match in key_pattern.finditer(text):
                value_text = text[match.end() :].lstrip()
                try:
                    value, _ = decoder.raw_decode(value_text)
                except json.JSONDecodeError:
                    continue
                reconstructed[key] = value
                break
        return reconstructed

    def _reasoning_effort(self, operation: str) -> str:
        if operation in {
            "structured_math_solution",
            "structured_math_steps_repair",
            "structured_math_solution_repair",
        }:
            return "medium"
        return "low"

    def _json_only_system_prompt(self, system_prompt: str) -> str:
        return (
            f"{system_prompt.strip()}\n\n"
            "Return exactly one JSON object as the entire response. Do not include "
            "reasoning, Markdown fences, raw scripts, or text outside the JSON object. "
            "JSON-escape every backslash inside strings (for example, emit "
            '"\\\\angle" in raw JSON).'
        )
