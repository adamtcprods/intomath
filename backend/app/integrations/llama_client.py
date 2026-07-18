from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, ClassVar

import httpx
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.integrations.errors import IntegrationRequestError, exception_diagnostics


logger = logging.getLogger(__name__)


class LlamaClient:
    """Local llama-server client using OpenAI's completion library.

    Used for local-first routing/normalization hints and zero-cost local solving.
    Connectivity failures open a short process-wide circuit keyed by base URL so one
    stopped server does not add the same failed hop to every stage of every request.
    """

    _unavailable_until_by_base_url: ClassVar[dict[str, float]] = {}

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.local_llama_enabled)

    @property
    def model(self) -> str:
        return self.settings.local_solver_llama_model

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        unavailable_until = self._unavailable_until_by_base_url.get(
            self._availability_key(), 0.0
        )
        return time.monotonic() >= unavailable_until

    async def probe_health(
        self, *, timeout_seconds: float | None = None, trace_id: str | None = None
    ) -> bool:
        """Probe llama-server once and update the shared availability circuit."""
        if not self.enabled:
            return False
        request_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "local_llama_startup_probe_timeout_seconds", 1.0)
        )
        base_url = self._configured_base_url()
        health_url = f"{base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await asyncio.wait_for(
                    client.get(health_url), timeout=request_timeout
                )
            if response.status_code >= 400:
                raise IntegrationRequestError(
                    f"Llama-server health probe returned HTTP {response.status_code}.",
                    provider="llama.cpp",
                    model=self.model,
                    operation="local_llama_health_probe",
                    status_code=response.status_code,
                    response_body=response.text,
                )
        except Exception as exc:
            diagnostics = exception_diagnostics(exc)
            self._mark_unavailable()
            logger.warning(
                "Llama-server health probe failed operation=local_llama_health_probe "
                "trace_id=%s model=%s base_url=%s error_type=%s error_message=%s "
                "status_code=%s response_body=%s",
                trace_id,
                self.model,
                base_url,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            return False

        self._clear_unavailable()
        logger.info(
            "Llama-server health probe succeeded operation=local_llama_health_probe "
            "trace_id=%s model=%s base_url=%s",
            trace_id,
            self.model,
            base_url,
        )
        return True

    async def generate_json(
        self,
        *,
        prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        json_schema: dict[str, Any] | None = None,
        operation: str = "local_json_generation",
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Local llama.cpp integration is disabled.")
        if not self.available:
            raise IntegrationRequestError(
                "Llama-server is temporarily unavailable after a recent connectivity failure.",
                provider="llama.cpp",
                model=self.model,
                operation=operation,
            )

        request_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.local_solver_llama_timeout_seconds
        )
        request_max_tokens = max_tokens if max_tokens is not None else 500
        base_url = f"{self._configured_base_url()}/v1"

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="llama-server",
            max_retries=0,
        )

        response_format: dict[str, Any] = {"type": "json_object"}
        if json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "intomath_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        logger.info(
            "Llama-server request started operation=%s trace_id=%s model=%s "
            "base_url=%s timeout_seconds=%.1f sdk_retries=0",
            operation,
            trace_id,
            self.model,
            base_url,
            request_timeout,
        )
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=request_max_tokens,
                ),
                timeout=request_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self._mark_unavailable()
            error = IntegrationRequestError(
                f"Llama-server request timed out after {request_timeout:.1f}s for model {self.model}.",
                provider="llama.cpp",
                model=self.model,
                operation=operation,
            )
            diagnostics = exception_diagnostics(error)
            logger.warning(
                "Llama-server request failed operation=%s trace_id=%s model=%s "
                "error_type=%s error_message=%s status_code=%s response_body=%s",
                operation,
                trace_id,
                self.model,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            raise error from exc
        except Exception as exc:
            diagnostics = exception_diagnostics(exc)
            if self._is_connectivity_failure(exc):
                self._mark_unavailable()
            error = IntegrationRequestError(
                f"Llama-server request failed for model {self.model}: {diagnostics.error_message}",
                provider="llama.cpp",
                model=self.model,
                operation=operation,
                status_code=diagnostics.status_code,
                response_body=diagnostics.response_body,
            )
            logger.warning(
                "Llama-server request failed operation=%s trace_id=%s model=%s "
                "error_type=%s error_message=%s status_code=%s response_body=%s",
                operation,
                trace_id,
                self.model,
                diagnostics.error_type,
                diagnostics.error_message,
                diagnostics.status_code,
                diagnostics.response_body,
            )
            raise error from exc

        self._clear_unavailable()
        text = response.choices[0].message.content or ""
        text = text.strip()
        if not text:
            raise RuntimeError(f"Llama-server returned an empty response for {self.model}.")

        # Strip reasoning tags (<think>...</think>) if they are present in the response
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            payload = json.loads(self._strip_json_wrappers(text))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Llama-server returned invalid JSON for model {self.model} "
                f"at line {exc.lineno}, column {exc.colno}."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Llama-server returned JSON {type(payload).__name__} for model "
                f"{self.model}; expected object."
            )
        logger.info(
            "Llama-server request succeeded operation=%s trace_id=%s model=%s response_chars=%s",
            operation,
            trace_id,
            self.model,
            len(text),
        )
        return payload

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Adapt the shared structured-completion interface to llama-server."""
        _ = (model, temperature, schema_name)
        return await self.generate_json(
            prompt=f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip(),
            max_tokens=kwargs.get("max_tokens"),
            timeout_seconds=timeout_seconds,
            json_schema=json_schema,
            operation=operation,
            trace_id=trace_id,
        )

    def _configured_base_url(self) -> str:
        return self.settings.local_solver_llama_base_url.rstrip("/")

    def _mark_unavailable(self) -> None:
        cooldown = float(
            getattr(self.settings, "local_llama_unavailable_cooldown_seconds", 60.0)
        )
        self._unavailable_until_by_base_url[self._availability_key()] = (
            time.monotonic() + cooldown
        )

    def _clear_unavailable(self) -> None:
        self._unavailable_until_by_base_url.pop(self._availability_key(), None)

    def _availability_key(self) -> str:
        return f"{self._configured_base_url()}|{self.model}"

    def _is_connectivity_failure(self, error: BaseException) -> bool:
        diagnostics = exception_diagnostics(error)
        message = diagnostics.error_message.casefold()
        return diagnostics.status_code is None and any(
            marker in message
            for marker in (
                "apiconnectionerror",
                "connecterror",
                "connection error",
                "connection refused",
                "all connection attempts failed",
            )
        )

    def _strip_json_wrappers(self, payload: str) -> str:
        payload = payload.strip()
        payload = re.sub(r"^```json\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"^```\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        return (match.group(0) if match else payload).strip()
