from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    async def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        logger.info(
            "OpenRouter JSON completion started model=%s schema_name=%s has_json_schema=%s",
            model,
            schema_name,
            json_schema is not None,
        )
        data = await self._post_json_chat_with_fallbacks(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=6000,
            timeout_seconds=120.0,
        )

        text = self._extract_response_text(data, model=model)
        logger.info(
            "OpenRouter JSON completion returned text model=%s chars=%s",
            model,
            len(text),
        )
        try:
            return self._loads_json_response(text, model=model)
        except RuntimeError as exc:
            if not self._is_invalid_json_error(exc):
                raise
            logger.warning(
                "OpenRouter JSON completion returned invalid JSON; attempting repair model=%s",
                model,
            )
            return await self._repair_invalid_json_response(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                invalid_response=text,
                json_schema=json_schema,
                schema_name=schema_name,
            )

    async def vision_extract(
        self,
        *,
        model: str,
        prompt: str,
        image_base64: str,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        logger.info(
            "OpenRouter vision extraction started model=%s mime_type=%s image_chars=%s",
            model,
            mime_type,
            len(image_base64),
        )
        payload = {
            "model": model,
            "temperature": 0.1,
            "instructions": "Extract the requested information and return exactly one valid JSON object.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_base64}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": 2000,
            "store": False,
        }

        data = await self._post_responses(payload, model=model, timeout_seconds=90.0)
        text = self._extract_response_text(data, model=model)
        logger.info(
            "OpenRouter vision extraction returned text model=%s chars=%s",
            model,
            len(text),
        )
        return self._loads_json_response(text, model=model)

    async def _post_json_chat_with_fallbacks(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        payload = self._chat_json_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
        )

        try:
            return await self._post_chat_completions(
                payload, model=model, timeout_seconds=timeout_seconds
            )
        except RuntimeError as exc:
            if self._is_optional_json_control_unsupported_error(exc):
                logger.warning(
                    "OpenRouter optional JSON controls unsupported; retrying with reduced controls model=%s",
                    model,
                )
                include_reasoning_controls, enable_response_healing = (
                    self._supported_optional_json_controls_after_error(exc)
                )
                payload = self._chat_json_payload(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    max_tokens=max_tokens,
                    include_reasoning_controls=include_reasoning_controls,
                    enable_response_healing=enable_response_healing,
                )
                try:
                    return await self._post_chat_completions(
                        payload, model=model, timeout_seconds=timeout_seconds
                    )
                except RuntimeError as retry_exc:
                    if self._is_optional_json_control_unsupported_error(retry_exc):
                        logger.warning(
                            "OpenRouter optional JSON controls still unsupported; retrying with controls disabled model=%s",
                            model,
                        )
                        payload = self._chat_json_payload(
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            json_schema=json_schema,
                            schema_name=schema_name,
                            max_tokens=max_tokens,
                            include_reasoning_controls=False,
                            enable_response_healing=False,
                        )
                        try:
                            return await self._post_chat_completions(
                                payload, model=model, timeout_seconds=timeout_seconds
                            )
                        except RuntimeError as final_optional_exc:
                            retry_exc = final_optional_exc
                    if json_schema is None or not self._is_schema_unsupported_error(
                        retry_exc
                    ):
                        raise retry_exc
                    exc = retry_exc
            if json_schema is None or not self._is_schema_unsupported_error(exc):
                raise
            logger.warning(
                "OpenRouter JSON schema response_format unsupported; retrying with json_object model=%s",
                model,
            )
            payload = self._chat_json_payload(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                json_schema=None,
                schema_name=schema_name,
                max_tokens=max_tokens,
            )
            return await self._post_chat_completions(
                payload, model=model, timeout_seconds=timeout_seconds
            )

    async def _post_chat_completions(
        self, payload: dict[str, Any], *, model: str, timeout_seconds: float
    ) -> dict[str, Any]:
        return await self._post_openrouter(
            "chat/completions", payload, model=model, timeout_seconds=timeout_seconds
        )

    async def _post_responses(
        self, payload: dict[str, Any], *, model: str, timeout_seconds: float
    ) -> dict[str, Any]:
        return await self._post_openrouter(
            "responses", payload, model=model, timeout_seconds=timeout_seconds
        )

    async def _post_openrouter(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        model: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        logger.info(
            "OpenRouter request started path=%s model=%s timeout_seconds=%.1f",
            path,
            model,
            timeout_seconds,
        )
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=10.0,
            read=timeout_seconds,
            write=10.0,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await asyncio.wait_for(
                    client.post(
                        f"{self.settings.openrouter_base_url}/{path}",
                        headers=self._headers(),
                        json=payload,
                    ),
                    timeout=timeout_seconds + 5.0,
                )
            except (TimeoutError, httpx.TimeoutException) as exc:
                logger.warning(
                    "OpenRouter request timed out path=%s model=%s timeout_seconds=%.1f",
                    path,
                    model,
                    timeout_seconds,
                )
                raise RuntimeError(
                    f"OpenRouter request timed out for model {model}."
                ) from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.warning(
                "OpenRouter returned non-JSON HTTP response path=%s model=%s status=%s response_chars=%s",
                path,
                model,
                response.status_code,
                len(response.text),
            )
            raise RuntimeError(
                f"OpenRouter returned a non-JSON HTTP {response.status_code} response "
                f"for model {model}."
            ) from exc

        if response.status_code >= 400:
            logger.warning(
                "OpenRouter request failed path=%s model=%s status=%s error_type=%s",
                path,
                model,
                response.status_code,
                data.get("error_type") or type(data.get("error")).__name__,
            )
            raise RuntimeError(
                f"OpenRouter request failed for model {model}: "
                f"HTTP {response.status_code} {self._format_api_error(data)}"
            )
        self._raise_response_error(data, model=model)
        logger.info(
            "OpenRouter request completed path=%s model=%s status=%s",
            path,
            model,
            response.status_code,
        )
        return data

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_site_url,
            "X-Title": self.settings.openrouter_app_name,
            "X-OpenRouter-Metadata": "enabled",
        }

    def _chat_json_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        max_tokens: int,
        include_reasoning_controls: bool = True,
        enable_response_healing: bool = True,
    ) -> dict[str, Any]:
        response_format: dict[str, Any]
        if json_schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            }

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": self._json_only_system_prompt(system_prompt),
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if enable_response_healing:
            payload["plugins"] = [{"id": "response-healing"}]
        if include_reasoning_controls:
            payload["reasoning"] = {"exclude": True}
        return payload

    async def _repair_invalid_json_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        invalid_response: str,
        json_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> dict[str, Any]:
        repair_prompt = (
            "The previous assistant response was invalid JSON. Convert the work into the "
            "required final JSON object only. If the previous response contains only analysis, "
            "solve the original task and output the JSON now.\n\n"
            "Original task:\n"
            f"{user_prompt}\n\n"
            "Invalid previous response:\n"
            f"{invalid_response[:4000]}\n\n"
            "Return only one valid JSON object. The first character must be { and the last must be }."
        )
        logger.info(
            "OpenRouter JSON repair started model=%s schema_name=%s invalid_response_chars=%s",
            model,
            schema_name,
            len(invalid_response),
        )
        data = await self._post_json_chat_with_fallbacks(
            model=model,
            system_prompt=system_prompt,
            user_prompt=repair_prompt,
            temperature=0.0,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=6000,
            timeout_seconds=120.0,
        )
        text = self._extract_response_text(data, model=model)
        try:
            repaired = self._loads_json_response(text, model=model)
            logger.info("OpenRouter JSON repair succeeded model=%s", model)
            return repaired
        except RuntimeError as repair_error:
            logger.warning(
                "OpenRouter JSON repair failed model=%s error_type=%s",
                model,
                type(repair_error).__name__,
            )
            raise RuntimeError(
                f"OpenRouter returned invalid JSON for model {model}; "
                "the JSON repair attempt also failed."
            ) from repair_error

    def _json_only_system_prompt(self, system_prompt: str) -> str:
        return (
            f"{system_prompt.strip()}\n\n"
            "Critical output constraints:\n"
            "- The first character of your response must be { and the last must be }.\n"
            "- Output the final JSON object directly as the entire message.\n"
            "- Do not start with phrases like 'We need', 'I need', 'Let's', or 'Here is'.\n"
            "- Do not explain your plan, chain of thought, or reasoning outside JSON.\n"
            "- Do not echo the schema or use placeholders such as ..., string, or null-as-text.\n"
            "- If uncertain, still return valid JSON and put uncertainty in warnings."
        )

    def _is_schema_unsupported_error(self, error: RuntimeError) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "json_schema",
                "structured output",
                "structured outputs",
                "response_format",
                "unsupported parameter",
                "unsupported params",
            )
        )

    def _is_optional_json_control_unsupported_error(self, error: RuntimeError) -> bool:
        message = str(error).lower()
        optional_controls = ("response-healing", "plugins", "reasoning")
        unsupported_markers = (
            "unsupported parameter",
            "unsupported params",
            "unrecognized request argument",
            "unknown parameter",
            "invalid request",
        )
        return any(control in message for control in optional_controls) and any(
            marker in message for marker in unsupported_markers
        )

    def _supported_optional_json_controls_after_error(
        self, error: RuntimeError
    ) -> tuple[bool, bool]:
        message = str(error).lower()
        mentions_reasoning = "reasoning" in message
        mentions_healing = "response-healing" in message or "plugins" in message
        if not mentions_reasoning and not mentions_healing:
            return False, False
        return not mentions_reasoning, not mentions_healing

    def _is_invalid_json_error(self, error: RuntimeError) -> bool:
        return "returned invalid JSON" in str(error)

    def _extract_response_text(self, data: dict[str, Any], *, model: str) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = data.get("output")
        if isinstance(output, list):
            text_parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                    continue
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    refusal = part.get("refusal")
                    if isinstance(refusal, str):
                        text_parts.append(refusal)
            text = "".join(text_parts).strip()
            if text:
                return text

        # Be defensive if a compatibility proxy or older endpoint returns the
        # legacy Chat Completions shape despite this client using /responses.
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    text = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ).strip()
                    if text:
                        return text
                if isinstance(content, str) and content.strip():
                    return content

        status = data.get("status")
        incomplete_details = data.get("incomplete_details")
        raise RuntimeError(
            f"OpenRouter returned no output text for model {model} "
            f"(status={status}, incomplete_details={incomplete_details})."
        )

    def _loads_json_response(self, text: str, *, model: str) -> dict[str, Any]:
        stripped = self._strip_json_wrappers(text)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning(
                "OpenRouter returned invalid JSON model=%s line=%s column=%s response_chars=%s",
                model,
                exc.lineno,
                exc.colno,
                len(stripped),
            )
            raise RuntimeError(
                f"OpenRouter returned invalid JSON for model {model} "
                f"at line {exc.lineno}, column {exc.colno}."
            ) from exc
        if not isinstance(parsed, dict):
            logger.warning(
                "OpenRouter returned non-object JSON model=%s json_type=%s",
                model,
                type(parsed).__name__,
            )
            raise RuntimeError(
                f"OpenRouter returned JSON {type(parsed).__name__} for model {model}; expected object."
            )
        return parsed

    def _raise_response_error(self, data: dict[str, Any], *, model: str) -> None:
        error = data.get("error")
        if error in (None, {}):
            return
        logger.warning(
            "OpenRouter response contained error model=%s error_type=%s",
            model,
            data.get("error_type") or type(error).__name__,
        )
        raise RuntimeError(
            f"OpenRouter returned an error for model {model}: {self._format_api_error(data)}"
        )

    def _format_api_error(self, data: dict[str, Any]) -> str:
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or error
            error_type = data.get("error_type")
            if error_type:
                return f"{message} (type={error_type})"
            return str(message)
        if isinstance(error, str):
            return error
        return json.dumps(data, ensure_ascii=False)[:500]

    def _strip_json_wrappers(self, payload: str) -> str:
        payload = payload.strip()
        fenced = self._first_fenced_json_block(payload)
        if fenced is not None:
            payload = fenced.strip()
        else:
            payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
            payload = re.sub(r"\s*```$", "", payload)

        valid_object = self._first_valid_json_object(payload)
        if valid_object is not None:
            return valid_object.strip()

        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        return (match.group(0) if match else payload).strip()

    def _first_fenced_json_block(self, payload: str) -> str | None:
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.IGNORECASE | re.DOTALL
        )
        if match is None:
            return None
        return match.group(1)

    def _first_valid_json_object(self, payload: str) -> str | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", payload):
            candidate = payload[match.start() :]
            try:
                parsed, end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return candidate[:end]
        return None
