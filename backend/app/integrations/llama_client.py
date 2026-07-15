from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings


class LlamaClient:
    """Local llama-server client using OpenAI's completion library.

    Used for local-first routing/normalization hints and zero-cost local solving.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.local_llama_enabled)

    @property
    def model(self) -> str:
        return self.settings.local_solver_llama_model

    async def generate_json(
        self,
        *,
        prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Local llama.cpp integration is disabled.")

        request_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.local_solver_llama_timeout_seconds
        )
        request_max_tokens = max_tokens if max_tokens is not None else 500
        base_url = f"{self.settings.local_solver_llama_base_url.rstrip('/')}/v1"

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="llama-server",
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

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=request_max_tokens,
                ),
                timeout=request_timeout + 0.5,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise RuntimeError(
                f"Llama-server request timed out for model {self.model}."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Llama-server request failed: {exc}"
            ) from exc

        text = response.choices[0].message.content or ""
        text = text.strip()
        if not text:
            raise RuntimeError(f"Llama-server returned an empty response for {self.model}.")

        # Strip reasoning tags (<think>...</think>) if they are present in the response
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        return json.loads(self._strip_json_wrappers(text))

    def _strip_json_wrappers(self, payload: str) -> str:
        payload = payload.strip()
        payload = re.sub(r"^```json\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"^```\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        return (match.group(0) if match else payload).strip()
