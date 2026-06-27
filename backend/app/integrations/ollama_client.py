from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from app.core.config import get_settings


class OllamaClient:
    """Tiny local Ollama JSON helper used only for routing/normalization hints.

    The deterministic solver remains the source of truth. If Ollama is missing,
    slow, or returns invalid JSON, callers should ignore the hint and continue
    with the regular model route.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.local_solver_ollama_detection_enabled)

    @property
    def model(self) -> str:
        return self.settings.local_solver_ollama_model

    async def generate_json(self, *, prompt: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Local Ollama detection is disabled.")

        timeout_seconds = self.settings.local_solver_ollama_timeout_seconds
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(1.0, timeout_seconds),
            read=timeout_seconds,
            write=min(1.0, timeout_seconds),
            pool=min(1.0, timeout_seconds),
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 180},
        }

        base_url = self.settings.local_solver_ollama_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await asyncio.wait_for(
                    client.post(f"{base_url}/api/generate", json=payload),
                    timeout=timeout_seconds + 0.5,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Ollama detection timed out for model {self.model}."
                ) from exc
            response.raise_for_status()
            data = response.json()

        text = str(data.get("response", "")).strip()
        if not text:
            raise RuntimeError(f"Ollama returned an empty response for {self.model}.")
        return json.loads(self._strip_json_wrappers(text))

    def _strip_json_wrappers(self, payload: str) -> str:
        payload = payload.strip()
        payload = re.sub(r"^```json\s*", "", payload)
        payload = re.sub(r"^```\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        return (match.group(0) if match else payload).strip()
