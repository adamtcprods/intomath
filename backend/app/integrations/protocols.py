"""Shared structural interfaces for external model integrations."""

from __future__ import annotations

from typing import Any, Protocol


class StructuredCompletionClient(Protocol):
    """Protocol for any LLM client that can produce structured JSON completions."""

    @property
    def enabled(self) -> bool: ...

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
    ) -> dict[str, Any]: ...


__all__ = ["StructuredCompletionClient"]
