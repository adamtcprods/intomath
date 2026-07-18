"""Shared structural interfaces for external model integrations."""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable


class _StructuredCompletionProtocolMeta(type(Protocol)):
    """Allow class checks for this data protocol as well as instance checks."""

    def __subclasscheck__(cls, subclass: type[Any]) -> bool:
        if getattr(cls, "_is_runtime_protocol", False):
            missing = object()
            enabled = inspect.getattr_static(subclass, "enabled", missing)
            complete_json = inspect.getattr_static(subclass, "complete_json", None)
            return enabled is not missing and callable(complete_json)
        return super().__subclasscheck__(subclass)


@runtime_checkable
class StructuredCompletionClient(
    Protocol, metaclass=_StructuredCompletionProtocolMeta
):
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
