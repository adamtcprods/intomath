from __future__ import annotations

from dataclasses import dataclass


MAX_LOGGED_ERROR_BODY_CHARS = 2_000
MAX_LOGGED_ERROR_MESSAGE_CHARS = 1_000


def compact_log_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


@dataclass(frozen=True)
class ExceptionDiagnostics:
    error_type: str
    error_message: str
    status_code: int | None
    response_body: str | None


class IntegrationRequestError(RuntimeError):
    """An integration failure with safe, structured diagnostics for callers/logs."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        operation: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.operation = operation
        self.status_code = status_code
        self.response_body = compact_log_text(
            response_body, limit=MAX_LOGGED_ERROR_BODY_CHARS
        )


def exception_diagnostics(error: BaseException) -> ExceptionDiagnostics:
    status_code: int | None = None
    response_body: str | None = None
    chain_messages: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current_message = compact_log_text(
            str(current), limit=MAX_LOGGED_ERROR_MESSAGE_CHARS
        )
        chain_entry = type(current).__name__
        if current_message and current_message != chain_entry:
            chain_entry = (
                current_message
                if not chain_messages
                else f"{type(current).__name__}: {current_message}"
            )
        if chain_entry not in chain_messages:
            chain_messages.append(chain_entry)

        candidate_status = getattr(current, "status_code", None)
        if status_code is None and isinstance(candidate_status, int):
            status_code = candidate_status

        candidate_body = getattr(current, "response_body", None)
        response = getattr(current, "response", None)
        if candidate_body is None and response is not None:
            candidate_body = getattr(response, "text", None)
        if response_body is None and candidate_body is not None:
            response_body = compact_log_text(
                candidate_body, limit=MAX_LOGGED_ERROR_BODY_CHARS
            )

        # Follow explicit causal chains only. Implicit ``__context__`` can point at a
        # prior provider failure while a fallback is running inside its except block,
        # which would incorrectly attach one integration's error to another.
        current = current.__cause__

    return ExceptionDiagnostics(
        error_type=type(error).__name__,
        error_message=compact_log_text(
            " <- ".join(chain_messages), limit=MAX_LOGGED_ERROR_MESSAGE_CHARS
        )
        or type(error).__name__,
        status_code=status_code,
        response_body=response_body,
    )
