"""Controlled errors raised by the solve orchestration boundary."""

from __future__ import annotations


class SolveRequestTimeoutError(TimeoutError):
    def __init__(self, *, request_id: str, stage: str) -> None:
        super().__init__("The solve request exceeded its total execution deadline.")
        self.request_id = request_id
        self.stage = stage


__all__ = ["SolveRequestTimeoutError"]
