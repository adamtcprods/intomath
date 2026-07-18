"""Reusable stages and data types for the solver pipeline."""

from app.services.solver_pipeline.content_quality import (
    StructuredContentIssue,
    contains_clear_math_notation,
    looks_like_scratch_work,
    structured_content_issues,
)
from app.services.solver_pipeline.response_builder import StructuredSolveDraft
from app.services.solver_pipeline.subquestion import DetectedSubquestion


__all__ = [
    "DetectedSubquestion",
    "StructuredContentIssue",
    "StructuredSolveDraft",
    "contains_clear_math_notation",
    "looks_like_scratch_work",
    "structured_content_issues",
]
