"""Detection and formatting helpers for multi-part problem statements."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedSubquestion:
    label: str
    question: str


def detect_subquestions(text: str) -> list[DetectedSubquestion]:
    marker_pattern = re.compile(
        r"^[ \t]*(?:\(([A-Za-z]|\d{1,2}|[ivxlcdm]+)\)|([A-Za-z]|\d{1,2}|[ivxlcdm]+)[\).:])\s+",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(marker_pattern.finditer(text))
    if len(matches) <= 1:
        return []

    subquestions: list[DetectedSubquestion] = []
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        question = text[body_start:body_end].strip()
        if question:
            subquestions.append(
                DetectedSubquestion(label=alpha_label(index), question=question)
            )
    return subquestions if len(subquestions) > 1 else []


def format_subquestion_hint(subquestions: list[DetectedSubquestion]) -> str:
    if len(subquestions) <= 1:
        return "Detected subquestions: none."
    formatted = "\n".join(
        f"{subquestion.label}) {subquestion.question}"
        for subquestion in subquestions
    )
    return (
        "Detected subquestions (use these normalized labels and solve each separately):\n"
        f"{formatted}"
    )


def subquestion_with_context(full_text: str, question: str) -> str:
    position = full_text.find(question)
    if position <= 0:
        return question
    marker_pattern = re.compile(
        r"^[ \t]*(?:\(([A-Za-z]|\d{1,2}|[ivxlcdm]+)\)|([A-Za-z]|\d{1,2}|[ivxlcdm]+)[\).:])\s+",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    first_marker = marker_pattern.search(full_text)
    context_end = (
        first_marker.start()
        if first_marker is not None and first_marker.start() < position
        else position
    )
    context = re.sub(
        r"^[ \t]*(?:\([A-Za-z\d]+\)|[A-Za-z\d]+[\).:])\s*$",
        "",
        full_text[:context_end].strip(),
        flags=re.MULTILINE,
    ).strip()
    return f"{context}\n{question}".strip() if context else question


def alpha_label(index: int) -> str:
    label = ""
    cursor = index
    while cursor >= 0:
        label = f"{chr(97 + (cursor % 26))}{label}"
        cursor = cursor // 26 - 1
    return label


__all__ = [
    "DetectedSubquestion",
    "alpha_label",
    "detect_subquestions",
    "format_subquestion_hint",
    "subquestion_with_context",
]
