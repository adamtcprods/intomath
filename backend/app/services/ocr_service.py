from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from app.core.config import get_settings
from app.core.solve_metrics import record_model_attempt
from app.integrations.local_deepseek_ocr import LocalDeepSeekOCR
from app.services.cache import TTLCache

logger = logging.getLogger(__name__)
_OCR_CACHE_VERSION = 1
_OCR_OPTIONS = {
    "as_markdown": False,
    "prompt": "free_ocr",
    "base_size": 1024,
    "image_size": 768,
    "crop_mode": True,
}

_settings = get_settings()
_OCR_CACHE: TTLCache["OCRResult"] = TTLCache(
    ttl_seconds=_settings.ocr_cache_ttl_seconds,
    max_size=_settings.ocr_cache_max_size,
)


@dataclass
class OCRResult:
    raw_text: str
    cleaned_text: str
    confidence: float
    warning: str | None = None
    cached: bool = False


class OCRService:
    def __init__(
        self,
        *,
        local_ocr: Any | None = None,
        cache: TTLCache[OCRResult] | None = None,
    ) -> None:
        self.local_ocr = local_ocr or LocalDeepSeekOCR()
        self.cache = cache if cache is not None else _OCR_CACHE

    async def extract_problem_text(
        self, image_bytes: bytes | None, mime_type: str | None
    ) -> OCRResult | None:
        if not image_bytes:
            return None

        normalized_mime_type = (
            (mime_type or "image/png").split(";", 1)[0].strip().lower()
        )
        cache_key = self._build_cache_key(image_bytes, normalized_mime_type)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                "OCR cache lookup cache=ocr result=hit cache_key_prefix=%s",
                cache_key[:12],
            )
            return replace(cached_result, cached=True)

        logger.info(
            "OCR cache lookup cache=ocr result=miss cache_key_prefix=%s",
            cache_key[:12],
        )
        try:
            record_model_attempt(
                provider="local",
                model=str(self.local_ocr.model_id),
                operation="ocr",
            )
            raw_text = await self.local_ocr.extract_text(
                image_bytes=image_bytes,
                mime_type=normalized_mime_type,
                as_markdown=False,
            )
        except Exception as exc:
            return OCRResult(
                raw_text="",
                cleaned_text="",
                confidence=0.0,
                warning=f"Local DeepSeek OCR failed — image text was not extracted: {exc}",
            )

        if not raw_text:
            # Model returned nothing — surface this clearly rather than
            # passing empty strings silently to the solver router.
            return None

        cleaned_text = _clean_for_solver(raw_text)
        confidence = _estimate_confidence(raw_text)

        result = OCRResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            confidence=confidence,
        )
        self.cache.set(cache_key, result)
        return result

    def _build_cache_key(self, image_bytes: bytes, mime_type: str) -> str:
        payload = {
            "cache_version": _OCR_CACHE_VERSION,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "mime_type": mime_type,
            "model": self.local_ocr.model_id,
            "options": _OCR_OPTIONS,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

# Patterns that suggest the OCR captured a real math problem.
_MATH_SIGNALS = re.compile(
    r"[\d\+\-\*/=\^√∫∑\(\)\[\]]"  # operators and brackets
    r"|\\[a-zA-Z]+"  # LaTeX commands like \frac, \sqrt
    r"|(?:solve|find|simplify|evaluate|calculate|proof|prove)",
    re.IGNORECASE,
)


def _clean_for_solver(text: str) -> str:
    """Normalise raw OCR output into a solver-friendly string.

    - Collapse stray whitespace and line breaks inside expressions.
    - Replace common OCR confusables (e.g. 'x' as multiplication vs variable
      is left to the solver; we only fix unambiguous visual artefacts).
    - Strip leading/trailing noise.
    """
    # Collapse runs of whitespace into a single space, but preserve
    # intentional paragraph breaks (double newline) for multi-part problems.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Common OCR confusables in math contexts
    replacements = [
        (r"(?<!\w)O(?!\w)", "0"),  # standalone letter O → digit 0
        (r"(?<!\w)l(?!\w)", "1"),  # standalone lowercase l → digit 1
        (r"\u00d7", "*"),  # × → *
        (r"\u00f7", "/"),  # ÷ → /
        (r"\u2212", "-"),  # − (minus sign) → hyphen-minus
        (r"\u2019", "'"),  # right single quote → apostrophe
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def _estimate_confidence(text: str) -> float:
    """Heuristic confidence score based on the extracted text.

    The model itself has no self-reporting mechanism for confidence, so we
    approximate it from observable signals in the output.

    Returns a float in [0.0, 1.0].
    """
    if not text:
        return 0.0

    score = 0.5  # baseline: something was extracted

    # Math signals present → more likely a valid extraction
    if _MATH_SIGNALS.search(text):
        score += 0.3

    # Reasonable length for a problem (5–500 chars) → more likely valid
    if 5 <= len(text) <= 500:
        score += 0.1

    # Excessive punctuation/symbols → possible garbage output
    noise_ratio = len(re.findall(r"[^\w\s]", text)) / max(len(text), 1)
    if noise_ratio > 0.4:
        score -= 0.2

    return round(min(max(score, 0.0), 1.0), 2)
