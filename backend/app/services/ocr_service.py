from __future__ import annotations

from dataclasses import dataclass

from app.integrations.local_deepseek_ocr import LocalDeepSeekOCR
from app.integrations.openrouter_client import OpenRouterClient


@dataclass
class OCRResult:
    raw_text: str
    cleaned_text: str
    confidence: float
    warning: str | None = None


class OCRService:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client
        self.local_ocr = LocalDeepSeekOCR()

    async def extract_problem_text(
        self, image_base64: str | None, mime_type: str | None
    ) -> OCRResult | None:
        if not image_base64:
            return None
        try:
            payload = await self.local_ocr.extract_json(
                prompt=(
                    "Extract the mathematics problem from the image or PDF. Return JSON only with keys: "
                    "raw_text, cleaned_text, confidence. Use cleaned_text for a solver-friendly version."
                ),
                image_base64=image_base64,
                mime_type=mime_type or "image/png",
            )
            return OCRResult(
                raw_text=str(payload.get("raw_text", "")).strip(),
                cleaned_text=str(
                    payload.get("cleaned_text", payload.get("raw_text", ""))
                ).strip(),
                confidence=float(payload.get("confidence", 0.0) or 0.0),
            )
        except Exception as exc:
            return OCRResult(
                raw_text="",
                cleaned_text="",
                confidence=0.0,
                warning=f"Local DeepSeek OCR failed, so image text was not extracted: {exc}",
            )
