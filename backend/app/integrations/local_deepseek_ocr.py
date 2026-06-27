from __future__ import annotations

import base64
import io
import json
import re
from functools import lru_cache
from typing import Any

from app.core.config import get_settings


class LocalDeepSeekOCR:
    """Lazy local DeepSeek OCR runner.

    The model is intentionally loaded only when OCR is requested so normal API
    startup and text-only solve requests do not download or initialize the large
    vision model.
    """

    @property
    def model_id(self) -> str:
        return get_settings().deepseek_ocr_model_id

    async def extract_json(
        self,
        *,
        prompt: str,
        image_base64: str,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        try:
            import anyio
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Local DeepSeek OCR requires optional dependencies: pillow and anyio."
            ) from exc

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return await anyio.to_thread.run_sync(self._extract_json_sync, prompt, image)

    def _extract_json_sync(self, prompt: str, image: Any) -> dict[str, Any]:
        model, processor_or_tokenizer = _load_model(self.model_id)

        if hasattr(model, "chat"):
            response = model.chat(processor_or_tokenizer, image, prompt)
        else:
            response = self._generate_with_transformers(
                model, processor_or_tokenizer, prompt, image
            )

        return json.loads(self._strip_json_wrappers(str(response)))

    def _generate_with_transformers(
        self, model: Any, processor: Any, prompt: str, image: Any
    ) -> str:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def _strip_json_wrappers(self, payload: str) -> str:
        payload = payload.strip()
        payload = re.sub(r"^```json\s*", "", payload)
        payload = re.sub(r"^```\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        return (match.group(0) if match else payload).strip()


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local DeepSeek OCR requires optional dependencies: torch and transformers."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        processor_or_tokenizer = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
    except Exception:
        processor_or_tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    return model, processor_or_tokenizer
