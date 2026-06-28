from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from functools import lru_cache
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Prompts per the official model card (README.md):
#   "<image>\nFree OCR. "                              — plain text, no layout
#   "<image>\n<|grounding|>Convert the document to markdown. " — structured markdown
# For student math photos we want plain text.
PROMPT_FREE_OCR = "<image>\nFree OCR. "
PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "


class LocalDeepSeekOCR:
    """Lazy local DeepSeek-OCR-2 runner.

    The model is intentionally loaded only when OCR is requested so normal API
    startup and text-only solve requests do not download or initialize the large
    vision model.
    """

    @property
    def model_id(self) -> str:
        return get_settings().deepseek_ocr_model_id

    async def extract_text(
        self,
        *,
        image_base64: str,
        mime_type: str = "image/png",
        as_markdown: bool = False,
    ) -> str:
        """Return raw OCR text from DeepSeek-OCR-2.

        Args:
            image_base64: Base64-encoded image or PDF bytes.
            mime_type: MIME type of the input ("image/png", "image/jpeg",
                       "application/pdf", etc.).
            as_markdown: If True, use the document-to-markdown prompt. Useful
                         for structured worksheets; leave False for handwritten
                         photos.

        Raises:
            RuntimeError: If required dependencies are missing.
            ValueError: If the image cannot be decoded or OCR output is missing.
        """
        try:
            import anyio
        except ImportError as exc:
            raise RuntimeError(
                "Local DeepSeek OCR requires anyio. Install with: pip install anyio"
            ) from exc

        image_bytes = base64.b64decode(image_base64)
        image_path = _write_temp_image(image_bytes, mime_type)
        prompt = PROMPT_MARKDOWN if as_markdown else PROMPT_FREE_OCR

        return await anyio.to_thread.run_sync(
            lambda: self._extract_text_sync(image_path, prompt)
        )

    def _extract_text_sync(self, image_path: str, prompt: str) -> str:
        model, tokenizer = _load_model(self.model_id)

        # model.infer() always writes output to disk (save_results=True is the
        # only documented usage). We provide a temp directory and read the
        # result file back ourselves rather than assuming a return value.
        with tempfile.TemporaryDirectory() as output_dir:
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )
            return _read_ocr_output(output_dir, image_path)


def _read_ocr_output(output_dir: str, image_path: str) -> str:
    """Read the text file that model.infer() writes to output_dir.

    The model writes a .txt file named after the input image stem.
    E.g. input: /tmp/abc123.png → output: <output_dir>/abc123.txt
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(output_dir, f"{stem}.txt")

    if not os.path.exists(result_path):
        # Fall back: scan for any .txt file written to the output dir
        txt_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        if not txt_files:
            raise ValueError(
                f"DeepSeek-OCR-2 produced no output file in {output_dir}. "
                "Check that the model and image loaded correctly."
            )
        result_path = os.path.join(output_dir, txt_files[0])

    with open(result_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _write_temp_image(image_bytes: bytes, mime_type: str) -> str:
    """Write image bytes to a temp file and return its path.

    Handles raster images directly. PDFs are rasterized to the first page.
    Uses delete=False so the path remains valid after close.
    """
    if mime_type == "application/pdf":
        try:
            from pdf2image import convert_from_bytes
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pdf2image and poppler. "
                "Install with: pip install pdf2image"
            ) from exc
        pages = convert_from_bytes(image_bytes, dpi=200)
        if not pages:
            raise ValueError("PDF contained no renderable pages.")
        buf = io.BytesIO()
        pages[0].convert("RGB").save(buf, format="PNG")
        image_bytes = buf.getvalue()
        suffix = ".png"
    else:
        from PIL import Image as _PILImage

        _PILImage.open(io.BytesIO(image_bytes)).verify()  # validate before writing
        suffix = _mime_to_suffix(mime_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(image_bytes)
        return f.name


def _mime_to_suffix(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
    }.get(mime_type, ".png")


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load DeepSeek-OCR-2 model and tokenizer, cached after first load.

    NOTE: model_id is the cache key. If settings change between calls
    (e.g. in tests), clear the cache with _load_model.cache_clear().
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local DeepSeek OCR requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from exc

    if not torch.cuda.is_available():
        logger.warning(
            "DeepSeek-OCR-2 is running on CPU. This will be very slow and "
            "may OOM on typical hardware. A CUDA GPU is strongly recommended."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # _attn_implementation, use_safetensors, bfloat16, and .cuda() are all
    # required per the official model card. flash-attn must be installed
    # separately: pip install flash-attn==2.7.3 --no-build-isolation
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2",
    )
    model = model.eval().cuda().to(torch.bfloat16)

    return model, tokenizer
