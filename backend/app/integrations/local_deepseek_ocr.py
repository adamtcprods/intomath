from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import importlib.util
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
        image_bytes = _decode_base64_payload(image_base64)
        image_path = _write_temp_image(image_bytes, mime_type)
        prompt = PROMPT_MARKDOWN if as_markdown else PROMPT_FREE_OCR

        worker = asyncio.create_task(
            asyncio.to_thread(self._extract_text_sync, image_path, prompt)
        )
        cancelled = False
        try:
            while not worker.done():
                try:
                    _ = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    if worker.cancelled():
                        raise
                    # asyncio cannot stop a running worker thread. Delay parent
                    # cancellation until inference releases the image it is reading.
                    cancelled = True
                except Exception:
                    break

            if cancelled:
                if not worker.cancelled():
                    _ = worker.exception()  # consume a failure after parent cancellation
                raise asyncio.CancelledError
            return worker.result()
        finally:
            _safe_unlink(image_path)

    def _extract_text_sync(self, image_path: str, prompt: str) -> str:
        model, tokenizer = _load_model(self.model_id)

        # model.infer() always writes output to disk (save_results=True is the
        # only documented usage). We provide a temp directory and read the
        # result file back ourselves rather than assuming a return value.
        with tempfile.TemporaryDirectory() as output_dir:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )
            if isinstance(result, str) and result.strip():
                return result.strip()
            return _read_ocr_output(output_dir, image_path)


def _decode_base64_payload(image_base64: str) -> bytes:
    """Decode raw base64 or a data URL payload into bytes."""
    payload = image_base64.strip()
    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data.") from exc


def _read_ocr_output(output_dir: str, image_path: str) -> str:
    """Read the text/markdown file that model.infer() writes to output_dir."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    extensions = (".txt", ".md", ".mmd")

    for extension in extensions:
        result_path = os.path.join(output_dir, f"{stem}{extension}")
        if os.path.exists(result_path):
            return _read_text_file(result_path)

    # Fall back: scan recursively because some model revisions write nested
    # result files or use markdown-like extensions for structured OCR output.
    for root, _dirs, files in os.walk(output_dir):
        for name in sorted(files):
            if name.endswith(extensions):
                return _read_text_file(os.path.join(root, name))

    raise ValueError(
        f"DeepSeek-OCR-2 produced no output file in {output_dir}. "
        "Check that the model and image loaded correctly."
    )


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _write_temp_image(image_bytes: bytes, mime_type: str) -> str:
    """Write image bytes to a temp file and return its path.

    Handles raster images directly. PDFs are rasterized to the first page.
    Uses delete=False so the path remains valid after close.
    """
    mime_type = _normalize_mime_type(mime_type)

    if mime_type == "application/pdf":
        pdf2image = _import_optional_module(
            "pdf2image",
            "PDF support requires pdf2image and poppler. "
            "Install with: pip install pdf2image",
        )
        pages = pdf2image.convert_from_bytes(image_bytes, dpi=200)
        if not pages:
            raise ValueError("PDF contained no renderable pages.")
        buf = io.BytesIO()
        pages[0].convert("RGB").save(buf, format="PNG")
        image_bytes = buf.getvalue()
        suffix = ".png"
    else:
        image_module = _import_optional_module(
            "PIL.Image",
            "Image validation requires Pillow. Install with: pip install pillow",
        )
        try:
            with image_module.open(io.BytesIO(image_bytes)) as image:
                image.verify()
        except Exception as exc:
            raise ValueError("Input image bytes are not a valid image.") from exc
        suffix = _mime_to_suffix(mime_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(image_bytes)
        return f.name


def _normalize_mime_type(mime_type: str) -> str:
    return mime_type.split(";", 1)[0].strip().lower()


def _mime_to_suffix(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/tiff": ".tiff",
    }.get(_normalize_mime_type(mime_type), ".png")


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Failed to delete temporary OCR image %s", path, exc_info=True)


def _import_optional_module(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(install_hint) from exc


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load DeepSeek-OCR-2 model and tokenizer, cached after first load.

    NOTE: model_id is the cache key. If settings change between calls
    (e.g. in tests), clear the cache with _load_model.cache_clear().
    """
    torch = _import_optional_module(
        "torch",
        "Local DeepSeek OCR requires torch. Install with: pip install torch",
    )
    transformers = _import_optional_module(
        "transformers",
        "Local DeepSeek OCR requires transformers. "
        "Install with: pip install transformers",
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Local DeepSeek OCR requires a CUDA GPU. DeepSeek-OCR-2 uses "
            "flash_attention_2 and .cuda() per the official model card."
        )
    if importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            "Local DeepSeek OCR requires flash-attn for flash_attention_2. "
            "Install the project's OCR extra with --no-build-isolation as "
            "documented in docs/SETUP.md."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # _attn_implementation, use_safetensors, bfloat16, and .cuda() are all
    # required per the official model card.
    model = transformers.AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2",
    )
    model = model.eval().cuda().to(torch.bfloat16)

    return model, tokenizer
