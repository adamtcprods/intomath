"""Bounded validation for solve request text and image inputs."""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.schemas.solve import ProblemInput


_MIME_TO_PIL_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


class SolveInputError(ValueError):
    """A safe, client-facing solve input validation error."""

    status_code = 422


class SolveInputTooLargeError(SolveInputError):
    """A solve input exceeded a configured resource boundary."""

    status_code = 413


@dataclass(frozen=True)
class ValidatedSolveInput:
    image_bytes: bytes | None = None
    image_mime_type: str | None = None


def validate_solve_input(
    problem_input: ProblemInput, settings: Any
) -> ValidatedSolveInput:
    text_limit = int(getattr(settings, "max_solve_text_length", 20_000))
    if len(problem_input.text) > text_limit:
        raise SolveInputTooLargeError(
            f"Input text must not exceed {text_limit} characters."
        )

    if not problem_input.image_base64:
        return ValidatedSolveInput()

    mime_type = (problem_input.image_mime_type or "").strip().lower()
    if mime_type not in _MIME_TO_PIL_FORMAT:
        raise SolveInputError("A supported image_mime_type is required for image input.")

    base64_limit = int(getattr(settings, "max_image_base64_length", 14_000_000))
    if len(problem_input.image_base64) > base64_limit:
        raise SolveInputTooLargeError(
            f"Base64 image payload must not exceed {base64_limit} characters."
        )
    raw_payload = problem_input.image_base64.strip()

    payload = raw_payload
    if raw_payload.lower().startswith("data:"):
        header, separator, payload = raw_payload.partition(",")
        if not separator or not header.lower().endswith(";base64"):
            raise SolveInputError("Invalid Base64 image data.")
        declared_data_mime = header[5:].rsplit(";", 1)[0].strip().lower()
        if declared_data_mime != mime_type:
            raise SolveInputError(
                "The image data URL MIME type does not match image_mime_type."
            )

    if not payload or any(character.isspace() for character in payload):
        raise SolveInputError("Invalid Base64 image data.")

    decoded_limit = int(
        getattr(settings, "max_decoded_image_bytes", 10_485_760)
    )
    padding = len(payload) - len(payload.rstrip("="))
    estimated_decoded_size = max(0, (len(payload) * 3) // 4 - padding)
    if estimated_decoded_size > decoded_limit:
        raise SolveInputTooLargeError(
            f"Decoded image must not exceed {decoded_limit} bytes."
        )

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SolveInputError("Invalid Base64 image data.") from exc
    if len(image_bytes) > decoded_limit:
        raise SolveInputTooLargeError(
            f"Decoded image must not exceed {decoded_limit} bytes."
        )

    max_width = int(getattr(settings, "max_image_width", 8_192))
    max_height = int(getattr(settings, "max_image_height", 8_192))
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            actual_format = (image.format or "").upper()
            if actual_format != _MIME_TO_PIL_FORMAT[mime_type]:
                raise SolveInputError(
                    "Decoded image format does not match image_mime_type."
                )
            if width > max_width or height > max_height:
                raise SolveInputTooLargeError(
                    "Image dimensions must not exceed "
                    f"{max_width}x{max_height} pixels."
                )
            image.verify()
    except SolveInputError:
        raise
    except Image.DecompressionBombError as exc:
        raise SolveInputTooLargeError(
            "Decoded image exceeds the safe pixel-count limit."
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise SolveInputError("Decoded bytes are not a valid image.") from exc

    return ValidatedSolveInput(
        image_bytes=image_bytes,
        image_mime_type=mime_type,
    )


__all__ = [
    "SolveInputError",
    "SolveInputTooLargeError",
    "ValidatedSolveInput",
    "validate_solve_input",
]
