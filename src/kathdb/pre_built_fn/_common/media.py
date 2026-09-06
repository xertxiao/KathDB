"""Image and audio loaders for the pre_built_fn semantic operators.

Resolves DataFrame cell values (file paths or existing data URIs) into the
forms LiteLLM expects: data-URI strings for images and ``(base64, format)``
tuples for audio. Unsupported extensions log a warning and return ``None`` so
the caller can drop the media from the prompt.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from kathdb.common.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

AUDIO_EXT_TO_FORMAT: dict[str, str] = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
    ".m4a": "m4a",
}


def load_image_as_base64(image_path: Path) -> str:
    """Load an image from disk and return a data URI."""
    raw_bytes = image_path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    return f"data:{mime};base64,{b64}"


def resolve_image_uri(value: str) -> str | None:
    """Return a data URI from a file path or existing data URI, or None."""
    if value.startswith("data:image"):
        return value
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning(
            "Unsupported image format '%s': only .jpg and .png are supported, skipping",
            suffix or value,
        )
        return None
    if path.exists():
        return load_image_as_base64(path)
    return None


def load_audio_as_base64(audio_path: Path) -> tuple[str, str]:
    """Load an audio file from disk and return ``(base64_data, format)``."""
    raw_bytes = audio_path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    fmt = AUDIO_EXT_TO_FORMAT.get(audio_path.suffix.lower(), "wav")
    return b64, fmt


def resolve_audio_data(value: str) -> tuple[str, str] | None:
    """Return ``(base64_data, format)`` from a file path, or None."""
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning(
            "Unsupported audio format '%s': only %s are supported, skipping",
            suffix or value,
            ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS)),
        )
        return None
    if path.exists():
        return load_audio_as_base64(path)
    return None
