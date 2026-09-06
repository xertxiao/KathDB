"""The single entry point for per-record model calls.

Generated code, saved functions and prebuilt functions all call the AI-op model
through :func:`call_model`; :func:`run_batch` fans such calls out with bounded
concurrency. Calls go through a ``litellm.Router`` so per-model rpm/tpm caps are
enforced client-side and transport errors are retried.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

from litellm import Router

from kathdb.common.logger import get_logger

logger = get_logger(__name__)

__all__ = ["call_model", "run_batch", "Media"]

REQUEST_TIMEOUT_S: float = 40.0
MAX_RETRIES: int = 3
RETRY_BACKOFF_S: float = 2.0

# Client-side rpm/tpm caps applied to every model: requests beyond them are deferred
# instead of bouncing back as 429s. Override with KATHDB_AIOP_RPM / KATHDB_AIOP_TPM.
DEFAULT_RPM: int = int(os.environ.get("KATHDB_AIOP_RPM", "500"))
DEFAULT_TPM: int = int(os.environ.get("KATHDB_AIOP_TPM", "200000"))

ALLOWED_IMAGE_DETAILS: tuple[str, ...] = ("low", "high", "auto")
ALLOWED_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
AUDIO_FORMATS: dict[str, str] = {
    ".wav": "wav", ".mp3": "mp3", ".m4a": "m4a", ".flac": "flac", ".ogg": "ogg", ".webm": "webm"
}

# One media item: a local path, an http(s) URL or a data URI (images), a local path
# (audio), or an already encoded ``(base64_data, format)`` audio tuple.
Media = "str | tuple[str, str]"

_router_lock = threading.Lock()
_router: Router | None = None
_router_models: set[str] = set()


def _deployment(model: str) -> dict[str, Any]:
    return {
        "model_name": model,
        "litellm_params": {
            "model": model,
            "rpm": DEFAULT_RPM,
            "tpm": DEFAULT_TPM,
            "timeout": REQUEST_TIMEOUT_S,
        },
    }


def _router_for(model: str) -> Router:
    """Router that knows ``model``; models are registered on first use.

    Any LiteLLM model id works (``openai/gpt-4o-mini``, ``azure/<deployment>``,
    ``anthropic/...``, ``vertex_ai/...``); credentials come from the provider's
    usual environment variables.
    """
    global _router
    with _router_lock:
        if _router is None or model not in _router_models:
            _router_models.add(model)
            _router = Router(
                model_list=[_deployment(m) for m in sorted(_router_models)],
                num_retries=MAX_RETRIES,
                retry_after=int(RETRY_BACKOFF_S),
                timeout=REQUEST_TIMEOUT_S,
            )
        return _router


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


def modality_of(item: Any) -> str:
    """``"image"`` or ``"audio"`` for one media item (tuples are encoded audio)."""
    if isinstance(item, tuple):
        return "audio"
    v = str(item).strip()
    if v.startswith("data:audio"):
        return "audio"
    if v.startswith(("data:", "http://", "https://")):
        return "image"
    return "audio" if Path(v).suffix.lower() in AUDIO_FORMATS else "image"


def image_url(value: str) -> str:
    """A LiteLLM ``image_url``: data URIs and http(s) URLs pass through, a local
    file path is read and base64-encoded."""
    v = str(value).strip()
    if v.startswith(("data:", "http://", "https://")):
        return v
    path = Path(v).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {value!r}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def audio_data(value: Any) -> tuple[str, str]:
    """``(base64_data, format)`` from a local path or an already encoded tuple."""
    if isinstance(value, tuple):
        return str(value[0]), str(value[1])
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"audio not found: {value!r}")
    fmt = AUDIO_FORMATS.get(path.suffix.lower(), "wav")
    return base64.b64encode(path.read_bytes()).decode("ascii"), fmt


def _content_part(item: Any, modality: str | None, image_detail: str) -> dict[str, Any]:
    kind = modality or modality_of(item)
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": image_url(item), "detail": image_detail}}
    if kind == "audio":
        data, fmt = audio_data(item)
        return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}
    raise ValueError(f"modality must be 'image' or 'audio', got {kind!r}")


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def _describe_exc(exc: BaseException) -> str:
    parts: list[str] = [f"type={type(exc).__name__}"]
    for attr in ("status_code", "code", "llm_provider", "model"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"{attr}={val!r}")
    parts.append(f"msg={str(exc)!r}")
    return " ".join(parts)


def _extract_content(response: Any, model: str) -> str:
    """Reply text, or ``""`` when the provider returned no content (e.g. a safety block)."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning("model %s returned no choices; using empty string", model)
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        logger.warning(
            "model %s returned no content (finish_reason=%r); using empty string",
            model,
            getattr(choices[0], "finish_reason", None),
        )
        return ""
    return content.strip()


def call_model(
    prompt: str,
    model: str,
    media: Any = None,
    *,
    modality: str | None = None,
    image_detail: str = "low",
    reasoning_effort: str = "minimal",
    temperature: float = 0.0,
) -> str:
    """One model call; returns the reply text (``""`` if the provider returned none).

    ``model`` is any LiteLLM model id (``openai/gpt-4o-mini``, ``azure/<deployment>``, ...),
    credentials from the provider's environment variables. ``media`` is one item or a
    list: image paths / URLs / data URIs, audio paths, or ``(base64, format)`` audio
    tuples; text goes in ``prompt``. ``modality`` forces ``"image"`` / ``"audio"`` for
    every item (default: inferred per item from its extension or URI).
    Raises ``RuntimeError`` after the router's retries are exhausted.
    """
    if image_detail not in ALLOWED_IMAGE_DETAILS:
        raise ValueError(f"image_detail must be one of {ALLOWED_IMAGE_DETAILS!r}, got {image_detail!r}")
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {ALLOWED_REASONING_EFFORTS!r}, got {reasoning_effort!r}"
        )
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty LiteLLM model id, got {model!r}")
    items = [] if media is None else ([media] if isinstance(media, (str, tuple)) else list(media))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(_content_part(item, modality, image_detail) for item in items)
    model = model.strip()
    try:
        response = _router_for(model).completion(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            drop_params=True,  # providers that reject a parameter get it dropped, not an error
        )
    except Exception as exc:
        logger.warning("model call failed (model=%s): %s", model, _describe_exc(exc))
        raise RuntimeError(
            f"model call failed for {model!r} after num_retries={MAX_RETRIES} "
            f"(timeout={REQUEST_TIMEOUT_S:.1f}s): {_describe_exc(exc)}"
        ) from exc
    return _extract_content(response, model)


T = TypeVar("T")


def run_batch(fns: list[Callable[[], T]], max_concurrency: int) -> list[T]:
    """Run blocking thunks concurrently (at most ``max_concurrency`` at once), preserving order."""
    if not fns:
        return []
    workers = max(1, min(max_concurrency, len(fns)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn) for fn in fns]
        return [f.result() for f in futures]
