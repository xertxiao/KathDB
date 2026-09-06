"""Synchronous LiteLLM helpers used by library functions.

``call_text`` is a blocking single-shot completion through a ``litellm.Router``
(client-side rpm/tpm caps, retries, per-request timeout); ``run_batch`` fans
blocking thunks out across a bounded thread pool preserving input order. The
sync API is used deliberately: async httpx transports cached at litellm module
scope outlive the event loop that created them.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from litellm import Router

from kathdb.common.logger import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT_S: float = 40.0
MAX_RETRIES: int = 3
RETRY_BACKOFF_S: float = 2.0

# Client-side rpm/tpm caps per model; override with KATHDB_AIOP_RPM / KATHDB_AIOP_TPM.
DEFAULT_RPM: int = int(os.environ.get("KATHDB_AIOP_RPM", "500"))
DEFAULT_TPM: int = int(os.environ.get("KATHDB_AIOP_TPM", "200000"))

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
    """Router that knows ``model``; any LiteLLM model id is registered on first use."""
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


T = TypeVar("T")


def to_litellm_model(model: str) -> str:
    """Validate a model id: any LiteLLM ``provider/model`` id, or a bare OpenAI name."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty LiteLLM model id, got {model!r}")
    return model.strip()


def _describe_exc(exc: BaseException) -> str:
    """One-line description of a LiteLLM exception (type, status, provider, message)."""
    parts: list[str] = [f"type={type(exc).__name__}"]
    for attr in ("status_code", "code", "llm_provider", "model"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"{attr}={val!r}")
    parts.append(f"msg={str(exc)!r}")
    return " ".join(parts)


def _extract_content(response: Any, litellm_model: str) -> str:
    """Response text of a LiteLLM completion; empty string (logged) when the
    provider returns no content, e.g. a safety-filter block."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        logger.warning(
            "litellm.completion returned no choices (model=%s); using empty string",
            litellm_model,
        )
        return ""
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        finish_reason = getattr(choice, "finish_reason", None)
        logger.warning(
            "litellm.completion returned None content (model=%s, finish_reason=%r); "
            "using empty string",
            litellm_model,
            finish_reason,
        )
        return ""
    return content.strip()


ALLOWED_IMAGE_DETAILS: tuple[str, ...] = ("low", "high", "auto")
ALLOWED_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")


def call_text(
    prompt: str,
    model: str,
    images: list[str],
    audios: list[tuple[str, str]] | None = None,
    image_detail: str = "low",
    reasoning_effort: str = "minimal",
    temperature: float = 0.0,
) -> str:
    """Blocking single-shot model call with retry and per-request timeout.

    ``model`` is any LiteLLM model id (credentials from the provider's env vars);
    ``images`` are data URIs, ``audios`` are ``(base64_data, format)`` tuples;
    ``image_detail`` / ``reasoning_effort`` must be in the ALLOWED_* tuples.
    Returns the stripped response text (empty when the provider returns no
    content); raises RuntimeError once the router's retries are exhausted.
    """
    if image_detail not in ALLOWED_IMAGE_DETAILS:
        raise ValueError(
            f"image_detail must be one of {ALLOWED_IMAGE_DETAILS!r}, "
            f"got {image_detail!r}"
        )
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {ALLOWED_REASONING_EFFORTS!r}, "
            f"got {reasoning_effort!r}"
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {"type": "image_url", "image_url": {"url": img, "detail": image_detail}}
        )
    for data, fmt in audios or []:
        content.append(
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}
        )
    litellm_model = to_litellm_model(model)
    try:
        response = _router_for(litellm_model).completion(
            model=litellm_model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            # Silently drop params the target provider does not accept.
            drop_params=True,
        )
    except Exception as exc:
        logger.warning(
            "router.completion failed (model=%s, num_retries=%d, timeout=%.1fs): %s",
            litellm_model,
            MAX_RETRIES,
            REQUEST_TIMEOUT_S,
            _describe_exc(exc),
        )
        raise RuntimeError(
            f"router.completion failed for model {litellm_model!r} after "
            f"num_retries={MAX_RETRIES} (timeout={REQUEST_TIMEOUT_S:.1f}s): "
            f"{_describe_exc(exc)}"
        ) from exc
    return _extract_content(response, litellm_model)


def run_batch(
    fns: list[Callable[[], T]],
    max_concurrency: int,
) -> list[T]:
    """Run blocking thunks on a thread pool of *max_concurrency*, preserving input order."""
    if not fns:
        return []
    workers = max(1, min(max_concurrency, len(fns)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn) for fn in fns]
        return [f.result() for f in futures]
