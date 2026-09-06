"""Common utilities for KathDB."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import re
import shutil
import time
import typing
from typing import Any, Sequence, TypeVar

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..common.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "message_to_text",
    "parse_typed_value",
    "sample_sequence",
    "sample_dataframe",
    "get_device_info",
    "invoke_structured_with_retry",
    "ainvoke_structured_with_retry",
    "abatch_structured_with_retry",
    "get_rate_limit_stats",
    "reset_rate_limit_stats",
    "is_rate_limit_error",
]

T = TypeVar("T")
TModel = TypeVar("TModel", bound=BaseModel)


# --- HTTP 429 rate-limit backoff -------------------------------------------
_RATE_LIMIT_MAX_WAITS = 6
_RATE_LIMIT_WAIT_RE = re.compile(r"wait\s+(\d+)\s*second", re.IGNORECASE)

_RATE_LIMIT_STATS: dict[str, float] = {
    "retries": 0,  # 429s slept through and retried
    "exhausted": 0,  # 429s that ran out of waits and propagated
    "wait_seconds": 0.0,
}


def reset_rate_limit_stats() -> None:
    _RATE_LIMIT_STATS.update(retries=0, exhausted=0, wait_seconds=0.0)


def get_rate_limit_stats() -> dict[str, Any]:
    """Snapshot of this process's 429 accounting (``degraded`` = a call failed despite backoff)."""
    s = dict(_RATE_LIMIT_STATS)
    s["hit"] = bool(s["retries"] or s["exhausted"])
    s["degraded"] = bool(s["exhausted"])
    return s


def _is_rate_limit(exc: BaseException) -> bool:
    """True for an HTTP 429 / rate-limit error from any provider."""
    if type(exc).__name__ == "RateLimitError":
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if code == 429:
        return True
    s = str(exc).lower()
    return "429" in s and (
        "rate limit" in s or "ratelimit" in s or "too many requests" in s
    )


def is_rate_limit_error(exc: BaseException) -> bool:
    """True for an HTTP 429 / rate-limit error from any provider."""
    return _is_rate_limit(exc)


def _rate_limit_wait_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to sleep before retrying a 429: the server's "wait N seconds" hint
    if present, else capped exponential backoff; plus jitter."""
    m = _RATE_LIMIT_WAIT_RE.search(str(exc))
    base = (float(m.group(1)) + 1.0) if m else min(60.0, 5.0 * (2**attempt))
    return min(75.0, base) + random.uniform(0.0, 3.0)


def _invoke_with_rate_limit_retry(structured_llm: Any, prompt: Any, config: Any) -> Any:
    """Invoke, sleeping and retrying on HTTP 429; other exceptions propagate."""
    for attempt in range(_RATE_LIMIT_MAX_WAITS + 1):
        try:
            return structured_llm.invoke(prompt, config=config)
        except Exception as exc:  # noqa: BLE001 - re-raise non-429 immediately
            if not _is_rate_limit(exc):
                raise
            if attempt >= _RATE_LIMIT_MAX_WAITS:
                _RATE_LIMIT_STATS["exhausted"] += 1
                raise
            wait = _rate_limit_wait_seconds(exc, attempt)
            _RATE_LIMIT_STATS["retries"] += 1
            _RATE_LIMIT_STATS["wait_seconds"] += wait
            logger.warning(
                "Rate limited (429); sleeping %.0fs then retry %d/%d: %s",
                wait,
                attempt + 1,
                _RATE_LIMIT_MAX_WAITS,
                str(exc)[:140],
            )
            time.sleep(wait)


async def _ainvoke_with_rate_limit_retry(
    structured_llm: Any, prompt: Any, config: Any
) -> Any:
    """Async sibling of :func:`_invoke_with_rate_limit_retry`."""
    for attempt in range(_RATE_LIMIT_MAX_WAITS + 1):
        try:
            return await structured_llm.ainvoke(prompt, config=config)
        except Exception as exc:  # noqa: BLE001 - re-raise non-429 immediately
            if not _is_rate_limit(exc):
                raise
            if attempt >= _RATE_LIMIT_MAX_WAITS:
                _RATE_LIMIT_STATS["exhausted"] += 1
                raise
            wait = _rate_limit_wait_seconds(exc, attempt)
            _RATE_LIMIT_STATS["retries"] += 1
            _RATE_LIMIT_STATS["wait_seconds"] += wait
            logger.warning(
                "Rate limited (429, async); sleeping %.0fs then retry %d/%d: %s",
                wait,
                attempt + 1,
                _RATE_LIMIT_MAX_WAITS,
                str(exc)[:140],
            )
            await asyncio.sleep(wait)


def invoke_structured_with_retry(
    prompt: Any,
    *,
    llm: BaseChatModel,
    schema: type[TModel],
    max_retries: int = 3,
    config: RunnableConfig | None = None,
) -> TModel:
    """Invoke ``llm`` with structured output, retrying validation errors up to
    ``max_retries`` times; ``config`` is forwarded into every call."""
    structured_llm = llm.with_structured_output(schema)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _invoke_with_rate_limit_retry(structured_llm, prompt, config)
        except (ValidationError, ValueError, TypeError) as exc:
            last_exc = exc
            logger.warning(
                "Structured output validation failed for %s (attempt %d/%d): %s",
                schema.__name__,
                attempt,
                max_retries,
                exc,
            )
    assert last_exc is not None
    raise last_exc


async def ainvoke_structured_with_retry(
    prompt: Any,
    *,
    llm: BaseChatModel,
    schema: type[TModel],
    max_retries: int = 3,
    config: RunnableConfig | None = None,
) -> TModel:
    """Async sibling of :func:`invoke_structured_with_retry`."""
    structured_llm = llm.with_structured_output(schema)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await _ainvoke_with_rate_limit_retry(structured_llm, prompt, config)
        except (ValidationError, ValueError, TypeError) as exc:
            last_exc = exc
            logger.warning(
                "Async structured output validation failed for %s "
                "(attempt %d/%d): %s",
                schema.__name__,
                attempt,
                max_retries,
                exc,
            )
    assert last_exc is not None
    raise last_exc


_VALIDATION_EXC_TYPES: tuple[type[BaseException], ...] = (
    ValidationError,
    ValueError,
    TypeError,
)


async def _abatch_with_retry_core(
    prompts: Sequence[Any],
    *,
    llm: BaseChatModel,
    schema: type[TModel],
    max_retries: int,
    config: RunnableConfig | None,
    tolerant: bool,
) -> list[TModel | Exception]:
    """Batched structured invoke; only indices that failed validation are retried.
    Non-validation exceptions raise immediately unless ``tolerant``."""
    n = len(prompts)
    if n == 0:
        return []
    structured_llm = llm.with_structured_output(schema)
    results: list[TModel | Exception | None] = [None] * n
    pending: list[int] = list(range(n))
    last_exc_by_idx: dict[int, Exception] = {}
    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        sub_prompts = [prompts[i] for i in pending]
        sub_results = await structured_llm.abatch(
            sub_prompts, config=config, return_exceptions=True
        )
        next_pending: list[int] = []
        for local_i, res in enumerate(sub_results):
            global_i = pending[local_i]
            if isinstance(res, _VALIDATION_EXC_TYPES):
                last_exc_by_idx[global_i] = res
                next_pending.append(global_i)
            elif isinstance(res, BaseException):
                if tolerant:
                    last_exc_by_idx[global_i] = (
                        res if isinstance(res, Exception) else Exception(str(res))
                    )
                    results[global_i] = last_exc_by_idx[global_i]
                else:
                    raise res
            else:
                results[global_i] = res
                last_exc_by_idx.pop(global_i, None)
        if next_pending:
            logger.warning(
                "abatch validation: %d/%d still pending after attempt %d/%d for %s",
                len(next_pending),
                n,
                attempt,
                max_retries,
                schema.__name__,
            )
        pending = next_pending
    if pending:
        if tolerant:
            for idx in pending:
                results[idx] = last_exc_by_idx[idx]
        else:
            raise last_exc_by_idx[pending[0]]
    return results  # type: ignore[return-value]


async def abatch_structured_with_retry(
    prompts: Sequence[Any],
    *,
    llm: BaseChatModel,
    schema: type[TModel],
    max_retries: int = 3,
    config: RunnableConfig | None = None,
) -> list[TModel]:
    """Batched structured invoke in input order; raises the first failure left after retries."""
    out = await _abatch_with_retry_core(
        prompts,
        llm=llm,
        schema=schema,
        max_retries=max_retries,
        config=config,
        tolerant=False,
    )
    return out  # type: ignore[return-value]


def message_to_text(message: Any) -> str:
    """Normalize LangChain message responses into text."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        if parts:
            return "".join(parts).strip()
    if content is not None:
        return str(content).strip()
    return str(message).strip()


# Seeded RNG so the samples shown in prompts are reproducible across runs.
_SAMPLE_RNG = random.Random(17)


def sample_sequence(items: Sequence[T], sample_rows: int) -> list[T]:
    """Up to ``sample_rows`` random elements of ``items``, in original order."""
    max_rows = sample_rows if sample_rows > 0 else 0
    if max_rows <= 0 or not items:
        return []
    if max_rows >= len(items):
        return list(items)
    indices = sorted(_SAMPLE_RNG.sample(range(len(items)), max_rows))
    return [items[i] for i in indices]


def sample_dataframe(df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    """Up to ``sample_rows`` random rows of ``df``, in original order."""
    max_rows = sample_rows if sample_rows > 0 else 0
    if max_rows <= 0:
        return df.head(0)
    row_count = len(df)
    if max_rows >= row_count:
        return df
    indices = sorted(_SAMPLE_RNG.sample(range(row_count), max_rows))
    return df.iloc[indices]


_TYPE_NAMESPACE: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "None": type(None),
    "NoneType": type(None),
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Any": typing.Any,
}


def _resolve_type(type_str: str) -> type:
    """Convert a Python type-annotation string to an actual type object."""
    try:
        return eval(type_str, {"__builtins__": {}}, _TYPE_NAMESPACE)  # noqa: S307
    except Exception:
        logger.warning(
            "Cannot resolve type annotation %r; falling back to Any", type_str
        )
        return typing.Any


def parse_typed_value(value_str: str, type_str: str) -> Any:
    """Parse *value_str* into the Python type described by *type_str*."""
    resolved_type = _resolve_type(type_str)
    adapter = TypeAdapter(resolved_type)

    if type_str.strip() == "str":
        return adapter.validate_python(value_str)

    try:
        return adapter.validate_python(json.loads(value_str))
    except Exception:
        return adapter.validate_python(value_str)


def get_device_info() -> dict[str, Any]:
    """Lightweight snapshot of local device characteristics."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
    }

    try:
        stats = shutil.disk_usage(".")
        info["disk_total_gb"] = round(stats.total / (1024**3), 2)
        info["disk_free_gb"] = round(stats.free / (1024**3), 2)
    except OSError:
        pass

    try:
        import psutil  # type: ignore
    except Exception:  # pragma: no cover
        psutil = None  # type: ignore

    if psutil:
        try:
            virtual_mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(virtual_mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(virtual_mem.available / (1024**3), 2)
        except Exception:
            pass

    try:
        import torch  # type: ignore

        has_gpu = bool(torch.cuda.is_available())
        info["gpu_available"] = has_gpu
        if has_gpu:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info.setdefault("gpu_available", False)

    return info
