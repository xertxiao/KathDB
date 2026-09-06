"""Async structured-output helpers: retry on validation errors, strict batching."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from kathdb.common.utils import (
    abatch_structured_with_retry,
    ainvoke_structured_with_retry,
)


class _Toy(BaseModel):
    n: int


class _FakeStructuredRunnable:
    """Stand-in for ``llm.with_structured_output(schema)`` with injectable handlers."""

    def __init__(self, *, ainvoke_handler=None, abatch_handler=None):
        self._ainvoke_handler = ainvoke_handler
        self._abatch_handler = abatch_handler

    async def ainvoke(self, prompt, config=None):
        return self._ainvoke_handler(prompt)

    async def abatch(self, prompts, config=None, return_exceptions=False):
        results = self._abatch_handler(prompts)
        if return_exceptions:
            return [
                r if not isinstance(r, BaseException) else r
                for r in results
            ]
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return results


class _FakeLLM:
    """Stand-in for ``BaseChatModel`` exposing only ``with_structured_output``."""

    def __init__(self, runnable: _FakeStructuredRunnable):
        self._runnable = runnable

    def with_structured_output(self, schema):  # noqa: ARG002 — schema unused
        return self._runnable


def test_ainvoke_happy_path_returns_first_result():
    runnable = _FakeStructuredRunnable(ainvoke_handler=lambda p: _Toy(n=1))
    out = asyncio.run(
        ainvoke_structured_with_retry(
            "p", llm=_FakeLLM(runnable), schema=_Toy, max_retries=3
        )
    )
    assert out.n == 1


def test_ainvoke_retries_on_validation_then_succeeds():
    attempts = {"n": 0}

    def _handle(_prompt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("first try fails")
        return _Toy(n=42)

    runnable = _FakeStructuredRunnable(ainvoke_handler=_handle)
    out = asyncio.run(
        ainvoke_structured_with_retry(
            "p", llm=_FakeLLM(runnable), schema=_Toy, max_retries=3
        )
    )
    assert attempts["n"] == 2
    assert out.n == 42


def test_ainvoke_raises_after_exhaustion():
    def _handle(_prompt):
        raise ValueError("always fails")

    runnable = _FakeStructuredRunnable(ainvoke_handler=_handle)
    with pytest.raises(ValueError, match="always fails"):
        asyncio.run(
            ainvoke_structured_with_retry(
                "p", llm=_FakeLLM(runnable), schema=_Toy, max_retries=2
            )
        )


def test_abatch_strict_happy_path_preserves_order():
    runnable = _FakeStructuredRunnable(
        abatch_handler=lambda prompts: [_Toy(n=int(p)) for p in prompts]
    )
    out = asyncio.run(
        abatch_structured_with_retry(
            ["1", "2", "3"], llm=_FakeLLM(runnable), schema=_Toy, max_retries=2
        )
    )
    assert [t.n for t in out] == [1, 2, 3]


def test_abatch_strict_retries_only_failed_indices():
    """Only the index that failed validation is re-sent."""
    calls: list[list[str]] = []

    def _handle(prompts):
        calls.append(list(prompts))
        if len(calls) == 1:
            return [
                _Toy(n=1),
                ValueError("transient on idx 1"),
                _Toy(n=3),
            ]
        return [_Toy(n=2)]

    runnable = _FakeStructuredRunnable(abatch_handler=_handle)
    out = asyncio.run(
        abatch_structured_with_retry(
            ["a", "b", "c"], llm=_FakeLLM(runnable), schema=_Toy, max_retries=3
        )
    )
    assert [t.n for t in out] == [1, 2, 3]
    assert calls[0] == ["a", "b", "c"]
    assert calls[1] == ["b"]


def test_abatch_strict_raises_first_persistent_failure():
    """A persistently failing index raises after exhaustion."""

    def _handle(prompts):
        out = []
        for p in prompts:
            if p == "b":
                out.append(ValueError("idx 1 persistently bad"))
            elif p == "a":
                out.append(_Toy(n=1))
            else:  # "c"
                out.append(_Toy(n=3))
        return out

    runnable = _FakeStructuredRunnable(abatch_handler=_handle)
    with pytest.raises(ValueError, match="idx 1 persistently bad"):
        asyncio.run(
            abatch_structured_with_retry(
                ["a", "b", "c"], llm=_FakeLLM(runnable), schema=_Toy, max_retries=2
            )
        )


def test_abatch_strict_raises_immediately_on_non_retryable_error():
    def _handle(_prompts):
        return [_Toy(n=1), RuntimeError("transport down"), _Toy(n=3)]

    runnable = _FakeStructuredRunnable(abatch_handler=_handle)
    with pytest.raises(RuntimeError, match="transport down"):
        asyncio.run(
            abatch_structured_with_retry(
                ["a", "b", "c"], llm=_FakeLLM(runnable), schema=_Toy, max_retries=3
            )
        )


def test_abatch_strict_empty_input_returns_empty_list():
    runnable = _FakeStructuredRunnable(
        abatch_handler=lambda _prompts: pytest.fail("should not be called")
    )
    out = asyncio.run(
        abatch_structured_with_retry(
            [], llm=_FakeLLM(runnable), schema=_Toy, max_retries=3
        )
    )
    assert out == []
