"""HTTP 429 detection, backoff and accounting in ``invoke_structured_with_retry``."""

from __future__ import annotations

import pytest

import kathdb.common.utils as U


class _RateLimitError(Exception):
    """Class name matches the provider SDK error."""


_MSG = (
    "Error code: 429 - {'error': {'code': 'RateLimitReached', 'message': "
    "'Rate limit of 2000000 per 60s exceeded. Please wait 3 seconds before retrying.'}}"
)


def test_detection():
    assert U.is_rate_limit_error(_RateLimitError(_MSG))
    assert U.is_rate_limit_error(Exception(_MSG))

    class _Http(Exception):
        status_code = 429

    assert U.is_rate_limit_error(_Http("boom"))
    assert not U.is_rate_limit_error(ValueError("schema validation failed"))
    assert not U.is_rate_limit_error(Exception("500 internal error"))


def test_wait_honors_server_hint():
    w = U._rate_limit_wait_seconds(Exception(_MSG), 0)
    assert 4.0 <= w <= 7.0


def test_retry_then_succeed_counts_retries(monkeypatch):
    monkeypatch.setattr(U, "_rate_limit_wait_seconds", lambda exc, a: 0.0)
    U.reset_rate_limit_stats()

    class _Flaky:
        n = 0

        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt, config=None):
            self.n += 1
            if self.n < 3:
                raise _RateLimitError("429 RateLimitReached wait 1 seconds")
            return "OK"

    r = U.invoke_structured_with_retry("p", llm=_Flaky(), schema=str, max_retries=1)
    st = U.get_rate_limit_stats()
    assert r == "OK"
    assert st["retries"] == 2 and st["exhausted"] == 0
    assert st["hit"] is True and st["degraded"] is False


def test_exhaustion_sets_degraded(monkeypatch):
    monkeypatch.setattr(U, "_rate_limit_wait_seconds", lambda exc, a: 0.0)
    U.reset_rate_limit_stats()

    class _Always:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt, config=None):
            raise _RateLimitError("429 RateLimitReached wait 1 seconds")

    with pytest.raises(Exception):
        U.invoke_structured_with_retry("p", llm=_Always(), schema=str, max_retries=1)
    st = U.get_rate_limit_stats()
    assert st["exhausted"] == 1 and st["degraded"] is True


def test_validation_error_still_retried_not_treated_as_rate_limit(monkeypatch):
    U.reset_rate_limit_stats()

    class _Bad:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt, config=None):
            raise ValueError("not valid json for schema")

    with pytest.raises(ValueError):
        U.invoke_structured_with_retry("p", llm=_Bad(), schema=str, max_retries=2)
    st = U.get_rate_limit_stats()
    assert st["retries"] == 0 and st["exhausted"] == 0 and st["hit"] is False
