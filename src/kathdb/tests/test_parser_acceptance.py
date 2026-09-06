"""Human-review acceptance must be an exact-intent match, not a substring hit."""

from __future__ import annotations

from kathdb.parser.parser import _is_acceptance


def test_exact_acceptance_replies():
    for reply in ("accept", "Accept", " ACCEPT ", "accepted", "ok", "lgtm", "yes"):
        assert _is_acceptance(reply)


def test_corrective_feedback_containing_accept_is_not_acceptance():
    assert not _is_acceptance("I can't accept step 2, change the join")
    assert not _is_acceptance("this is not acceptable")
    assert not _is_acceptance("accept step 1 but redo step 2")


def test_empty_and_none_are_not_acceptance():
    assert not _is_acceptance(None)
    assert not _is_acceptance("")
    assert not _is_acceptance("   ")
