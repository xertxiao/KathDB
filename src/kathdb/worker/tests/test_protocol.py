"""Worker IPC framing helpers over an in-memory socketpair."""

from __future__ import annotations

import socket
import threading

import pandas as pd
import pytest

from kathdb.worker._worker import (
    _recv_dataframe,
    _recv_message,
    _recv_pickle,
    _send_dataframe,
    _send_message,
    _send_pickle,
    _serialize_dataframe,
    _serialize_pickle,
)


def _rw_pair():
    a, b = socket.socketpair()
    return a.makefile("wb", buffering=0), b.makefile("rb", buffering=0), (a, b)


def test_message_roundtrip():
    wf, rf, _ = _rw_pair()
    _send_message(wf, "execute", {"entrypoint": "f", "df_names": ["x"]})
    msg = _recv_message(rf)
    assert msg["type"] == "execute"
    assert msg["entrypoint"] == "f"
    assert msg["df_names"] == ["x"]


def test_dataframe_roundtrip():
    wf, rf, _ = _rw_pair()
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    _send_dataframe(wf, df)
    out = _recv_dataframe(rf)
    pd.testing.assert_frame_equal(out, df)


def test_pickle_roundtrip():
    wf, rf, _ = _rw_pair()
    _send_pickle(wf, {"k": [1, 2, 3]})
    assert _recv_pickle(rf) == {"k": [1, 2, 3]}


def test_serialize_dataframe_returns_bytes():
    assert isinstance(_serialize_dataframe(pd.DataFrame({"x": [1]})), bytes)


def test_serialize_pickle_raises_before_send_on_unpicklable():
    # Serialization raises before any frame is sent (no desync).
    with pytest.raises(Exception):
        _serialize_pickle(threading.Lock())


def test_recv_message_raises_eof_on_closed_channel():
    wf, rf, (a, b) = _rw_pair()
    # shutdown() forces EOF even though the makefile still references the write fd.
    a.shutdown(socket.SHUT_RDWR)
    a.close()
    with pytest.raises(EOFError):
        _recv_message(rf)
