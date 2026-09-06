"""WorkerManager leasing: slot limit, release, poisoned replacement."""

from __future__ import annotations

import threading
import time

from kathdb.worker._manager import WorkerManager


class _FakeWorker:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def shutdown(self):
        self.alive = False


def _manager(max_workers):
    m = WorkerManager(max_workers=max_workers)
    m._spawn = lambda: _FakeWorker()  # type: ignore[method-assign]
    return m


def test_acquire_blocks_at_max_and_release_unblocks():
    m = _manager(2)
    w1, w2 = m.acquire(), m.acquire()
    got = []

    def waiter():
        got.append(m.acquire())

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.1)
    assert got == []  # blocked: both slots busy
    m.release(w1)
    t.join(timeout=2)
    assert got == [w1]  # reused the released worker, no third process
    m.release(w2)
    m.release(got[0])
    assert len(m._all) == 2


def test_replace_swaps_poisoned_worker_keeping_slot():
    m = _manager(1)
    w = m.acquire()
    w.alive = False
    fresh = m.replace(w)
    assert fresh is not w and fresh.is_alive()
    assert m._all == [fresh]
    m.release(fresh)
    assert m.get_worker() is fresh  # idle again, no extra spawn


def test_dead_idle_worker_is_dropped_and_respawned():
    m = _manager(1)
    w = m.get_worker()
    w.alive = False
    w2 = m.get_worker()
    assert w2 is not w and w2.is_alive()
    assert m._n_slots_taken == 1
