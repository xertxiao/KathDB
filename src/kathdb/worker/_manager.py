"""Worker lifecycle manager: a pool of up to ``max_workers`` worker subprocesses.

* ``acquire()`` / ``release()`` lease a worker for one operator's execution; blocks
  while every slot is busy (this is what lets independent operators run in parallel).
* ``replace(worker)`` swaps a poisoned worker for a fresh one in the same slot.
* ``get_worker()`` returns an idle worker without leasing it (sequential callers).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ..common.logger import get_logger
from ._worker import (
    WorkerClient,
    remove_conda_env,
    spawn_worker,
    spawn_worker_conda,
)

logger = get_logger(__name__)

__all__ = ["WorkerManager"]


class WorkerManager:
    """Owns the worker subprocesses (and their conda env) for one KathDB instance.

    Parameters
    ----------
    conda_env_name:
        If set, reuse this existing conda environment.
    requirements_path:
        Path to ``requirements.txt`` used when creating a new environment.
    max_workers:
        Number of worker processes, i.e. how many plan operators may execute at
        the same time.
    """

    def __init__(
        self,
        *,
        conda_env_name: str | None = None,
        requirements_path: str | Path | None = None,
        connect_timeout_s: float | None = None,
        exec_timeout_s: float | None = None,
        max_workers: int = 1,
    ) -> None:
        self._conda_env_name = conda_env_name
        self._requirements_path = Path(requirements_path) if requirements_path else None
        self._connect_timeout_s = connect_timeout_s
        self._exec_timeout_s = exec_timeout_s
        self.max_workers = max(1, int(max_workers))
        self._env_name: str | None = None
        self._created_env: bool = False

        self._cond = threading.Condition()
        self._idle: list[WorkerClient] = []  # live, not leased
        self._all: list[WorkerClient] = []  # every live worker (idle + leased)
        self._n_slots_taken = 0  # live + currently spawning
        # Spawns are serialized: the first one may CREATE the conda env.
        self._spawn_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Leasing (parallel execution)
    # ------------------------------------------------------------------

    def acquire(self) -> WorkerClient:
        """Lease a live worker; blocks while all ``max_workers`` slots are busy."""
        with self._cond:
            while True:
                self._drop_dead_idle()
                if self._idle:
                    return self._idle.pop()
                if self._n_slots_taken < self.max_workers:
                    self._n_slots_taken += 1
                    break
                self._cond.wait()
        try:
            worker = self._spawn_with_retry()
        except Exception:
            with self._cond:
                self._n_slots_taken -= 1
                self._cond.notify()
            raise
        with self._cond:
            self._all.append(worker)
        return worker

    def release(self, worker: WorkerClient) -> None:
        """Return a leased worker to the pool (a dead one is dropped)."""
        with self._cond:
            if worker.is_alive():
                self._idle.append(worker)
            else:
                self._forget(worker)
            self._cond.notify()

    def replace(self, worker: WorkerClient) -> WorkerClient:
        """Swap a poisoned leased worker for a fresh one (the lease is kept)."""
        try:
            worker.shutdown()
        except Exception:  # noqa: BLE001 - already dead in the common case
            pass
        with self._cond:
            if worker in self._all:
                self._all.remove(worker)
        fresh = self._spawn_with_retry()
        with self._cond:
            self._all.append(fresh)
        return fresh

    # ------------------------------------------------------------------
    # Sequential convenience
    # ------------------------------------------------------------------

    def get_worker(self) -> WorkerClient:
        """Return a live idle worker without leasing it, spawning one if needed.

        For callers that run alone (plan-time profiling, warm-start); parallel
        execution uses :meth:`acquire` / :meth:`release`.
        """
        with self._cond:
            self._drop_dead_idle()
            if self._idle:
                return self._idle[-1]
            self._n_slots_taken += 1
        try:
            worker = self._spawn_with_retry()
        except Exception:
            with self._cond:
                self._n_slots_taken -= 1
            raise
        with self._cond:
            self._all.append(worker)
            self._idle.append(worker)
        return worker

    def shutdown(self, *, remove_env: bool = False) -> None:
        """Shut down every worker subprocess and optionally remove the env."""
        with self._cond:
            workers = list(self._all)
            self._all.clear()
            self._idle.clear()
            self._n_slots_taken = 0
            self._cond.notify_all()
        for w in workers:
            try:
                w.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error shutting down worker: %s", exc)

        if remove_env and self._created_env and self._env_name:
            remove_conda_env(self._env_name)
            logger.info("Removed conda env: %s", self._env_name)
            self._env_name = None
            self._created_env = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drop_dead_idle(self) -> None:
        # Caller holds ``self._cond``.
        dead = [w for w in self._idle if not w.is_alive()]
        for w in dead:
            self._idle.remove(w)
            self._forget(w)

    def _forget(self, worker: WorkerClient) -> None:
        # Caller holds ``self._cond``.
        if worker in self._all:
            self._all.remove(worker)
        self._n_slots_taken = max(0, self._n_slots_taken - 1)

    def _spawn_with_retry(self) -> WorkerClient:
        """Spawn one worker, retrying transient failures (e.g. a slow connect while
        several processes provision from the same conda env).

        KATHDB_WORKER_SPAWN_RETRIES (default 2 extra attempts) and
        KATHDB_WORKER_SPAWN_BACKOFF_S (default 30s, doubled each attempt).
        """
        attempts = int(os.environ.get("KATHDB_WORKER_SPAWN_RETRIES", "2")) + 1
        backoff = float(os.environ.get("KATHDB_WORKER_SPAWN_BACKOFF_S", "30"))
        last: Exception | None = None
        for i in range(attempts):
            try:
                with self._spawn_lock:
                    return self._spawn()
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if final
                last = exc
                if i == attempts - 1:
                    break
                logger.warning(
                    "Worker provision attempt %d/%d failed (%s); retrying in %.0fs",
                    i + 1,
                    attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        assert last is not None
        raise last

    def _spawn(self) -> WorkerClient:
        """Spawn a new worker subprocess (in the existing or a freshly created env)."""
        if self._conda_env_name is not None:
            worker = spawn_worker(
                self._conda_env_name,
                connect_timeout_s=self._connect_timeout_s,
                exec_timeout_s=self._exec_timeout_s,
            )
            self._env_name = self._conda_env_name
            self._created_env = False
            logger.info("Worker started in existing conda env: %s", self._conda_env_name)
            return worker
        worker, env_name = spawn_worker_conda(
            requirements_path=self._requirements_path,
            env_prefix="kathdb_worker",
            connect_timeout_s=self._connect_timeout_s,
            exec_timeout_s=self._exec_timeout_s,
        )
        self._env_name = env_name
        self._created_env = True
        logger.info("Worker started in conda env: %s", env_name)
        return worker
