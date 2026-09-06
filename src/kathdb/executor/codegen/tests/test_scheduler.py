"""Dependency-driven scheduling in Executor.run (parallel branches)."""

from __future__ import annotations

import threading
import time

import pandas as pd

from kathdb.executor.codegen.codegen import CodeGenerator
from kathdb.executor.executor import Executor
from kathdb.plan_gen.plan_node import FAONode


class _Pool:
    """WorkerManager stand-in: acquire/release with a slot limit, no processes."""

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self._sem = threading.BoundedSemaphore(max_workers)
        self.peak = 0
        self.active = 0
        self._lock = threading.Lock()

    def acquire(self):
        self._sem.acquire()
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        return object()

    def release(self, w):
        with self._lock:
            self.active -= 1
        self._sem.release()

    def replace(self, w):  # never poisoned here
        return w


class _Exec:
    def __init__(self, op, outputs):
        self.op = op
        self.outputs = outputs
        self.children = []
        self.metadata = {}
        self.function = None

    def replace_children(self, deps):
        self.children = list(deps)


def _plan():
    # in -> A -> A2 -> A3 -\
    #                       > C
    # in -> B -> B2 -------/
    a = FAONode(op="A", inputs=["in_t"], outputs=["a"])
    a2 = FAONode(op="A2", inputs=["a"], outputs=["a2"], children=[a])
    a3 = FAONode(op="A3", inputs=["a2"], outputs=["a3"], children=[a2])
    b = FAONode(op="B", inputs=["in_t"], outputs=["b"])
    b2 = FAONode(op="B2", inputs=["b"], outputs=["b2"], children=[b])
    c = FAONode(op="C", inputs=["a3", "b2"], outputs=["c"], children=[a3, b2])
    return FAONode(op="logical_plan", children=[c])


def _max_in_flight(events):
    """Peak number of operators executing at once, from the start/end timeline."""
    timeline = sorted((t, 1 if kind == "start" else -1) for _, kind, t in events)
    cur = peak = 0
    for _, delta in timeline:
        cur += delta
        peak = max(peak, cur)
    return peak


def _run(max_workers, durations):
    cg = CodeGenerator(generation_llm=None, diagnosis_llm=None, revision_llm=None)
    ex = Executor(code_gen=cg, save_functions=False)
    events: list[tuple[str, str, float]] = []
    lock = threading.Lock()
    t0 = time.time()

    def fake_codegen(node, materialized, siblings, parent, cg_in, cdm, depth):
        assert all(i in materialized for i in node.inputs)  # inputs ready
        return _Exec(node.op, list(node.outputs)), "code"

    def fake_exec(plan_node, exec_ctx, *, layer_idx):
        with lock:
            events.append((plan_node.op, "start", time.time() - t0))
        time.sleep(durations[plan_node.op])
        for o in plan_node.outputs:
            exec_ctx[o] = pd.DataFrame({"x": [1]})
        with lock:
            events.append((plan_node.op, "end", time.time() - t0))
        return plan_node

    cg._codegen_layered_node = fake_codegen  # type: ignore[assignment]
    ex._execute_with_regen = fake_exec  # type: ignore[assignment]
    pool = _Pool(max_workers)
    out = ex.run(
        {
            "q_in": "q",
            "actions": [],
            "relation_context": None,
            "input_rel_names": ["in_t"],
            "input_rel": [pd.DataFrame({"x": [0]})],
            "logical_plan": _plan(),
        },
        worker_manager=pool,
    )
    return out, ex, events, pool


def test_two_slots_run_branches_concurrently_and_join_waits():
    d = {"A": 0.15, "A2": 0.15, "A3": 0.15, "B": 0.5, "B2": 0.15, "C": 0.05}
    out, ex, events, pool = _run(2, d)
    start = {op: t for op, kind, t in events if kind == "start"}
    end = {op: t for op, kind, t in events if kind == "end"}
    # A's chain progressed while B was still running.
    assert start["A2"] < end["B"] and start["A3"] < end["B"]
    # The join started only after both producers finished.
    assert start["C"] >= max(end["A3"], end["B2"]) - 1e-3
    # Never more than 2 operators in flight; but 2 at once did happen.
    assert _max_in_flight(events) == 2
    assert ex._last_code_tree.op == "C"
    assert set(out) == {"in_t", "a", "a2", "a3", "b", "b2", "c"}


def test_one_slot_is_sequential():
    d = {k: 0.02 for k in ("A", "A2", "A3", "B", "B2", "C")}
    out, ex, events, pool = _run(1, d)
    assert _max_in_flight(events) == 1
    ends = [t for op, kind, t in events if kind == "end"]
    starts = [t for op, kind, t in events if kind == "start"][1:]
    assert all(s >= e - 1e-3 for s, e in zip(starts, ends))  # no overlap
