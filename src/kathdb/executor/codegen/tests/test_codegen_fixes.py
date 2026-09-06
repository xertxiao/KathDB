"""Worker re-fetch per attempt, infra-error retry, diamond dedupe, output-schema block."""

from __future__ import annotations

import pandas as pd
import pytest

from kathdb.executor.codegen.codegen import CodeGenerator
from kathdb.executor.codegen.prompts import _format_output_schema_block
from kathdb.plan_gen.plan_node import FAONode
from kathdb.worker import KathDBWorkerExecuteError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeManager:
    """WorkerManager stand-in that hands out a fresh worker per get_worker()."""

    def __init__(self) -> None:
        self.calls = 0

    def get_worker(self):
        self.calls += 1
        return f"worker-{self.calls}"


class _FakeFunction:
    name = "n1"
    script_path = None
    str_impl = "def n1(df):\n    return df"


class _FakeNode:
    """Minimal FAOExecutableNode surface for _execute_with_regen."""

    def __init__(self, fail_on_workers: set[str], error: Exception) -> None:
        self.op = "n1"
        self.outputs = ["out1"]
        self.children: list = []
        self.metadata: dict = {}
        self.function = _FakeFunction()
        self.implementation_guidance = ""
        self._fail_on_workers = fail_on_workers
        self._error = error
        self.seen_workers: list = []

    def execute(self, ctx, *, profile, worker):
        self.seen_workers.append(worker)
        if worker in self._fail_on_workers:
            raise self._error
        ctx["out1"] = pd.DataFrame({"a": [1]})
        return ctx


def _make_code_gen(**kwargs) -> CodeGenerator:
    # LLMs are never invoked on the infra-error paths under test.
    return CodeGenerator(
        generation_llm=None,
        diagnosis_llm=None,
        revision_llm=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Poisoned worker is re-fetched per execution attempt
# ---------------------------------------------------------------------------


def test_execute_with_regen_refetches_worker_per_attempt():
    cg = _make_code_gen(max_retries=2)
    mgr = _FakeManager()
    cg._worker_manager = mgr

    # worker-1 dies with a framing error; worker-2 succeeds.
    node = _FakeNode({"worker-1"}, EOFError("Worker connection closed"))
    ctx: dict = {}
    out = cg._execute_with_regen(node, ctx, layer_idx=0)

    assert out is node
    assert node.seen_workers == ["worker-1", "worker-2"]
    assert mgr.calls == 2
    assert "out1" in ctx


def test_execute_with_regen_static_worker_backward_compat():
    cg = _make_code_gen()
    cg._worker = "static-worker"  # no manager: static worker= path

    node = _FakeNode(set(), RuntimeError("unused"))
    cg._execute_with_regen(node, {}, layer_idx=0)
    assert node.seen_workers == ["static-worker"]


def test_run_requires_worker_or_manager():
    cg = _make_code_gen()
    with pytest.raises(ValueError, match="worker or worker_manager"):
        cg.run({}, worker=None, worker_manager=None)


# ---------------------------------------------------------------------------
# Infra errors retry the SAME code (no diagnosis / regeneration)
# ---------------------------------------------------------------------------


def test_timeout_retries_same_code_without_regen():
    cg = _make_code_gen(max_retries=2)
    mgr = _FakeManager()
    cg._worker_manager = mgr

    # diagnosis_llm is None: any regeneration attempt would raise.
    timeout_err = KathDBWorkerExecuteError("n1", "worker timed out after 5.0s")
    node = _FakeNode({"worker-1"}, timeout_err)
    out = cg._execute_with_regen(node, {}, layer_idx=0)

    assert out is node  # same node, same code
    assert node.seen_workers == ["worker-1", "worker-2"]


def test_infra_error_exhausting_retries_raises():
    cg = _make_code_gen(max_retries=1)
    mgr = _FakeManager()
    cg._worker_manager = mgr

    node = _FakeNode({"worker-1", "worker-2"}, EOFError("dead"))
    with pytest.raises(EOFError):
        cg._execute_with_regen(node, {}, layer_idx=0)
    assert mgr.calls == 2  # one fresh worker per attempt


def test_is_infra_error_classification():
    assert CodeGenerator._is_infra_error(EOFError("x"), "x")
    assert CodeGenerator._is_infra_error(BrokenPipeError(), "")
    timeout = KathDBWorkerExecuteError("fn", "worker timed out after 3s")
    assert CodeGenerator._is_infra_error(timeout, timeout.underlying_error)
    genuine = KathDBWorkerExecuteError("fn", "KeyError: 'col'")
    assert not CodeGenerator._is_infra_error(genuine, genuine.underlying_error)


# ---------------------------------------------------------------------------
# Diamond plans dedupe duplicated logical nodes by op
# ---------------------------------------------------------------------------


def test_topo_layers_dedupes_diamond_duplicates():
    # Shared `scan` node duplicated as two instances (same .op).
    scan_a = FAONode(op="scan", inputs=["base"], outputs=["scanned"])
    scan_b = FAONode(op="scan", inputs=["base"], outputs=["scanned"])
    left = FAONode(op="left", inputs=["scanned"], outputs=["l_out"])
    right = FAONode(op="right", inputs=["scanned"], outputs=["r_out"])
    left.add_child(scan_a)
    right.add_child(scan_b)
    join = FAONode(op="join", inputs=["l_out", "r_out"], outputs=["joined"])
    join.add_child(left)
    join.add_child(right)
    root = FAONode(op="logical_plan")
    root.add_child(join)

    layers = CodeGenerator._topo_layers(root)
    flat = [n.op for layer in layers for n in layer]
    assert flat.count("scan") == 1  # executed exactly once
    assert sorted(flat) == ["join", "left", "right", "scan"]
    # Both branches still come after the (single) scan, join last.
    assert layers[0] == [scan_a] or layers[0] == [scan_b]
    assert {n.op for n in layers[1]} == {"left", "right"}
    assert [n.op for n in layers[2]] == ["join"]


# ---------------------------------------------------------------------------
# _format_output_schema_block rendering
# ---------------------------------------------------------------------------


def test_output_schema_block_renders_string():
    block = _format_output_schema_block("answer: brand (object), score (float64)")
    assert isinstance(block, str)
    assert "Predicted output schema" in block
    assert "brand (object)" in block


def test_output_schema_block_renders_list():
    block = _format_output_schema_block(["col_a (int64)", "col_b (object)"])
    assert "col_a (int64)\ncol_b (object)" in block


def test_output_schema_block_rewrites_zero_rows_and_handles_empty():
    block = _format_output_schema_block("answer (0 rows)")
    assert "(0 rows)" not in block
    assert "schema only" in block
    assert _format_output_schema_block(None) == ""
    assert _format_output_schema_block("") == ""
    assert _format_output_schema_block([]) == ""
