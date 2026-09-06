"""Save-time smoke test for finalized generated functions.

The finalizer emits a ``smoke`` script defining ``canned_response(prompt) -> str``
(the fake model) and ``run(fn)`` (build rows, call ``fn``, assert). :func:`run_smoke`
runs it in a subprocess with every model endpoint replaced by a signature-faithful
fake; the save is rejected unless it passes and, for code that references a model
client, at least one fake call was made.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# kathdb and litellm are replaced by sys.modules stubs before the candidate module
# is imported; the fake call_model must mirror the real signature so an invented
# kwarg raises TypeError as it would at reuse time.
_HARNESS = r"""
import importlib.util, inspect, logging, sys, types

module_path, fn_name, smoke_path = sys.argv[1], sys.argv[2], sys.argv[3]
smoke_src = open(smoke_path).read()
ns = {}
exec(compile(smoke_src, smoke_path, "exec"), ns)
_raw_canned = ns.get("canned_response") or (lambda prompt: "")
run = ns.get("run")
if run is None:
    print("SMOKE FAIL: smoke script defines no run(fn)")
    sys.exit(2)

calls = {"client": 0, "canned": 0}

def _counted_canned(prompt):
    calls["canned"] += 1
    return _raw_canned(prompt)

# The smoke script's own run()/lambdas resolve `canned_response` through the
# script's globals (= ns) at call time; rebinding it to the counting wrapper
# means even a script that patches the fake client with its own
# canned_response-backed lambda still proves model calls happened.
ns["canned_response"] = _counted_canned

def call_model(
    prompt,
    model,
    media=None,
    *,
    modality=None,
    image_detail="low",
    reasoning_effort="minimal",
    temperature=0.0,
):
    calls["client"] += 1
    return str(_raw_canned(prompt))

_pkg = types.ModuleType("kathdb"); _pkg.__path__ = []
_common = types.ModuleType("kathdb.common"); _common.__path__ = []
_ls = types.ModuleType("kathdb.common.model_call")
_ls.call_model = call_model
_logmod = types.ModuleType("kathdb.common.logger")
_logmod.get_logger = logging.getLogger
_pkg.common = _common
_common.model_call = _ls
_common.logger = _logmod

class _Msg:
    def __init__(self, content):
        self.content = content
    def __getitem__(self, k):
        return getattr(self, k)

class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"
    def __getitem__(self, k):
        return getattr(self, k)

class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
    def __getitem__(self, k):
        return getattr(self, k)

def _fake_completion(*args, **kwargs):
    calls["client"] += 1
    prompt = ""
    for m in kwargs.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            prompt = c
    return _Resp(str(_raw_canned(prompt)))

_litellm = types.ModuleType("litellm")
_litellm.completion = _fake_completion

sys.modules.update({
    "kathdb": _pkg,
    "kathdb.common": _common,
    "kathdb.common.model_call": _ls,
    "kathdb.common.logger": _logmod,
    "litellm": _litellm,
})

spec = importlib.util.spec_from_file_location("fn_module", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, fn_name)

# Some finalizer-written smoke scripts insist on importing the fn's module as
# `scripts.fn` or patching "scripts.fn.call_model" (unittest.mock) instead of
# trusting the injected fn. Alias the loaded module under those names so that
# pattern resolves to the same already-faked module instead of failing.
_scripts_pkg = types.ModuleType("scripts")
_scripts_pkg.__path__ = []
_scripts_pkg.fn = mod
sys.modules.setdefault("scripts", _scripts_pkg)
sys.modules.setdefault("scripts.fn", mod)
sys.modules.setdefault("fn_module", mod)

run(fn)

code_src = open(module_path).read()
n_calls = calls["client"] or calls["canned"]
if ("call_model" in code_src or "litellm" in code_src) and n_calls == 0:
    print("SMOKE FAIL: body references a model client but made ZERO model "
          "calls (a swallowed per-row exception?)")
    sys.exit(3)
print(f"SMOKE OK: {n_calls} model calls")
"""

# Must track the real call_model signature (mirrored by the harness fake).
FAKE_CALL_MODEL_PARAMS = (
    ("prompt", None),
    ("model", None),
    ("media", None),
    ("modality", None),
    ("image_detail", "low"),
    ("reasoning_effort", "minimal"),
    ("temperature", 0.0),
)


def run_smoke(
    code: str,
    fn_name: str,
    smoke_src: str,
    timeout_s: float = 120.0,
) -> tuple[bool, str]:
    """Run the smoke script against ``code`` in a subprocess; returns ``(ok, detail)``, never raises."""
    if not (smoke_src or "").strip():
        return False, "finalizer produced no smoke script"
    try:
        with tempfile.TemporaryDirectory(prefix="kathdb_fn_smoke_") as td:
            tdir = Path(td)
            module_path = tdir / "fn_module.py"
            smoke_path = tdir / "smoke.py"
            harness_path = tdir / "harness.py"
            module_path.write_text(code, encoding="utf-8")
            smoke_path.write_text(smoke_src, encoding="utf-8")
            harness_path.write_text(_HARNESS, encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(harness_path),
                    str(module_path),
                    fn_name,
                    str(smoke_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        return False, f"smoke test timed out after {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"smoke harness could not run: {exc}"
    if proc.returncode == 0:
        return True, (proc.stdout or "").strip()
    tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return False, tail[-2000:]
