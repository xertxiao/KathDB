"""Finalizer for saved functions: one structured LLM call rewrites a query-time
function into a reusable ``scripts/fn.py`` (descriptive name, typed signature with
query-specific literals lifted into parameters, module-level ``CONTRACT``).
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)

__all__ = ["SaveFinalizerRecord", "finalize_with_llm"]


_KATHDB_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EXAMPLE_FN = _KATHDB_PKG_ROOT / "pre_built_fn" / "_example" / "sem_map"


@dataclass
class SaveFinalizerRecord:
    """Outcome of one function save (``fn_name`` = plan op, ``canonical_name`` = saved name)."""

    fn_name: str
    status: str  # "success" | "failed"
    wall_time_sec: float
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""
    canonical_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fn_name": self.fn_name,
            "canonical_name": self.canonical_name,
            "status": self.status,
            "wall_time_sec": round(self.wall_time_sec, 3),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **({"extra": self.extra} if self.extra else {}),
        }


def _read_text(path: Path, max_bytes: int = 200_000) -> str:
    try:
        data = path.read_text()
    except Exception as e:  # noqa: BLE001
        return f"<could not read {path}: {e}>"
    if len(data) > max_bytes:
        return data[:max_bytes] + f"\n... [truncated from {len(data)} bytes]"
    return data


_MODEL_DEFAULT_RE = re.compile(r"""model\s*:\s*str\s*=\s*["']([^"']+)["']""")


def model_default_violation(code_out: str, source_code: str) -> str | None:
    """Error string if a ``model`` default in ``code_out`` does not appear verbatim
    as a string literal in ``source_code``; else None."""
    for m in _MODEL_DEFAULT_RE.finditer(code_out):
        lit = m.group(1)
        # Match the quoted literal so a dropped provider prefix is not accepted.
        if f'"{lit}"' not in source_code and f"'{lit}'" not in source_code:
            return (
                f"model default {lit!r} does not appear as a string literal "
                "in the source code — copy the origin model string verbatim"
            )
    return None


def _iso_now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _build_llm_finalize_prompt(
    *, fn_name: str, code: str, example_fn_dir: Path, existing_names: list[str]
) -> str:
    example_script_fn = _read_text(example_fn_dir / "scripts" / "fn.py")
    existing_block = ", ".join(sorted(existing_names)) if existing_names else "(none)"
    return f"""You are refactoring KathDB-generated code into a REUSABLE function so \
future queries can reuse it. Return structured output only.

# Source (op `{fn_name}`)
```python
{code}
```

# What to produce
- `canonical_name`: snake_case, lowercase, valid Python identifier, 2-5 words, \
describing the reusable ACTION (not the query-specific target). If the body \
commits to a semantic choice a sibling query might want the other way \
(same-class vs cross-class, bounded top-k vs all rows), that constraint MUST \
appear in the name and the first words of `purpose`. Must NOT collide with any \
existing function: {existing_block}.
- Preserve the body's data-transformation logic (same result for equivalent \
inputs). You MAY rename arguments and restructure the signature. Lift \
query-specific literals (thresholds, keywords, prompt strings, category lists) \
into REQUIRED typed parameters — NO defaults on semantic parameters (only infra \
knobs like `model`/`temperature` keep defaults), so every future caller must \
consciously choose its own label set / cap / prompt / columns. Any cap/limit \
parameter must be `int | None` with `None` = unlimited. Annotate every param; \
use `pd.DataFrame` for DataFrame inputs. The CONTRACT `example` must use \
NEUTRAL placeholder values, never the origin query's literals.
- MODEL-CALL RULE: every model call in the body MUST go through \
`from kathdb.common.model_call import call_model`, whose EXACT signature is \
`call_model(prompt: str, model: str, media=None, *, modality: str | None = None, image_detail: str = "low", reasoning_effort: str = "minimal", temperature: float = 0.0) -> str`. \
`media` is one item or a list: image paths / URLs / data URIs, audio paths, or \
`(base64, format)` audio tuples (text goes in `prompt`); give the function a \
`modality` parameter when the media kind is the caller's choice. Call it with \
EXACTLY these parameters — do NOT invent keyword arguments, and do NOT call any \
other client. A hallucinated kwarg raises TypeError at reuse time, and a per-row \
`except` silently turns that into an all-empty result — so also keep `except` \
clauses NARROW around the `call_model` line only, never around whole loop bodies. The `model` parameter's default MUST be the exact \
model string that appears in the source code, copied VERBATIM (keep any \
provider prefix like `azure/`) — a "simplified" model name can route to a \
provider this deployment cannot reach.
- Add a module-level `CONTRACT` dict (the prose the signature can't carry). Keep \
each note to one terse line:
```python
CONTRACT = {{
    "purpose": "one-line summary of the reusable action",
    "params": {{"<name>": "one-line note", ...}},   # one per non-system param
    "sys_params": ["model", ...],                    # params that are infra knobs
    "output": "what it returns",
    "behavior": "tiny worked run: 3-5 concrete input rows -> the EXACT output rows",
    "example": "<one canonical call, as source>",
    "cost": "cost note (optional)",
    "use_when": "when to pick this op",
    "not_when": "when not to (optional)",
}}
```
`behavior` must be a concrete micro-trace exposing the fn's shape decisions (how \
items pair up — once each or every combination; what a cap does; which rows are \
excluded; how groups qualify), with neutral values — it is what a future code \
generator reads to decide whether this fn does ITS logic.
- `code`: the COMPLETE `scripts/fn.py` — imports + the `CONTRACT` dict + \
`def <canonical_name>(...)`. Do NOT include fn.md or spec.py (KathDB generates \
them from CONTRACT). No markdown fences inside `code`.
- `smoke`: a standalone smoke-test script, executed in a sandbox at save time; \
the function is REJECTED if it fails, so it must be self-contained and \
deterministic. It must define EXACTLY two top-level functions:
```python
def canned_response(prompt: str) -> str:
    ...  # the fake model: map each input row's prompt to the answer your
         # behavior trace assumes (e.g. return 'pos' if 'great' in prompt else 'neg')

def run(fn) -> None:
    import pandas as pd
    ...  # build the SAME 3-5 rows as `behavior`, call fn(...) with explicit
         # kwargs for EVERY semantic parameter, and assert the EXACT expected
         # output (row count AND key values), matching the `behavior` trace
```
`fn` arrives ALREADY imported and ALREADY wired to a fake model that answers \
via `canned_response` — do NOT import the function's module (no `import \
scripts.fn`; it does not exist at test time), do NOT use unittest.mock/patch, \
do NOT re-implement any interception, and do not import litellm. Just build \
rows, call `fn(...)`, assert. This is the `behavior` trace made executable: \
if `run` cannot assert the behavior trace's exact output, the behavior trace \
is wrong. Keep it under ~30 lines. No markdown fences inside `smoke`.

# Canonical example shape (sem_map/scripts/fn.py)
```python
{example_script_fn}
```
"""


def finalize_with_llm(
    *,
    fn_name: str,
    code: str,
    llm,
    example_fn_dir: Path | None = None,
    existing_names: list[str] | None = None,
    config=None,
) -> tuple[SaveFinalizerRecord, str, str]:
    """Return ``(record, canonical_name, code_out)``; on any error the record is
    failed and ``code`` is returned unchanged."""
    from pydantic import BaseModel, Field

    from .utils import invoke_structured_with_retry

    class _FinalizedFn(BaseModel):
        canonical_name: str = Field(description="snake_case reusable function name")
        code: str = Field(description="full scripts/fn.py: imports + CONTRACT + def")
        summary: str = Field(description="one sentence on the reusable action")
        smoke: str = Field(
            description=(
                "standalone smoke script: def canned_response(prompt)->str and "
                "def run(fn)->None asserting the behavior trace's exact output"
            )
        )

    example_fn_dir = example_fn_dir or _DEFAULT_EXAMPLE_FN
    existing = list(existing_names or [])
    started = _iso_now()

    def _fail(err: str, wall: float = 0.0):
        return (
            SaveFinalizerRecord(
                fn_name=fn_name,
                status="failed",
                wall_time_sec=wall,
                error=err,
                started_at=started,
                finished_at=_iso_now(),
                canonical_name=None,
            ),
            fn_name,
            code,
        )

    prompt = _build_llm_finalize_prompt(
        fn_name=fn_name,
        code=code,
        example_fn_dir=example_fn_dir,
        existing_names=existing,
    )
    t0 = time.time()
    try:
        out = invoke_structured_with_retry(
            prompt, llm=llm, schema=_FinalizedFn, config=config
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"LLM finalize call failed: {exc}", time.time() - t0)
    wall = time.time() - t0

    name = (out.canonical_name or "").strip()
    code_out = out.code or ""
    if not _IDENTIFIER_RE.match(name):
        return _fail(f"invalid canonical_name {name!r}", wall)
    if name in set(existing):
        return _fail(f"canonical_name {name!r} collides with existing fn", wall)
    if f"def {name}(" not in code_out:
        return _fail(f"code missing 'def {name}('", wall)
    if "CONTRACT" not in code_out:
        return _fail("code has no module-level CONTRACT dict", wall)
    smoke = (out.smoke or "").strip()
    if "def run(" not in smoke:
        return _fail("smoke script missing 'def run(fn)'", wall)
    model_err = model_default_violation(code_out, code)
    if model_err:
        return _fail(model_err, wall)

    return (
        SaveFinalizerRecord(
            fn_name=fn_name,
            status="success",
            wall_time_sec=wall,
            error=None,
            started_at=started,
            finished_at=_iso_now(),
            canonical_name=name,
            extra={"summary": out.summary, "smoke": smoke},
        ),
        name,
        code_out,
    )
