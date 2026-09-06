"""Token/USD accounting across the KathDB stages.

Parent-process LangChain calls are captured via ``UsageMetadataCallbackHandler``
(tokens only); the worker logs one JSONL record per generated-code model call to
``KATHDB_INFERENCE_LOG_PATH`` (tokens + ``cost_usd``). USD for parent stages is
derived from the same litellm pricing table. The buckets in :data:`STAGES` are
disjoint and sum to the total; ``plan_gen`` excludes ``grouping``.
"""

from __future__ import annotations

import time

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping

# Disjoint accounting buckets in pipeline order; they sum to the total.
STAGES: tuple[str, ...] = ("parser", "plan_gen", "grouping", "codegen", "execution")


def price_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD for the given tokens of ``model`` via ``litellm.cost_per_token``;
    ``None`` (not 0.0) when litellm is unavailable or does not know the model."""
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=int(input_tokens),
            completion_tokens=int(output_tokens),
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return None


@dataclass
class StageCost:
    """Running token/cost totals for one stage. ``money_cost_known`` turns False once
    any call could not be priced (USD is then a lower bound; tokens stay exact).
    ``calls`` is 0 for LangChain stages, which report no per-call count.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    money_cost_usd: float = 0.0
    money_cost_known: bool = True
    wall_time_sec: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _bump(
        self,
        model: str,
        d_in: int,
        d_out: int,
        d_calls: int,
        d_cost: float | None,
    ) -> None:
        self.input_tokens = max(0, self.input_tokens + d_in)
        self.output_tokens = max(0, self.output_tokens + d_out)
        self.calls = max(0, self.calls + d_calls)
        if d_cost is None:
            self.money_cost_known = False
        else:
            self.money_cost_usd = max(0.0, self.money_cost_usd + d_cost)
        m = self.by_model.setdefault(
            model,
            {"input_tokens": 0, "output_tokens": 0, "calls": 0, "money_cost_usd": 0.0},
        )
        m["input_tokens"] = max(0, int(m["input_tokens"]) + d_in)
        m["output_tokens"] = max(0, int(m["output_tokens"]) + d_out)
        m["calls"] = max(0, int(m["calls"]) + d_calls)
        if d_cost is not None:
            m["money_cost_usd"] = max(0.0, float(m["money_cost_usd"]) + d_cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "money_cost_usd": round(self.money_cost_usd, 6),
            "money_cost_known": self.money_cost_known,
            "wall_time_sec": round(self.wall_time_sec, 3),
            "by_model": {
                k: {**v, "money_cost_usd": round(float(v["money_cost_usd"]), 6)}
                for k, v in self.by_model.items()
            },
        }


@dataclass
class CostTracker:
    """Per-stage token/USD totals; see module docstring."""

    stages: dict[str, StageCost] = field(
        default_factory=lambda: {s: StageCost() for s in STAGES}
    )

    def record_usage_metadata(
        self,
        stage: str,
        usage_metadata: Mapping[str, Mapping[str, Any]] | None,
        *,
        sign: int = 1,
        price: bool = True,
    ) -> None:
        """Add (``sign=1``) or subtract (``sign=-1``) a ``UsageMetadataCallbackHandler``
        ``usage_metadata`` dict ``{model: {input_tokens, output_tokens}}`` into ``stage``."""
        if not usage_metadata:
            return
        bucket = self._bucket(stage)
        for model, um in usage_metadata.items():
            d_in = int(um.get("input_tokens", 0) or 0)
            d_out = int(um.get("output_tokens", 0) or 0)
            if d_in == 0 and d_out == 0:
                continue
            cost = price_usd(model, d_in, d_out) if price else None
            bucket._bump(
                model,
                sign * d_in,
                sign * d_out,
                0,
                None if cost is None else sign * cost,
            )

    def record_inference_totals(
        self,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        money_cost_usd: float,
        *,
        calls: int = 0,
        model: str = "_worker",
    ) -> None:
        """Add pre-summed worker-log totals (already priced by the worker) into ``stage``."""
        if input_tokens == 0 and output_tokens == 0 and not money_cost_usd:
            return
        self._bucket(stage)._bump(
            model, int(input_tokens), int(output_tokens), int(calls), float(money_cost_usd)
        )

    def add_wall_time(self, stage: str, seconds: float) -> None:
        self._bucket(stage).wall_time_sec += max(0.0, float(seconds))

    def totals(self) -> StageCost:
        """Grand total across all stages."""
        return self._sum(STAGES)

    def _sum(self, names: tuple[str, ...]) -> StageCost:
        agg = StageCost()
        for name in names:
            st = self.stages[name]
            agg._bump(
                "_total",
                st.input_tokens,
                st.output_tokens,
                st.calls,
                st.money_cost_usd if st.money_cost_known else None,
            )
            agg.wall_time_sec += st.wall_time_sec
        return agg

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view: per-stage breakdown + grand total."""
        return {
            "stages": {name: self.stages[name].to_dict() for name in STAGES},
            "total": self.totals().to_dict(),
        }

    def summary(self) -> str:
        """Compact one-line-per-stage human summary."""
        lines = []
        for name in (*STAGES, "_total"):
            st = self.totals() if name == "_total" else self.stages[name]
            flag = "" if st.money_cost_known else "  (USD lower-bound)"
            lines.append(
                f"  {name:<10} in={st.input_tokens:>8}  out={st.output_tokens:>8}  "
                f"calls={st.calls:>4}  ${st.money_cost_usd:.6f}  {st.wall_time_sec:>7.1f}s{flag}"
            )
        return "CostTracker:\n" + "\n".join(lines)

    def _bucket(self, stage: str) -> StageCost:
        if stage not in self.stages:
            raise KeyError(f"unknown cost stage {stage!r}; expected one of {STAGES}")
        return self.stages[stage]


@contextmanager
def capture_into(tracker: "CostTracker", stage: str, *, sign: int = 1):
    """Record every LangChain LLM call made inside the ``with`` block into
    ``tracker[stage]`` (contextvar-based, so calls need not forward ``config``)."""
    from langchain_core.callbacks import get_usage_metadata_callback

    t0 = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        yield cb
    tracker.record_usage_metadata(stage, cb.usage_metadata, sign=sign)
    tracker.add_wall_time(stage, time.perf_counter() - t0)


def new_stage_handler():
    """Fresh ``UsageMetadataCallbackHandler``; one instance per stage is required."""
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    return UsageMetadataCallbackHandler()


def attach_handler(config: dict | None, handler: Any) -> dict:
    """``RunnableConfig`` copy with ``handler`` appended to ``callbacks``."""
    cfg = dict(config) if config else {}
    existing = list(cfg.get("callbacks") or [])
    cfg["callbacks"] = [*existing, handler]
    return cfg


def read_inference_log_file(
    path: str | None, *, reset: bool = True
) -> tuple[int, int, float]:
    """Sum ``(input_tokens, output_tokens, cost_usd)`` of a worker inference log;
    with ``reset`` the file is truncated so successive calls partition it by stage."""
    if not path or not os.path.exists(path):
        return (0, 0, 0.0)
    ti = to = 0
    cost = 0.0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ti += int(rec.get("prompt_tokens", 0) or 0)
                to += int(rec.get("completion_tokens", 0) or 0)
                cost += float(rec.get("cost_usd", 0.0) or 0.0)
    except Exception:
        return (ti, to, cost)
    if reset:
        try:
            open(path, "w").close()
        except Exception:
            pass
    return (ti, to, cost)
