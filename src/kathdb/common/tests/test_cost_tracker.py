"""CostTracker: pricing, disjoint stage buckets, worker-log ingestion."""

from __future__ import annotations

import json

from kathdb.common.cost_tracker import (
    STAGES,
    CostTracker,
    attach_handler,
    price_usd,
    read_inference_log_file,
)

_GPT = "gpt-4o-mini"


def test_price_usd_known_vs_unknown():
    c1 = price_usd(_GPT, 1000, 0)
    c2 = price_usd(_GPT, 2000, 0)
    assert c1 is not None and c2 is not None
    assert c2 == c1 * 2
    assert price_usd("totally-made-up-model-xyz", 1000, 1000) is None


def test_record_usage_metadata_prices_and_splits():
    t = CostTracker()
    total = {_GPT: {"input_tokens": 1000, "output_tokens": 500}}
    grouping = {_GPT: {"input_tokens": 400, "output_tokens": 100}}
    t.record_usage_metadata("plan_gen", total)
    t.record_usage_metadata("plan_gen", grouping, sign=-1)
    t.record_usage_metadata("grouping", grouping)

    pg = t.stages["plan_gen"]
    gp = t.stages["grouping"]
    assert pg.input_tokens == 600 and pg.output_tokens == 400
    assert gp.input_tokens == 400 and gp.output_tokens == 100
    assert abs((pg.money_cost_usd + gp.money_cost_usd) - price_usd(_GPT, 1000, 500)) < 1e-9
    assert pg.money_cost_known and gp.money_cost_known


def test_unknown_model_flags_lower_bound_but_keeps_tokens():
    t = CostTracker()
    t.record_usage_metadata("codegen", {"mystery-model": {"input_tokens": 10, "output_tokens": 5}})
    cg = t.stages["codegen"]
    assert cg.input_tokens == 10 and cg.output_tokens == 5
    assert cg.money_cost_known is False


def test_inference_totals_and_grand_total():
    t = CostTracker()
    t.record_inference_totals("execution", 3000, 1500, 0.0123, calls=7)
    ex = t.stages["execution"]
    assert ex.input_tokens == 3000 and ex.output_tokens == 1500
    assert ex.calls == 7 and abs(ex.money_cost_usd - 0.0123) < 1e-9

    t.record_inference_totals("grouping", 150, 30, 0.0015, calls=2)
    gp = t.stages["grouping"]
    assert gp.input_tokens == 150 and gp.output_tokens == 30 and gp.calls == 2

    tot = t.totals()
    assert tot.input_tokens == 3150 and tot.output_tokens == 1530
    d = t.to_dict()
    assert set(d["stages"]) == set(STAGES)
    assert d["total"]["input_tokens"] == 3150


def test_attach_handler_preserves_existing_callbacks():
    sentinel = object()
    cfg = attach_handler({"callbacks": [sentinel], "tags": ["x"]}, "H")
    assert cfg["callbacks"] == [sentinel, "H"]
    assert cfg["tags"] == ["x"]
    assert attach_handler(None, "H")["callbacks"] == ["H"]


def test_read_inference_log_file_sums_and_resets(tmp_path):
    p = tmp_path / "inf.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"prompt_tokens": 100, "completion_tokens": 40, "cost_usd": 0.002},
                {"prompt_tokens": 60, "completion_tokens": 10, "cost_usd": 0.001},
            ]
        )
        + "\n"
    )
    ti, to, cost = read_inference_log_file(str(p), reset=True)
    assert ti == 160 and to == 50 and abs(cost - 0.003) < 1e-9
    assert read_inference_log_file(str(p), reset=False) == (0, 0, 0.0)
    assert read_inference_log_file(None) == (0, 0, 0.0)
