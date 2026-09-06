# KathDB

KathDB answers natural-language questions over multimodal data (tables, text,
images) by planning like a database and executing like an agent: an LLM
decomposes the question into atomic operators, an optimizer fuses operators so
the generated code can push filters ahead of model calls and stop early, and the
generated Python runs in a sandboxed worker.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=...     # planner (default: anthropic/claude-opus-5)
export OPENAI_API_KEY=...        # the model the generated code calls (default: openai/gpt-4o-mini)
```

The generated code runs in a separate conda environment ("worker"). By default
KathDB provisions one from `src/kathdb/worker/requirements.txt` (its name is
`db.worker_env_name`); pass `worker_env="<name>"` to reuse an existing environment.

## Quick start

```python
import pandas as pd
from kathdb import KathDB
from kathdb.common.view_schema import Modality

with KathDB("catalog.duckdb", human_in_the_loop=False) as db:
    df = pd.read_csv("products.csv")      # has an image_path column
    db.register_table(df, "products", column_modalities={"image_path": Modality.IMAGE})
    relations = db.query("Which products priced under 50 show a logo in their product image?")
    print(db.last_result(relations))      # the answer DataFrame (relations = every relation the plan produced)
    print(db.last_cost().summary())       # tokens / USD / seconds per stage
    print(db.last_grouping_trace())       # what the optimizer fused
```

## Configuration

**Basic settings** — keyword arguments of `KathDB(...)` (also fields of
`KathDBConfig`):

| setting | default | what it does |
|---|---|---|
| `planner_model` | `anthropic/claude-opus-5` | Model KathDB reasons with (parse, plan, optimize, generate code). `provider/model`; providers: `openai`, `anthropic`, `google`, `azure_anthropic`. |
| `ai_op_model` | `openai/gpt-4o-mini` | LiteLLM model id the *generated code* calls for each record's semantic operation. |
| `human_in_the_loop` | `False` | Ask clarification questions, review the plan, confirm saves/persistence. |
| `logical_rewrite` | `True` | Run the grouping optimizer. `False` executes the atomic plan as-is. |
| `phy_opt` | `False` | `True` lets generated code batch, parallelize, cascade and cache model calls; `False` pins one model call per item. |
| `prebuilt_functions` | `True` | Use hand-written functions from `pre_built_fn/` (ships empty; add your own). |
| `generated_functions` | `True` | Reuse functions saved from prior queries and save new reusable ones. |
| `max_generated_functions` | `10` | Library size before least-used saved functions are evicted. |
| `worker_env` | `None` | Existing conda env for the worker; `None` provisions one. |
| `num_executor_workers` | `1` | Worker processes: how many independent plan operators execute at the same time (an operator starts as soon as its inputs are ready). |

**Advanced settings** — edit the defaults in `src/kathdb/config.py` (or pass a
`KathDBConfig`): `parser_type` (`action` / `action_with_functions` /
`action_with_functions_with_coarsening`), `grouping_rank_k=10`,
`grouping_max_group_size=None`, `grouping_base_plan_profiling=True`,
`grouping_sample_rows=10`, `image_quality_low_ai_op=True`, `codegen_concurrency=1`,
per-stage model overrides, temperatures, retry budgets, worker timeouts,
`generated_fn_dir`, `runtime_dir`, `log_level`. Everything can be changed live with
`db.configure(**overrides)`.

## Layout

```
src/kathdb/
  kathdb.py            KathDB facade: register tables, query, configure
  config.py            KathDBConfig (basic / advanced settings) + LLM factory
  parser/              NL question -> atomic action sketch (+ optional human review,
                       + library-steered sketch)
  plan_gen/            PlanGenerator: DAG build, function threading, demand
                       propagation (demand_propagation.py), grouping
    optimizer/         enumerate convex partitions of the plan, LLM-rank them
  executor/            Executor: dependency-driven scheduling, sandboxed execution
                       with diagnose-and-regenerate, function saves; persistence.py
    codegen/           prompts, per-operator code generator, plan-time base-plan
                       codegen/profiling
  worker/              conda-isolated subprocess that runs generated code
  common/              catalog (DuckDB), function library, cost tracking, utils,
                       fn_helpers/ (shared helpers for hand-written operators)
  fn/                  NOT a store: the import resolver behind `from kathdb.fn
                       import <name>` (looks in pre_built_fn/, then generated_fn/)
  pre_built_fn/        your hand-written operators (see its README; ships empty)
  generated_fn/        functions saved from prior queries (relocatable via
                       generated_fn_dir / KATHDB_GENERATED_FN_DIR)
```

## Tests

```bash
pip install pytest
pytest src
```
