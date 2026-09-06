# Running KathDB on SemBench

The experiments use [SemBench](https://github.com/SemBench/SemBench) at commit
`995738f` plus one patch: a KathDB runner, the **Additional** scenario (10 queries),
and small fixes to the original query texts / evaluator.

```bash
# 1. KathDB (this repo, any branch) in a Python >= 3.10 environment
pip install -e .

# 2. SemBench at the pinned commit, with the patch
git clone https://github.com/SemBench/SemBench.git && cd SemBench
git checkout 995738f
git apply ../KathDB/experiments/sembench-vldb27.patch
pip install -r requirements.txt

# 3. Keys: planner model + the model the generated code calls
export ANTHROPIC_API_KEY=...    # or AZURE_ANTHROPIC_ENDPOINT / AZURE_ANTHROPIC_API_KEY
export OPENAI_API_KEY=...       # KATHDB_AI_OP_MODEL defaults to openai/gpt-4o-mini
```

Run one query (data is downloaded and scaled on first use; add `--skip-setup` afterwards):

```bash
python3 src/run.py --systems kathdb --use-cases ecomm --queries 1 \
    --scale-factor 500 --model anthropic/claude-opus-5
```

Results land in `files/ecomm/raw_results/kathdb/Q1.csv`, metrics (time, tokens, USD,
plan-generation vs execution split) in `files/ecomm/metrics/kathdb.json`, quality in
`files/ecomm/metrics/kathdb.json` after the evaluator runs.

Scenarios: `movie` (Q1–10), `ecomm` (Q1–14), `mmqa` (Q1–11), `additional` (Q1–10,
fixed scale factors: e-com 1000 / MMQA 400, so `--scale-factor` is ignored). A few
original queries are not supported by the SemBench harness and are excluded from the
paper's tables.

KathDB settings are environment variables read by the runner
(`src/runner/generic_kathdb_runner/generic_kathdb_runner.py`):
`KATHDB_AI_OP_MODEL`, `KATHDB_LOGICAL_REWRITE`, `KATHDB_PHY_OPT`,
`KATHDB_PREBUILT_FUNCTIONS`, `KATHDB_GENERATED_FUNCTIONS`,
`KATHDB_NUM_EXECUTOR_WORKERS`, `KATHDB_WORKER_ENV`, `KATHDB_FN_DIR`.
For example, the atomic plan without the optimizer:

```bash
KATHDB_LOGICAL_REWRITE=false python3 src/run.py --systems kathdb --use-cases additional \
    --queries 5 --model anthropic/claude-opus-5
```

## Additional scenario

`files/additional/queries/q{1..10}.toml` hold the natural-language queries.
Q1–Q4 are E-Commerce queries at scale factor 1000; their ground truth is the SQL in
the same file, run over `files/ecomm/data/sf_1000`. Q5–Q10 are MMQA image queries at
scale factor 400 over albums (`files/additional/data/mmqa_albums_sf400.csv`: image →
album, plus the frozen per-image labels `subject` / `has_people`); their ground truth is
the rule named in each file applied to those labels (`python3 files/additional/make_gt.py`).
All ten are scored as set-F1 over the `id` column.
