# KathDB — VLDB'27 submission

Reproduce the experiments with [SemBench](https://github.com/SemBench/SemBench) at commit
`995738f` plus the patch in `experiments/` (KathDB runner, the *Additional* scenario, and
small fixes to the original query texts and evaluator).

```bash
# KathDB
git clone -b vldb27-submission git@github.com:xertxiao/KathDB.git
pip install -e KathDB

# SemBench at the pinned commit, with the patch
git clone https://github.com/SemBench/SemBench.git
cd SemBench
git checkout 995738f
git apply ../KathDB/experiments/sembench-vldb27.patch
pip install -r requirements.txt

# API keys: the planner, and the model the generated code calls
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

Run E-Commerce Q1 (data is downloaded and scaled on first use):

```bash
python3 src/run.py --systems kathdb --use-cases ecomm --queries 1 \
    --scale-factor 500 --model anthropic/claude-opus-5
```

Result: `files/ecomm/raw_results/kathdb/Q1.csv`; time, tokens, cost and quality:
`files/ecomm/metrics/kathdb.json`.

Use cases: `movie`, `ecomm`, `mmqa`, `additional` (Q1–Q10; `--scale-factor` is ignored,
the scenario fixes e-com 1000 / MMQA 400).
