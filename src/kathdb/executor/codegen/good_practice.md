# Codegen good practices

Stable rules for any function emitted by KathDB codegen. Apply unless an explicit instruction overrides them.

## Code structure
- Single top-level function; no nested functions. All imports inside the function.
- No try/except ANYWHERE — let bugs surface for diagnosis — EXCEPT a narrow one wrapping each per-record `call_model(...)` call (and nothing else). There, catch the error so a single row the model refuses or that errors (content-safety / ContentPolicyViolation block, unreadable image, transient provider error) yields a sentinel (`None` / `"unknown"` / `""`, per the column's downstream meaning) and the loop continues. One bad row must never abort the whole operation.
- A row whose model call failed is unknown, not a category: leave it out of group checks (all the same kind, majority, counts per class). A real answer such as 'other' or 'unclear' is a normal category and stays in. Use a different value for the two.
- For relational work (filter, join, group-by, aggregate, project), use pandas on the input DataFrames; never use duckdb/SQL.
- If a row fails a filter or has no match, leave it out of the output. Do not keep it with blank columns.

## Schemas & columns
- Column names are case-sensitive. Use exact casing from input data (e.g., `customerId` not `customerid`); input casing wins over the predicted output schema.
- Use only columns shown under `## Inputs`. Do not reference columns from tables not listed there.
- Touch only columns needed for the node; avoid unnecessary reshaping.
- Preserve id columns on transform/filter; include keys + new columns on aggregate/join.
- Deduplicate on the key you loop over, pair, or count; the same record can appear twice in the input.
- A self-join never pairs a record with itself and emits each pair once ((a, b) but not also (b, a)) unless the query asks otherwise.

## LLM/VLM prompts
- Keep prompts succinct and direct — one instruction, no preamble or politeness.
- LLM/VLM outputs may drift in casing/whitespace for the same entity; you may strip whitespace and lowercase.
- Before you join, group, or count model-extracted values, map the distinct values so one entity gets one label (e.g., 'NYC' and 'New York City' become one). Do not compare raw phrases, and do not decide a match by substring or shared words.
- Temperature is set by the per-run model-constraint block; do not hardcode a different one.

## From observed data
- A shortcut built from values you saw (a keyword test, an alias map, a fixed list) may decide only the rows it covers; every other row still goes to the model.
- Do not assume a column's values are clean or complete unless the input shows it.
