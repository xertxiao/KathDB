# Codegen good practices

Stable rules for any function emitted by KathDB codegen. Apply unless an explicit instruction overrides them.

## Code structure
- Single top-level function; no nested functions. All imports inside the function.
- No try/except ANYWHERE — let bugs surface for diagnosis — EXCEPT a narrow one wrapping each per-record `call_model(...)` call (and nothing else). There, catch the error so a single row the model refuses or that errors (content-safety / ContentPolicyViolation block, unreadable image, transient provider error) yields a sentinel (`None` / `"unknown"` / `""`, per the column's downstream meaning) and the loop continues. One bad row must never abort the whole operation.
- Error-fallback sentinels (the fallback rows above — rows where the model call itself failed) must never participate in aggregate predicates — uniformity, majority, same-kind, counts-per-class. Exclude those rows from the check: a group whose items are all *unclassifiable* is NOT "all the same kind". This applies ONLY to error fallbacks: a genuine semantic answer of `other`/`unclear` — where the model looked and judged the record to fall outside the named classes — is a real classification and MUST participate like any other class (e.g. it breaks "all the same kind"). Keep the error-fallback sentinel value distinct from any semantic catch-all class so the two are separable.
- For relational work (filter, join, group-by, aggregate, project), use pandas on the input DataFrames; never use duckdb/SQL.

## Schemas & columns
- Column names are case-sensitive. Use exact casing from input data (e.g., `customerId` not `customerid`); input casing wins over the predicted output schema.
- Use only columns shown in `<Inputs>`. Do not reference columns from source tables not listed there.
- Touch only columns needed for the node; avoid unnecessary reshaping.
- Preserve id columns on transform/filter; include keys + new columns on aggregate/join.

## LLM/VLM prompts
- Keep prompts succinct and direct — one instruction, no preamble or politeness.
- LLM/VLM outputs may drift in casing/whitespace for the same entity; you may strip whitespace and lowercase.
- Temperature is set by the per-run model-constraint block; do not hardcode a different one.

## From observed data
- You MAY add optimizations or reconciliation in the code (e.g. normalize/canonicalize values, an alias map, prune, dedup) when you believe it helps answer the query — BUT ONLY IF it is grounded in the actual input data this node observes. A sample of the data is fine to ground on; if you rely on one, keep the model call as the fallback for values beyond the sample — a shortcut may decide the rows it covers, never drop or mislabel the rows it does not.
- NEVER bake in such logic (an alias/synonym/canonicalization map, a fixed candidate vocabulary, a closed/clean value-set assumption) without having seen the actual, immediately-relevant input data it depends on.
