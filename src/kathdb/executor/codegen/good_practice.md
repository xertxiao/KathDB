# Codegen good practices

Stable rules for any function emitted by KathDB codegen. Apply unless an explicit instruction overrides them.

## Code structure
- Single top-level function; no nested functions. All imports inside the function.
- No try/except ANYWHERE — let bugs surface for diagnosis — EXCEPT a narrow one wrapping each per-record `call_model(...)` call (and nothing else). There, catch the error so a single row the model refuses or that errors (content-safety / ContentPolicyViolation block, unreadable image, transient provider error) yields a sentinel (`None` / `"unknown"` / `""`, per the column's downstream meaning) and the loop continues. One bad row must never abort the whole operation.
- Error-fallback sentinels (the fallback rows above — rows where the model call itself failed) must never participate in aggregate predicates — uniformity, majority, same-kind, counts-per-class. Exclude those rows from the check: a group whose items are all *unclassifiable* is NOT "all the same kind". This applies ONLY to error fallbacks: a genuine semantic answer of `other`/`unclear` — where the model looked and judged the record to fall outside the named classes — is a real classification and MUST participate like any other class (e.g. it breaks "all the same kind"). Keep the error-fallback sentinel value distinct from any semantic catch-all class so the two are separable.
- For relational work (filter, join, group-by, aggregate, project), use pandas on the input DataFrames; never use duckdb/SQL.
- A memoization cache stores exactly the value the function returns, with the same type on the cached and the computed path (e.g., do not cache the string 'no' and return a boolean elsewhere — a non-empty string is truthy).
- A fused operator returns exactly the rows its member operators would return: nothing added, nothing kept that they would drop (e.g., an unmatched record is dropped, not emitted with null fields).

## Schemas & columns
- Column names are case-sensitive. Use exact casing from input data (e.g., `customerId` not `customerid`); input casing wins over the predicted output schema.
- Use only columns shown in `<Inputs>`. Do not reference columns from source tables not listed there.
- Touch only columns needed for the node; avoid unnecessary reshaping.
- Preserve id columns on transform/filter; include keys + new columns on aggregate/join.
- Deduplicate on the key you iterate over, pair, or count, since a record can occur more than once in the input; a self-join never pairs a record with itself and emits each unordered pair once (e.g., (a, b) but not also (b, a)) unless the query asks otherwise.

## LLM/VLM prompts
- Keep prompts succinct and direct — one instruction, no preamble or politeness.
- LLM/VLM outputs may drift in casing/whitespace for the same entity; you may strip whitespace and lowercase.
- Values that will be compared for equality (a join, a group-by, a distinct count, a pairing) must be canonical strings: extract freely, then canonicalize the DISTINCT extracted values before comparing — map them onto the query's categories or a stored column's values when such a list exists, otherwise ask the model once to group the distinct values into labels (one call over the value list, never one per record or per pair). Case/whitespace/plural normalization is fine; never compare raw phrases and never use token overlap or substring tests as the judgement. Keep the label set coarse (e.g., 'NYC' and 'New York City' must map to one label).
- Temperature is set by the per-run model-constraint block; do not hardcode a different one.
- When you ask the model per item instead of per record (splitting a list or text so you can stop early), ask the same question with the same context the record gave and never strip or shorten a value first (e.g., 'Springfield (IL)' must not become 'Springfield'); a cheaper call that changes the question is not equivalent.

## From observed data
- You MAY add optimizations or reconciliation in the code (e.g. normalize/canonicalize values, an alias map, prune, dedup) when you believe it helps answer the query — BUT ONLY IF it is grounded in the actual input data this node observes. A sample of the data is fine to ground on; if you rely on one, keep the model call as the fallback for values beyond the sample — a shortcut may decide the rows it covers, never drop or mislabel the rows it does not.
- NEVER bake in such logic (an alias/synonym/canonicalization map, a fixed candidate vocabulary, a closed/clean value-set assumption) without having seen the actual, immediately-relevant input data it depends on.
