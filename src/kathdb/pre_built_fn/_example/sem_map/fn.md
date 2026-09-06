# sem_map

Add one new column by applying an LLM prompt to each row (transform / extract / classify).

## Signature
`sem_map(df: pd.DataFrame, prompt: str, out_col: str = 'sem_map', modality_map: dict[str, str] | None = None, model: str = 'gpt-4o-mini', max_concurrency: int = 20, image_detail: str = 'low', reasoning_effort: str = 'minimal') -> pd.DataFrame`

## Arguments
- `df` (pd.DataFrame, required): Input rows.
- `prompt` (str, required): Template using {col} placeholders (single braces; each col must exist in df); ask for a single-line plain-text answer. Do NOT call .format() yourself — substitution is per-row.
- `out_col` (str, default 'sem_map'): Name of the new column (use a descriptive name, e.g. 'sentiment', not the default).
- `modality_map` (dict[str, str] | None, default None): Map media columns to 'image'/'audio'; the column must also appear as a {placeholder}. Omit for text-only.
- `model` (str, system)
- `max_concurrency` (int, system)
- `image_detail` (str, system)
- `reasoning_effort` (str, system)

## Output
All input columns unchanged, plus one new text column `out_col` with the per-row LLM response.

## Examples
```python
from kathdb.fn import sem_map
result = sem_map(df, prompt="Summarize in one sentence: {text}", out_col="summary")
```

## Cost Warning
O(N) LLM calls, no early-exit. If downstream limits discard most rows, process only the needed rows.

## Selection Guidance
Use when: derive a new per-row column via LLM (summarize / classify / extract).
Do NOT use when: the value is computable with plain pandas/SQL.
