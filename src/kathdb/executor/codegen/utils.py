from __future__ import annotations

import json
from pandas import DataFrame

from typing import (
    Any,
    Mapping,
    Sequence,
    List,
    Dict,
)


from ...common.utils import sample_sequence
from ...common.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "normalize_quotes",
    "relation_sample_texts",
    "map_input_relation_objects",
    "convert_relation_to_records",
    "sanitize_records",
    "json_default",
    "truncate_io_text",
    "format_df_as_bordered_table",
]


# Unicode quote characters that should be normalized to ASCII equivalents
_QUOTE_NORMALIZE_MAP = {
    # Single quote variants -> '
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2039": "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203a": "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    # Do NOT map U+0060 (backtick): it has no quoting meaning in Python, and
    # rewriting it corrupts string literals that contain markdown fences.
    "\u00b4": "'",  # ACUTE ACCENT
    # Double quote variants -> "
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u00ab": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
}

_QUOTE_TRANS_TABLE = str.maketrans(_QUOTE_NORMALIZE_MAP)


def normalize_quotes(code: str) -> str:
    """Normalize Unicode quote characters (which break ``ast.parse``) to ASCII quotes."""
    return code.translate(_QUOTE_TRANS_TABLE)


def relation_sample_texts(
    names: Sequence[str],
    relations: Sequence[Any],
    *,
    sample_rows: int = 3,
    char_limit: int = 60,
) -> Dict[str, str]:

    def truncate_value(v: Any, limit: int = 200) -> Any:
        if isinstance(v, Mapping):
            return {k: truncate_value(val, limit) for k, val in v.items()}
        if isinstance(v, str):
            if limit <= 3:
                return v[:limit]
            return v if len(v) <= limit else (v[: limit - 3] + "...")
        if isinstance(v, (bytes, bytearray)):
            text = v.decode("utf-8", errors="replace")
            return truncate_value(text, limit)
        if isinstance(v, Sequence):
            if len(v) == 0:
                return {"length": 0, "sample_item": None}
            sample_item = truncate_value(v[0], limit)
            return {
                "sample_item": sample_item,
                "length": len(v),
                "note": "only first item shown",
            }
        return v

    samples: Dict[str, str] = {}
    for idx, (name, relation) in enumerate(zip(names, relations)):
        assert name is not None, f"Name at index {idx} is None."
        records = convert_relation_to_records(relation)
        subset = sample_sequence(records, sample_rows)
        subset = [truncate_value(row, char_limit) for row in subset]
        sample_text = (
            json.dumps(subset, indent=None, ensure_ascii=False) if subset else "(no rows)"
        )
        samples[name] = sample_text
    return samples


def map_input_relation_objects(
    names: Sequence[str], relations: Sequence[Any]
) -> Dict[str, Any]:
    """Map table names to the raw relation objects for execution contexts."""
    mapping: Dict[str, Any] = {}
    for idx, (name, relation) in enumerate(zip(names, relations)):
        mapping[name or f"relation_{idx}"] = relation
    return mapping


def convert_relation_to_records(relation: Any) -> List[Dict[str, Any]]:
    if relation is None:
        return []

    records: List[Any]

    if isinstance(relation, DataFrame):
        records = relation.reset_index(drop=True).to_dict(orient="records")
    elif isinstance(relation, Mapping):
        records = [dict(relation)]
    elif isinstance(relation, Sequence) and not isinstance(
        relation, (str, bytes, bytearray)
    ):
        if not relation:
            return []
        first = relation[0]
        if isinstance(first, Mapping):
            records = [dict(item) for item in relation]  # type: ignore[arg-type]
        else:
            records = [{"value": item} for item in relation]
    else:
        records = [{"value": relation}]

    return sanitize_records(records)


def sanitize_records(records: Sequence[Any]) -> List[Dict[str, Any]]:
    serialized = json.dumps(records, default=json_default, ensure_ascii=False)
    result = json.loads(serialized)

    if isinstance(result, list):
        sanitized: List[Dict[str, Any]] = []
        for item in result:
            if isinstance(item, Mapping):
                sanitized.append(dict(item))
            else:
                sanitized.append({"value": item})
        return sanitized
    if isinstance(result, Mapping):
        return [dict(result)]
    return [{"value": result}]


def json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def truncate_io_text(text: str | None, *, max_length: int = 2000) -> str:
    """
    Return a truncated string preserving both the beginning and ending segments.
    """
    if not text or max_length <= 0:
        return text or ""
    if len(text) <= max_length:
        return text

    head_len = max(1, max_length // 2)
    tail_len = max(1, max_length - head_len)
    if head_len + tail_len > len(text):
        head_len = len(text) // 2 or 1
        tail_len = len(text) - head_len
    omitted = len(text) - (head_len + tail_len)
    head = text[:head_len]
    tail = text[-tail_len:]
    marker = f"\n... <truncated {omitted} chars> ...\n"
    return f"{head}{marker}{tail}"


def format_df_as_bordered_table(df: DataFrame) -> str:
    """Format a DataFrame as a bordered ASCII table for readable review output."""
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    rows = [[str(df.iloc[i][c]) for c in cols] for i in range(len(df))]
    str_cols = [str(c) for c in cols]
    widths = [
        max(
            len(str_cols[j]),
            max((len(rows[i][j]) for i in range(len(rows))), default=0),
        )
        for j in range(len(cols))
    ]
    top = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_cell(values: list[str]) -> str:
        return (
            "|"
            + "|".join(f" {values[j].ljust(widths[j])} " for j in range(len(cols)))
            + "|"
        )

    lines = [top, row_cell(str_cols), sep]
    for r in rows:
        lines.append(row_cell(r))
    lines.append(top)
    return "\n".join(lines)
