"""Shared core for semantic extraction operators: extract a value from one column
into a new column via an LLM/VLM; with a fixed ``vocab`` the answer is snapped to the
closest label (case-insensitive), else ``"other"``."""

from __future__ import annotations

import pandas as pd

from kathdb.common.model_call import call_model, run_batch

from .prompt_render import build_row_prompt_and_media


def _esc(s: object) -> str:
    """Escape literal braces so str.format() leaves them intact (only {in_col} is live)."""
    return str(s).replace("{", "{{").replace("}", "}}")


def snap_to_vocab(value: str | None, vocab: list[str]) -> str:
    """Map a raw answer to a canonical vocab term (case-insensitive), else 'other'."""
    v = (value or "").strip()
    low = v.lower()
    canon = {t.lower(): t for t in vocab}
    if low in canon:
        return canon[low]
    # the model sometimes wraps the label ("Category: jeans" / quotes); match by contains
    for t in vocab:
        if t.lower() in low:
            return t
    return "other"


def extract_column(
    df: pd.DataFrame,
    in_col: str,
    out_col: str,
    extract: str,
    vocab: list[str] | None,
    *,
    modality: str | None,
    model: str,
    max_concurrency: int,
    image_detail: str,
    reasoning_effort: str,
) -> pd.DataFrame:
    """Per-row extraction from ``in_col`` into ``out_col``. ``modality`` is 'image' for
    image inputs, None for text. With ``vocab`` the output is snapped to vocab ∪ {'other'}."""
    out = df.copy()
    if df.empty:
        out[out_col] = []
        return out
    if in_col not in df.columns:
        raise KeyError(f"Column '{in_col}' not found in df")

    if vocab:
        allowed = ", ".join(f'"{_esc(t)}"' for t in vocab)
        instr = (
            f"Answer with EXACTLY one of these values: {allowed}. "
            f'If none of them apply, answer "other". '
            f"Output only the chosen value, nothing else."
        )
    else:
        instr = "Output only the extracted value as a single short phrase, nothing else."
    template = f"Extract {_esc(extract)} from the following:\n{{{in_col}}}\n\n{instr}"

    columns = [in_col]
    modality_map = {in_col: modality} if modality else None
    prompts: list[str] = []
    image_lists: list[list[str]] = []
    audio_lists: list[list[tuple[str, str]]] = []
    for _, row in df.iterrows():
        text, imgs, auds = build_row_prompt_and_media(template, row, columns, modality_map)
        prompts.append(text)
        image_lists.append(imgs)
        audio_lists.append(auds)

    fns = [
        (
            lambda p=p, im=im, au=au: call_model(
                p, model, im + au, image_detail=image_detail, reasoning_effort=reasoning_effort
            )
        )
        for p, im, au in zip(prompts, image_lists, audio_lists)
    ]
    raw = run_batch(fns, max_concurrency)
    out[out_col] = (
        [snap_to_vocab(r, vocab) for r in raw] if vocab else [(r or "").strip() for r in raw]
    )
    return out
