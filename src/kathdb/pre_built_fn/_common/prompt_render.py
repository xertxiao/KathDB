"""Prompt-template parsing and per-row rendering helpers.

These take a template like ``"Is {title} about {topic}?"`` plus a pandas row
and a modality map, and produce ``(prompt_text, image_uris, audio_payloads)``
ready to hand to ``litellm_sync.call_text``.
"""

from __future__ import annotations

import string
from typing import Any

import pandas as pd

from .media import resolve_audio_data, resolve_image_uri
from .modality import lookup_modality

_FORMATTER = string.Formatter()
_MAX_STR_DEFAULT = 500


def parse_prompt_columns(prompt: str) -> list[str]:
    """Column names from ``{col_name}`` placeholders in *prompt* (first-seen order, deduped)."""
    seen: set[str] = set()
    cols: list[str] = []
    for _, field_name, _, _ in _FORMATTER.parse(prompt):
        if field_name is not None and field_name != "" and field_name not in seen:
            seen.add(field_name)
            cols.append(field_name)
    return cols


def parse_join_prompt_columns(prompt: str) -> dict[str, list[str]]:
    """``{df_name.col}`` refs from a join prompt -> ``{df_name: [col, ...]}`` (first-seen order)."""
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for _, field_name, _, _ in _FORMATTER.parse(prompt):
        if field_name is None or field_name == "" or field_name in seen:
            continue
        seen.add(field_name)
        dot = field_name.find(".")
        if dot < 1:
            raise ValueError(
                f"sem_join prompt placeholders must use '{{df_name.col}}' "
                f"format, got '{{{field_name}}}'"
            )
        df_name = field_name[:dot]
        col_name = field_name[dot + 1 :]
        result.setdefault(df_name, []).append(col_name)
    return result


def truncate(value: Any, max_len: int = _MAX_STR_DEFAULT) -> str:
    """Stringify *value* and truncate to *max_len* chars with an ellipsis."""
    s = "" if value is None else str(value)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def build_row_prompt_and_media(
    prompt_template: str,
    row: pd.Series,
    columns: list[str],
    modality_map: dict[str, str] | None,
) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Render one row into (prompt text, images, audios): text columns are inlined,
    media columns become the token ``[image]`` / ``[audio]`` and are collected separately."""
    substitutions: dict[str, str] = {}
    images: list[str] = []
    audios: list[tuple[str, str]] = []
    for col in columns:
        modality = lookup_modality(col, modality_map)
        if modality == "image":
            uri = resolve_image_uri(str(row[col]))
            if uri is not None:
                images.append(uri)
            substitutions[col] = "[image]"
        elif modality == "audio":
            audio_data = resolve_audio_data(str(row[col]))
            if audio_data is not None:
                audios.append(audio_data)
            substitutions[col] = "[audio]"
        else:
            substitutions[col] = truncate(row[col])
    return prompt_template.format(**substitutions), images, audios


def row_to_text_and_media(
    row: pd.Series,
    columns: list[str],
    modality_map: dict[str, str] | None,
) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Render a row as multi-line ``col: value`` text plus media
    (media columns become ``col: [image]`` / ``col: [audio]``)."""
    parts: list[str] = []
    images: list[str] = []
    audios: list[tuple[str, str]] = []
    for col in columns:
        modality = lookup_modality(col, modality_map)
        if modality == "image":
            uri = resolve_image_uri(str(row[col]))
            if uri is not None:
                images.append(uri)
            parts.append(f"{col}: [image]")
        elif modality == "audio":
            audio_data = resolve_audio_data(str(row[col]))
            if audio_data is not None:
                audios.append(audio_data)
            parts.append(f"{col}: [audio]")
        else:
            parts.append(f"{col}: {truncate(row[col])}")
    return "\n".join(parts), images, audios
