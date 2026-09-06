"""Column-modality lookup for the pre_built_fn semantic operators."""

from __future__ import annotations

SUPPORTED_MODALITIES: set[str] = {"text", "image", "audio"}


def lookup_modality(
    col_name: str,
    modality_map: dict[str, str] | None,
    table_name: str | None = None,
) -> str | None:
    """Return the modality for *col_name*, or None if not registered.

    With *table_name*, ``"table_name.col_name"`` is tried first, then the bare name.
    """
    if modality_map is None:
        return None
    if table_name:
        qualified = modality_map.get(f"{table_name}.{col_name}")
        if qualified is not None:
            if qualified not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported modality '{qualified}' for column "
                    f"'{table_name}.{col_name}'. Currently only "
                    f"{SUPPORTED_MODALITIES} are supported."
                )
            return qualified
    bare = modality_map.get(col_name)
    if bare is not None:
        if bare not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{bare}' for column '{col_name}'. "
                f"Currently only {SUPPORTED_MODALITIES} are supported."
            )
        return bare
    return None
