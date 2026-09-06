"""Auto-discovery: a media folder a table already references is not registered twice."""

from __future__ import annotations

from kathdb.common.auto_discovery import AutoDiscovery
from kathdb.common.view_schema import Modality


class _Ctx:
    _llm = None

    def __init__(self):
        self.registered = {}

    def list_tables(self):
        return list(self.registered)

    def register_table(self, df, name, *, column_modalities=None, description=None):
        self.registered[name] = (df, column_modalities or {})


def _make(tmp_path, reference_images: bool):
    (tmp_path / "images").mkdir()
    for i in range(3):
        (tmp_path / "images" / f"{i}.jpg").write_bytes(b"\xff\xd8\xff")
    col = "image_path" if reference_images else "note"
    rows = "\n".join(f"{i},images/{i}.jpg" if reference_images else f"{i},n{i}" for i in range(3))
    (tmp_path / "products.csv").write_text(f"id,{col}\n{rows}\n")


def test_referenced_media_is_not_a_second_table(tmp_path):
    _make(tmp_path, reference_images=True)
    ctx = _Ctx()
    names = AutoDiscovery(ctx, llm_assist=False).discover(tmp_path)
    assert names == ["products"]
    assert ctx.registered["products"][1] == {"image_path": Modality.IMAGE}


def test_loose_media_becomes_a_media_table(tmp_path):
    _make(tmp_path, reference_images=False)
    ctx = _Ctx()
    names = AutoDiscovery(ctx, llm_assist=False).discover(tmp_path)
    assert "products" in names and len(names) == 2
    media = [n for n in names if n != "products"][0]
    assert Modality.IMAGE in ctx.registered[media][1].values()
