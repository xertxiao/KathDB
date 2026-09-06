"""Canonical multimodal view schemas (image, video, text) used by :class:`DBContext`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

__all__ = ["Modality", "ViewSource"]


class Modality(Enum):
    """Supported multimodal data types."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    AUDIO = "audio"


@dataclass(frozen=True)
class ViewSource:
    """Origin (table, column, modality) of an auto-expanded view table."""

    source_table: str
    source_column: str
    modality: Modality


# (view_suffix, columns, dtypes, summary, primary_key)
ViewDef = Tuple[str, Tuple[str, ...], Dict[str, str], str, Tuple[str, ...]]

_IMAGE_VIEWS: List[ViewDef] = [
    (
        "objects",
        ("iid", "oid", "label", "x1", "y1", "x2", "y2"),
        {
            "iid": "BIGINT",
            "oid": "BIGINT",
            "label": "VARCHAR",
            "x1": "DOUBLE",
            "y1": "DOUBLE",
            "x2": "DOUBLE",
            "y2": "DOUBLE",
        },
        "Detected objects in images with bounding boxes.",
        ("iid", "oid"),
    ),
    (
        "relationships",
        ("iid", "rid", "oid_i", "predicate", "oid_j"),
        {
            "iid": "BIGINT",
            "rid": "BIGINT",
            "oid_i": "BIGINT",
            "predicate": "VARCHAR",
            "oid_j": "BIGINT",
        },
        "Pairwise relationships between objects within an image.",
        ("iid", "rid"),
    ),
    (
        "attributes",
        ("iid", "oid", "k", "v"),
        {
            "iid": "BIGINT",
            "oid": "BIGINT",
            "k": "VARCHAR",
            "v": "VARCHAR",
        },
        "Key-value attributes of detected objects within an image.",
        ("iid", "oid"),
    ),
]

_VIDEO_VIEWS: List[ViewDef] = [
    (
        "objects",
        ("vid", "fid", "oid", "label", "x1", "y1", "x2", "y2"),
        {
            "vid": "BIGINT",
            "fid": "BIGINT",
            "oid": "BIGINT",
            "label": "VARCHAR",
            "x1": "DOUBLE",
            "y1": "DOUBLE",
            "x2": "DOUBLE",
            "y2": "DOUBLE",
        },
        "Detected objects in video frames with bounding boxes.",
        ("vid", "fid", "oid"),
    ),
    (
        "relationships",
        ("vid", "fid", "rid", "oid_i", "predicate", "oid_j"),
        {
            "vid": "BIGINT",
            "fid": "BIGINT",
            "rid": "BIGINT",
            "oid_i": "BIGINT",
            "predicate": "VARCHAR",
            "oid_j": "BIGINT",
        },
        "Pairwise relationships between objects within a video frame.",
        ("vid", "fid", "rid"),
    ),
    (
        "attributes",
        ("vid", "fid", "oid", "k", "v"),
        {
            "vid": "BIGINT",
            "fid": "BIGINT",
            "oid": "BIGINT",
            "k": "VARCHAR",
            "v": "VARCHAR",
        },
        "Key-value attributes of detected objects within a video frame.",
        ("vid", "fid", "oid"),
    ),
]

_TEXT_VIEWS: List[ViewDef] = [
    (
        "entities",
        ("did", "eid", "label"),
        {"did": "BIGINT", "eid": "BIGINT", "label": "VARCHAR"},
        "Named entities extracted from a document.",
        ("did", "eid"),
    ),
    (
        "mentions",
        ("did", "sid", "mid", "eid", "span1", "span2"),
        {
            "did": "BIGINT",
            "sid": "BIGINT",
            "mid": "BIGINT",
            "eid": "BIGINT",
            "span1": "BIGINT",
            "span2": "BIGINT",
        },
        "Textual mention spans linked to entities within sentences.",
        ("did", "sid", "mid"),
    ),
    (
        "relationships",
        ("did", "sid", "rid", "eid_i", "predicate", "eid_j"),
        {
            "did": "BIGINT",
            "sid": "BIGINT",
            "rid": "BIGINT",
            "eid_i": "BIGINT",
            "predicate": "VARCHAR",
            "eid_j": "BIGINT",
        },
        "Pairwise relationships between entities within a document.",
        ("did", "sid", "rid"),
    ),
    (
        "attributes",
        ("did", "sid", "eid", "k", "v"),
        {
            "did": "BIGINT",
            "sid": "BIGINT",
            "eid": "BIGINT",
            "k": "VARCHAR",
            "v": "VARCHAR",
        },
        "Key-value attributes of entities within a document.",
        ("did", "sid", "eid"),
    ),
]
