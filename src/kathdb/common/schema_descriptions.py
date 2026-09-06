"""Pre-written LLM-facing descriptions of the canonical multimodal view columns."""

from __future__ import annotations

__all__ = ["SCHEMA_COLUMN_DESCRIPTIONS"]

SCHEMA_COLUMN_DESCRIPTIONS: dict[str, str] = {
    # ======================
    # Core identifiers
    # ======================
    "did": (
        "Column `did` is a unique document identifier. "
        "All text-related rows (sentences, mentions, entities, relationships, attributes) "
        "belonging to the same document share the same `did`. "
        "Example: one news article or one Wikipedia page."
    ),
    "iid": (
        "Column `iid` is a unique image identifier. "
        "Each standalone image has its own unique `iid`. "
        "This column only applies to IMAGE modality."
    ),
    "vid": (
        "Column `vid` is a unique video identifier. "
        "Each video has its own unique `vid`. "
        "This column only applies to VIDEO modality; IMAGE modality tables use `iid` instead."
    ),
    "fid": (
        "Column `fid` is a frame identifier within a video (`vid`). "
        "`fid` is unique only within the same video. "
        "This column only applies to VIDEO modality; IMAGE modality tables do not have `fid`."
    ),
    "sid": (
        "Column `sid` is a sentence identifier within a document (`did`). "
        "`sid` is unique only within that document. "
        "Example: sid=5 refers to the fifth sentence in document did=12."
    ),
    "eid": (
        "Column `eid` is an entity identifier within a document (`did`). "
        "All mentions referring to the same real-world entity in that document "
        "share the same `eid`. "
        "Example: 'Taylor Swift', 'she', and 'the artist behind the Eras Tour' "
        "share one `eid` in a document."
    ),
    "mid": (
        "Column `mid` is a mention identifier within a document (`did`). "
        "Each concrete textual mention (a character span in a sentence) "
        "has its own `mid`, even if multiple mentions map to the same `eid`."
    ),
    "rid": (
        "Column `rid` is a relationship identifier within a document (`did`) "
        "or within an image (`iid`) or video frame (`vid`, `fid`) depending on modality. "
        "Each relationship instance has a unique `rid` within its scope."
    ),
    "oid": (
        "Column `oid` is an object identifier within a single image (`iid`) or "
        "video frame (`vid`, `fid`). "
        "`oid` is unique only within that image or frame. "
        "Example: two different images may both contain oid=1 referring to different objects."
    ),
    # ======================
    # Semantic labels
    # ======================
    "label": (
        "Column `label` is a string-based class label. "
        "For text, it is the entity class (e.g., 'person', 'location'). "
        "For images or videos, it is the object class (e.g., 'person', 'car')."
    ),
    "predicate": (
        "Column `predicate` is a string-based predicate or relation type. "
        "It describes the semantic meaning of a relationship. "
        "Examples: 'born_in', 'directed_by', 'next_to'."
    ),
    # ======================
    # Subject / object roles
    # ======================
    "eid_i": (
        "Column `eid_i` is the subject entity identifier in a text relationship. "
        "It references an `eid`."
    ),
    "eid_j": (
        "Column `eid_j` is the object entity identifier in a text relationship. "
        "It references an `eid`."
    ),
    "oid_i": (
        "Column `oid_i` is the subject object identifier in a visual relationship. "
        "It references an `oid` within the same image or frame."
    ),
    "oid_j": (
        "Column `oid_j` is the object object identifier in a visual relationship. "
        "It references an `oid` within the same image or frame."
    ),
    # ======================
    # Attributes
    # ======================
    "k": (
        "Column `k` is an attribute key describing the type of an attribute. "
        "Examples: 'color', 'budget', 'confidence', 'age'."
    ),
    "v": (
        "Column `v` is the attribute value corresponding to key `k`. "
        "Values may be strings, numbers, or structured values. "
        "Examples: 'black', '130M', 0.87."
    ),
    # ======================
    # Text spans
    # ======================
    "span1": (
        "Column `span1` is the starting character offset of a mention "
        "within a sentence."
    ),
    "span2": (
        "Column `span2` is the ending character offset (exclusive) of a mention "
        "within a sentence."
    ),
    # ======================
    # Visual geometry
    # ======================
    "x1": (
        "Column `x1` is the left x-coordinate of an object's bounding box "
        "in image pixel space."
    ),
    "y1": (
        "Column `y1` is the top y-coordinate of an object's bounding box "
        "in image pixel space."
    ),
    "x2": (
        "Column `x2` is the right x-coordinate of an object's bounding box "
        "in image pixel space."
    ),
    "y2": (
        "Column `y2` is the bottom y-coordinate of an object's bounding box "
        "in image pixel space."
    ),
}
