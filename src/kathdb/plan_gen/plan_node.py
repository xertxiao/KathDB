"""FAONode — function-as-operator tree for logical query plans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "FAONode",
    "build_fao_dag",
    "load_fao_dag",
]


# Every input/output of a node is a relation name.
@dataclass(slots=True)
class FAONode:
    """Minimal tree node for logical plan construction."""

    op: str
    description: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    children: list["FAONode"] = field(default_factory=list)
    selected_functions: list[str] = field(default_factory=list)
    type: str = "OTHER"
    # "SEMANTIC" (needs model inference) or "RELATIONAL[-<op>]"; None on synthetic nodes.
    op_kind: str | None = None
    consumer_demands: list[dict] = field(default_factory=list)
    # Fused (GROUPED) nodes only: member atom ops, their descriptions (same order),
    # and the optimizer's rationale for the fusion.
    member_atoms: list[str] = field(default_factory=list)
    member_descriptions: list[str] = field(default_factory=list)
    merge_rationale: str | None = None
    # Audit trail when demand propagation changed op_kind: keys from, to, rationale, evidence.
    op_kind_rewrite: dict[str, str] | None = None

    def add_child(self, child: "FAONode") -> "FAONode":
        """Append a child node and return it for chaining."""
        self.children.append(child)
        return child

    def replace_child(self, old: "FAONode", new: "FAONode") -> bool:
        """Replace old child with new in children list. Returns True if found."""
        for i, child in enumerate(self.children):
            if child is old:
                self.children[i] = new
                return True
        return False

    def to_dict(self) -> dict[str, object]:
        """Serialize the logical plan to a nested dict."""
        data: dict[str, object] = {
            "op": self.op,
            "children": [child.to_dict() for child in self.children],
        }
        if self.description:
            data["description"] = self.description
        if self.inputs:
            data["input"] = list(self.inputs)
        if self.outputs:
            data["output"] = list(self.outputs)
        if self.selected_functions:
            data["selected_functions"] = list(self.selected_functions)
        if self.type != "OTHER":
            data["type"] = self.type
        if self.op_kind:
            data["op_kind"] = self.op_kind
        if self.consumer_demands:
            data["consumer_demands"] = list(self.consumer_demands)
        if self.member_atoms:
            data["member_atoms"] = list(self.member_atoms)
        if self.member_descriptions:
            data["member_descriptions"] = list(self.member_descriptions)
        if self.merge_rationale:
            data["merge_rationale"] = self.merge_rationale
        if self.op_kind_rewrite:
            data["op_kind_rewrite"] = dict(self.op_kind_rewrite)
        return data

    def pretty(self, indent: str = "") -> str:
        """Return a human-readable text representation of the plan tree."""
        parts = [f"{indent}{self.op}"]
        if self.description:
            parts.append(f" ({self.description})")
        lines = ["".join(parts)]

        if self.inputs:
            lines.append(f"{indent}  inputs: {', '.join(self.inputs)}")
        if self.outputs:
            lines.append(f"{indent}  outputs: {', '.join(self.outputs)}")
        if self.selected_functions:
            lines.append(f"{indent}  functions: {', '.join(self.selected_functions)}")
        if self.consumer_demands:
            for cd in self.consumer_demands:
                consumer = cd.get("consumer", "?")
                cols = [c.get("name", "?") for c in cd.get("required_columns", [])]
                constraints = [
                    f"{vc.get('column', '?')}={vc.get('constraint', '')}"
                    for vc in cd.get("value_constraints", [])
                ]
                parts_cd = []
                if cols:
                    parts_cd.append(f"columns=[{', '.join(cols)}]")
                if constraints:
                    parts_cd.append(f"constraints=[{', '.join(constraints)}]")
                lines.append(
                    f"{indent}  demand[{consumer}]: {'; '.join(parts_cd) or '(none)'}"
                )
        if self.op_kind_rewrite:
            rw = self.op_kind_rewrite
            lines.append(
                f"{indent}  op_kind_rewrite: {rw.get('from', '?')} -> {rw.get('to', '?')}"
            )

        for child in self.children:
            child_lines = child.pretty(indent + "    ").splitlines()
            if not child_lines:
                continue
            child_lines[0] = f"{indent}  ↳ {child_lines[0].strip()}"
            lines.extend(child_lines)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FAONode(op={self.op!r}, children={len(self.children)})"

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FAONode":
        node = cls(op=str(data.get("op", "")))
        description = data.get("description")
        if isinstance(description, str) and description.strip():
            node.description = description

        inputs = _coerce_names(data.get("input"))
        outputs = _coerce_names(data.get("output"))
        if inputs:
            node.inputs = inputs
        if outputs:
            node.outputs = outputs

        raw_selected = data.get("selected_functions")
        if isinstance(raw_selected, (list, tuple)):
            node.selected_functions = [str(s) for s in raw_selected if s]
        raw_type = data.get("type")
        if isinstance(raw_type, str) and raw_type.strip():
            node.type = raw_type.strip()

        raw_op_kind = data.get("op_kind")
        if isinstance(raw_op_kind, str) and raw_op_kind.strip():
            node.op_kind = raw_op_kind.strip()

        raw_demands = data.get("consumer_demands")
        if isinstance(raw_demands, (list, tuple)):
            node.consumer_demands = [
                dict(x) for x in raw_demands if isinstance(x, dict)
            ]

        raw_members = data.get("member_atoms")
        if isinstance(raw_members, (list, tuple)):
            node.member_atoms = [str(s) for s in raw_members if s]

        raw_member_descs = data.get("member_descriptions")
        if isinstance(raw_member_descs, (list, tuple)):
            node.member_descriptions = [str(s) for s in raw_member_descs]

        raw_rationale = data.get("merge_rationale")
        if isinstance(raw_rationale, str) and raw_rationale.strip():
            node.merge_rationale = raw_rationale

        raw_rewrite = data.get("op_kind_rewrite")
        if isinstance(raw_rewrite, Mapping):
            rewrite = {
                str(k): str(v)
                for k, v in raw_rewrite.items()
                if isinstance(k, str) and v is not None
            }
            if rewrite:
                node.op_kind_rewrite = rewrite

        for child_data in data.get("children", []):
            if isinstance(child_data, Mapping):
                node.add_child(cls.from_dict(child_data))
        return node

    def iter_postorder(self) -> Iterable["FAONode"]:
        """Yield nodes in post-order (children before parent).

        Deduplicates shared DAG nodes so each node is yielded exactly once,
        even when multiple parents reference the same child.
        """
        seen: set[int] = set()
        stack: list[tuple["FAONode", int]] = []
        stack.append((self, 0))
        while stack:
            node, i = stack.pop()
            if i < len(node.children):
                stack.append((node, i + 1))
                child = node.children[i]
                if id(child) not in seen:
                    stack.append((child, 0))
            else:
                if id(node) not in seen:
                    seen.add(id(node))
                    yield node

    def iter_preorder(self) -> Iterable["FAONode"]:
        """Yield nodes in pre-order (parent before children).

        Deduplicates shared DAG nodes so each node is yielded exactly once,
        even when multiple parents reference the same child.
        """
        seen: set[int] = set()
        stack: list["FAONode"] = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            # Push children in reverse so leftmost child is visited first
            for child in reversed(node.children):
                if id(child) not in seen:
                    stack.append(child)


def load_fao_dag(path: str) -> FAONode:
    """Load a logical plan tree from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FAONode.from_dict(payload)


def build_fao_dag(
    plan_entries: Sequence[Mapping[str, object]],
    input_schema_names: Sequence[str],
) -> FAONode:
    root = FAONode(op="logical_plan")
    if not plan_entries:
        raise ValueError("Empty logical plan entries.")

    schema_names = {name for name in input_schema_names if name}
    schema_nodes = {
        name: FAONode(op="input_relation", outputs=[name])
        for name in schema_names
    }

    nodes_in_order: list[FAONode] = []
    output_providers: dict[str, FAONode] = {}

    for index, entry in enumerate(plan_entries):
        op_name = str(entry.get("name") or f"step_{index}")
        node = FAONode(op=op_name)
        description = entry.get("description")
        if isinstance(description, str) and description.strip():
            node.description = description.strip()

        raw_type = entry.get("type")
        if isinstance(raw_type, str) and raw_type.strip():
            node.type = raw_type.strip()

        raw_op_kind = entry.get("op_kind")
        if isinstance(raw_op_kind, str) and raw_op_kind.strip():
            node.op_kind = raw_op_kind.strip()

        inputs = _coerce_names(entry.get("input"))
        outputs = _coerce_names(entry.get("output"))
        if inputs:
            node.inputs = inputs
        if outputs:
            node.outputs = outputs
            for output_name in outputs:
                # Output names wire consumers to producers; a duplicate would misroute silently.
                existing = output_providers.get(output_name)
                if existing is not None and existing is not node:
                    raise ValueError(
                        f"Duplicate output relation {output_name!r}: produced by "
                        f"both {existing.op!r} and {node.op!r}. Action outputs must "
                        f"be unique within a plan sketch."
                    )
                output_providers[output_name] = node

        nodes_in_order.append(node)

    nodes_with_parent: set[int] = set()

    for node in nodes_in_order:
        attached_ids: set[int] = set()
        inputs_with_producer: set[str] = set()
        for input_name in node.inputs:
            producer = output_providers.get(input_name)
            if producer and producer is not node:
                producer_id = id(producer)
                if producer_id not in attached_ids:
                    node.add_child(producer)
                    attached_ids.add(producer_id)
                    nodes_with_parent.add(producer_id)
                inputs_with_producer.add(input_name)

        # Attach input_relation schema nodes only for inputs that
        # don't already have a producer step in the plan.
        for input_name in node.inputs:
            if input_name in inputs_with_producer:
                continue
            schema_node = schema_nodes.get(input_name)
            if schema_node is None:
                continue
            schema_id = id(schema_node)
            if schema_id not in attached_ids:
                node.add_child(schema_node)
                attached_ids.add(schema_id)
                nodes_with_parent.add(schema_id)

    top_level_nodes = [
        node for node in nodes_in_order if id(node) not in nodes_with_parent
    ] or nodes_in_order

    for node in top_level_nodes:
        root.add_child(node)
    return root


# ------------------------------------------------------------------
# Helpers


def _coerce_names(value: object) -> list[str]:
    """Normalize input/output names to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
    return []

