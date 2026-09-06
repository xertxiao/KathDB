"""Function contracts: the typed signature + module-level ``CONTRACT`` dict in a
function's ``scripts/fn.py`` are its single source of truth. ``fn.md``,
``df_params`` and the semantic-param hint are derived from them by AST parsing
(the module is never imported).

``CONTRACT`` schema (all values are plain strings unless noted)::

    CONTRACT = {
        "purpose":   "one-line summary of what the fn does",
        "params":    {"<name>": "one-line note", ...},   # prose; types come from the signature
        "output":    "what it returns",
        "example":   "<one canonical call, as source>",
        "cost":      "cost note (optional)",
        "use_when":  "when the planner should pick this op",
        "not_when":  "when it should NOT (optional)",
        "sys_params": ["model", ...],                      # optional: which params are SYSTEM knobs
    }
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = [
    "ParsedFn",
    "Param",
    "parse_fn_source",
    "derive_df_params",
    "format_signature",
    "render_fn_md",
    "semantic_params",
    "validate_contract",
]


@dataclass(frozen=True)
class Param:
    name: str
    annotation: str | None
    default: str | None  # source string, or None when required
    kw_only: bool
    variadic: str | None = None  # None | "*" (var-positional) | "**" (var-keyword)

    @property
    def is_df(self) -> bool:
        return bool(self.annotation) and "DataFrame" in self.annotation


@dataclass(frozen=True)
class ParsedFn:
    name: str
    params: tuple[Param, ...]
    returns: str | None
    contract: dict


def parse_fn_source(code: str, fn_name: str) -> ParsedFn:
    """Parse source into a :class:`ParsedFn`; ValueError if the function or CONTRACT is missing."""
    tree = ast.parse(code)

    contract: dict | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CONTRACT" for t in node.targets
        ):
            try:
                contract = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"CONTRACT for {fn_name!r} is not a literal: {exc}")
            break
    if contract is None:
        raise ValueError(f"No module-level CONTRACT dict found for {fn_name!r}")

    fn_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            fn_node = node
            break
    if fn_node is None:
        raise ValueError(f"Function {fn_name!r} not found in source")

    a = fn_node.args
    params: list[Param] = []

    pos = list(a.posonlyargs) + list(a.args)
    pos_defaults = list(a.defaults)
    n_required = len(pos) - len(pos_defaults)
    for i, arg in enumerate(pos):
        default = ast.unparse(pos_defaults[i - n_required]) if i >= n_required else None
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        params.append(Param(arg.arg, ann, default, kw_only=False))

    if a.vararg is not None:
        ann = ast.unparse(a.vararg.annotation) if a.vararg.annotation else None
        params.append(Param(a.vararg.arg, ann, None, kw_only=False, variadic="*"))

    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        default = ast.unparse(dflt) if dflt is not None else None
        params.append(Param(arg.arg, ann, default, kw_only=True))

    if a.kwarg is not None:
        ann = ast.unparse(a.kwarg.annotation) if a.kwarg.annotation else None
        params.append(Param(a.kwarg.arg, ann, None, kw_only=True, variadic="**"))

    returns = ast.unparse(fn_node.returns) if fn_node.returns else None
    return ParsedFn(fn_name, tuple(params), returns, contract)


def derive_df_params(parsed: ParsedFn) -> tuple[str, ...]:
    """Names of DataFrame-typed params; a ``**dfs: pd.DataFrame`` yields the ``"**"`` sentinel."""
    fixed = tuple(p.name for p in parsed.params if p.is_df and not p.variadic)
    var_kw_df = any(p.variadic == "**" and p.is_df for p in parsed.params)
    return fixed + (("**",) if var_kw_df else ())


def _sys_params(parsed: ParsedFn) -> set[str]:
    return set(parsed.contract.get("sys_params", ()) or ())


def format_signature(parsed: ParsedFn) -> str:
    """One-line typed signature string."""
    parts: list[str] = []
    star_emitted = False
    for p in parsed.params:
        if p.variadic == "*":
            star_emitted = True
        elif p.kw_only and not star_emitted and p.variadic != "**":
            parts.append("*")
            star_emitted = True
        prefix = p.variadic or ""
        s = f"{prefix}{p.name}"
        if p.annotation:
            s += f": {p.annotation}"
        if p.default is not None:
            s += f" = {p.default}"
        parts.append(s)
    ret = f" -> {parsed.returns}" if parsed.returns else ""
    return f"{parsed.name}({', '.join(parts)}){ret}"


def render_fn_md(parsed: ParsedFn) -> str:
    """Render ``fn.md`` from signature + CONTRACT. The headings are consumed by the
    parser/codegen renderers; keep them in sync."""
    c = parsed.contract
    sysp = _sys_params(parsed)
    notes = c.get("params", {}) or {}
    out: list[str] = [f"# {parsed.name}", "", c.get("purpose", "").strip(), ""]

    out.append("## Signature")
    out.append(f"`{format_signature(parsed)}`")
    out.append("")

    out.append("## Arguments")
    for p in parsed.params:
        bits = [p.annotation or "Any"]
        if p.variadic:
            bits.append("variadic")
        elif p.name in sysp:
            bits.append("system")
        else:
            bits.append("required" if p.default is None else f"default {p.default}")
        note = notes.get(p.name, "").strip()
        line = f"- `{(p.variadic or '')}{p.name}` ({', '.join(bits)})"
        if note:
            line += f": {note}"
        out.append(line)
    out.append("")

    out.append("## Output")
    out.append(c.get("output", "").strip())
    out.append("")

    behavior = c.get("behavior", "").strip()
    if behavior:
        out.append("## Behavior")
        out.append(behavior)
        out.append("")

    ex = c.get("example", "").strip()
    if ex:
        out.append("## Examples")
        out.append("```python")
        out.append(f"from kathdb.fn import {parsed.name}")
        out.append(ex)
        out.append("```")
        out.append("")

    guidance = c.get("guidance", "").strip()
    if guidance:
        out.append("## Guidance")
        out.append(guidance)
        out.append("")

    cost = c.get("cost", "").strip()
    if cost:
        out.append("## Cost Warning")
        out.append(cost)
        out.append("")

    use_when = c.get("use_when", "").strip()
    not_when = c.get("not_when", "").strip()
    if use_when or not_when:
        out.append("## Selection Guidance")
        if use_when:
            out.append(f"Use when: {use_when}")
        if not_when:
            out.append(f"Do NOT use when: {not_when}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def semantic_params(parsed: ParsedFn) -> list[Param]:
    """Params that are not a DataFrame input, a variadic, or a system knob (the
    kwargs the code generator must supply)."""
    sysp = _sys_params(parsed)
    return [
        p
        for p in parsed.params
        if not p.is_df and not p.variadic and p.name not in sysp
    ]


def validate_contract(parsed: ParsedFn) -> list[str]:
    """Signature/CONTRACT consistency problems (empty == consistent)."""
    issues: list[str] = []
    sig_names = {p.name for p in parsed.params}
    notes = parsed.contract.get("params", {}) or {}
    sysp = _sys_params(parsed)

    if not parsed.contract.get("utility", False) and not derive_df_params(parsed):
        issues.append(
            "no DataFrame-typed parameter (need a df input, or mark CONTRACT['utility']=True "
            "for a low-level non-DataFrame utility)"
        )
    for k in notes:
        if k not in sig_names:
            issues.append(f"CONTRACT['params'] documents unknown param {k!r}")
    for s in sysp:
        if s not in sig_names:
            issues.append(f"CONTRACT['sys_params'] lists unknown param {s!r}")
    for p in parsed.params:
        if p.is_df or p.name in sysp:
            continue
        if p.name not in notes:
            issues.append(
                f"param {p.name!r} is undocumented (add to CONTRACT['params'] or sys_params)"
            )
    if not parsed.contract.get("purpose", "").strip():
        issues.append("CONTRACT['purpose'] is empty")
    if not parsed.contract.get("output", "").strip():
        issues.append("CONTRACT['output'] is empty")
    return issues
