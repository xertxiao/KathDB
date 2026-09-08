"""``list_rank``: choose the partition of the atomic plan with the lowest expected cost.

Candidates are every legal convex partition (:mod:`.enumerate`) or, for wide plans,
LLM-proposed ones. An LLM ranks up to ``rank_k`` candidates per call (tournament
beyond that), seeing each atom's base-plan code and, when profiled, measured
cardinalities. The pick is validated/repaired; an invalid pick falls back to atomic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ...common.logger import get_logger
from .dag import all_atoms_of, nodes_by_op
from .enumerate import enumerate_convex_partitions
from .rewriter import compute_convex_closure, verify_fusable_subset
from .types import Partition

logger = get_logger(__name__)

__all__ = ["ListRankSelector", "SelectionResult", "select_partition"]

# Wide plans whose convex-partition lattice exceeds this many partitions are not
# enumerated; the LLM proposes candidates instead.
ENUMERATION_CAP = 200

# Number of LLM-proposed candidate partitions for wide plans, and the per-candidate
# nudges that make the proposals differ from one another.
_N_PROPOSED_CANDIDATES = 4
_PROPOSAL_NUDGES = [
    "Give a sensible grouping that fuses where model calls can be shared or skipped.",
    "Give a DIFFERENT, more AGGRESSIVE grouping (fuse more atoms into fewer groups).",
    "Give a more CONSERVATIVE grouping (fuse only the single clearest opportunity).",
    "Give an ALTERNATIVE grouping that targets a different sharing/early-exit chance.",
]


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------


class PartitionGroup(BaseModel):
    """One group of a proposed partition: a set of atom op-names to fuse."""

    atoms: list[str] = Field(
        description=(
            "Op-names of the atomic operators fused into this one group, copied "
            "EXACTLY from the plan listing. A single-element list is a standalone "
            "(unfused) atom. Every group must be a convex (gap-free) subgraph: do "
            "not fuse two atoms while leaving an operator that sits between them on "
            "a data-flow path in a different group."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Short reason this grouping cuts model calls (e.g. 'classify once per "
            "brand via a cheap sort+break instead of per-row')."
        ),
    )


class PartitionProposalResponse(BaseModel):
    """A single proposed partition: a disjoint cover of ALL atoms by groups."""

    groups: list[PartitionGroup] = Field(
        default_factory=list,
        description=(
            "The groups partitioning the plan. Their atoms together must cover EVERY "
            "atom exactly once. Prefer the MINIMAL grouping that still removes the "
            "most model calls — fuse only where fusion enables a real saving "
            "(shared/early-exit/cached model calls); leave everything else unfused."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One or two sentences on why this partition minimizes model calls.",
    )


class GroupRewrite(BaseModel):
    """The rewrite one fused group of the top-ranked candidate stands for."""

    atoms: list[str] = Field(
        description="Op-names of ONE fused group of the TOP-ranked candidate, copied exactly."
    )
    rewrite: str = Field(
        description=(
            "At most two sentences, concrete: what the fused code computes first, per "
            "which key it calls the model, and where it stops early — then why the "
            "result is unchanged. E.g. 'Classify each brand's items one at a time and "
            "stop at the second distinct kind; a brand qualifies iff all its kinds "
            "agree, so the first disagreement already decides it.'"
        ),
    )


class PartitionCandidateRanking(BaseModel):
    """Listwise ranking of several candidate partitions (best first)."""

    ranked_candidate_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Candidate ids (as labeled in the prompt) ordered BEST first — the "
            "partition expected to make the fewest full-scale model calls while "
            "preserving the query result comes first."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief justification for the top-ranked candidate.",
    )
    rewrites: list[GroupRewrite] = Field(
        default_factory=list,
        description="One entry per fused group of the TOP-ranked candidate.",
    )


# ---------------------------------------------------------------------------
# Entry point + validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionResult:
    partition: Partition
    # ``None`` on success; ``"no_fusion_candidates"`` when no fusion can help;
    # ``"partition_rejected"`` when the selector's pick could not be repaired.
    reason: str | None
    n_fused_groups: int
    n_candidates: int
    # fused group -> the rewrite the ranker chose it for (handed to the fused codegen)
    rewrites: dict[frozenset[str], str] = field(default_factory=dict)


def select_partition(
    root, selector: "ListRankSelector", *, max_group_size: int | None
) -> SelectionResult:
    """Run the selector and return a validated partition (atomic on any failure)."""
    atoms = all_atoms_of(root)
    try:
        proposed = selector.select(root)
    except Exception as exc:  # noqa: BLE001 - any selector failure -> atomic fallback
        logger.warning("[list_rank] selector failed: %s; falling back to atomic", exc)
        proposed = None

    if proposed is None and selector.n_candidates == 0:
        singleton = tuple(frozenset({a}) for a in atoms)
        logger.info("[list_rank] no fusion candidate can cut model calls; atomic plan")
        return SelectionResult(singleton, "no_fusion_candidates", 0, 0)
    rewrites = dict(getattr(selector, "group_rewrites", {}) or {})

    partition, reason, n_multi = _validate_or_repair(
        proposed, root, atoms, max_group_size=max_group_size
    )
    logger.info(
        "[list_rank] %d group(s), %d fused; outcome=%s",
        len(partition),
        n_multi,
        reason or "ok",
    )
    return SelectionResult(partition, reason, n_multi, selector.n_candidates, rewrites)


def _validate_or_repair(
    proposed, root, atoms, max_group_size: int | None = None
) -> tuple[Partition, str | None, int]:
    """Coerce a proposed partition into a valid one (cover + convex + size cap), or
    fall back to atomic. Returns ``(partition, reason, n_multi_atom_groups)``;
    ``reason`` is ``None`` on success, ``"partition_rejected"`` on fallback.
    """
    atom_set = set(atoms)
    singleton = tuple(frozenset({a}) for a in atoms)
    if not proposed:
        return singleton, "partition_rejected", 0

    by_op = nodes_by_op(root)
    assigned: set[str] = set()
    groups: list[set[str]] = []
    for g in proposed:
        members = {a for a in g if a in atom_set and a not in assigned}
        if not members:
            continue
        if len(members) >= 2:
            nodes = [by_op[m] for m in members if m in by_op]
            if len(nodes) != len(members) or not verify_fusable_subset(nodes, root):
                closure = {n.op for n in compute_convex_closure(nodes, root)}
                if closure and closure <= (atom_set - assigned - members):
                    members |= closure
                else:
                    # Cannot repair without stealing assigned atoms -> split to singletons.
                    for m in members:
                        groups.append({m})
                        assigned.add(m)
                    continue
        if max_group_size is not None and len(members) > max_group_size:
            logger.info(
                "[list_rank] group of %d atoms exceeds max_group_size=%d; "
                "splitting to singletons",
                len(members),
                max_group_size,
            )
            for m in members:
                groups.append({m})
                assigned.add(m)
            continue
        groups.append(members)
        assigned |= members

    for a in atoms:  # cover anything the LLM dropped
        if a not in assigned:
            groups.append({a})
            assigned.add(a)

    partition = tuple(frozenset(g) for g in groups)
    if not _is_valid_partition(partition, root, atom_set):
        return singleton, "partition_rejected", 0
    n_multi = sum(1 for g in partition if len(g) >= 2)
    return partition, None, n_multi


def _is_valid_partition(partition: Partition, root, atom_set: set[str]) -> bool:
    by_op = nodes_by_op(root)
    covered: set[str] = set()
    for g in partition:
        if not g or (covered & g):
            return False
        covered |= set(g)
        if len(g) >= 2:
            nodes = [by_op[m] for m in g if m in by_op]
            if len(nodes) != len(g) or not verify_fusable_subset(nodes, root):
                return False
    return covered == atom_set


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class ListRankSelector:
    """Rank candidate partitions of one atomic plan and return the best.

    Built per query via :func:`make_selector_factory`; the base-plan code-gen
    engine is built lazily on the first :meth:`select`.
    """

    def __init__(
        self,
        *,
        atomic_root,
        code_gen,
        rc,
        nl_query: str | None,
        grouping_cache,
        sample_rows: int,
        rank_k: int = 10,
        max_group_size: int | None = None,
        profile: bool = False,
        worker_manager=None,
    ) -> None:
        self._atomic_root = atomic_root
        self._code_gen = code_gen
        self._rc = rc
        self._nl_query = nl_query
        self._cache = grouping_cache
        self._sample_rows = sample_rows
        self._profile = profile
        self._worker_manager = worker_manager
        self._rank_k = max(2, int(rank_k))
        self._max_group_size = max_group_size
        self._engine = None
        self._engine_built = False
        self._sem_atoms: frozenset[str] | None = None
        # Number of candidate partitions ranked by the last :meth:`select`.
        self.n_candidates = 0

    # -- base-plan code-gen ------------------------------------------------

    def _ensure_engine(self):
        if self._engine_built:
            return self._engine
        self._engine_built = True
        # Lazy import: plan_gen must not import the executor at module load.
        from ...executor.codegen.base_plan_codegen import BasePlanCodegen

        try:
            self._engine = BasePlanCodegen(
                code_gen=self._code_gen,
                rc=self._rc,
                atomic_root=self._atomic_root,
                nl_query=self._nl_query,
                grouping_cache=self._cache,
                sample_rows=self._sample_rows,
                profile=self._profile,
                worker_manager=self._worker_manager,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to structure-only ranking
            logger.warning("[list_rank] base-plan code-gen failed: %s; structure only", exc)
            self._engine = None
        return self._engine

    # -- plan rendering ----------------------------------------------------

    def _render_plan(self, root) -> str:
        by_op = nodes_by_op(root)
        engine = self._engine
        lines = ["## Atomic plan — operators to partition (fuse a convex subset):"]
        for op in all_atoms_of(root):
            node = by_op.get(op)
            if node is None:
                continue
            kind = "S" if (node.op_kind or "").upper().startswith("SEMANTIC") else "R"
            piece = (
                f"- {op} [{kind}]: {node.description or '(no description)'}; "
                f"inputs={list(node.inputs)} outputs={list(node.outputs)}"
            )
            if engine is not None:
                in_fulls = [
                    engine.full_cardinality[i]
                    for i in node.inputs
                    if i in engine.full_cardinality
                ]
                if in_fulls:
                    piece += f"; full_in~{max(in_fulls):.0f} rows"
                in_rel = node.inputs[0] if node.inputs else None
                out_rel = node.outputs[0] if node.outputs else None
                in_df = engine.materialized.get(in_rel) if in_rel else None
                out_df = engine.materialized.get(out_rel) if out_rel else None
                # Profiled: output cardinality + per-op selectivity (out / in rows).
                if out_df is not None and not out_df.empty:
                    out_full = engine.full_cardinality.get(out_rel)
                    piece += (
                        f"; out~{len(out_df)}"
                        + (f" [~{int(out_full)}]" if out_full else "")
                        + " rows"
                    )
                    if in_df is not None and len(in_df):
                        piece += f"; selectivity {len(out_df) / len(in_df):.2f}"
                if in_df is not None and not in_df.empty:
                    vstats = engine.column_value_stats(in_rel, list(in_df.columns))
                    if vstats:
                        piece += f"; {vstats}"
            lines.append(piece)
        lines.append("")
        if engine is not None:
            lines.append(self._stats_provenance(engine))
        lines.append(
            "[S] = semantic operator (one model call per record by default); "
            "[R] = relational (free). Fuse to cut TOTAL execution LLM tokens (calls × "
            "prompt length): share / early-exit / cache calls and shorten prompts."
        )
        lines.append(
            "When judging a group, score it by the BEST cost the codegen can reach AFTER "
            "an intra-group logic rewrite — not the naive per-record cost. In particular, "
            "if the information a semantic [S] op infers can be read directly from an "
            "existing column, the model need not be called over the multimodal data at "
            "all, so that op's true cost is ~0. Prefer groupings that unlock this, push a "
            "relational filter ahead of a model call, or expose an early-exit; a grouping "
            "whose group still needs a full model pass after rewrite is worth less. A "
            "shortcut keyed on OBSERVED values (a column test or lookup that replaces a "
            "model call) only counts when it keeps the model call as the fallback for "
            "every row it does not cover — the sample never shows the full vocabulary, so "
            "do not credit a grouping with eliminating model calls for unseen values."
        )
        if engine is not None:
            code_lines = ["", "## Generated base-plan code (per atomic operator):"]
            for op in all_atoms_of(root):
                code_str = engine.code_by_op.get(op, "")
                if code_str.strip():
                    code_lines.append(f"### {op}\n```python\n{code_str.strip()}\n```")
            if len(code_lines) > 2:
                lines.extend(code_lines)
        return "\n".join(lines)

    @staticmethod
    def _stats_provenance(engine) -> str:
        """One line telling the ranker how much of the data the statistics reflect."""
        k = engine._k
        base_n = [
            engine.full_cardinality[r]
            for r in engine.materialized
            if engine.full_cardinality.get(r) and r not in engine.code_by_op
        ]
        largest = max(base_n) if base_n else None
        share = f" (about {100.0 * k / largest:.1f}% of the largest input)" if largest else ""
        if engine.profile:
            return (
                f"Statistics: full_in~ is the exact full-data row count. out~ rows, "
                f"selectivity and values{{...}} were MEASURED on a random sample of {k} "
                f"rows per base table{share}, then scaled; treat them as estimates, and "
                "value lists as the sample's most frequent values, not the full vocabulary."
            )
        return (
            f"Statistics: full_in~ is the exact full-data row count. values{{...}} come "
            f"from the first {k} rows of each base table{share}: the sample's most "
            "frequent values, not the full vocabulary."
        )

    # -- candidate generation ----------------------------------------------

    def select(self, root) -> Partition | None:
        """Return the best partition, or ``None`` when nothing can be fused."""
        self.n_candidates = 0
        self.group_rewrites: dict[frozenset[str], str] = {}
        self._ensure_engine()
        candidates = self._enumerate_candidates(root)
        candidates = self._canonicalize_candidates(root, candidates)
        if not candidates:
            return None  # no fusion can cut model tokens -> keep the atomic plan
        self.n_candidates = len(candidates)
        if len(candidates) == 1:
            return tuple(candidates[0])
        return tuple(self._rank(root, candidates)[0])

    def _semantic_atoms(self, root) -> frozenset[str]:
        if self._sem_atoms is None:
            self._sem_atoms = frozenset(
                op
                for op, node in nodes_by_op(root).items()
                if (node.op_kind or "").upper().startswith("SEMANTIC")
            )
        return self._sem_atoms

    def _canonicalize_candidates(self, root, candidates):
        """Split relational-only groups (no model call to save) into singletons; drop
        candidates left with no fusion and duplicates."""
        sem = self._semantic_atoms(root)
        out: list[list[frozenset]] = []
        seen: set = set()
        for cand in candidates:
            canon: list[frozenset] = []
            for g in cand:
                if len(g) >= 2 and not (g & sem):
                    canon.extend(frozenset({a}) for a in g)
                else:
                    canon.append(g)
            if not any(len(g) >= 2 for g in canon):
                continue
            key = frozenset(canon)
            if key in seen:
                continue
            seen.add(key)
            out.append(canon)
        return out

    def _enumerate_candidates(self, root):
        parts, capped = enumerate_convex_partitions(
            root, max_group_size=self._max_group_size, hard_cap=ENUMERATION_CAP
        )
        if capped or not parts:
            logger.warning(
                "[list_rank] convex-partition lattice exceeds %d for %d atoms; "
                "asking the LLM to propose candidates instead",
                ENUMERATION_CAP,
                len(all_atoms_of(root)),
            )
            return self._propose_candidates(root)
        return [list(p) for p in parts]

    def _propose_candidates(self, root):
        candidates: list[list[frozenset]] = []
        seen: set = set()
        for nudge in _PROPOSAL_NUDGES[:_N_PROPOSED_CANDIDATES]:
            groups = self._propose(root, nudge)
            if not groups:
                continue
            cand = [frozenset(g) for g in groups]
            key = frozenset(cand)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
        return candidates

    def _propose(self, root, nudge: str):
        cap = (
            f" with at most {self._max_group_size} atoms"
            if self._max_group_size is not None
            else ""
        )
        query = f"\n## Query\n{self._nl_query}\n" if self._nl_query else ""
        prompt = (
            "You are choosing how to PARTITION an atomic query plan into fused groups so "
            "the generated code uses the LEAST total execution LLM tokens (fewest model "
            "calls and shortest prompts) while preserving "
            "the query result.\n"
            f"{query}\n{self._render_plan(root)}\n\n"
            f"{nudge}\n"
            "Return groups covering EVERY atom exactly once. Each group must be a convex "
            f"(gap-free) subgraph{cap}. Copy atom op-names EXACTLY."
        )
        try:
            resp = self._code_gen._invoke_structured(
                prompt,
                llm=self._code_gen.generation_llm,
                schema=PartitionProposalResponse,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[list_rank] proposal call failed: %s", exc)
            return None
        groups = [set(g.atoms) for g in (resp.groups or []) if g.atoms]
        return groups or None

    # -- ranking -------------------------------------------------------------

    @staticmethod
    def _fmt_fused(cand) -> str:
        """``(a+b) (c+d)`` — fused groups only; ``(no fusion)`` when nothing is fused."""
        fused = ["(" + "+".join(sorted(g)) + ")" for g in cand if len(g) >= 2]
        return " ".join(fused) or "(no fusion)"

    def _render_candidates(self, cand_list) -> str:
        return "\n\n".join(
            f"Candidate {i}: {self._fmt_fused(cand)}" for i, cand in enumerate(cand_list)
        )

    def _rank(self, root, candidates):
        """Return candidates best-first (``[0]`` is the champion)."""
        try:
            order = self._tournament_order(root, list(candidates))
        except Exception as exc:  # noqa: BLE001 - ranking failure -> enumeration order
            logger.warning("[list_rank] ranking failed: %s", exc)
            return candidates
        in_order = {id(c) for c in order}
        rest = [c for c in candidates if id(c) not in in_order]
        return [*order, *rest]

    def _tournament_order(self, root, candidates):
        """Rank chunks of ≤ rank_k, recurse on the winners; return the final ordering."""
        k = self._rank_k
        if len(candidates) <= k:
            return self._rank_chunk(root, candidates)
        winners = [
            self._rank_chunk(root, candidates[s : s + k])[0]
            for s in range(0, len(candidates), k)
        ]
        return self._tournament_order(root, winners)

    def _rank_chunk(self, root, candidates):
        """One listwise ranking call over ≤ rank_k candidates; returns them best-first."""
        prompt = (
            f"{self._render_plan(root)}\n\n"
            "Rank these candidate partitions of the above plan from LEAST to most "
            "TOTAL execution LLM token usage (model calls × prompt+completion length; "
            "best first). Each parenthesized group (a+b) is fused: its operators are "
            "implemented together in one piece of code. Cost = calls × (payload + prompt "
            "+ answer); the payload (an image, a record's text) is paid on every call, so "
            "tokens fall only by sending each payload fewer times: one call per item that "
            "answers everything the group needs of it, a relational filter that removes "
            "rows before their payload is sent, a stop at a reached LIMIT. Splitting an "
            "item's judgement into several calls re-sends its payload and saves nothing. "
            "TIE-BREAK: when two candidates would cut total tokens by the SAME amount, "
            "prefer the one that fuses FEWER atoms (smaller groups) — simpler generated "
            "code, lower risk.\n\n"
            f"{self._render_candidates(candidates)}\n\n"
            "Return candidate ids best-first. For every fused group of your top candidate, "
            "add a rewrite of at most two sentences: what the fused code computes first, "
            "per which key it calls the model, where it stops early (only where the "
            "skipped rows cannot change the output; a query that asks for all rows or "
            "pairs has no early exit), and why the result is unchanged. A rewrite may only "
            "remove or shorten calls: one pass over a single input, never every row of one "
            "input against every row of the other, never a re-scan of one input per item "
            "of another. Code takes over a judgement only when it is exact (e.g., equality "
            "on a stored column), never by guessing from words in free text. Count the "
            "items the model sees with the full_in~ row counts, not the sample counts."
        )
        resp = self._code_gen._invoke_structured(
            prompt, llm=self._code_gen.generation_llm, schema=PartitionCandidateRanking
        )
        for rw in resp.rewrites or []:
            if rw.atoms and rw.rewrite.strip():
                self.group_rewrites[frozenset(rw.atoms)] = rw.rewrite.strip()
        order = [
            i for i in (resp.ranked_candidate_ids or []) if 0 <= i < len(candidates)
        ]
        seen = set(order)
        order += [i for i in range(len(candidates)) if i not in seen]
        ranked = [candidates[i] for i in order]
        logger.info(
            "[list_rank] ranked %d candidate(s); top=%s; rewrites=%s",
            len(candidates),
            self._fmt_fused(ranked[0]),
            [rw.rewrite.strip() for rw in (resp.rewrites or []) if rw.rewrite.strip()],
        )
        return ranked


def make_selector_factory(
    *,
    code_gen,
    rc,
    nl_query: str | None,
    grouping_cache,
    sample_rows: int,
    rank_k: int,
    max_group_size: int | None,
    profile: bool = False,
    worker_manager=None,
):
    """Return ``atomic_root -> ListRankSelector`` bound to this query's context."""

    def factory(atomic_root) -> ListRankSelector:
        return ListRankSelector(
            atomic_root=atomic_root,
            code_gen=code_gen,
            rc=rc,
            nl_query=nl_query,
            grouping_cache=grouping_cache,
            sample_rows=sample_rows,
            rank_k=rank_k,
            max_group_size=max_group_size,
            profile=profile,
            worker_manager=worker_manager,
        )

    return factory
