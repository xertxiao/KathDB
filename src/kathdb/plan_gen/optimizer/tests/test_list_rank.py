"""Validation / repair / ranking plumbing of the list_rank selector."""

from __future__ import annotations

from ..dag import all_atoms_of
from ..list_rank import (
    ListRankSelector,
    _is_valid_partition,
    _validate_or_repair,
    select_partition,
)
from ._fixtures import linear_chain


def test_tournament_order_picks_global_winner():
    # rank_k=2 -> bracket of pairs; _rank_chunk stubbed to sort by a synthetic cost.
    sel = ListRankSelector.__new__(ListRankSelector)
    sel._rank_k = 2
    cost: dict[int, int] = {}
    sel._rank_chunk = lambda root, cands: sorted(cands, key=lambda c: cost[id(c)])
    cands = [[frozenset({chr(65 + i)})] for i in range(5)]
    for i, c in enumerate(cands):
        cost[id(c)] = (i - 2) ** 2  # minimum at i == 2
    order = sel._tournament_order(None, list(cands))
    assert order[0] is cands[2]


def test_tournament_single_chunk_is_one_call():
    # <= rank_k candidates -> a single ranking call returning the FULL ordering.
    calls = []
    sel = ListRankSelector.__new__(ListRankSelector)
    sel._rank_k = 5

    def chunk(root, cands):
        calls.append(len(cands))
        return list(reversed(cands))  # best = last

    sel._rank_chunk = chunk
    cands = [[frozenset({"A"})], [frozenset({"B"})], [frozenset({"C"})]]
    order = sel._tournament_order(None, list(cands))
    assert calls == [3]  # one call over all 3
    assert order[0] is cands[-1]
    assert len(order) == 3


# -- validation / repair -----------------------------------------------------


def test_validate_accepts_convex_cover():
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, n_multi = _validate_or_repair([{"A", "B"}, {"C"}], root, atoms)
    assert reason is None
    assert set().union(*part) == set(atoms)
    assert frozenset({"A", "B"}) in part
    assert n_multi == 1
    assert _is_valid_partition(part, root, set(atoms))


def test_validate_adds_dropped_atoms_as_singletons():
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, _ = _validate_or_repair([{"A", "B"}], root, atoms)  # C dropped
    assert reason is None
    assert frozenset({"C"}) in part
    assert set().union(*part) == set(atoms)


def test_validate_none_falls_back_to_atomic():
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, n_multi = _validate_or_repair(None, root, atoms)
    assert reason == "partition_rejected"
    assert all(len(g) == 1 for g in part)
    assert n_multi == 0


def test_validate_repairs_nonconvex_via_closure():
    # A->B->C: {A,C} is non-convex (B bridges). B is unassigned, so closure absorbs it.
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, n_multi = _validate_or_repair([{"A", "C"}], root, atoms)
    assert reason is None
    assert frozenset({"A", "B", "C"}) in part
    assert n_multi == 1


def test_validate_splits_nonconvex_when_bridge_already_assigned():
    # B is assigned first, so {A,C} cannot absorb it -> split A,C to singletons.
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, n_multi = _validate_or_repair([{"B"}, {"A", "C"}], root, atoms)
    assert reason is None
    assert all(len(g) == 1 for g in part)
    assert n_multi == 0


def test_validate_drops_duplicate_atom_across_groups():
    # A appears twice; first-wins keeps it in the first group only.
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, _ = _validate_or_repair([{"A", "B"}, {"A", "C"}], root, atoms)
    assert reason is None
    assert set().union(*part) == set(atoms)
    flat = [a for g in part for a in g]
    assert len(flat) == len(set(flat)) == len(atoms)


def test_validate_splits_groups_over_max_group_size():
    root = linear_chain("A", "B", "C")
    atoms = all_atoms_of(root)
    part, reason, n_multi = _validate_or_repair(
        [{"A", "B", "C"}], root, atoms, max_group_size=2
    )
    assert reason is None
    assert all(len(g) == 1 for g in part)
    assert n_multi == 0
    # None = no cap
    part, _, n_multi = _validate_or_repair(
        [{"A", "B", "C"}], root, atoms, max_group_size=None
    )
    assert n_multi == 1


# -- select_partition -------------------------------------------------------


class _FakeSelector:
    def __init__(self, proposed, n_candidates):
        self._proposed = proposed
        self.n_candidates = n_candidates

    def select(self, root):
        return self._proposed


def test_select_partition_no_candidates_reports_reason():
    root = linear_chain("A", "B", "C")
    res = select_partition(root, _FakeSelector(None, 0), max_group_size=None)
    assert res.reason == "no_fusion_candidates"
    assert res.n_fused_groups == 0
    assert all(len(g) == 1 for g in res.partition)


def test_select_partition_applies_validated_pick():
    root = linear_chain("A", "B", "C")
    res = select_partition(
        root, _FakeSelector((frozenset({"A", "B"}), frozenset({"C"})), 3),
        max_group_size=None,
    )
    assert res.reason is None
    assert res.n_fused_groups == 1
    assert res.n_candidates == 3


def test_select_partition_selector_crash_falls_back_to_atomic():
    class _Boom:
        n_candidates = 2

        def select(self, root):
            raise RuntimeError("llm down")

    root = linear_chain("A", "B", "C")
    res = select_partition(root, _Boom(), max_group_size=None)
    assert res.reason == "partition_rejected"
    assert all(len(g) == 1 for g in res.partition)


# -- rendering / canonicalization -------------------------------------------


def test_fmt_fused_uses_parens():
    f = ListRankSelector._fmt_fused
    assert f([frozenset({"A", "B"}), frozenset({"C"})]) == "(A+B)"
    assert f([frozenset({"A", "B"}), frozenset({"C", "D"})]) == "(A+B) (C+D)"
    assert f([frozenset({"A"}), frozenset({"B"})]) == "(no fusion)"


def test_canonicalize_splits_relational_only_and_dedups():
    # Only A is semantic. A+B keeps (has [S]); B+C is relational-only -> split to
    # singletons -> that candidate collapses to atomic -> dropped; the duplicate dedups.
    sel = ListRankSelector.__new__(ListRankSelector)
    sel._sem_atoms = frozenset({"A"})
    cands = [
        [frozenset({"A", "B"}), frozenset({"C"})],
        [frozenset({"A"}), frozenset({"B", "C"})],
        [frozenset({"A", "B"}), frozenset({"C"})],
    ]
    out = sel._canonicalize_candidates(None, cands)
    assert len(out) == 1
    assert frozenset(out[0]) == frozenset({frozenset({"A", "B"}), frozenset({"C"})})


def test_canonicalize_splits_only_the_relational_group():
    sel = ListRankSelector.__new__(ListRankSelector)
    sel._sem_atoms = frozenset({"D"})
    out = sel._canonicalize_candidates(
        None, [[frozenset({"A", "B"}), frozenset({"C", "D"})]]
    )
    assert len(out) == 1
    groups = {tuple(sorted(g)) for g in out[0]}
    assert ("A",) in groups and ("B",) in groups and ("C", "D") in groups


def test_stats_provenance_mentions_sample_share():
    class _Eng:
        _k = 50
        profile = True
        materialized = {"products": None, "labeled": None}
        full_cardinality = {"products": 1000.0, "labeled": 400.0}
        code_by_op = {"labeled": "..."}  # 'labeled' is produced by an atom, not a base table

    line = ListRankSelector._stats_provenance(_Eng())
    assert "50 rows per base table" in line and "5.0%" in line and "MEASURED" in line
    _Eng.profile = False
    assert "first 50 rows" in ListRankSelector._stats_provenance(_Eng())
