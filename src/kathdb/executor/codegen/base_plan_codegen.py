"""Plan-time code generation (and optional profiling) of the atomic base plan.

Every atomic operator is code-generated once against a sample of its inputs before
the grouping optimizer ranks candidate groupings: the ranker reads the code, and the
:class:`~.grouping_cache.GroupingCache` keeps it so unfused atoms are not regenerated
at execution time. With ``profile=True`` each atom also runs on the sample (real
model calls) so the ranker sees measured cardinalities, selectivities and value
distributions; with ``profile=False`` base tables are seeded with ``head(k)``,
intermediates with an empty frame of their demanded columns, at unit selectivity.
"""

from __future__ import annotations

import pandas as pd

from ...common.context import DBContext
from ...common.logger import get_logger
from ...plan_gen.plan_node import FAONode
from ...worker import WorkerManager
from .codegen import CodeGenerator, _build_consumer_demands_map
from .grouping_cache import GroupingCache

logger = get_logger(__name__)

__all__ = ["BasePlanCodegen"]

_SYNTHETIC_OPS = {"input_relation", "logical_plan"}

# Fixed seed: the profiling sample (hence the chosen grouping) must be deterministic.
_SAMPLE_SEED = 17

# Max characters of a sample value in the stats shown to the ranker.
_VALUE_PREVIEW_LIMIT = 30


def _truncate_value(value: object, limit: int = _VALUE_PREVIEW_LIMIT) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class BasePlanCodegen:
    """Code-generate (and optionally run) every atom of ``atomic_root`` once."""

    def __init__(
        self,
        *,
        code_gen: CodeGenerator,
        rc: DBContext,
        atomic_root: FAONode,
        nl_query: str | None,
        grouping_cache: GroupingCache,
        sample_rows: int,
        profile: bool = False,
        worker_manager: WorkerManager | None = None,
    ) -> None:
        if profile and worker_manager is None:
            raise ValueError("profile=True requires a worker_manager to run the code")
        self._code_gen = code_gen
        self._rc = rc
        self._atomic_root = atomic_root
        self._k = max(1, int(sample_rows))
        self.profile = bool(profile)
        self._worker_manager = worker_manager

        # relation name -> sampled / placeholder / profiled DataFrame
        self.materialized: dict[str, pd.DataFrame] = {}
        # atom op-name -> generated source
        self.code_by_op: dict[str, str] = {}
        # relation name -> estimated full-data row count
        self.full_cardinality: dict[str, float] = {}

        self._by_op: dict[str, FAONode] = {
            n.op: n
            for n in atomic_root.iter_preorder()
            if n is not atomic_root and n.op and n.op not in _SYNTHETIC_OPS
        }
        self._consumer_demands_map = _build_consumer_demands_map(atomic_root)
        # ``_grouping_base_plan``: plan-time pass, generated under the same objective as fused groups.
        self._cg_in = {
            "relation_context": rc,
            "q_in": nl_query,
            "_grouping_cache": grouping_cache,
            "_grouping_base_plan": True,
        }

        if self.profile:
            self._seed_base_inputs()
        else:
            self._seed_placeholders()
        self._codegen_all()
        self._build_full_cardinality()

    # ------------------------------------------------------------------
    # Input seeding
    # ------------------------------------------------------------------

    def _base_relations(self) -> set[str]:
        produced = {o for n in self._by_op.values() for o in n.outputs if o}
        return {
            inp for n in self._by_op.values() for inp in n.inputs if inp and inp not in produced
        }

    def _seed_base_inputs(self) -> None:
        """Profiling: a deterministic random sample (not ``head``) of every base table."""
        for name in self._base_relations():
            try:
                df = self._rc.load_table(name)
            except Exception as exc:  # noqa: BLE001 - codegen falls back to the catalog
                logger.warning("[base-plan] cannot load base table %r: %s", name, exc)
                continue
            if isinstance(df, pd.DataFrame):
                if len(df) > self._k:
                    df = df.sample(n=self._k, random_state=_SAMPLE_SEED)
                # Contiguous 0..n-1 index so positional indexing behaves as on head(k).
                self.materialized[name] = df.reset_index(drop=True)

    def _seed_placeholders(self) -> None:
        """No profiling: ``head(k)`` of each base table, empty frames for intermediates."""
        produced = {o for n in self._by_op.values() for o in n.outputs if o}
        cols_by_rel: dict[str, list[str]] = {}
        for inp in self._base_relations():
            try:
                cols_by_rel[inp] = list(self._rc.get_columns(inp))
            except Exception:  # noqa: BLE001 - best effort schema
                cols_by_rel[inp] = []
        for n in self._by_op.values():
            for o in n.outputs:
                if o and o not in cols_by_rel:
                    cols_by_rel[o] = self._demand_columns(o)

        for rel, cols in cols_by_rel.items():
            seeded = None
            if rel not in produced:
                # Base table: a real head(k) sample so codegen sees actual values / struct columns.
                try:
                    if self._rc.has_table(rel):
                        df = self._rc.execute(
                            f'SELECT * FROM "{rel}" LIMIT {self._k}'
                        ).fetchdf()
                        if cols:
                            df = df[[c for c in cols if c in df.columns]]
                        seeded = df
                except Exception:  # noqa: BLE001 - fall back to the schema-only frame
                    seeded = None
            if seeded is None:
                seeded = (
                    pd.DataFrame({c: pd.Series([], dtype=object) for c in cols})
                    if cols
                    else pd.DataFrame()
                )
            self.materialized[rel] = seeded

    def _demand_columns(self, rel: str) -> list[str]:
        """Columns downstream consumers demand from ``rel`` (a usable schema proxy)."""
        cols: list[str] = []
        seen: set[str] = set()
        for cd in self._consumer_demands_map.get(rel, []) or []:
            for col in cd.get("required_columns", []) or []:
                name = col.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    cols.append(name)
        return cols

    # ------------------------------------------------------------------
    # Code generation (+ profiling)
    # ------------------------------------------------------------------

    def _codegen_all(self) -> None:
        layers = self._code_gen._topo_layers(self._atomic_root)
        for layer_idx, layer in enumerate(layers):
            for node in layer:
                try:
                    code_node, code_str = self._code_gen._codegen_layered_node(
                        node,
                        self.materialized,
                        [],
                        None,
                        self._cg_in,
                        self._consumer_demands_map,
                        layer_idx,
                    )
                except Exception as exc:  # noqa: BLE001 - one atom failing is not fatal
                    logger.warning(
                        "[base-plan] code-gen failed for %s: %s", node.op, exc
                    )
                    continue
                self.code_by_op[node.op] = code_str
                if self.profile:
                    self._run_on_sample(node, code_node)
        logger.info(
            "[base-plan] %d atom(s) code-gen'd on %s (k=%d)",
            len(self.code_by_op),
            "the profiling sample" if self.profile else "head(k) samples",
            self._k,
        )

    def _run_on_sample(self, node: FAONode, code_node) -> None:
        """Execute one atom on the sample; materialize its outputs for its consumers.

        On failure the cached code is dropped so the execution-time run regenerates
        it instead of reusing a function that already crashed once.
        """
        exec_ctx = dict(self.materialized)
        try:
            worker = self._worker_manager.get_worker()
            code_node.execute(exec_ctx, profile=True, worker=worker)
        except Exception as exc:  # noqa: BLE001 - profiling is best effort
            self._cg_in["_grouping_cache"].codegen.pop(node.op, None)
            logger.warning("[base-plan] sample run failed for %s: %s", node.op, exc)
            return
        for out in node.outputs:
            if out in exec_ctx:
                self.materialized[out] = exec_ctx[out]

    # ------------------------------------------------------------------
    # Statistics shown to the ranker
    # ------------------------------------------------------------------

    def column_value_stats(self, rel: str | None, cols: list[str], top_k: int = 3) -> str:
        """Distinct count, top values and null fraction of ``cols`` in the sample of ``rel``."""
        df = self.materialized.get(rel) if rel else None
        if not isinstance(df, pd.DataFrame) or df.empty:
            return ""
        chunks: list[str] = []
        for col in cols:
            if col not in df.columns:
                continue
            try:
                series = df[col]
                n = len(series)
                n_null = int(series.isna().sum())
                non_null = series.dropna()
                n_distinct = int(non_null.nunique())
                counts = non_null.value_counts().head(top_k)
                top_str = ", ".join(
                    f"{_truncate_value(v)}x{int(c)}" for v, c in counts.items()
                )
                piece = f"{col}: {n_distinct} distinct in sample"
                if top_str:
                    piece += f", top [{top_str}]"
                if n_null:
                    piece += f", {n_null / n * 100:.0f}% null"
                chunks.append(piece)
            except Exception:  # noqa: BLE001 - stats are best-effort
                continue
        return "values{ " + " | ".join(chunks) + " }" if chunks else ""

    def _build_full_cardinality(self) -> None:
        """Estimate each relation's FULL-data cardinality.

        Base tables: ``COUNT(*)``. Intermediates: ``full(primary input) × sample
        selectivity`` when profiled (selectivity = sample out-rows / in-rows of the
        largest input), else the primary input's count (unit selectivity).
        """
        producers = {o for n in self._by_op.values() for o in n.outputs if o}
        full: dict[str, float] = {}
        for n in self._by_op.values():
            for inp in n.inputs:
                if inp and inp not in producers and inp not in full:
                    full[inp] = self._base_count(inp)
        for layer in self._code_gen._topo_layers(self._atomic_root):
            for n in layer:
                in_fulls = [full[i] for i in n.inputs if i in full]
                primary_full = max(in_fulls) if in_fulls else float(self._k)
                in_samples = [
                    len(self.materialized[i]) for i in n.inputs if i in self.materialized
                ]
                in_sample = max(in_samples) if in_samples else self._k
                for o in n.outputs:
                    if not o or o in full:
                        continue
                    if self.profile and o in self.materialized:
                        sel = len(self.materialized[o]) / max(1, in_sample)
                        full[o] = primary_full * sel
                    else:
                        full[o] = primary_full
        self.full_cardinality = full

    def _base_count(self, rel: str) -> float:
        try:
            if self._rc.has_table(rel):
                row = self._rc.execute(f'SELECT COUNT(*) FROM "{rel}"').fetchone()
                if row:
                    return float(row[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[base-plan] COUNT(*) failed for %r: %s", rel, exc)
        df = self.materialized.get(rel)
        return float(len(df)) if df is not None else float(self._k)
