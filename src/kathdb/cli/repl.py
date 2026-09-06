"""Interactive shell for KathDB: type a question, or a slash command.

    $ kathdb                       # catalog in ./kathdb.duckdb
    kathdb> /register-data products.csv --image image_path
    kathdb> /model anthropic/claude-opus-5
    kathdb> Which products under 50 show a logo in their image?
    kathdb> /config logical_rewrite=false phy_opt=true

Commands: /help /model /register-data /tables /config /hitl /cost /plan /functions /export /clear-data /clear-fn /exit
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import os
import re
import shlex
import sys
import threading
import time
import typing
import warnings
from pathlib import Path
from typing import Any, Callable

from ..config import KathDBConfig, split_model_id

_BASIC = (
    "planner_model",
    "ai_op_model",
    "human_in_the_loop",
    "logical_rewrite",
    "phy_opt",
    "prebuilt_functions",
    "generated_functions",
    "max_generated_functions",
    "worker_env",
    "num_executor_workers",
)


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def accent(self, t: str) -> str:
        return self._wrap("38;5;208", t)  # orange

    def ok(self, t: str) -> str:
        return self._wrap("32", t)

    def err(self, t: str) -> str:
        return self._wrap("31", t)

    def key(self, t: str) -> str:
        return self._wrap("36", t)


class _Spinner:
    """Braille spinner on stderr while a long call runs (off when stdin is not a tty)."""

    def __init__(self, label: str, style: _Style, enabled: bool) -> None:
        self.label = label
        self._style = style
        self._enabled = enabled and sys.stderr.isatty()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r\033[K{self._style.accent(ch)} {self.label} ")
            sys.stderr.flush()
            time.sleep(0.08)
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def __enter__(self) -> "_Spinner":
        if self._enabled:
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._enabled:
            self._thread.join()


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _field_types() -> dict[str, Any]:
    hints = typing.get_type_hints(KathDBConfig)
    return {f.name: hints[f.name] for f in dataclasses.fields(KathDBConfig)}


def coerce_setting(name: str, raw: str) -> Any:
    """Convert a ``key=value`` string to the type of the config field ``name``."""
    types = _field_types()
    if name not in types:
        raise KeyError(f"unknown setting {name!r}; see /config")
    hint = types[name]
    args = typing.get_args(hint)
    base = next((a for a in args if a is not type(None)), hint) if args else hint
    value = raw.strip()
    if value.lower() in ("none", "null") and (type(None) in args):
        return None
    if base is bool:
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{name} expects true/false, got {raw!r}")
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    return value


def parse_settings(tokens: list[str]) -> dict[str, Any]:
    """``["k=v", "k2=v2"]`` -> typed overrides."""
    out: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected key=value, got {tok!r}")
        key, raw = tok.split("=", 1)
        out[key.strip()] = coerce_setting(key.strip(), raw)
    return out


def _register_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="/register-data", add_help=False)
    p.add_argument("path")
    p.add_argument("--name", help="table name (default: file stem)")
    p.add_argument("--image", action="append", default=[], help="column holding image paths")
    p.add_argument("--text", action="append", default=[], help="column holding long text")
    p.add_argument("--audio", action="append", default=[], help="column holding audio paths")
    p.add_argument("--video", action="append", default=[], help="column holding video paths")
    p.add_argument("--describe", help="one-line table description for the planner")
    return p


def parse_register_args(tokens: list[str]) -> argparse.Namespace:
    return _register_parser().parse_args(tokens)


_STAGE_LABELS = (
    ("Stage 1/3", "parsing the question"),
    ("Stage 2/3", "planning"),
    ("[base-plan]", "profiling the plan on a sample"),
    ("[list_rank]", "ranking groupings"),
    ("[optimizer]", "optimizing"),
    ("Stage 3/3", "generating code"),
    ("persist", "deciding what to keep"),
    ("[exec]", "executing"),
    ("[save]", "saving reusable functions"),
    ("finaliz", "saving reusable functions"),
)


def stage_label(message: str) -> str | None:
    """Progress label for a KathDB log line, or None if the line is not a stage marker."""
    m = re.search(r"\[codegen\] op=(\S+)", message)
    if m:
        return f"generating code for {m.group(1)}"
    m = re.search(r"\[run\] dispatched (\S+)", message)
    if m:
        return f"executing {m.group(1)}"
    for key, label in _STAGE_LABELS:
        if key in message:
            return label
    return None


_INTERACT_LEVEL = logging.INFO + 5  # KathDB's human-in-the-loop dialogue level


class _Progress(logging.Handler):
    """Feeds KathDB's stage log lines into the spinner label; prints its dialogue."""

    def __init__(self, spinner: _Spinner, style: _Style) -> None:
        super().__init__(logging.INFO)
        self._spinner = spinner
        self._style = style
        self.warnings: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno == _INTERACT_LEVEL:
            # Human-in-the-loop dialogue (clarification menu, plan review, save approval).
            print(self._style.key(message))
            return
        if record.levelno >= logging.WARNING:
            self.warnings.append(message.splitlines()[0][:160])
        label = stage_label(message)
        if label:
            self._spinner.label = label


_PLANNER_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "azure_anthropic": ("AZURE_ANTHROPIC_API_KEY", "AZURE_ANTHROPIC_ENDPOINT"),
}
_AI_OP_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "vertex_ai": ("GOOGLE_APPLICATION_CREDENTIALS",),
}


def required_env(model_id: str, *, planner: bool) -> tuple[str, ...]:
    """Environment variables a model id needs (first one is the API key)."""
    provider = model_id.split("/", 1)[0].lower() if "/" in model_id else "openai"
    return (_PLANNER_ENV if planner else _AI_OP_ENV).get(provider, ())


def missing_env(model_id: str, *, planner: bool) -> list[str]:
    return [k for k in required_env(model_id, planner=planner) if not os.environ.get(k)]


def load_dotenv(path: str | Path = ".env") -> int:
    """Export ``KEY=VALUE`` lines from *path*; variables already set win. Returns the count."""
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


def _quiet_library() -> None:
    """Show only warnings from KathDB (the shell renders progress itself)."""
    warnings.filterwarnings("ignore")
    from ..common.logger import configure_logger

    klog = configure_logger(level="WARNING")
    for h in klog.handlers:
        h.setLevel(logging.WARNING)


class KathDBShell:
    """REPL state: a lazily opened KathDB plus settings chosen before it opens."""

    def __init__(
        self,
        db_path: str | Path = "kathdb.duckdb",
        *,
        settings: dict[str, Any] | None = None,
        color: bool = True,
        verbose: bool = False,
        open_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.verbose = verbose
        # The function library sits next to the catalog, one per catalog.
        self.fn_dir = self.db_path.with_name(self.db_path.stem + "_functions")
        if not verbose:
            _quiet_library()
        self.pending: dict[str, Any] = dict(settings or {})
        self.db: Any = None
        self.style = _Style(color and sys.stdout.isatty())
        self._open_fn = open_fn
        self.commands: dict[str, tuple[Callable[[list[str]], None], str]] = {
            "/help": (self.cmd_help, "show this help"),
            "/model": (self.cmd_model, "[<planner id> | planner <id> | ai-op <id>]  show or set models"),
            "/register-data": (
                self.cmd_register,
                "<file.csv|file.parquet|dir> [--name N] [--image COL] [--text COL] ...",
            ),
            "/tables": (self.cmd_tables, "[name]  list tables or inspect one"),
            "/config": (self.cmd_config, "[key=value ...]  show or change any setting"),
            "/hitl": (self.cmd_hitl, "[on|off]  human in the loop: clarifying questions, plan review, save approval"),
            "/cost": (self.cmd_cost, "tokens / USD / seconds of the last query"),
            "/plan": (self.cmd_plan, "what the optimizer fused in the last query"),
            "/functions": (self.cmd_functions, "[name]  prebuilt + saved functions, or one function's docs and code"),
            "/export": (self.cmd_export, "[file.csv]  save the last answer"),
            "/clear-data": (self.cmd_clear_data, "[table]  drop one table, or every registered table"),
            "/clear-fn": (self.cmd_clear_fn, "[name]  delete one saved function, or all of them (prebuilt stay)"),
            "/exit": (self.cmd_exit, "quit"),
            "/quit": (self.cmd_exit, "quit"),
        }
        self._running = True
        self.last_df: Any = None

    # -- output helpers -----------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text)

    def error(self, text: str) -> None:
        print(self.style.err(f"✗ {text}"))

    # -- settings -----------------------------------------------------------

    def setting(self, name: str) -> Any:
        if self.db is not None:
            return getattr(self.db.config, name)
        if name in self.pending:
            return self.pending[name]
        return getattr(KathDBConfig(), name)

    def apply(self, overrides: dict[str, Any]) -> None:
        if self.db is None:
            KathDBConfig(**{**self.pending, **overrides}).validate()
            self.pending.update(overrides)
        else:
            self.db.configure(**overrides)

    def open(self) -> Any:
        if self.db is None:
            self.pending.setdefault("generated_fn_dir", str(self.fn_dir))
            opener = self._open_fn or self._default_open
            with _Spinner("starting KathDB (first start loads the LLM libraries)", self.style, enabled=True):
                self.db = opener(self.db_path, self.pending)
        return self.db

    @staticmethod
    def _default_open(db_path: Path, settings: dict[str, Any]) -> Any:
        from ..kathdb import KathDB

        basic = {k: v for k, v in settings.items() if k in _BASIC}
        advanced = {k: v for k, v in settings.items() if k not in _BASIC}
        return KathDB(db_path, config=KathDBConfig(**advanced), **basic)

    def _base_tables(self) -> list[str]:
        return [n for n in self.db.list_tables() if not self.db.is_view(n)]

    # -- status line --------------------------------------------------------

    def status(self) -> str:
        s = self.style
        n_tables = len(self._base_tables()) if self.db is not None else 0
        parts = [
            f"planner {s.key(str(self.setting('planner_model')))}",
            f"ai-op {s.key(str(self.setting('ai_op_model')))}",
            f"optimizer {s.ok('on') if self.setting('logical_rewrite') else s.dim('off')}",
            f"phy-opt {s.ok('on') if self.setting('phy_opt') else s.dim('off')}",
            f"hitl {s.ok('on') if self.setting('human_in_the_loop') else s.dim('off')}",
            f"tables {n_tables}",
        ]
        return s.dim("  ·  ").join(parts)

    # -- commands -----------------------------------------------------------

    def cmd_help(self, args: list[str]) -> None:
        self.say(self.style.bold("Type a question in plain English, or one of:"))
        for name, (_, usage) in self.commands.items():
            if name == "/quit":
                continue
            self.say(f"  {self.style.accent(name):<24} {usage}")

    def cmd_model(self, args: list[str]) -> None:
        api_key, check, rest = None, True, []
        it = iter(args)
        for tok in it:
            if tok == "--key":
                api_key = next(it, None)
            elif tok == "--no-check":
                check = False
            else:
                rest.append(tok)
        if not rest:
            for label, name in (("planner", "planner_model"), ("ai-op", "ai_op_model")):
                model = self.setting(name)
                miss = missing_env(model, planner=name == "planner_model")
                note = self.style.err(f"  missing {', '.join(miss)}") if miss else self.style.ok("  ✓ credentials set")
                self.say(f"{label + ':':<9}{model}{note}")
            return
        if rest[0] in ("ai-op", "ai_op"):
            key, model = "ai_op_model", " ".join(rest[1:])
        elif rest[0] == "planner":
            key, model = "planner_model", " ".join(rest[1:])
        else:
            key, model = "planner_model", " ".join(rest)
        if not model:
            raise ValueError("missing model id (provider/model)")
        planner = key == "planner_model"
        if planner:
            split_model_id(model)  # validates the provider
        needed = required_env(model, planner=planner)
        if api_key:
            if not needed:
                raise ValueError(f"no known credential variable for {model!r}")
            os.environ[needed[0]] = api_key
        miss = missing_env(model, planner=planner)
        if miss:
            self.say(self.style.err(f"  set {', '.join(miss)} in the environment or pass --key"))
        elif planner and check:
            self._ping_planner(model)
        self.apply({key: model})
        self.say(self.style.ok(f"✓ {key.replace('_', ' ')} = {model}"))
        self.say(self.style.dim(self.status()))

    def _ping_planner(self, model: str) -> None:
        from ..config import make_llm

        try:
            with _Spinner(f"checking {model}", self.style, enabled=True):
                make_llm(model).invoke("Reply with the single word OK.")
            self.say(self.style.ok("  ✓ model responds"))
        except Exception as exc:  # noqa: BLE001 - report, keep the setting
            self.say(self.style.err(f"  model check failed: {type(exc).__name__}: {str(exc)[:200]}"))

    def cmd_register(self, args: list[str]) -> None:
        if not args:
            self.cmd_tables([])
            return
        try:
            ns = parse_register_args(args)
        except SystemExit:
            raise ValueError(self.commands["/register-data"][1])
        from ..common.view_schema import Modality

        modalities = {}
        for col in ns.image:
            modalities[col] = Modality.IMAGE
        for col in ns.text:
            modalities[col] = Modality.TEXT
        for col in ns.audio:
            modalities[col] = Modality.AUDIO
        for col in ns.video:
            modalities[col] = Modality.VIDEO
        path = Path(ns.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        db = self.open()
        if path.is_dir():
            with _Spinner(f"discovering {path}", self.style, enabled=True):
                names = db.discover(path)
            self.say(self.style.ok(f"✓ registered {len(names)} table(s)"))
            self._table_summary(names)
            self.say(self.style.dim(self.status()))
            return
        name = ns.name or path.stem
        if path.suffix.lower() == ".csv":
            db.register_csv(path, name, column_modalities=modalities, description=ns.describe)
        elif path.suffix.lower() in (".parquet", ".pq"):
            db.register_parquet(path, name, column_modalities=modalities, description=ns.describe)
        else:
            raise ValueError("expected a .csv, .parquet file, or a directory")
        self.say(self.style.ok(f"✓ registered table {name}"))
        self._table_summary([name])
        self.say(self.style.dim(self.status()))

    def _table_summary(self, names: list[str]) -> None:
        for n in names:
            info = self.db.table_info(n)
            mods = ", ".join(f"{c}:{m}" for c, m in info["modalities"].items())
            cols = ", ".join(info["columns"][:8]) + (" …" if len(info["columns"]) > 8 else "")
            self.say(f"  {self.style.key(n):<32} {info['rows']:>7} rows   {cols}")
            if mods:
                self.say(self.style.dim(f"  {'':<32} media: {mods}"))

    def cmd_tables(self, args: list[str]) -> None:
        db = self.open()
        if args:
            db.inspect(args[0])
            return
        names = self._base_tables()
        if not names:
            self.say(self.style.dim("no tables yet; /register-data <file or dir>"))
            return
        self._table_summary(names)

    def cmd_config(self, args: list[str]) -> None:
        if args:
            overrides = parse_settings(args)
            self.apply(overrides)
            for k, v in overrides.items():
                self.say(self.style.ok(f"✓ {k} = {v}"))
            self.say(self.style.dim(self.status()))
            return
        defaults = KathDBConfig()
        self.say(self.style.bold("Basic settings"))
        for f in dataclasses.fields(KathDBConfig):
            if f.name == _BASIC[-1]:
                self._config_row(f.name, defaults)
                self.say(self.style.bold("Advanced settings"))
                continue
            self._config_row(f.name, defaults)

    def _config_row(self, name: str, defaults: KathDBConfig) -> None:
        value = self.setting(name)
        marker = "" if value == getattr(defaults, name) else self.style.accent("  (changed)")
        self.say(f"  {self.style.key(name):<44} {value!r}{marker}")

    def cmd_hitl(self, args: list[str]) -> None:
        if args:
            value = coerce_setting("human_in_the_loop", args[0])
        else:
            value = not bool(self.setting("human_in_the_loop"))
        self.apply({"human_in_the_loop": value})
        self.say(self.style.ok("✓ human in the loop " + ("on: KathDB will ask clarifying questions, show the plan for review and ask before saving functions" if value else "off: fully automatic")))
        self.say(self.style.dim(self.status()))

    def cmd_cost(self, args: list[str]) -> None:
        cost = self.db.last_cost() if self.db is not None else None
        if cost is None:
            self.say(self.style.dim("no query yet"))
            return
        self.say(cost.summary())

    def cmd_plan(self, args: list[str]) -> None:
        trace = self.db.last_grouping_trace() if self.db is not None else None
        if not trace:
            self.say(self.style.dim("no optimizer trace (no query yet, or logical_rewrite=false)"))
            return
        groups = trace.get("fused_groups") or []
        self.say(f"atomic operators: {trace.get('n_atoms')}  ·  candidate groupings: {trace.get('n_candidates')}  ·  fused groups: {len(groups)}")
        for g in groups:
            members = g.get("members", g) if isinstance(g, dict) else g
            self.say(f"  {self.style.accent('⊕')} " + " + ".join(map(str, members)))
            if isinstance(g, dict) and g.get("rationale"):
                self.say(self.style.dim(f"     {g['rationale']}"))
        if trace.get("short_circuit_reason"):
            self.say(self.style.dim(f"  ({trace['short_circuit_reason']})"))

    def cmd_functions(self, args: list[str]) -> None:
        db = self.open()
        if args:
            name = args[0]
            docs, code = db.function_docs(name), db.function_code(name)
            if docs is None and code is None:
                raise ValueError(f"no function {name!r}; /functions lists them")
            if docs:
                self.say(docs.rstrip())
                self.say()
            if code:
                self.say(self.style.bold("── scripts/fn.py ") + self.style.dim("─" * 44))
                self.say(code.rstrip())
            return
        entries = db.list_functions()
        if not entries:
            self.say(self.style.dim("function library is empty (functions are saved after queries when generated_functions=true)"))
            return
        for e in entries:
            tag = self.style.accent(e["source"]) if e["source"] == "prebuilt" else self.style.key(e["source"])
            uses = self.style.dim(f"  used {e['uses']}x") if e["source"] == "generated" else ""
            self.say(f"  {tag:<20} {self.style.bold(e['name'])}{uses}")
            if e["purpose"]:
                self.say(self.style.dim(f"      {e['purpose'][:110]}"))
            if e["members"]:
                self.say(self.style.dim(f"      from: {' + '.join(e['members'])}"))

    def _confirm(self, what: str) -> bool:
        if not sys.stdin.isatty():
            return True
        answer = input(self.style.accent(f"{what} [y/N] ")).strip().lower()
        return answer in ("y", "yes")

    def cmd_clear_data(self, args: list[str]) -> None:
        db = self.open()
        if args:
            if not db.has_table(args[0]):
                raise ValueError(f"no table {args[0]!r}")
            db.drop_table(args[0])
            self.say(self.style.ok(f"✓ dropped {args[0]}"))
        else:
            names = self._base_tables()
            if not names:
                self.say(self.style.dim("no tables to drop"))
                return
            if not self._confirm(f"drop {len(names)} table(s): {', '.join(names)}?"):
                return
            db.clear_tables()
            self.say(self.style.ok(f"✓ dropped {len(names)} table(s)"))
        self.say(self.style.dim(self.status()))

    def cmd_clear_fn(self, args: list[str]) -> None:
        db = self.open()
        if args:
            if not db.remove_function(args[0]):
                raise ValueError(f"no saved function {args[0]!r} (prebuilt functions cannot be removed)")
            self.say(self.style.ok(f"✓ removed {args[0]}"))
            return
        saved = [e["name"] for e in db.list_functions() if e["source"] == "generated"]
        if not saved:
            self.say(self.style.dim("no saved functions"))
            return
        if not self._confirm(f"delete {len(saved)} saved function(s): {', '.join(saved)}?"):
            return
        removed = db.clear_functions()
        self.say(self.style.ok(f"✓ removed {len(removed)} function(s)"))

    def cmd_export(self, args: list[str]) -> None:
        if self.last_df is None:
            raise RuntimeError("nothing to export yet")
        target = Path(args[0]) if args else Path(f"{self.db.last_result_name() or 'result'}.csv")
        self.last_df.to_csv(target, index=False)
        self.say(self.style.ok(f"✓ wrote {len(self.last_df)} rows to {target}"))

    def cmd_exit(self, args: list[str]) -> None:
        self._running = False

    # -- natural-language query ---------------------------------------------

    def run_query(self, question: str) -> None:
        db = self.open()
        if not db.list_tables():
            raise RuntimeError("no tables registered; /register-data <file or dir> first")
        spinner_on = not bool(self.setting("human_in_the_loop"))
        t0 = time.perf_counter()
        klog = logging.getLogger("kathdb")
        old_level = klog.level
        spinner = _Spinner("thinking", self.style, enabled=spinner_on)
        progress = _Progress(spinner, self.style)
        klog.addHandler(progress)
        if klog.level == logging.NOTSET or klog.level > logging.INFO:
            klog.setLevel(logging.INFO)
        muted = [] if self.verbose else [h for h in klog.handlers if h is not progress]
        old_handler_levels = [h.level for h in muted]
        for h in muted:
            h.setLevel(logging.ERROR)
        try:
            with spinner:
                relations = db.query(question)
        finally:
            klog.removeHandler(progress)
            klog.setLevel(old_level)
            for h, lvl in zip(muted, old_handler_levels):
                h.setLevel(lvl)
        wall = time.perf_counter() - t0
        result = db.last_result(relations)
        self.last_df = result
        self.say()
        if result is None:
            self.say(self.style.dim("(no result relation)"))
        else:
            self.say(self.style.bold(f"{db.last_result_name()}  ({len(result)} rows)"))
            self.say(result.head(20).to_string(index=False))
            if len(result) > 20:
                self.say(self.style.dim(f"  … {len(result) - 20} more rows"))
        if progress.warnings:
            shown = progress.warnings[:3]
            for w in shown:
                self.say(self.style.dim(f"⚠ {w}"))
            if len(progress.warnings) > 3:
                self.say(self.style.dim(f"⚠ … {len(progress.warnings) - 3} more (run with --verbose)"))
        cost = db.last_cost()
        if cost is not None:
            total = cost.totals()
            self.say(
                self.style.dim(
                    f"{wall:.1f}s · {total.total_tokens} tokens · ${total.money_cost_usd:.4f}"
                    "  (/cost for the breakdown, /plan for the optimizer)"
                )
            )

    # -- loop -----------------------------------------------------------------

    def dispatch(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if line.startswith("/"):
            try:
                tokens = shlex.split(line)
            except ValueError as exc:
                raise ValueError(f"cannot parse command: {exc}")
            name, args = tokens[0], tokens[1:]
            if name not in self.commands:
                raise ValueError(f"unknown command {name}; /help lists them")
            self.commands[name][0](args)
        else:
            self.run_query(line)

    def banner(self) -> None:
        from .. import __version__

        s = self.style
        self.say(s.bold("KathDB") + s.dim(f" v{__version__}") + s.dim("  ·  kdb shell"))
        self.say(
            s.dim("planner ") + s.key(str(self.setting("planner_model")))
            + s.dim("  ·  ai-op ") + s.key(str(self.setting("ai_op_model")))
        )
        self.say(s.dim(str(Path.cwd())))
        n_fns = sum(1 for p in self.fn_dir.iterdir() if p.is_dir() and not p.name.startswith("_")) if self.fn_dir.is_dir() else 0
        self.say(s.dim(f"catalog {self.db_path}  ·  functions {self.fn_dir} ({n_fns})  ·  optimizer ")
                 + (s.ok("on") if self.setting("logical_rewrite") else s.dim("off"))
                 + s.dim("  ·  phy-opt ") + (s.ok("on") if self.setting("phy_opt") else s.dim("off")))
        self.say()
        self.say(s.dim("Type a question over your tables, or /help. /register-data <dir> adds data."))
        self.say()

    def loop(self) -> int:
        self._install_readline()
        while self._running:
            try:
                line = input(self.style.accent("kdb> "))
            except EOFError:
                self.say()
                break
            except KeyboardInterrupt:
                self.say()
                continue
            try:
                self.dispatch(line)
            except KeyboardInterrupt:
                self.error("interrupted")
            except Exception as exc:  # noqa: BLE001 - the shell must survive any error
                self.error(f"{type(exc).__name__}: {exc}")
        if self.db is not None:
            self.db.close()
        return 0

    def _install_readline(self) -> None:
        try:
            import readline
        except ImportError:  # pragma: no cover - Windows
            return
        names = sorted(self.commands)

        def complete(text: str, state: int):
            matches = [n for n in names if n.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer_delims(" \t\n")
        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kdb", description="Interactive KathDB shell.")
    p.add_argument("--env-file", default=".env", help="KEY=VALUE file exported before start (default: ./.env)")
    p.add_argument("--db", default="kathdb.duckdb", help="catalog file (default: ./kathdb.duckdb)")
    p.add_argument("--planner-model", dest="planner_model", help="provider/model")
    p.add_argument("--ai-op-model", dest="ai_op_model", help="LiteLLM model id for the generated code")
    p.add_argument("--worker-env", dest="worker_env", help="existing conda env for the worker")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show KathDB's own log lines")
    p.add_argument("-c", "--command", action="append", default=[], help="run this line, then continue interactively")
    ns = p.parse_args(argv)
    load_dotenv(ns.env_file)
    settings = {
        k: v
        for k, v in vars(ns).items()
        if k in ("planner_model", "ai_op_model", "worker_env") and v
    }
    shell = KathDBShell(ns.db, settings=settings, color=not ns.no_color, verbose=ns.verbose)
    shell.banner()
    for line in ns.command:
        try:
            shell.dispatch(line)
        except Exception as exc:  # noqa: BLE001
            shell.error(f"{type(exc).__name__}: {exc}")
    return shell.loop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
