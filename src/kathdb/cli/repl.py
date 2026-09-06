"""Interactive shell for KathDB: type a question, or a slash command.

    $ kathdb                       # catalog in ./kathdb.duckdb
    kathdb> /register-data products.csv --image image_path
    kathdb> /model anthropic/claude-opus-5
    kathdb> Which products under 50 show a logo in their image?
    kathdb> /config logical_rewrite=false phy_opt=true

Commands: /help /model /register-data /tables /config /cost /plan /exit
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import shlex
import sys
import threading
import time
import typing
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
        self._label = label
        self._style = style
        self._enabled = enabled and sys.stderr.isatty()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r{self._style.accent(ch)} {self._label} ")
            sys.stderr.flush()
            time.sleep(0.08)
        sys.stderr.write("\r" + " " * (len(self._label) + 4) + "\r")
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


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


class KathDBShell:
    """REPL state: a lazily opened KathDB plus settings chosen before it opens."""

    def __init__(
        self,
        db_path: str | Path = "kathdb.duckdb",
        *,
        settings: dict[str, Any] | None = None,
        color: bool = True,
        open_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
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
            "/cost": (self.cmd_cost, "tokens / USD / seconds of the last query"),
            "/plan": (self.cmd_plan, "what the optimizer fused in the last query"),
            "/exit": (self.cmd_exit, "quit"),
            "/quit": (self.cmd_exit, "quit"),
        }
        self._running = True

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
            opener = self._open_fn or self._default_open
            with _Spinner("starting KathDB", self.style, enabled=True):
                self.db = opener(self.db_path, self.pending)
        return self.db

    @staticmethod
    def _default_open(db_path: Path, settings: dict[str, Any]) -> Any:
        from ..kathdb import KathDB

        basic = {k: v for k, v in settings.items() if k in _BASIC}
        advanced = {k: v for k, v in settings.items() if k not in _BASIC}
        config = KathDBConfig(**advanced) if advanced else None
        return KathDB(db_path, config=config, **basic)

    # -- status line --------------------------------------------------------

    def status(self) -> str:
        s = self.style
        n_tables = len(self.db.list_tables()) if self.db is not None else 0
        parts = [
            f"planner {s.key(str(self.setting('planner_model')))}",
            f"ai-op {s.key(str(self.setting('ai_op_model')))}",
            f"optimizer {s.ok('on') if self.setting('logical_rewrite') else s.dim('off')}",
            f"phy-opt {s.ok('on') if self.setting('phy_opt') else s.dim('off')}",
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
        if not args:
            self.say(f"planner: {self.setting('planner_model')}")
            self.say(f"ai-op:   {self.setting('ai_op_model')}")
            return
        if args[0] in ("ai-op", "ai_op"):
            key, model = "ai_op_model", " ".join(args[1:])
        elif args[0] == "planner":
            key, model = "planner_model", " ".join(args[1:])
        else:
            key, model = "planner_model", " ".join(args)
        if not model:
            raise ValueError("missing model id (provider/model)")
        if key == "planner_model":
            split_model_id(model)  # validates the provider
        self.apply({key: model})
        self.say(self.style.ok(f"✓ {key.replace('_', ' ')} = {model}"))
        self.say(self.style.dim(self.status()))

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
            names = db.discover(path)
            self.say(self.style.ok(f"✓ registered {len(names)} table(s): {', '.join(names)}"))
            return
        name = ns.name or path.stem
        if path.suffix.lower() == ".csv":
            db.register_csv(path, name, column_modalities=modalities, description=ns.describe)
        elif path.suffix.lower() in (".parquet", ".pq"):
            db.register_parquet(path, name, column_modalities=modalities, description=ns.describe)
        else:
            raise ValueError("expected a .csv, .parquet file, or a directory")
        self.say(self.style.ok(f"✓ registered table {name}"))
        self.say(self.style.dim(self.status()))

    def cmd_tables(self, args: list[str]) -> None:
        db = self.open()
        if args:
            db.inspect(args[0])
            return
        names = db.list_tables()
        if not names:
            self.say(self.style.dim("no tables yet; /register-data <file or dir>"))
            return
        for n in names:
            self.say(f"  {n}")

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
        self.say(f"atoms: {trace.get('n_atoms')}  candidates: {trace.get('n_candidates')}  fused groups: {len(groups)}")
        for g in groups:
            self.say(f"  {self.style.accent('⊕')} {g}")
        if trace.get("short_circuit_reason"):
            self.say(self.style.dim(f"  ({trace['short_circuit_reason']})"))

    def cmd_exit(self, args: list[str]) -> None:
        self._running = False

    # -- natural-language query ---------------------------------------------

    def run_query(self, question: str) -> None:
        db = self.open()
        if not db.list_tables():
            raise RuntimeError("no tables registered; /register-data <file or dir> first")
        spinner_on = not bool(self.setting("human_in_the_loop"))
        t0 = time.perf_counter()
        with _Spinner("thinking", self.style, enabled=spinner_on):
            relations = db.query(question)
        wall = time.perf_counter() - t0
        result = db.last_result(relations)
        self.say()
        if result is None:
            self.say(self.style.dim("(no result relation)"))
        else:
            self.say(self.style.bold(f"{db.last_result_name()}  ({len(result)} rows)"))
            self.say(result.head(20).to_string(index=False))
            if len(result) > 20:
                self.say(self.style.dim(f"  … {len(result) - 20} more rows"))
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

    def loop(self) -> int:
        self._install_readline()
        self.say(self.style.bold("KathDB") + self.style.dim("  ·  ask questions over your tables; /help for commands"))
        self.say(self.style.dim(self.status()))
        while self._running:
            try:
                line = input(self.style.accent("kathdb> "))
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
    p = argparse.ArgumentParser(prog="kathdb", description="Interactive KathDB shell.")
    p.add_argument("--db", default="kathdb.duckdb", help="catalog file (default: ./kathdb.duckdb)")
    p.add_argument("--planner-model", dest="planner_model", help="provider/model")
    p.add_argument("--ai-op-model", dest="ai_op_model", help="LiteLLM model id for the generated code")
    p.add_argument("--worker-env", dest="worker_env", help="existing conda env for the worker")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("-c", "--command", action="append", default=[], help="run this line, then continue interactively")
    ns = p.parse_args(argv)
    settings = {
        k: v
        for k, v in vars(ns).items()
        if k in ("planner_model", "ai_op_model", "worker_env") and v
    }
    shell = KathDBShell(ns.db, settings=settings, color=not ns.no_color)
    for line in ns.command:
        try:
            shell.dispatch(line)
        except Exception as exc:  # noqa: BLE001
            shell.error(f"{type(exc).__name__}: {exc}")
    return shell.loop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
