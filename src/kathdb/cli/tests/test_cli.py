"""Shell command parsing and dispatch (no LLM, no worker)."""

from __future__ import annotations

import pandas as pd
import pytest

from kathdb.cli.repl import KathDBShell, coerce_setting, parse_register_args, parse_settings
from kathdb.common.view_schema import Modality
from kathdb.config import KathDBConfig


class _FakeDB:
    def __init__(self, db_path, settings):
        self.config = KathDBConfig(**settings)
        self.tables = {}
        self.queries = []
        self.closed = False

    def configure(self, **overrides):
        self.config.update(**overrides)

    def list_tables(self):
        return list(self.tables)

    def register_csv(self, path, name, *, column_modalities=None, description=None):
        self.tables[name] = column_modalities or {}

    def register_parquet(self, path, name, *, column_modalities=None, description=None):
        self.tables[name] = column_modalities or {}

    def discover(self, root):
        self.tables["found"] = {}
        return ["found"]

    def query(self, q):
        self.queries.append(q)
        return {"answer": pd.DataFrame({"id": [1, 2]})}

    def last_result(self, ctx):
        return ctx["answer"]

    def last_result_name(self):
        return "answer"

    def last_cost(self):
        return None

    def last_grouping_trace(self):
        return {"n_atoms": 2, "n_candidates": 1, "fused_groups": [["a", "b"]]}

    def inspect(self, name):
        pass

    def is_view(self, name):
        return False

    def has_table(self, name):
        return name in self.tables

    def drop_table(self, name):
        self.tables.pop(name)

    def clear_tables(self):
        names = list(self.tables); self.tables.clear(); return names

    def remove_function(self, name):
        return name == "saved_fn"

    def clear_functions(self):
        return ["saved_fn"]

    def list_functions(self):
        return [{"name": "sem_map", "source": "prebuilt", "purpose": "Map", "uses": 0, "members": []},
                {"name": "saved_fn", "source": "generated", "purpose": "Saved", "uses": 2, "members": ["a", "b"]}]

    def function_docs(self, name):
        return "# sem_map\n\nMap docs" if name == "sem_map" else None

    def function_code(self, name):
        return "def sem_map(df): ..." if name == "sem_map" else None

    def table_info(self, name):
        return {"rows": 1, "columns": ["id"], "modalities": {k: v.value for k, v in self.tables[name].items()}}

    def close(self):
        self.closed = True


def _shell(tmp_path):
    return KathDBShell(tmp_path / "c.duckdb", color=False, open_fn=_FakeDB)


def test_coerce_setting_uses_config_field_types():
    assert coerce_setting("logical_rewrite", "false") is False
    assert coerce_setting("grouping_rank_k", "5") == 5
    assert coerce_setting("llm_temperature", "0.5") == 0.5
    assert coerce_setting("grouping_max_group_size", "none") is None
    assert coerce_setting("planner_model", "openai/gpt-4o") == "openai/gpt-4o"
    with pytest.raises(KeyError):
        coerce_setting("nope", "1")
    with pytest.raises(ValueError):
        coerce_setting("phy_opt", "maybe")


def test_parse_settings_requires_key_value():
    assert parse_settings(["phy_opt=true", "grouping_rank_k=3"]) == {
        "phy_opt": True,
        "grouping_rank_k": 3,
    }
    with pytest.raises(ValueError):
        parse_settings(["phy_opt"])


def test_parse_register_args_modalities():
    ns = parse_register_args(["p.csv", "--image", "img", "--text", "body", "--name", "t"])
    assert ns.path == "p.csv" and ns.name == "t" and ns.image == ["img"] and ns.text == ["body"]


def test_settings_before_open_are_applied_at_open(tmp_path):
    sh = _shell(tmp_path)
    sh.dispatch("/model anthropic/claude-opus-5")
    sh.dispatch("/model ai-op openai/gpt-4o")
    sh.dispatch("/config logical_rewrite=false phy_opt=true")
    assert sh.db is None
    assert sh.setting("logical_rewrite") is False
    db = sh.open()
    assert db.config.planner_model == "anthropic/claude-opus-5"
    assert db.config.ai_op_model == "openai/gpt-4o"
    assert db.config.logical_rewrite is False and db.config.phy_opt is True


def test_invalid_setting_before_open_is_rejected(tmp_path):
    sh = _shell(tmp_path)
    with pytest.raises(ValueError):
        sh.dispatch("/config grouping_rank_k=1")
    with pytest.raises(ValueError):
        sh.dispatch("/model bogus/model")
    assert "grouping_rank_k" not in sh.pending


def test_config_after_open_reconfigures_live(tmp_path):
    sh = _shell(tmp_path)
    sh.open()
    sh.dispatch("/config grouping_rank_k=7")
    assert sh.db.config.grouping_rank_k == 7


def test_register_data_csv_and_dir(tmp_path):
    sh = _shell(tmp_path)
    (tmp_path / "products.csv").write_text("id,image_path\n1,a.jpg\n")
    sh.dispatch(f"/register-data {tmp_path / 'products.csv'} --image image_path")
    assert sh.db.tables["products"] == {"image_path": Modality.IMAGE}
    sh.dispatch(f"/register-data {tmp_path}")
    assert "found" in sh.db.tables
    with pytest.raises(FileNotFoundError):
        sh.dispatch("/register-data missing.csv")


def test_query_requires_tables_then_runs(tmp_path, capsys):
    sh = _shell(tmp_path)
    sh.open()
    with pytest.raises(RuntimeError):
        sh.dispatch("Which rows?")
    sh.db.tables["t"] = {}
    sh.dispatch("Which rows?")
    assert sh.db.queries == ["Which rows?"]
    out = capsys.readouterr().out
    assert "answer  (2 rows)" in out


def test_unknown_command_and_exit(tmp_path):
    sh = _shell(tmp_path)
    with pytest.raises(ValueError):
        sh.dispatch("/nope")
    sh.dispatch("/exit")
    assert sh._running is False


def test_load_dotenv_and_missing_env(tmp_path, monkeypatch):
    from kathdb.cli.repl import load_dotenv, missing_env

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# keys\nexport ANTHROPIC_API_KEY='abc'\nAZURE_API_KEY=x\n\nbad line\n")
    assert load_dotenv(env) == 2
    assert missing_env("anthropic/claude-opus-5", planner=True) == []
    assert missing_env("azure/gpt-4o-mini", planner=False) == ["AZURE_API_BASE", "AZURE_API_VERSION"]
    assert load_dotenv(tmp_path / "nope") == 0


def test_model_key_flag_sets_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sh = _shell(tmp_path)
    sh.dispatch("/model openai/gpt-4o --key sk-test --no-check")
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert sh.setting("planner_model") == "openai/gpt-4o"


def test_stage_labels():
    from kathdb.cli.repl import stage_label

    assert stage_label("=== Stage 2/3: Plan Gen ===") == "planning"
    assert stage_label("[codegen] op=classify_x layer=1 codegen_time=1s") == "generating code for classify_x"
    assert stage_label("[run] dispatched g_1 (level 0; 1 running, 0 waiting)") == "executing g_1"
    assert stage_label("LLM persistence decision failed") == "deciding what to keep"
    assert stage_label("unrelated") is None


def test_export_after_query(tmp_path):
    sh = _shell(tmp_path)
    sh.open()
    sh.db.tables["t"] = {}
    with pytest.raises(RuntimeError):
        sh.dispatch("/export")
    sh.dispatch("Which rows?")
    out = tmp_path / "out.csv"
    sh.dispatch(f"/export {out}")
    assert out.read_text().splitlines() == ["id", "1", "2"]


def test_functions_commands(tmp_path, capsys):
    sh = _shell(tmp_path)
    sh.open()
    sh.dispatch("/functions")
    out = capsys.readouterr().out
    assert "sem_map" in out and "saved_fn" in out and "used 2x" in out and "a + b" in out
    sh.dispatch("/functions sem_map")
    assert "Map docs" in capsys.readouterr().out
    with pytest.raises(ValueError):
        sh.dispatch("/functions nope")


def test_clear_data_and_clear_fn(tmp_path, capsys):
    sh = _shell(tmp_path)
    sh.open()
    sh.db.tables.update({"t1": {}, "t2": {}})
    sh.dispatch("/clear-data t1")
    assert list(sh.db.tables) == ["t2"]
    sh.dispatch("/clear-data")  # stdin is not a tty under pytest -> no confirmation prompt
    assert sh.db.tables == {}
    with pytest.raises(ValueError):
        sh.dispatch("/clear-data nope")
    sh.dispatch("/clear-fn saved_fn")
    with pytest.raises(ValueError):
        sh.dispatch("/clear-fn sem_map")
    sh.dispatch("/clear-fn")
    assert "removed 1 function" in capsys.readouterr().out


def test_hitl_toggle(tmp_path):
    sh = _shell(tmp_path)
    sh.dispatch("/hitl")
    assert sh.setting("human_in_the_loop") is True
    sh.dispatch("/hitl off")
    assert sh.setting("human_in_the_loop") is False
    sh.open()
    sh.dispatch("/hitl on")
    assert sh.db.config.human_in_the_loop is True


def test_function_library_sits_next_to_catalog(tmp_path):
    sh = _shell(tmp_path)
    db = sh.open()
    assert db.config.generated_fn_dir == str(tmp_path / "c_functions")


def test_hitl_dialogue_is_printed_during_query(tmp_path, capsys):
    import logging

    class _TalkingDB(_FakeDB):
        def query(self, q):
            logging.getLogger("kathdb.test").log(logging.INFO + 5, "❓ Clarification needed")
            logging.getLogger("kathdb.test").info("=== Stage 1/3: Parsing ===")
            return super().query(q)

    sh = KathDBShell(tmp_path / "c.duckdb", color=False, open_fn=_TalkingDB)
    sh.open()
    sh.db.tables["t"] = {}
    sh.dispatch("/hitl on")
    sh.dispatch("Which rows?")
    out = capsys.readouterr().out
    assert "Clarification needed" in out and "Stage 1/3" not in out
