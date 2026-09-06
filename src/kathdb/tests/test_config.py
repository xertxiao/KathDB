"""Tests for KathDBConfig validation and the LLM factory."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kathdb.config import KathDBConfig, make_llm, split_model_id


class _FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_defaults_validate():
    cfg = KathDBConfig()
    cfg.validate()
    assert cfg.planner_model == "anthropic/claude-opus-5"
    assert cfg.ai_op_model == "openai/gpt-4o-mini"
    assert cfg.human_in_the_loop is False
    assert cfg.logical_rewrite is True
    assert cfg.phy_opt is False
    assert cfg.prebuilt_functions and cfg.generated_functions
    assert cfg.max_generated_functions == 10
    assert cfg.parser_type == "action_with_functions_with_coarsening"
    assert cfg.grouping_rank_k == 10
    assert cfg.grouping_max_group_size is None
    assert cfg.grouping_base_plan_profiling is True
    assert cfg.image_quality_low_ai_op is True
    assert cfg.num_executor_workers == 1
    assert cfg.persist_results is False


def test_split_model_id():
    assert split_model_id("anthropic/claude-opus-5") == ("anthropic", "claude-opus-5")
    assert split_model_id("gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert split_model_id("azure_anthropic/my-deploy") == ("azure_anthropic", "my-deploy")
    with pytest.raises(ValueError, match="provider"):
        split_model_id("nope/model")


def test_make_llm_forwards_default_temperature(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=_FakeChatModel)
    )
    llm = make_llm("openai/gpt-4o-mini")
    assert llm.kwargs["model"] == "gpt-4o-mini"
    assert llm.kwargs["temperature"] == 0.0
    assert llm.kwargs["max_tokens"] == 16384


def test_make_llm_omits_temperature_for_claude_5(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=_FakeChatModel)
    )
    assert "temperature" not in make_llm("anthropic/claude-opus-5").kwargs
    assert "temperature" not in make_llm("anthropic/claude-sonnet-5").kwargs
    assert make_llm("anthropic/claude-3-5-sonnet-latest").kwargs["temperature"] == 0.0


def test_make_llm_azure_anthropic_sets_max_tokens(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=_FakeChatModel)
    )
    monkeypatch.setenv("AZURE_ANTHROPIC_ENDPOINT", "https://x/")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "k")
    llm = make_llm("azure_anthropic/claude-opus-4-7")
    assert llm.kwargs["max_tokens"] == 16384 and llm.kwargs["base_url"] == "https://x"


def test_planner_model_is_validated():
    with pytest.raises(ValueError, match="provider"):
        KathDBConfig(planner_model="bogus/x").validate()
    with pytest.raises(ValueError, match="provider"):
        KathDBConfig(plan_gen_llm_model="bogus/x").validate()


def test_parser_type_is_validated():
    for t in ("action", "action_with_functions", "action_with_functions_with_coarsening"):
        KathDBConfig(parser_type=t).validate()
    with pytest.raises(ValueError, match="parser_type"):
        KathDBConfig(parser_type="bogus").validate()


def test_temperatures_and_timeouts_are_validated():
    with pytest.raises(ValueError, match="llm_temperature"):
        KathDBConfig(llm_temperature=-0.1).validate()
    with pytest.raises(ValueError, match="ai_op_temperature"):
        KathDBConfig(ai_op_temperature=-0.1).validate()
    with pytest.raises(ValueError, match="worker_exec_timeout_s"):
        KathDBConfig(worker_exec_timeout_s=0).validate()


def test_grouping_knobs_are_validated():
    with pytest.raises(ValueError, match="grouping_rank_k"):
        KathDBConfig(grouping_rank_k=1).validate()
    KathDBConfig(grouping_max_group_size=None).validate()
    with pytest.raises(ValueError, match="grouping_max_group_size"):
        KathDBConfig(grouping_max_group_size=0).validate()
    with pytest.raises(ValueError, match="max_generated_functions"):
        KathDBConfig(max_generated_functions=0).validate()


def test_update_rolls_back_on_invalid_override():
    cfg = KathDBConfig()
    with pytest.raises(ValueError):
        cfg.update(grouping_rank_k=1)
    assert cfg.grouping_rank_k == 10
    with pytest.raises(TypeError, match="Unknown config key"):
        cfg.update(grouping_strategy="list_rank")
    assert cfg.update(grouping_rank_k=5) == {"grouping_rank_k"}
