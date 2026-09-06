"""Save-time smoke gate outcomes."""

import inspect
import textwrap

from kathdb.common.fn_smoke import FAKE_CALL_TEXT_PARAMS, run_smoke


def test_fake_signature_matches_real():
    """The harness fake must track call_text's real signature."""
    from kathdb.common.litellm_sync import call_text

    real = [
        (p.name, None if p.default is inspect.Parameter.empty else p.default)
        for p in inspect.signature(call_text).parameters.values()
    ]
    assert real == list(FAKE_CALL_TEXT_PARAMS)


GOOD_FN = textwrap.dedent(
    """
    import pandas as pd
    from kathdb.common.litellm_sync import call_text

    CONTRACT = {"purpose": "label rows", "params": {}, "output": "df"}

    def label_rows(df: pd.DataFrame, text_col: str, model: str = "azure/gpt-4o-mini") -> pd.DataFrame:
        labels = []
        for _, row in df.iterrows():
            try:
                out = call_text(str(row[text_col]), model, [])
            except Exception:
                out = ""
            labels.append(out)
        return df.assign(label=labels)
    """
)

GOOD_SMOKE = textwrap.dedent(
    """
    def canned_response(prompt):
        return "pos" if "great" in prompt else "neg"

    def run(fn):
        import pandas as pd
        df = pd.DataFrame({"t": ["great movie", "bad movie"]})
        out = fn(df, text_col="t")
        assert list(out["label"]) == ["pos", "neg"], list(out["label"])
    """
)


def test_good_fn_passes():
    ok, detail = run_smoke(GOOD_FN, "label_rows", GOOD_SMOKE)
    assert ok, detail
    assert "2 model calls" in detail


def test_invented_kwarg_rejected():
    bad_fn = GOOD_FN.replace(
        "call_text(str(row[text_col]), model, [])",
        "call_text(str(row[text_col]), model, [], made_up_kwarg=1)",
    )
    ok, detail = run_smoke(bad_fn, "label_rows", GOOD_SMOKE)
    assert not ok
    assert "AssertionError" in detail or "ZERO model calls" in detail


def test_zero_model_calls_rejected():
    dead_fn = textwrap.dedent(
        """
        import pandas as pd
        from kathdb.common.litellm_sync import call_text

        CONTRACT = {"purpose": "label rows", "params": {}, "output": "df"}

        def label_rows(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
            return df.assign(label="")
        """
    )
    smoke = textwrap.dedent(
        """
        def canned_response(prompt):
            return "pos"

        def run(fn):
            import pandas as pd
            fn(pd.DataFrame({"t": ["x"]}), text_col="t")
        """
    )
    ok, detail = run_smoke(dead_fn, "label_rows", smoke)
    assert not ok
    assert "ZERO model calls" in detail


def test_relational_fn_needs_no_model_calls():
    rel_fn = textwrap.dedent(
        """
        import pandas as pd

        CONTRACT = {"purpose": "top-k rows", "params": {}, "output": "df"}

        def take_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
            return df.head(k)
        """
    )
    smoke = textwrap.dedent(
        """
        def canned_response(prompt):
            return ""

        def run(fn):
            import pandas as pd
            out = fn(pd.DataFrame({"a": [1, 2, 3]}), k=2)
            assert len(out) == 2
        """
    )
    ok, detail = run_smoke(rel_fn, "take_top_k", smoke)
    assert ok, detail


def test_missing_smoke_rejected():
    ok, detail = run_smoke(GOOD_FN, "label_rows", "")
    assert not ok
    assert "no smoke script" in detail


def test_model_default_must_be_verbatim():
    """A model default must be copied verbatim, provider prefix included."""
    from kathdb.common.function_finalizer import model_default_violation

    src = "out = call_text(p, 'azure/gpt-4o-mini', [])"
    ok_out = 'def f(model: str = "azure/gpt-4o-mini"): ...'
    bad_out = 'def f(model: str = "gpt-4o-mini"): ...'
    assert model_default_violation(ok_out, src) is None
    assert "gpt-4o-mini" in model_default_violation(bad_out, src)
    assert model_default_violation("def f(x: int = 3): ...", src) is None


def test_smoke_script_that_patches_scripts_fn_still_passes():
    """Smoke scripts that patch ``scripts.fn.call_text`` still pass."""
    smoke = textwrap.dedent(
        """
        def canned_response(prompt):
            return "pos" if "great" in prompt else "neg"

        def run(fn):
            import pandas as pd
            from unittest.mock import patch
            df = pd.DataFrame({"t": ["great movie", "bad movie"]})
            with patch(
                "scripts.fn.call_text",
                side_effect=lambda prompt, model, images, audios=None, **kw: canned_response(prompt),
            ):
                out = fn(df, text_col="t")
            assert list(out["label"]) == ["pos", "neg"], list(out["label"])
        """
    )
    ok, detail = run_smoke(GOOD_FN, "label_rows", smoke)
    assert ok, detail
