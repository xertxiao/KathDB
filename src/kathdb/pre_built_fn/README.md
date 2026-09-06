# Pre-built functions

Hand-written operators the planner may pick and the code generator may call
(`from kathdb.fn import <name>`). This folder ships **empty**; add your own.

Each function is a folder:

```
pre_built_fn/<name>/
  fn.md            LLM-facing documentation (generated from CONTRACT; see below)
  scripts/
    __init__.py    from .fn import <name>
    fn.py          the implementation: typed signature + a module-level CONTRACT dict
```

`scripts/fn.py` is the single source of truth: a fully typed `def <name>(...)`
whose DataFrame parameters are `pd.DataFrame` and a `CONTRACT` dict (purpose,
per-parameter docs, output, `use_when` / `not_when`, an example). `fn.md` is
rendered from it with `FunctionManager.save_function` /
`kathdb.common.fn_contract.render_fn_md`. Model calls go through
`kathdb.common.model_call.call_model`.

`_example/sem_map/` is a complete worked example (underscore-prefixed folders are
not discovered); it is also the template the function finalizer shows the LLM when
it turns a query-time function into a reusable one. `kathdb.common.fn_helpers` holds
shared helpers (image/audio loading, prompt rendering, modality lookup).
