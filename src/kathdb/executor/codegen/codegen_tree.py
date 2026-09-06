"""FAOExecutableNode — executable tree structure for physical query plans."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import textwrap
import shlex
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ...common.logger import get_logger
from ...common.utils import message_to_text
from .utils import normalize_quotes
from langchain_core.language_models import BaseChatModel

from ...worker import WorkerClient

__all__ = [
    "FAOExecutionError",
    "FAOFunction",
    "FAOExecutableNode",
    "parse_llm_function",
    "walk_nodes",
    "find_node",
]

_RUNTIME_MANIFEST_FILENAME = "_kathdb_runtime_manifest.json"
_RUNTIME_MANIFEST_CACHE: dict[str, Any] | None = None
_RUNTIME_SCRIPT_CACHE: dict[tuple[str, str], str] = {}
# Serializes manifest/cache mutation and runner-script writes across codegen threads.
_RUNTIME_MANIFEST_LOCK = threading.Lock()


logger = get_logger(__name__)


class FAOExecutionError(RuntimeError):
    """Raised when a physical operator cannot be executed."""


@dataclass(slots=True)
class FAOFunction:
    """Simple wrapper for an executable Python callable."""

    name: str
    impl: Callable[..., Any]
    str_impl: str | None = None
    description: str | None = None
    requirements: tuple[str, ...] = ()
    pip_commands: tuple[str, ...] = ()
    script_path: Path | None = None

    def __call__(self, **kwargs: Any) -> Any:
        return self.impl(**kwargs)

    def __repr__(self) -> str:
        return f"FAOFunction(name={self.name!r}, impl={self.impl!r})"


@dataclass(slots=True)
class FAOExecutableNode:
    """Basic tree node that executes a callable and stores its outputs."""

    op: str
    function: FAOFunction
    implementation_guidance: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    children: list["FAOExecutableNode"] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "FAOExecutableNode") -> "FAOExecutableNode":
        self.children.append(child)
        return child

    def extend(self, nodes: Iterable["FAOExecutableNode"]) -> None:
        self.children.extend(nodes)

    def execute(
        self,
        context: dict[str, Any],
        *,
        profile: bool = False,
        worker: WorkerClient | None = None,
    ) -> dict[str, Any]:
        """Execute this node: run children, execute function, store results.

        Args:
            context: Execution context with input/output relations.
            profile: If True, only execute this node (not children).
            worker: The worker subprocess client for isolated execution.
                    Required when executing with isolation.
        """
        if worker is None:
            raise ValueError("worker is required for execution")

        ctx: dict[str, Any] = context

        # Execute child nodes first unless profiling only this node
        child_names = [child.op for child in self.children]
        logger.info(f"[cg_tree] Enter `{self.op}` children={child_names}")
        if not profile:
            for child in self.children:
                # Skip if child's outputs are already computed (DAG shared node)
                if child.outputs and all(o in ctx for o in child.outputs):
                    logger.info(f"[cg_tree] -> skip child `{child.op}` (outputs already in ctx)")
                    continue
                logger.info(
                    f"[cg_tree] -> dispatch child `{child.op}` from parent `{self.op}`"
                )
                child.execute(ctx, worker=worker)

        # kwargs = DataFrame inputs resolved via ``bindings``.
        kwargs: dict[str, Any] = {}
        for param_name, ctx_key in self.arguments.get("bindings", {}).items():
            if ctx_key not in ctx:
                raise FAOExecutionError(
                    f"Input '{ctx_key}' not found in context "
                    f"(needed for parameter '{param_name}')"
                )
            kwargs[param_name] = ctx[ctx_key]

        # Execute the function via the worker
        logger.info(
            f"[cg_tree] Executing `{self.op}` with inputs: {list(kwargs.keys())}"
        )
        result = self.function(worker=worker, **kwargs)

        # Store results back to context
        self._store_result(result, ctx)
        logger.info(
            f"[cg_tree] <- `{self.op}` stored outputs {self.outputs} "
            f"(ctx keys now: {list(ctx.keys())})"
        )
        return ctx

    def replace_children(self, nodes: Iterable["FAOExecutableNode"]) -> None:
        """Replace child operators, typically after an LLM-driven refinement."""

        self.children = list(nodes)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "op": self.op,
            "function": self.function.name,
            "children": [child.to_dict() for child in self.children],
        }
        if self.implementation_guidance:
            data["implementation_guidance"] = self.implementation_guidance
        if self.inputs:
            data["inputs"] = list(self.inputs)
        if self.outputs:
            data["outputs"] = list(self.outputs)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.arguments:
            data["arguments"] = self.arguments
        return data

    def pretty(self, indent: str = "", style: str = "text") -> str:
        if style.lower() != "text":
            raise ValueError(f"Unsupported pretty style: {style!r}")
        lines = [f"{indent}{self.op}"]
        lines.append(f"{indent}  function: {self.function.name}")
        if self.implementation_guidance:
            lines.append(
                f"{indent}  implementation_guidance: {self.implementation_guidance}"
            )
        if self.inputs:
            rendered_inputs = ", ".join(self.inputs)
            lines.append(f"{indent}  inputs: {rendered_inputs}")
        if self.outputs:
            lines.append(f"{indent}  outputs: {', '.join(self.outputs)}")
        for child in self.children:
            lines.append(child.pretty(indent + "    ", style="text"))
        return "\n".join(lines)

    def _store_result(self, result: Any, context: dict[str, Any]) -> None:
        """Store function result directly into the context dict."""
        if not self.outputs:
            return

        if isinstance(result, list):
            if len(result) != len(self.outputs):
                raise FAOExecutionError(
                    f"Function {self.function.name!r} returned {len(result)} values; "
                    f"{len(self.outputs)} expected."
                )
            for name, value in zip(self.outputs, result):
                context[name] = value
            return

        if len(self.outputs) != 1:
            raise FAOExecutionError(
                f"Function {self.function.name!r} returned a single value; "
                f"{len(self.outputs)} outputs expected."
            )

        context[self.outputs[0]] = result

    def save(self, path: str) -> None:
        """Serialize this node and its children (code, metadata, dependencies) to a JSON file."""
        payload = _node_to_dict(self)
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved physical plan tree to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        llm: BaseChatModel | None = None,
    ) -> "FAOExecutableNode":
        """Rebuild a tree saved by :meth:`save` (``llm`` optionally fixes parse errors)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        root_node = _dict_to_node(payload, llm=llm)
        logger.info("Loaded physical plan tree from %s", path)
        return root_node


def parse_llm_function(
    payload: str,
    *,
    max_attempts: int = 3,
    llm: BaseChatModel | None = None,
    required_packages: list[str] | None = None,
    pip_install_commands: list[str] | None = None,
) -> FAOFunction:
    """Compile LLM-emitted Python into a :class:`FAOFunction`; ``llm`` may fix parse
    errors for up to ``max_attempts``. Explicit ``required_packages`` /
    ``pip_install_commands`` override those extracted from the code."""
    attempt_payload = payload
    for attempt in range(1, max_attempts + 1):
        code = _strip_fence(attempt_payload)
        code = normalize_quotes(code)
        try:
            module = ast.parse(code)
        except SyntaxError as exc:
            if llm is not None and attempt < max_attempts:
                attempt_payload = _request_function_parse_fix(llm, code, exc)
                continue
            raise FAOExecutionError(f"Failed to parse LLM function: {exc}") from exc

        target_fn = next(
            (node for node in module.body if isinstance(node, ast.FunctionDef)), None
        )

        if target_fn is None:
            error = FAOExecutionError("LLM response did not define a Python function.")
            if llm is not None and attempt < max_attempts:
                attempt_payload = _request_function_parse_fix(llm, code, error)
                continue
            raise error

        fn_name = target_fn.name
        description = ast.get_docstring(target_fn)
        description = description.strip() if description else None

        if required_packages is not None:
            requirements = list(required_packages)
        else:
            requirements = _extract_string_list(module, "REQUIRED_PACKAGES")

        if pip_install_commands is not None:
            pip_commands = list(pip_install_commands)
        else:
            pip_commands = _extract_string_list(module, "PIP_INSTALL_COMMANDS")

        if pip_commands and not requirements:
            requirements = _infer_packages_from_commands(pip_commands)
        if not pip_commands and requirements:
            pip_commands = [f"pip install {spec}" for spec in requirements]
        runner, script_path = _build_isolated_runner(
            code, fn_name, requirements, pip_commands
        )
        return FAOFunction(
            name=fn_name,
            impl=runner,
            str_impl=code,
            description=description,
            requirements=tuple(requirements),
            pip_commands=tuple(pip_commands),
            script_path=script_path,
        )

    raise FAOExecutionError("Failed to parse function after retries.")


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    if body.endswith("```"):
        body = body[:-3]
    body = body.lstrip("\n")
    if "\n" not in body:
        return body.strip()
    first_line, remainder = body.split("\n", 1)
    if first_line.strip() == "" or first_line.strip().isidentifier():
        code = remainder
    else:
        code = body
    return code.strip()


def _extract_string_list(module: ast.Module, name: str) -> list[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == name:
                    strings = _literal_string_list(node.value)
                    if strings is not None:
                        return strings
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                if node.value is None:
                    return []
                strings = _literal_string_list(node.value)
                if strings is not None:
                    return strings
    return []


def _literal_string_list(value: ast.AST | None) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        items: list[str] = []
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                items.append(element.value)
        return items
    return None


def _infer_packages_from_commands(commands: Iterable[str]) -> list[str]:
    packages: list[str] = []
    for command in commands:
        tokens = shlex.split(command)
        if not tokens:
            continue
        idx = 0
        # Strip leading python -m pip style launcher.
        if tokens[0].lower().startswith("python"):
            if len(tokens) >= 4 and tokens[1] == "-m" and tokens[2] == "pip":
                idx = 3
        elif tokens[0] in {"pip", "pip3"}:
            idx = 1
        while idx < len(tokens) and tokens[idx] != "install":
            idx += 1
        if idx >= len(tokens) or tokens[idx] != "install":
            continue
        idx += 1
        while idx < len(tokens):
            token = tokens[idx]
            idx += 1
            if token.startswith("-"):
                continue
            packages.append(token)
    return packages


def _runtime_dir() -> Path:
    override = os.environ.get("KATHDB_RUNTIME_DIR")
    path = Path(override) if override else Path.cwd() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(runtime_dir: Path) -> Path:
    return runtime_dir / _RUNTIME_MANIFEST_FILENAME


def _load_runtime_manifest(runtime_dir: Path) -> dict[str, Any]:
    global _RUNTIME_MANIFEST_CACHE
    if _RUNTIME_MANIFEST_CACHE is not None:
        return _RUNTIME_MANIFEST_CACHE
    manifest_file = _manifest_path(runtime_dir)
    if manifest_file.exists():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _RUNTIME_MANIFEST_CACHE = data
            else:
                _RUNTIME_MANIFEST_CACHE = {}
        except Exception:  # noqa: BLE001
            _RUNTIME_MANIFEST_CACHE = {}
    else:
        _RUNTIME_MANIFEST_CACHE = {}
    return _RUNTIME_MANIFEST_CACHE


def _save_runtime_manifest(runtime_dir: Path) -> None:
    if _RUNTIME_MANIFEST_CACHE is None:
        return
    manifest_file = _manifest_path(runtime_dir)
    try:
        manifest_file.write_text(
            json.dumps(_RUNTIME_MANIFEST_CACHE, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        # Best-effort persistence; ignore failures.
        pass


def _next_version_tag(entrypoint: str, source: str, runtime_dir: Path) -> str:
    with _RUNTIME_MANIFEST_LOCK:
        manifest = _load_runtime_manifest(runtime_dir)
        cache_key = (entrypoint, source)
        cached = _RUNTIME_SCRIPT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        source_hash = hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()
        entry = manifest.setdefault(
            entrypoint,
            {"hash_to_version": {}, "next_version": 0},
        )
        hash_to_version = entry.setdefault("hash_to_version", {})
        version_idx = hash_to_version.get(source_hash)
        if version_idx is None:
            next_version = int(entry.get("next_version", 0))
            version_idx = next_version
            hash_to_version[source_hash] = version_idx
            entry["next_version"] = next_version + 1
            _save_runtime_manifest(runtime_dir)

        version_tag = f"v{int(version_idx)}"
        _RUNTIME_SCRIPT_CACHE[cache_key] = version_tag
        return version_tag


def _build_isolated_runner(
    source: str,
    entrypoint: str,
    requirements: Sequence[str],
    pip_commands: Sequence[str],
) -> tuple[Callable[..., Any], Path]:
    """Build a runner function that executes via the worker subprocess."""
    script_path, _ = _compute_script_artifacts(source, entrypoint)

    # Serialized: concurrent codegen threads may target the same script_path.
    with _RUNTIME_MANIFEST_LOCK:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            _render_runner_script(source, entrypoint, requirements, pip_commands),
            encoding="utf-8",
        )

    def _runner(*, worker: Any, **kwargs: Any) -> Any:
        """Install requirements, load the script and run ``entrypoint`` in the worker."""
        if worker is None:
            raise ValueError("worker is required for execution")

        worker.install_packages(list(requirements), list(pip_commands))
        worker.load_script(str(script_path), entrypoint)
        return worker.execute(entrypoint, kwargs)

    return _runner, script_path


def _compute_script_artifacts(source: str, entrypoint: str) -> tuple[Path, str]:
    runtime_dir = _runtime_dir()
    version_tag = _next_version_tag(entrypoint, source, runtime_dir)
    return runtime_dir / f"{entrypoint}_{version_tag}.py", version_tag


def _render_runner_script(
    source: str,
    entrypoint: str,
    requirements: Sequence[str],
    pip_commands: Sequence[str],
) -> str:
    """Render the module the worker imports: header, package metadata, function source."""
    header = textwrap.dedent(
        f"""\
# Auto-generated by KathDB.
# Entry point: {entrypoint}

"""
    )
    metadata = f"REQUIRED_PACKAGES = {list(requirements)!r}\n"
    metadata += f"PIP_INSTALL_COMMANDS = {list(pip_commands)!r}\n\n"

    return header + metadata + source.rstrip() + "\n"


def _request_function_parse_fix(
    llm: BaseChatModel, source: str, error: Exception
) -> str:
    prompt = textwrap.dedent(
        f"""\
The following Python code failed to parse or was missing a function definition.
Error:
{error}

Please return ONLY the corrected Python code (no explanations or markdown fences) so it can be parsed successfully. Preserve the intent of the original implementation.

```python
{source}
```"""
    ).strip()
    response = llm.invoke(prompt)
    text = message_to_text(response)
    cleaned = text.strip()
    if not cleaned:
        return source
    return cleaned


def walk_nodes(root: FAOExecutableNode | None = None):
    """Yield all nodes in depth-first order starting from root."""
    if root is None:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def find_node(
    selector: str, root: FAOExecutableNode | None = None
) -> FAOExecutableNode | None:
    """Find by op, function name, or any output name."""

    for n in walk_nodes(root):
        fn = n.function
        if n.op == selector or getattr(fn, "name", None) == selector:
            return n
        if selector in (n.outputs or []):
            return n
    return None


def _node_to_dict(node: FAOExecutableNode) -> dict[str, Any]:
    """Recursively convert a FAOExecutableNode tree to a dictionary."""
    d: dict[str, Any] = {
        "op": node.op,
        "code": node.function.str_impl,
        "inputs": list(node.inputs) if node.inputs else [],
        "outputs": list(node.outputs) if node.outputs else [],
        "implementation_guidance": node.implementation_guidance or "",
        "metadata": dict(node.metadata) if node.metadata else {},
        "requirements": (
            list(node.function.requirements) if node.function.requirements else []
        ),
        "pip_commands": (
            list(node.function.pip_commands) if node.function.pip_commands else []
        ),
        "children": [_node_to_dict(child) for child in node.children],
    }
    if node.arguments:
        d["arguments"] = node.arguments
    return d


def _dict_to_node(
    data: dict[str, Any],
    llm: BaseChatModel | None = None,
) -> FAOExecutableNode:
    """Recursively reconstruct a FAOExecutableNode from a dictionary."""
    code = data.get("code", "")
    if not code:
        raise FAOExecutionError("Node data missing 'code' field")

    physical_fn = parse_llm_function(code, max_attempts=1, llm=llm)

    saved_requirements = data.get("requirements", [])
    saved_pip_commands = data.get("pip_commands", [])
    if saved_requirements:
        physical_fn.requirements = tuple(saved_requirements)
    if saved_pip_commands:
        physical_fn.pip_commands = tuple(saved_pip_commands)

    children = [
        _dict_to_node(child_data, llm=llm) for child_data in data.get("children", [])
    ]

    node = FAOExecutableNode(
        op=data.get("op", physical_fn.name),
        function=physical_fn,
        implementation_guidance=data.get("implementation_guidance", ""),
        inputs=data.get("inputs", []),
        outputs=data.get("outputs", []),
        children=children,
        metadata=data.get("metadata"),
        arguments=data.get("arguments", {}),
    )

    return node

