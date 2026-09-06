"""Worker subprocess for isolated execution of generated code.

``WorkerClient`` (parent side) talks to ``worker_main`` (subprocess) over a
unix-domain socket: length-prefixed JSON messages, DataFrames as Arrow Feather.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather

from ..common.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "KathDBWorkerError",
    "KathDBWorkerInstallError",
    "KathDBWorkerLoadError",
    "KathDBWorkerExecuteError",
    "WorkerClient",
    "spawn_worker",
    "spawn_worker_conda",
    "remove_conda_env",
    "worker_main",
]


class KathDBWorkerError(Exception):
    """Base error for worker execution failures."""

    def __init__(self, entrypoint: str, underlying_error: str, stage: str = "unknown"):
        self.entrypoint = entrypoint
        self.stage = stage
        self.underlying_error = underlying_error
        super().__init__(f"[{stage.upper()}] {entrypoint}: {underlying_error}")


class KathDBWorkerInstallError(KathDBWorkerError):
    """Error during pip install - includes the failed command in the message."""

    def __init__(self, underlying_error: str, failed_command: str):
        self.failed_command = failed_command
        rich_msg = (
            f"Pip install failed.\n"
            f"Failed command: {failed_command}\n"
            f"Error details:\n{underlying_error}"
        )
        super().__init__("", rich_msg, stage="install")
        self.underlying_error = rich_msg


class KathDBWorkerLoadError(KathDBWorkerError):
    """Error during script loading - includes failed library in the message."""

    def __init__(
        self, script_path: str, underlying_error: str, failed_import: str | None = None
    ):
        self.script_path = script_path
        self.failed_import = failed_import
        import_info = f"\nFailed import: {failed_import}" if failed_import else ""
        rich_msg = (
            f"Script load failed.\n"
            f"Script: {script_path}{import_info}\n"
            f"Error details:\n{underlying_error}"
        )
        super().__init__("", rich_msg, stage="load")
        self.underlying_error = rich_msg


class KathDBWorkerExecuteError(KathDBWorkerError):
    """Error during execution - includes offending line in the message."""

    def __init__(
        self, entrypoint: str, underlying_error: str, offending_line: str | None = None
    ):
        self.offending_line = offending_line
        line_info = f"\nOffending line: {offending_line}" if offending_line else ""
        rich_msg = f"Execution failed.{line_info}\nError details:\n{underlying_error}"
        super().__init__(entrypoint, rich_msg, stage="execute")
        self.underlying_error = rich_msg


# Message type constants
MSG_INSTALL = "install"
MSG_LOAD = "load"
MSG_EXECUTE = "execute"
MSG_SHUTDOWN = "shutdown"
MSG_RESPONSE = "response"
MSG_ERROR = "error"


def _read_exact(pipe: Any, n: int) -> bytes:
    """Read exactly *n* bytes from *pipe*, retrying on short reads."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_stderr_tail(stderr_path: str | None, max_bytes: int = 65536) -> str:
    """Return the last *max_bytes* of the worker's stderr file (best-effort)."""
    if not stderr_path:
        return ""
    try:
        with open(stderr_path, "rb") as f:
            try:
                f.seek(0, io.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
            except (OSError, ValueError):
                pass
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _eof_error(proc: subprocess.Popen | None, stderr_path: str | None) -> EOFError:
    """Build the 'worker connection closed' error, enriched with stderr tail."""
    if proc is None:
        return EOFError("Worker connection closed")
    exit_code = proc.poll()
    stderr_output = _read_stderr_tail(stderr_path)
    if stderr_output:
        return EOFError(
            f"Worker connection closed (exit code: {exit_code}).\n"
            f"Worker stderr:\n{stderr_output}"
        )
    return EOFError(f"Worker connection closed (exit code: {exit_code})")


def _send_message(pipe: Any, msg_type: str, payload: dict[str, Any]) -> None:
    """Send a JSON message with length prefix."""
    message = {"type": msg_type, **payload}
    data = json.dumps(message).encode("utf-8")
    pipe.write(struct.pack(">I", len(data)))
    pipe.write(data)
    pipe.flush()


def _recv_message(
    pipe: Any,
    proc: subprocess.Popen | None = None,
    stderr_path: str | None = None,
) -> dict[str, Any]:
    """Receive a JSON message with length prefix."""
    length_bytes = _read_exact(pipe, 4)
    if len(length_bytes) < 4:
        raise _eof_error(proc, stderr_path)
    length = struct.unpack(">I", length_bytes)[0]
    data = _read_exact(pipe, length)
    return json.loads(data.decode("utf-8"))


def _serialize_dataframe(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to PyArrow Feather bytes (may raise on bad dtypes)."""
    buffer = io.BytesIO()
    feather.write_feather(df, buffer)
    return buffer.getvalue()


def _serialize_pickle(obj: Any) -> bytes:
    """Serialize an arbitrary object to pickle bytes (may raise if unpicklable)."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _send_blob(pipe: Any, data: bytes) -> None:
    """Send pre-serialized bytes with a length prefix."""
    pipe.write(struct.pack(">I", len(data)))
    pipe.write(data)
    pipe.flush()


def _send_dataframe(pipe: Any, df: pd.DataFrame) -> None:
    """Serialize and send a DataFrame using PyArrow Feather format."""
    _send_blob(pipe, _serialize_dataframe(df))


def _recv_dataframe(
    pipe: Any,
    proc: subprocess.Popen | None = None,
    stderr_path: str | None = None,
) -> pd.DataFrame:
    """Receive a DataFrame using PyArrow Feather format."""
    length_bytes = _read_exact(pipe, 4)
    if len(length_bytes) < 4:
        raise _eof_error(proc, stderr_path)
    length = struct.unpack(">I", length_bytes)[0]
    data = _read_exact(pipe, length)
    buffer = io.BytesIO(data)
    return feather.read_feather(buffer)


def _send_pickle(pipe: Any, obj: Any) -> None:
    """Serialize and send an arbitrary Python object as a pickle blob."""
    _send_blob(pipe, _serialize_pickle(obj))


def _recv_pickle(
    pipe: Any,
    proc: subprocess.Popen | None = None,
    stderr_path: str | None = None,
) -> Any:
    """Receive a length-prefixed pickle blob and deserialize it."""
    length_bytes = _read_exact(pipe, 4)
    if len(length_bytes) < 4:
        raise _eof_error(proc, stderr_path)
    length = struct.unpack(">I", length_bytes)[0]
    data = _read_exact(pipe, length)
    return pickle.loads(data)


class WorkerClient:
    """Client for communicating with a worker subprocess."""

    def __init__(
        self,
        proc: subprocess.Popen,
        sock: socket.socket,
        sock_dir: Path,
        *,
        stderr_path: str | Path | None = None,
        exec_timeout_s: float | None = None,
    ) -> None:
        self.proc = proc
        self._sock = sock
        self._sock_dir = sock_dir
        # Worker stderr file; error paths read its tail.
        self._stderr_path = str(stderr_path) if stderr_path is not None else None
        # Wall-clock budget for one execute() round-trip; None disables it.
        self._exec_timeout_s = exec_timeout_s
        # Binary IPC over the unix socket (never stdin/stdout).
        self._ipc = sock.makefile("rwb", buffering=0)
        self._loaded_scripts: dict[str, str] = {}  # script_path -> entrypoint

    def _poison(self) -> None:
        """Kill the worker (the manager respawns a clean one) and release its resources.

        Used when the IPC channel may be desynced: a framing-level error or a timeout.
        """
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self._ipc.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        shutil.rmtree(self._sock_dir, ignore_errors=True)

    def install_packages(
        self, requirements: list[str], pip_commands: list[str]
    ) -> None:
        """Request the worker to install packages."""
        if not requirements and not pip_commands:
            return
        _send_message(
            self._ipc,
            MSG_INSTALL,
            {"requirements": requirements, "pip_commands": pip_commands},
        )
        response = _recv_message(self._ipc, self.proc, self._stderr_path)
        if response.get("type") == MSG_ERROR:
            error = response.get("error", "")
            if error.startswith("INSTALL_FAIL|"):
                parts = error.split("|", 2)
                failed_cmd = parts[1] if len(parts) > 1 else "unknown"
                actual_error = parts[2] if len(parts) > 2 else error
                raise KathDBWorkerInstallError(actual_error, failed_cmd)
            raise KathDBWorkerInstallError(error, "unknown")

    def load_script(self, script_path: str, entrypoint: str) -> None:
        """Request the worker to load a script."""
        _send_message(
            self._ipc,
            MSG_LOAD,
            {"script_path": script_path, "entrypoint": entrypoint},
        )
        response = _recv_message(self._ipc, self.proc, self._stderr_path)
        if response.get("type") == MSG_ERROR:
            error = response.get("error", "")
            if error.startswith("LOAD_FAIL|"):
                parts = error.split("|", 3)
                script = parts[1] if len(parts) > 1 else script_path
                failed_import = parts[2] if len(parts) > 2 and parts[2] else None
                actual_error = parts[3] if len(parts) > 3 else error
                raise KathDBWorkerLoadError(script, actual_error, failed_import)
            raise KathDBWorkerLoadError(script_path, error)
        self._loaded_scripts[script_path] = entrypoint

    def execute(self, entrypoint: str, inputs: dict[str, Any]) -> Any:
        """Execute a function in the worker with inputs."""
        df_inputs: dict[str, pd.DataFrame] = {}
        other_inputs: dict[str, Any] = {}
        for name, value in inputs.items():
            if isinstance(value, pd.DataFrame):
                df_inputs[name] = value
            else:
                other_inputs[name] = value

        # Serialize ALL inputs before sending the header, so a serialization failure
        # cannot leave a half-sent input stream (channel desync).
        try:
            df_blobs = [_serialize_dataframe(df) for df in df_inputs.values()]
            other_blobs = [_serialize_pickle(o) for o in other_inputs.values()]
        except Exception as exc:
            raise KathDBWorkerExecuteError(
                entrypoint, f"failed to serialize inputs: {exc}"
            ) from exc

        try:
            _send_message(
                self._ipc,
                MSG_EXECUTE,
                {
                    "entrypoint": entrypoint,
                    "df_names": list(df_inputs.keys()),
                    "other_names": list(other_inputs.keys()),
                },
            )
            for blob in df_blobs:
                _send_blob(self._ipc, blob)
            for blob in other_blobs:
                _send_blob(self._ipc, blob)

            if self._exec_timeout_s is not None:
                self._sock.settimeout(self._exec_timeout_s)
            try:
                response = _recv_message(self._ipc, self.proc, self._stderr_path)
                if response.get("type") == MSG_ERROR:
                    error = response.get("error", "Unknown error")
                    if error.startswith("EXEC_FAIL|"):
                        parts = error.split("|", 2)
                        offending_line = (
                            parts[1] if len(parts) > 1 and parts[1] else None
                        )
                        actual_error = parts[2] if len(parts) > 2 else error
                        raise KathDBWorkerExecuteError(
                            entrypoint, actual_error, offending_line
                        )
                    raise KathDBWorkerExecuteError(entrypoint, error)
                result_type = response.get("result_type", "dataframe")
                if result_type == "pickle":
                    return _recv_pickle(self._ipc, self.proc, self._stderr_path)
                return _recv_dataframe(self._ipc, self.proc, self._stderr_path)
            finally:
                if self._exec_timeout_s is not None:
                    try:
                        self._sock.settimeout(None)
                    except OSError:
                        pass
        except KathDBWorkerExecuteError:
            # Structured worker-side error: the channel is still in sync, worker reusable.
            raise
        except socket.timeout as exc:
            self._poison()
            raise KathDBWorkerExecuteError(
                entrypoint, f"worker timed out after {self._exec_timeout_s}s"
            ) from exc
        except Exception:
            # Framing-level failure: the channel may be desynced, so poison the worker.
            self._poison()
            raise

    def shutdown(self) -> None:
        """Request the worker to shut down."""
        try:
            _send_message(self._ipc, MSG_SHUTDOWN, {})
        except (BrokenPipeError, OSError):
            pass
        try:
            self._ipc.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGTERM ignored: force-kill so no orphan worker is leaked.
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                logger.warning(
                    "[worker] pid %s did not exit after SIGKILL", self.proc.pid
                )
        finally:
            shutil.rmtree(self._sock_dir, ignore_errors=True)

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self.proc.poll() is None


# Default worker connect timeout; override with KATHDB_WORKER_CONNECT_TIMEOUT_S
# (several processes provisioning from one shared conda env at once can need longer).
_WORKER_CONNECT_TIMEOUT_S = float(
    os.environ.get("KATHDB_WORKER_CONNECT_TIMEOUT_S", 3 * 180.0)
)


def spawn_worker(
    conda_env_name: str,
    *,
    connect_timeout_s: float | None = None,
    exec_timeout_s: float | None = None,
) -> WorkerClient:
    """Spawn a worker subprocess in the specified conda environment."""
    timeout_s = (
        connect_timeout_s
        if connect_timeout_s is not None
        else _WORKER_CONNECT_TIMEOUT_S
    )
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env_name,
        "--no-capture-output",
        "python",
        "-u",
        "-m",
        "kathdb.worker",
    ]
    logger.info("Spawning worker subprocess: %s", " ".join(cmd))

    # src/kathdb/worker/_worker.py -> src
    src_dir = str(Path(__file__).resolve().parent.parent.parent)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}:{existing_pythonpath}" if existing_pythonpath else src_dir
    )
    logger.info("PYTHONPATH for worker: %s", env["PYTHONPATH"])

    # Binary IPC over a dedicated unix-domain socket, so nothing the child prints
    # to stdout/stderr can corrupt the protocol.
    sock_dir = Path(tempfile.mkdtemp(prefix="kathdb_ipc_"))
    sock_path = sock_dir / "worker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(1)
    listener.settimeout(timeout_s)
    env["KATHDB_IPC_SOCK"] = str(sock_path)

    # stderr goes to a file, not a PIPE (an undrained pipe would deadlock the child);
    # its tail is read for diagnostics and it is removed with sock_dir.
    stderr_path = sock_dir / "worker.stderr"
    stderr_file = open(stderr_path, "wb")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            bufsize=0,
            env=env,
        )
    except Exception:
        # Spawn never happened: no later cleanup path will see these.
        try:
            listener.close()
        except Exception:
            pass
        shutil.rmtree(sock_dir, ignore_errors=True)
        raise
    finally:
        # The child holds its own dup of the fd; the parent reads the file by path.
        stderr_file.close()

    try:
        parent_sock, _ = listener.accept()
    except socket.timeout:
        proc.kill()
        proc.wait(timeout=5)
        stderr_output = _read_stderr_tail(str(stderr_path))
        shutil.rmtree(sock_dir, ignore_errors=True)
        raise RuntimeError(
            f"Worker did not connect to {sock_path} within "
            f"{timeout_s}s.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr:\n{stderr_output}"
        )
    finally:
        try:
            listener.close()
        except Exception:
            pass

    exit_code = proc.poll()
    if exit_code is not None:
        try:
            parent_sock.close()
        except Exception:
            pass
        stderr_output = _read_stderr_tail(str(stderr_path))
        shutil.rmtree(sock_dir, ignore_errors=True)
        raise RuntimeError(
            f"Worker subprocess failed to start (exit code: {exit_code}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"PYTHONPATH: {env['PYTHONPATH']}\n"
            f"Stderr:\n{stderr_output}"
        )

    return WorkerClient(
        proc,
        parent_sock,
        sock_dir,
        stderr_path=str(stderr_path),
        exec_timeout_s=exec_timeout_s,
    )


def spawn_worker_conda(
    *,
    requirements_path: Path | str | None = None,
    python_version: str | None = None,
    env_prefix: str = "kathdb_worker",
    connect_timeout_s: float | None = None,
    exec_timeout_s: float | None = None,
) -> tuple[WorkerClient, str]:
    """Create a new conda environment and spawn a worker in it."""
    env_name = f"{env_prefix}_{uuid.uuid4().hex[:8]}"

    if python_version is None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    logger.info("Creating conda environment: %s (python=%s)", env_name, python_version)
    create_cmd = [
        "conda",
        "create",
        "-n",
        env_name,
        f"python={python_version}",
        "-y",
    ]
    result = subprocess.run(create_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create conda environment {env_name}:\n{result.stderr}"
        )

    if requirements_path is not None:
        req_path = Path(requirements_path)
        if req_path.exists():
            logger.info(
                "Installing requirements from %s into %s",
                req_path,
                env_name,
            )
            install_cmd = [
                "conda",
                "run",
                "-n",
                env_name,
                "--no-capture-output",
                "pip",
                "install",
                "-r",
                str(req_path),
            ]
            result = subprocess.run(
                install_cmd, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                remove_conda_env(env_name)
                raise RuntimeError(
                    f"Failed to install requirements from {req_path} into {env_name}.\n"
                    f"Exit code: {result.returncode}\n"
                    f"Stderr:\n{result.stderr}\n"
                    f"Stdout:\n{result.stdout}"
                )
            logger.info("Successfully installed requirements into %s", env_name)
        else:
            raise FileNotFoundError(f"Requirements file not found: {req_path}")

    worker = spawn_worker(
        env_name,
        connect_timeout_s=connect_timeout_s,
        exec_timeout_s=exec_timeout_s,
    )
    return worker, env_name


def remove_conda_env(env_name: str) -> None:
    """Remove a conda environment."""
    logger.info("Removing conda environment: %s", env_name)
    remove_cmd = ["conda", "env", "remove", "-n", env_name, "-y"]
    result = subprocess.run(remove_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning(
            "Failed to remove conda environment %s: %s", env_name, result.stderr
        )
    else:
        logger.info("Successfully removed conda environment: %s", env_name)


# ---------------------------------------------------------------------------
# Worker subprocess main loop
# ---------------------------------------------------------------------------


def _setup_litellm_file_tracking() -> None:
    """Register a litellm callback that appends per-call usage to the JSONL file
    named by ``KATHDB_INFERENCE_LOG_PATH`` (read by the parent's cost tracker)."""
    log_path = os.environ.get("KATHDB_INFERENCE_LOG_PATH")
    if not log_path:
        return
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger

        class _FileUsageLogger(CustomLogger):
            def __init__(self, path: str) -> None:
                self._path = path

            def _log(self, response_obj: Any) -> None:
                try:
                    usage = getattr(response_obj, "usage", None)
                    if not usage:
                        return
                    record = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0)
                        or 0,
                    }
                    try:
                        record["cost_usd"] = (
                            litellm.completion_cost(completion_response=response_obj)
                            or 0.0
                        )
                    except Exception:
                        record["cost_usd"] = 0.0
                    with open(self._path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                except Exception:
                    pass

            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                self._log(response_obj)

            async def async_log_success_event(
                self, kwargs, response_obj, start_time, end_time
            ):
                self._log(response_obj)

        litellm.callbacks = [_FileUsageLogger(log_path)]
    except ImportError:
        pass


def worker_main() -> None:
    """Main loop for the worker subprocess."""
    import runpy

    _setup_litellm_file_tracking()

    loaded_modules: dict[str, dict[str, Any]] = {}

    # Protocol framing runs over the KATHDB_IPC_SOCK unix socket, never over
    # stdin/stdout, so print() from user code cannot corrupt the message stream.
    sock_path = os.environ["KATHDB_IPC_SOCK"]
    ipc_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ipc_sock.connect(sock_path)
    ipc = ipc_sock.makefile("rwb", buffering=0)
    stdin = ipc
    stdout = ipc

    def send_response(success: bool, error: str | None = None) -> None:
        if success:
            _send_message(stdout, MSG_RESPONSE, {"success": True})
        else:
            _send_message(stdout, MSG_ERROR, {"error": error or "Unknown error"})

    while True:
        try:
            msg = _recv_message(stdin)
        except EOFError:
            break

        msg_type = msg.get("type")

        if msg_type == MSG_SHUTDOWN:
            break

        elif msg_type == MSG_INSTALL:
            requirements = msg.get("requirements", [])
            pip_commands = msg.get("pip_commands", [])
            try:
                _do_install(requirements, pip_commands)
                send_response(True)
            except Exception as e:
                send_response(False, str(e))

        elif msg_type == MSG_LOAD:
            script_path = msg.get("script_path")
            entrypoint = msg.get("entrypoint")
            try:
                module_dict = runpy.run_path(script_path)
                loaded_modules[entrypoint] = module_dict
                send_response(True)
            except ImportError as e:
                failed_import = getattr(e, "name", None) or str(e)
                send_response(
                    False,
                    f"LOAD_FAIL|{script_path}|{failed_import}|{e}",
                )
            except Exception:
                import traceback

                tb = traceback.format_exc()
                send_response(False, f"LOAD_FAIL|{script_path}||{tb}")

        elif msg_type == MSG_EXECUTE:
            entrypoint = msg.get("entrypoint")
            df_names = msg.get("df_names", [])
            other_names = msg.get("other_names", [])
            # Read ALL declared input frames before replying: one MSG_ERROR reply means
            # "channel in sync" to the parent, so no blob may be left unread.
            kwargs: dict[str, Any] = {}
            total_frames = len(df_names) + len(other_names)
            frames_read = 0
            try:
                for name in df_names:
                    frames_read += 1
                    kwargs[name] = _recv_dataframe(stdin)
                for name in other_names:
                    frames_read += 1
                    kwargs[name] = _recv_pickle(stdin)
            except Exception:
                import traceback

                tb = traceback.format_exc()
                # Drain the undelivered frames so the next request parses cleanly.
                for _ in range(total_frames - frames_read):
                    try:
                        length_bytes = _read_exact(stdin, 4)
                        if len(length_bytes) < 4:
                            break
                        _read_exact(stdin, struct.unpack(">I", length_bytes)[0])
                    except Exception:
                        break
                send_response(False, f"EXEC_FAIL||failed to read inputs:\n{tb}")
                continue

            try:
                if entrypoint not in loaded_modules:
                    raise ValueError(f"Entrypoint {entrypoint!r} not loaded")
                module_dict = loaded_modules[entrypoint]
                fn = module_dict.get(entrypoint)
                if fn is None or not callable(fn):
                    raise ValueError(f"Function {entrypoint!r} not found in module")

                result = fn(**kwargs)

                # Serialize BEFORE sending the success response, so a serialization
                # failure becomes a clean MSG_ERROR instead of a half-sent frame.
                if isinstance(result, pd.DataFrame):
                    blob = _serialize_dataframe(result)
                    result_type = "dataframe"
                else:
                    blob = _serialize_pickle(result)
                    result_type = "pickle"
                _send_message(
                    stdout, MSG_RESPONSE, {"success": True, "result_type": result_type}
                )
                _send_blob(stdout, blob)

            except Exception:
                import traceback

                tb = traceback.format_exc()
                offending_line = _extract_offending_line(tb)
                send_response(False, f"EXEC_FAIL|{offending_line}|{tb}")


def _extract_offending_line(traceback_str: str) -> str:
    """Extract the offending code line from a traceback string."""
    lines = traceback_str.strip().split("\n")
    offending_line = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("File ") and ", line " in stripped:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.startswith("    ") or next_line.startswith("\t"):
                    offending_line = next_line.strip()
    return offending_line


def _do_install(requirements: list[str], pip_commands: list[str]) -> None:
    """Install packages in the worker's environment."""
    commands: list[list[str]] = []
    for cmd in pip_commands:
        if not cmd:
            continue
        parts = shlex.split(str(cmd))
        if not parts:
            continue
        if parts[0] in {"pip", "pip3"}:
            parts = [sys.executable, "-m", "pip"] + parts[1:]
        commands.append(parts)
    if not commands and requirements:
        commands.append([sys.executable, "-m", "pip", "install", *requirements])

    for parts in commands:
        rendered = " ".join(parts)
        proc = subprocess.run(parts, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            error_details = (
                f"Pip install failed (exit {proc.returncode}):\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
            raise RuntimeError(f"INSTALL_FAIL|{rendered}|{error_details}")
