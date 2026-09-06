"""Executor: per-operator code generation + sandboxed execution of a logical plan."""

from .error_handler import ExecutionErrorHandler
from .executor import Executor

__all__ = ["Executor", "ExecutionErrorHandler"]
