"""Narrow public runtime declarations; implementation modules remain internal."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from pllm._version import __version__

_EXPORTS = {
    "AsyncOpenAI": ("pllm.runtime.client", "AsyncOpenAI"),
    "AsyncPLLMTransport": ("pllm.runtime.transport", "AsyncPLLMTransport"),
    "CompiledRuntimeModel": ("pllm.runtime.model_binding", "CompiledRuntimeModel"),
    "CompiledRuntimeSession": ("pllm.runtime.model_execution", "CompiledRuntimeSession"),
    "ExecutionBudget": ("pllm.configuration", "ExecutionBudget"),
    "GatewayConfig": ("pllm.runtime.config", "GatewayConfig"),
    "OpenAI": ("pllm.runtime.client", "OpenAI"),
    "PLLMTransport": ("pllm.runtime.transport", "PLLMTransport"),
    "PrivacyMode": ("pllm.runtime.privacy", "PrivacyMode"),
    "ProprietaryProtocol": ("pllm.runtime.privacy", "ProprietaryProtocol"),
    "RuntimeBindingError": ("pllm.runtime.model_binding", "RuntimeBindingError"),
    "RuntimeExecutionError": ("pllm.runtime.model_execution", "RuntimeExecutionError"),
    "RuntimeStageBinding": ("pllm.runtime.model_binding", "RuntimeStageBinding"),
    "compile_runtime_model": ("pllm.runtime.model_binding", "compile_runtime_model"),
    "create_app": ("pllm.runtime.server", "create_app"),
    "create_sidecar_app": ("pllm.runtime.sidecar", "create_sidecar_app"),
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
