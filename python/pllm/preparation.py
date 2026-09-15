from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from pllm.configuration import ComponentDescriptor, ModelAwareCorrections

if TYPE_CHECKING:
    from pllm.runtime.preparation_server import create_preparation_app as create_preparation_app

__all__ = ["ComponentDescriptor", "ModelAwareCorrections", "create_preparation_app"]


def __getattr__(name: str) -> Any:
    if name != "create_preparation_app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("pllm.runtime.preparation_server").create_preparation_app
    globals()[name] = value
    return value
