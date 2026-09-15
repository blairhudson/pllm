from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pllm.plan import CompiledPlan, _wrap


class CompilationError(ValueError):
    """The compiler rejected an invalid or unsupported request."""


def compile(request: Mapping[str, Any] | bytes | str) -> CompiledPlan:
    """Compile one strict request into an immutable native plan."""
    from pllm import _native

    if isinstance(request, Mapping):
        try:
            document = json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CompilationError(f"compile request is not canonical JSON: {exc}") from exc
    elif type(request) is bytes:
        document = request
    elif type(request) is str:
        document = request.encode("utf-8")
    else:
        raise TypeError("request must be a mapping, bytes, or JSON string")
    try:
        return _wrap(_native.compile_plan(document))
    except ValueError as exc:
        raise CompilationError(str(exc)) from exc


__all__ = ["CompilationError", "compile"]
