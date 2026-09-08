"""Load the compiled module explicitly; a missing native build is not a benchmark."""
from __future__ import annotations

import importlib
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load():
    try:
        module = importlib.import_module("pllm._native")
    except ModuleNotFoundError as exc:
        if exc.name != "pllm._native":
            raise
        return None
    if module.capabilities().get("api_version") != 1:
        raise RuntimeError("PLLM native extension API mismatch; rebuild this checkout")
    return module


def extension():
    backend = os.environ.get("PLLM_KERNEL_BACKEND", "rust")
    if backend not in {"rust", "python"}:
        raise ValueError("PLLM_KERNEL_BACKEND must be rust or python")
    if backend == "python":
        if os.environ.get("PLLM_REQUIRE_RUST") == "1":
            raise RuntimeError("A Rust validation job cannot use the Python reference")
        return None
    module = _load()
    if module is None:
        raise RuntimeError(
            "PLLM's Rust extension is missing. Run `uv sync` to build with Maturin, "
            "or install a matching native wheel. For correctness research only, "
            "set PLLM_KERNEL_BACKEND=python."
        )
    return module


def capabilities() -> dict:
    module = extension()
    if module is None:
        return {"implementation": "python-reference", "compiled": False, "gpu": False}
    return {**module.capabilities(), "compiled": True}
