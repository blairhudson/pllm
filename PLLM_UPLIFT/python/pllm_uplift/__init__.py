"""Non-shadowing staging API. Performance work is delegated to the native extension."""
from __future__ import annotations
import json

def native_benchmark(*, dim: int = 64, lanes: int = 18, repeats: int = 7, bits: int = 24, tile: int = 32) -> dict:
    try:
        from . import _native
    except ImportError as exc:
        raise RuntimeError("Native extension unavailable. Build this package with Rust and maturin; no Python performance fallback is provided.") from exc
    return json.loads(_native.benchmark_json(dim, lanes, repeats, bits, tile))

__all__ = ["native_benchmark"]
