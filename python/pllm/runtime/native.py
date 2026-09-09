"""Native matrix execution. The Python branch is an explicit correctness reference.

The wire boundary uses immutable little endian byte buffers. Rust owns each
compiled matrix and one worker pool belongs to each engine, not each layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ._native_support import capabilities, extension


class NativeKernelError(ValueError):
    pass


def _integer_array(value, dtype, *, low: int, high: int, dimensions: int = 2):
    raw = np.asarray(value)
    if raw.ndim != dimensions or raw.dtype.kind not in {"i", "u"}:
        raise NativeKernelError(f"expected a rank {dimensions} integer tensor")
    target = np.dtype(dtype)
    bounds = np.iinfo(target)
    dtype_proves_bounds = raw.dtype == target and low <= bounds.min and high >= bounds.max
    if not dtype_proves_bounds and raw.size and (int(raw.min()) < low or int(raw.max()) > high):
        raise NativeKernelError(f"values must be in [{low}, {high}]")
    return np.ascontiguousarray(raw, dtype=target)


class CompiledMatrix:
    """An immutable matrix snapshot. Recompile explicitly after changing weights."""

    def __init__(self, owner: "MaskedGEMM", weights: np.ndarray):
        w = _integer_array(weights, np.int8, low=-128, high=127)
        if min(w.shape) <= 0:
            raise NativeKernelError("matrix dimensions must be positive")
        self.owner = owner
        self.shape = tuple(w.shape)
        self.weight_bytes = int(w.size)
        self._matrix = owner._extension.Matrix(w.tobytes(), *w.shape) if owner.native else None
        self._reference = None if owner.native else w.copy()
        if self._reference is not None:
            self._reference.flags.writeable = False

    def _execute(self, operation: str, *args):
        try:
            return getattr(self.owner._executor, operation)(self._matrix, *args)
        except ValueError as exc:
            raise NativeKernelError(str(exc)) from exc

    def _inputs(self, inputs, dtype, low, high, *, bounded=True):
        x = _integer_array(inputs, dtype, low=low, high=high)
        if x.shape[1] != self.shape[1]:
            raise NativeKernelError("expected weights [out,in] and inputs [batch,in]")
        # All callers, including the reference, have the same conservative bound.
        if bounded and self.shape[1] * 128 * high > np.iinfo(np.int64).max:
            raise NativeKernelError("dot product bound exceeds int64")
        return x

    def modular(self, inputs: np.ndarray, modulus: int) -> np.ndarray:
        p = int(modulus)
        if not 2 <= p < 2**31:
            raise NativeKernelError("prime-ring modulus must satisfy 2 <= modulus < 2^31")
        x = self._inputs(inputs, "<u4", 0, p - 1)
        if self._matrix is not None:
            data = self._execute("modular", x.tobytes(), x.shape[0], p)
            return np.frombuffer(data, dtype="<u4").reshape(x.shape[0], self.shape[0])
        return (x.astype(np.int64) @ self._reference.astype(np.int64).T % p).astype(np.uint32)

    def wrap32(self, inputs: np.ndarray) -> np.ndarray:
        x = self._inputs(inputs, "<u4", 0, 2**32 - 1, bounded=False)
        if self._matrix is not None:
            data = self._execute("wrap32", x.tobytes(), x.shape[0])
            return np.frombuffer(data, dtype="<u4").reshape(x.shape[0], self.shape[0])
        totals = x.astype(object) @ self._reference.astype(object).T
        return np.asarray(totals % (1 << 32), dtype=np.uint32)

    def wrap64(self, inputs: np.ndarray) -> np.ndarray:
        raw = np.asarray(inputs)
        if raw.ndim < 1 or raw.dtype.kind not in {"i", "u"}:
            raise NativeKernelError("expected an integer tensor")
        if raw.shape[-1] != self.shape[1]:
            raise NativeKernelError("expected weights [out,in] and inputs [...,in]")
        if raw.dtype.kind == "i" and raw.size and int(raw.min()) < 0:
            raise NativeKernelError("ring-64 inputs must be unsigned residues")
        x = np.ascontiguousarray(raw, dtype="<u8")
        leading = x.shape[:-1]
        rows = x.reshape(-1, self.shape[1])
        if self._matrix is not None:
            data = self._execute("wrap64", rows.tobytes(), rows.shape[0])
            return np.frombuffer(data, dtype="<u8").reshape(*leading, self.shape[0])
        totals = rows.astype(object) @ self._reference.astype(object).T
        return np.asarray(totals % (1 << 64), dtype=np.uint64).reshape(*leading, self.shape[0])

    def clear(self, inputs: np.ndarray) -> np.ndarray:
        x = self._inputs(inputs, np.int8, -128, 127)
        if self._matrix is not None:
            data = self._execute("clear", x.tobytes(), x.shape[0])
            return np.frombuffer(data, dtype="<i4").reshape(x.shape[0], self.shape[0])
        max_w = int(np.max(np.abs(self._reference.astype(np.int16))))
        if self.shape[1] * max_w * 128 > np.iinfo(np.int32).max:
            raise NativeKernelError("clear int32 result cannot represent the worst case")
        return (x.astype(np.int64) @ self._reference.astype(np.int64).T).astype(np.int32)

    def coefficients(self, inputs: np.ndarray, modulus: int) -> np.ndarray:
        q = int(modulus)
        if not 2 <= q < 2**54:
            raise NativeKernelError("ciphertext modulus must satisfy 2 <= q < 2^54")
        a = _integer_array(inputs, "<u8", low=0, high=q - 1)
        if a.shape[0] != self.shape[1] or a.shape[1] == 0:
            raise NativeKernelError("expected coefficients [in,columns]")
        if self._matrix is not None:
            data = self._execute("coefficients", a.tobytes(), a.shape[1], q)
            return np.frombuffer(data, dtype="<u8").reshape(self.shape[0], a.shape[1])
        w = self._reference.astype(np.int64)
        if self.shape[1] * int(np.max(np.abs(w))) * 128 > np.iinfo(np.int32).max:
            raise NativeKernelError("signed limb dot product exceeds int32")
        out = np.zeros((self.shape[0], a.shape[1]), dtype=np.int64)
        correction = 128 * w.sum(axis=1)[:, None]
        for limb in reversed(range(((q - 1).bit_length() + 7) // 8)):
            digits = ((a >> (8 * limb)) & 255).astype(np.int64) - 128
            out = (out * 256 + w @ digits + correction) % q
        return out.astype(np.uint64)

    __call__ = modular


class MaskedGEMM:
    """Exact modular kernels. Use compile() once for each immutable model stage."""

    def __init__(
        self,
        library: str | Path | None = None,
        *,
        threads: int = 0,
        build: bool = True,
        simd: bool = True,
    ) -> None:
        if library is not None:
            raise NativeKernelError(
                "External C++ libraries are no longer accepted; build pllm._native with Maturin"
            )
        default_threads = min(32, os.cpu_count() or 1)
        self.threads = int(threads or os.environ.get("PLLM_NATIVE_THREADS", str(default_threads)))
        if not 1 <= self.threads <= 32:
            raise NativeKernelError("threads must be in [1,32]")
        self._extension = extension()
        self._executor = self._extension.Executor(self.threads, simd) if self._extension else None
        self.library_path = Path(self._extension.__file__) if self._extension else None
        self.lib = self._extension  # Legacy inspection only; never a ctypes library.

    @property
    def native(self) -> bool:
        return self._extension is not None

    @property
    def available(self) -> bool:
        return self.native

    @property
    def backend(self) -> str:
        return "rust" if self.native else "python-reference"

    def compile(self, weights: np.ndarray) -> CompiledMatrix:
        return CompiledMatrix(self, weights)

    def modular(self, weights, inputs, modulus):
        return self.compile(weights).modular(inputs, modulus)

    def modular_limb12(self, weights, inputs, modulus):
        # The native dispatcher uses 12-bit limbs only when they are safe.
        return self.modular(weights, inputs, modulus)

    def wrap32(self, weights, inputs):
        return self.compile(weights).wrap32(inputs)

    def wrap64(self, weights, inputs):
        return self.compile(weights).wrap64(inputs)

    def clear(self, weights, inputs):
        return self.compile(weights).clear(inputs)

    def coefficients(self, weights, inputs, modulus):
        return self.compile(weights).coefficients(inputs, modulus)

    __call__ = modular


def build_native_library(output: str | Path | None = None) -> Path:
    """Compatibility check. Compilation happens during uv sync / wheel building."""
    if output is not None:
        raise NativeKernelError("Use `uv build` to choose a distribution output directory")
    module = extension()
    if module is None:
        raise NativeKernelError("The Python reference is not a native library")
    return Path(module.__file__)


def main() -> None:
    build_native_library()
    print(json.dumps(capabilities(), indent=2))


if __name__ == "__main__":
    main()
