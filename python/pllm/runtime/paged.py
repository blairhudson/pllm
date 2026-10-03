"""Bounded exact native execution over verified file-backed public weights."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._native_support import extension
from .native import NativeKernelError, _integer_array


class PagedGEMM:
    """Immutable private file snapshot; exact i8, modular and wrap32 kernels.

    Local private row gathers support token lookup without a resident vocabulary
    matrix. Source artifacts and the anonymous snapshot are separate disk owners.
    This direct kernel is not yet a selectable Pipeline storage backend.
    """

    __slots__ = ("_matrix",)

    def __init__(
        self, path: str | Path, artifact_digest: str, *, threads: int = 1, simd: bool = True
    ):
        if type(threads) is not int or not 1 <= threads <= 32 or type(simd) is not bool:
            raise NativeKernelError(
                "paged execution requires threads in [1,32] and a boolean SIMD choice"
            )
        native = extension()
        if native is None:
            raise NativeKernelError("paged execution requires the native extension")
        self._matrix = native.PagedMatrix(os.fspath(path), artifact_digest, threads, simd)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("paged matrix snapshots are immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def export(
        weights: np.ndarray, path: str | Path, *, page_rows: int = 512, compression: str = "raw"
    ) -> str:
        """Create a new public artifact; return its SHA-256. Never overwrite files.

        Export is a publisher operation and takes an existing weight array. Client
        import and later execution never materialize a full decoded weight array.
        Zlib chooses raw fallback independently for incompressible public pages.
        """
        if (
            type(page_rows) is not int
            or page_rows <= 0
            or type(compression) is not str
            or compression not in {"raw", "zlib"}
            or not isinstance(weights, np.ndarray)
            or weights.ndim != 2
        ):
            raise NativeKernelError("invalid paged export configuration")
        rows, cols = weights.shape
        if (
            min(rows, cols) <= 0
            or max(rows, cols) > 1048576
            or rows * cols > 2 * 1024**3
            or page_rows * cols > 4 * 1024**2
            or (rows + page_rows - 1) // page_rows > 65536
        ):
            raise NativeKernelError("paged export exceeds bounded shape policy")
        values = _integer_array(weights, np.int8, low=-128, high=127)
        native = extension()
        if native is None:
            raise NativeKernelError("paged export requires the native extension")
        return native.export_paged_weights(
            os.fspath(path), values.tobytes(), rows, cols, page_rows, compression == "zlib"
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    @property
    def weight_bytes(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def artifact_bytes(self) -> int:
        return self._matrix.artifact_bytes

    @property
    def metadata_bytes(self) -> int:
        return self._matrix.metadata_bytes

    @property
    def transient_weight_bytes(self) -> int:
        """Weight tile buffers only: excludes inputs, outputs, stacks and OS cache."""
        return self._matrix.transient_weight_bytes

    @property
    def weight_digest(self) -> str:
        return self._matrix.weight_digest

    def _inputs(self, values, dtype, low, high):
        inputs = _integer_array(values, dtype, low=low, high=high)
        if (
            inputs.shape[1] != self.shape[1]
            or inputs.shape[0] == 0
            or inputs.size > 4 * 1024**2
            or inputs.shape[0] * self.shape[0] > 4 * 1024**2
        ):
            raise NativeKernelError("paged input/output exceeds bounded shape policy")
        return inputs

    def clear(self, inputs: np.ndarray) -> np.ndarray:
        values = self._inputs(inputs, np.int8, -128, 127)
        result = self._matrix.clear(values.tobytes(), values.shape[0])
        return np.frombuffer(result, "<i4").reshape(values.shape[0], self.shape[0])

    def wrap32(self, inputs: np.ndarray) -> np.ndarray:
        values = self._inputs(inputs, "<u4", 0, 2**32 - 1)
        result = self._matrix.modular(values.tobytes(), values.shape[0])
        return np.frombuffer(result, "<u4").reshape(values.shape[0], self.shape[0])

    def modular(self, inputs: np.ndarray, modulus: int) -> np.ndarray:
        if type(modulus) is not int or not 2 <= modulus < 2**31:
            raise NativeKernelError("paged modulus must satisfy 2 <= modulus < 2^31")
        values = self._inputs(inputs, "<u4", 0, modulus - 1)
        result = self._matrix.modular(values.tobytes(), values.shape[0], modulus)
        return np.frombuffer(result, "<u4").reshape(values.shape[0], self.shape[0])

    def gather_rows(self, indices) -> np.ndarray:
        ids = _integer_array(indices, "<u8", low=0, high=self.shape[0] - 1, dimensions=1)
        if not 1 <= ids.size <= 4096 or ids.size * self.shape[1] > 4 * 1024**2:
            raise NativeKernelError("paged gather exceeds bounded shape policy")
        return np.frombuffer(self._matrix.gather(ids.tobytes()), np.int8).reshape(
            ids.size, self.shape[1]
        )

    def close(self) -> None:
        self._matrix.close()

    def __enter__(self) -> PagedGEMM:
        _ = self.shape
        return self

    def __exit__(self, *args) -> None:
        self.close()
