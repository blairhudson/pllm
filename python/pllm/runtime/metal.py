"""Opt-in Apple Silicon integer stage kernel; exact modulo-2**32 arithmetic."""

from __future__ import annotations

import platform

import numpy as np

from .native import NativeKernelError, _integer_array

_MAX_WORKING_BYTES = 512 * 1024 * 1024
_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint out_dim = weights_shape[0];
    uint in_dim = weights_shape[1];
    uint batch = inputs_shape[0];
    if (idx >= batch * out_dim) return;
    uint row = idx / out_dim;
    uint column = idx % out_dim;
    uint total = 0;
    for (uint k = 0; k < in_dim; ++k) {
        total += uint(int(weights[column * in_dim + k]))
               * uint(int(inputs[row * in_dim + k]));
    }
    result[idx] = total;
"""


class MetalGEMM:
    """Explicit MLX backend for bounded i8 weights and i8/u32 input stages."""

    def __init__(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise NativeKernelError("MLX Metal stages require Apple Silicon")
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise NativeKernelError("install pllm.run[metal] for MLX stages") from exc
        if not mx.metal.is_available():
            raise NativeKernelError("MLX Metal device is unavailable")
        self._mx = mx
        self._kernel = mx.fast.metal_kernel(
            name="pllm_exact_i8_u32_matmul_v1",
            input_names=["weights", "inputs"],
            output_names=["result"],
            source=_SOURCE,
        )

    @property
    def backend(self) -> str:
        return "mlx-metal"

    def compile(self, weights: np.ndarray) -> MetalCompiledMatrix:
        return MetalCompiledMatrix(self, weights)


class MetalCompiledMatrix:
    """GPU-owned immutable weight snapshot. Outputs materialize before returning."""

    def __init__(self, owner: MetalGEMM, weights: np.ndarray) -> None:
        raw = np.asarray(weights)
        if raw.ndim == 2 and raw.size > _MAX_WORKING_BYTES:
            raise NativeKernelError("Metal stage working set exceeds 512 MiB")
        w = _integer_array(weights, np.int8, low=-128, high=127)
        if min(w.shape) <= 0 or w.size > _MAX_WORKING_BYTES:
            raise NativeKernelError("matrix dimensions or Metal memory budget invalid")
        self.shape = tuple(w.shape)
        self.weight_bytes = int(w.nbytes)
        self._owner = owner
        # MLX owns a copy; mutation of the caller's array cannot change the plan.
        self._weights = owner._mx.array(w.copy())
        owner._mx.eval(self._weights)

    def _input(self, inputs: np.ndarray, dtype, low: int, high: int) -> np.ndarray:
        raw = np.asarray(inputs)
        if raw.ndim == 2 and (
            self.weight_bytes + raw.size * 4 + raw.shape[0] * self.shape[0] * 4
            > _MAX_WORKING_BYTES
        ):
            raise NativeKernelError("Metal stage working set exceeds 512 MiB")
        return _integer_array(raw, dtype, low=low, high=high)

    def _run(self, x: np.ndarray) -> np.ndarray:
        mx = self._owner._mx
        out_shape = (x.shape[0], self.shape[0])
        if self.weight_bytes + x.nbytes + out_shape[0] * out_shape[1] * 4 > _MAX_WORKING_BYTES:
            raise NativeKernelError("Metal stage working set exceeds 512 MiB")
        if not x.shape[0]:
            return np.empty(out_shape, dtype=np.uint32)
        gpu_x = mx.array(x)
        result = self._owner._kernel(
            inputs=[self._weights, gpu_x],
            template=[],
            grid=(out_shape[0] * out_shape[1], 1, 1),
            threadgroup=(min(256, out_shape[0] * out_shape[1]), 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.uint32],
        )[0]
        mx.eval(result)
        return np.asarray(result, dtype=np.uint32)

    def clear(self, inputs: np.ndarray) -> np.ndarray:
        x = self._input(inputs, np.int8, low=-128, high=127)
        if x.shape[1] != self.shape[1]:
            raise NativeKernelError("expected weights [out,in] and inputs [batch,in]")
        if self.shape[1] * 128 * 128 > np.iinfo(np.int32).max:
            raise NativeKernelError("clear int32 result cannot represent the worst case")
        return self._run(x).view(np.int32)

    def wrap32(self, inputs: np.ndarray) -> np.ndarray:
        x = self._input(inputs, "<u4", low=0, high=2**32 - 1)
        if x.shape[1] != self.shape[1]:
            raise NativeKernelError("expected weights [out,in] and inputs [batch,in]")
        return self._run(x)
