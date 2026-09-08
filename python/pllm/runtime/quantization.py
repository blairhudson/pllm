from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class QuantizationError(ValueError):
    pass


def signed_qmax(bits: int) -> int:
    if bits < 2 or bits > 8:
        raise QuantizationError("bits must be in [2, 8]")
    return (1 << (bits - 1)) - 1


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    values: np.ndarray
    scales: np.ndarray
    bits: int

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise QuantizationError("quantized weight must be rank 2")
        if self.values.dtype != np.int8:
            raise QuantizationError("quantized weight values must be int8")
        if self.scales.shape != (self.values.shape[0],):
            raise QuantizationError("one scale is required per output row")

    @property
    def out_features(self) -> int:
        return int(self.values.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.values.shape[1])

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float32) * self.scales[:, None]

    def metadata(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "shape": [self.out_features, self.in_features],
            "scales_dtype": "f4",
            "scales": self.scales.astype("<f4", copy=False).tobytes(),
        }


@dataclass(frozen=True, slots=True)
class QuantizedActivation:
    values: np.ndarray
    scales: np.ndarray
    bits: int
    original_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.dtype != np.int8:
            raise QuantizationError("quantized activations must be an int8 matrix")
        if self.scales.shape != (self.values.shape[0],):
            raise QuantizationError("one activation scale is required per row")
        if not self.original_shape or self.original_shape[-1] != self.values.shape[1]:
            raise QuantizationError("activation shape does not match flattened values")

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def features(self) -> int:
        return int(self.values.shape[1])


def quantize_weight_per_row(weight: np.ndarray, *, bits: int = 4) -> QuantizedWeight:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2:
        raise QuantizationError("weight must be rank 2")
    qmax = signed_qmax(bits)
    if value.shape[1] == 0 or not np.all(np.isfinite(value)):
        raise QuantizationError("weights require finite values and a nonzero input width")
    from ._native_support import extension

    native = extension()
    if native is not None:
        q, s = native.quantize(value.astype("<f4", copy=False).tobytes(), *value.shape, bits)
        return QuantizedWeight(
            np.frombuffer(q, np.int8).reshape(value.shape), np.frombuffer(s, "<f4"), bits
        )
    max_abs = np.max(np.abs(value), axis=1)
    scales = np.where(max_abs > 0, max_abs / qmax, 1.0).astype(np.float32)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise QuantizationError("weight scale underflow or overflow")
    quantized = np.rint(value / scales[:, None])
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
    return QuantizedWeight(np.ascontiguousarray(quantized), np.ascontiguousarray(scales), bits)


def quantize_activation_per_row(
    activation: np.ndarray,
    *,
    bits: int = 4,
    scales: float | np.ndarray | None = None,
) -> QuantizedActivation:
    value = np.asarray(activation, dtype=np.float32)
    if value.ndim < 1:
        raise QuantizationError("activation must have at least one dimension")
    original_shape = tuple(int(item) for item in value.shape)
    features = original_shape[-1]
    if features == 0 or not np.all(np.isfinite(value)):
        raise QuantizationError("activations require finite values and a nonzero input width")
    rows = int(value.size // features)
    flat = np.ascontiguousarray(value.reshape(rows, features))
    qmax = signed_qmax(bits)
    from ._native_support import extension

    native = extension()
    if native is not None:
        supplied = None
        if scales is not None:
            rs = np.asarray(scales, dtype=np.float32)
            rs = np.full(rows, float(rs), dtype=np.float32) if rs.ndim == 0 else rs.reshape(-1)
            if rs.shape != (rows,) or np.any(rs <= 0) or not np.all(np.isfinite(rs)):
                raise QuantizationError(
                    "activation scales must be finite positive values, one per row"
                )
            supplied = rs.astype("<f4", copy=False).tobytes()
        q, s = native.quantize(
            flat.astype("<f4", copy=False).tobytes(), rows, features, bits, supplied
        )
        return QuantizedActivation(
            np.frombuffer(q, np.int8).reshape(rows, features),
            np.frombuffer(s, "<f4"),
            bits,
            original_shape,
        )
    if scales is None:
        max_abs = np.max(np.abs(flat), axis=1)
        row_scales = np.where(max_abs > 0, max_abs / qmax, 1.0).astype(np.float32)
    else:
        row_scales = np.asarray(scales, dtype=np.float32)
        if row_scales.ndim == 0:
            row_scales = np.full(rows, float(row_scales), dtype=np.float32)
        row_scales = row_scales.reshape(-1)
        if (
            row_scales.shape != (rows,)
            or np.any(row_scales <= 0)
            or not np.all(np.isfinite(row_scales))
        ):
            raise QuantizationError("activation scales must be positive and have one value per row")
    if np.any(row_scales <= 0) or not np.all(np.isfinite(row_scales)):
        raise QuantizationError("activation scale underflow or overflow")
    quantized = np.rint(flat / row_scales[:, None])
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
    return QuantizedActivation(
        np.ascontiguousarray(quantized),
        np.ascontiguousarray(row_scales),
        bits,
        original_shape,
    )


def quantize_activation(
    activation: np.ndarray,
    *,
    bits: int = 4,
    scale: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning the quantized tensor and row scales."""
    result = quantize_activation_per_row(activation, bits=bits, scales=scale)
    values = result.values.reshape(result.original_shape)
    if len(result.original_shape) == 1:
        values = values.reshape(-1)
    return np.ascontiguousarray(values), result.scales


def dequantize_matmul(
    integer_output: np.ndarray,
    activation_scales: float | np.ndarray,
    weight_scales: np.ndarray,
    *,
    output_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    value = np.asarray(integer_output, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise QuantizationError("integer output must be a vector or matrix")
    w_scales = np.asarray(weight_scales, dtype=np.float32).reshape(-1)
    if value.shape[1] != w_scales.size:
        raise QuantizationError("output width and weight scales differ")
    a_scales = np.asarray(activation_scales, dtype=np.float32).reshape(-1)
    if a_scales.size == 1 and value.shape[0] != 1:
        a_scales = np.full(value.shape[0], float(a_scales[0]), dtype=np.float32)
    if a_scales.shape != (value.shape[0],):
        raise QuantizationError("one activation scale is required per row")
    result = value * a_scales[:, None] * w_scales[None, :]
    if output_shape is not None:
        return np.ascontiguousarray(result.reshape(output_shape))
    return np.ascontiguousarray(result[0] if np.asarray(integer_output).ndim == 1 else result)


def dequantize_linear_output(
    integer_output: np.ndarray,
    activation_scale: float | np.ndarray,
    weight_scales: np.ndarray,
) -> np.ndarray:
    return dequantize_matmul(integer_output, activation_scale, weight_scales)


def signed_dot_bound(in_features: int, weight_bits: int = 4, activation_bits: int = 4) -> int:
    return int(in_features) * signed_qmax(weight_bits) * signed_qmax(activation_bits)


def minimum_ring_bits(in_features: int, weight_bits: int = 4, activation_bits: int = 4) -> int:
    bound = signed_dot_bound(in_features, weight_bits, activation_bits)
    return 16 if bound < (1 << 15) else 32


# NTT-friendly primes p = k * 16384 + 1 for BFV batching at polynomial
# degree 8192. The list covers W8A8 dot products up to Qwen's widest stage.
_PLAIN_MODULI = (
    65_537,
    786_433,
    1_032_193,
    2_424_833,
    4_079_617,
    6_537_217,
    10_027_009,
    33_538_049,
    60_014_593,
    268_369_921,
    536_690_689,
    1_073_692_673,
)


def choose_plain_modulus(
    in_features: int,
    *,
    weight_bits: int = 4,
    activation_bits: int = 4,
    safety_factor: int = 2,
) -> int:
    required = signed_dot_bound(in_features, weight_bits, activation_bits) * int(safety_factor) + 1
    for modulus in _PLAIN_MODULI:
        if modulus > required:
            return modulus
    raise QuantizationError(f"no configured plaintext modulus covers bound {required}")


def choose_wire_bits(modulus: int) -> int:
    if modulus <= 0:
        raise QuantizationError("modulus must be positive")
    if modulus <= (1 << 16):
        return 16
    if modulus <= (1 << 24):
        return 24
    if modulus <= (1 << 32):
        return 32
    raise QuantizationError("modulus does not fit the wire codec")


def positive_residues(values: np.ndarray, modulus: int) -> np.ndarray:
    if modulus <= 1:
        raise QuantizationError("modulus must be greater than one")
    return np.mod(np.asarray(values, dtype=np.int64), int(modulus)).astype(np.uint32)


def centered_residues(values: np.ndarray, modulus: int) -> np.ndarray:
    if modulus <= 1:
        raise QuantizationError("modulus must be greater than one")
    raw = np.mod(np.asarray(values, dtype=np.int64), int(modulus))
    return np.where(raw > modulus // 2, raw - modulus, raw).astype(np.int64)
