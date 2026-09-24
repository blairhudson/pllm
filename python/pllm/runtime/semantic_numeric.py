"""Bounded local numeric contracts shared by semantic decoder operators."""

from __future__ import annotations

import math
from typing import Any

import ml_dtypes
import numpy as np


class SemanticNumericError(ValueError):
    pass


def _finite_float32(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise SemanticNumericError("semantic numeric operand must be finite")
    return array


def round_bfloat16(value: np.ndarray | float) -> np.ndarray:
    """Round finite float32 values to BF16 (ties to even), retaining float32 storage."""
    rounded = _finite_float32(value).astype(ml_dtypes.bfloat16).astype(np.float32)
    if not np.all(np.isfinite(rounded)):
        raise SemanticNumericError("semantic numeric result exceeds the BF16 domain")
    return rounded


def bfloat16_scale(value: np.ndarray, factor: dict[str, Any], weight: np.ndarray | None = None) -> np.ndarray:
    if (
        type(factor) is not dict
        or factor.get("factor_rounding_dtype") != "bfloat16"
        or factor.get("compute_dtype") != "bfloat16"
        or factor.get("output_dtype") != "bfloat16"
    ):
        raise SemanticNumericError("unsupported semantic scale dtype contract")
    kind = factor.get("kind")
    if kind in {"sqrt", "inverse_sqrt"}:
        if factor.get("factor_source_dtype") not in {"float32_buffer", "python_float64"}:
            raise SemanticNumericError("unsupported semantic scale factor source")
        radicand = factor.get("radicand")
        if type(radicand) is not int or radicand <= 0:
            raise SemanticNumericError("semantic scale radicand must be positive")
        coefficient = math.sqrt(radicand)
        if kind == "inverse_sqrt":
            coefficient = 1.0 / coefficient
    elif kind == "rational":
        if factor.get("factor_source_dtype") != "exact_integer":
            raise SemanticNumericError("unsupported rational scale factor source")
        numerator, denominator = factor.get("numerator"), factor.get("denominator")
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise SemanticNumericError("semantic rational scale requires bounded integers")
        coefficient = numerator / denominator
    elif kind == "learned":
        if factor.get("factor_source_dtype") != "bfloat16" or weight is None:
            raise SemanticNumericError("semantic learned scale requires a bound BF16 weight")
        coefficient = weight
    else:
        raise SemanticNumericError("unsupported semantic scale factor")
    input_value = round_bfloat16(value)
    rounded_factor = round_bfloat16(coefficient)
    if rounded_factor.size not in {1, input_value.shape[-1]}:
        raise SemanticNumericError("semantic scale factor width differs from input")
    return round_bfloat16(input_value * rounded_factor)


def bfloat16_gelu_tanh(value: np.ndarray) -> np.ndarray:
    source = round_bfloat16(value)
    # The cubic overflows at extreme finite BF16 inputs; tanh(±inf) gives the
    # correct limiting value. Reject a non-finite final result instead.
    with np.errstate(over="ignore", under="ignore"):
        expression = np.float32(math.sqrt(2.0 / math.pi)) * (
            source + np.float32(0.044715) * source * source * source
        )
        return round_bfloat16(np.float32(0.5) * source * (np.float32(1.0) + np.tanh(expression)))


def bfloat16_softcap(value: np.ndarray, cap: int) -> np.ndarray:
    if type(cap) is not int or cap <= 0:
        raise SemanticNumericError("softcap requires a positive public integer cap")
    source = round_bfloat16(value)
    rounded_cap = round_bfloat16(float(cap))
    ratio = round_bfloat16(source / rounded_cap)
    return round_bfloat16(round_bfloat16(np.tanh(ratio)) * rounded_cap)


__all__ = [
    "SemanticNumericError",
    "bfloat16_gelu_tanh",
    "bfloat16_scale",
    "bfloat16_softcap",
    "round_bfloat16",
]
