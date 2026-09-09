from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shared_mpc import SharedMPCError, SharedTensor
from .shared_party import SharedComputeParty


@dataclass(frozen=True, slots=True)
class RMSNormApproximation:
    """Public fixed-point tangent used for inverse square root."""

    weight: np.ndarray
    intercept: int
    slope: int
    epsilon: int
    scale: int
    minimum_statistic: int
    maximum_statistic: int
    maximum_sampled_relative_error: float

    def __post_init__(self) -> None:
        raw_weight = np.asarray(self.weight)
        if raw_weight.ndim != 1 or raw_weight.size == 0:
            raise SharedMPCError("RMSNorm weight must be a non-empty vector")
        if not np.issubdtype(raw_weight.dtype, np.integer):
            raise SharedMPCError("RMSNorm weight must contain integers")
        if int(raw_weight.min()) < -(1 << 63) or int(raw_weight.max()) >= 1 << 63:
            raise SharedMPCError("RMSNorm weight exceeds signed ring range")
        weight = np.ascontiguousarray(raw_weight, dtype=np.int64)
        weight.setflags(write=False)
        if self.scale < 2 or self.scale & (self.scale - 1):
            raise SharedMPCError("RMSNorm scale must be a power of two of at least two")
        if self.epsilon < 0:
            raise SharedMPCError("RMSNorm epsilon cannot be negative")
        if not 0 < self.minimum_statistic < self.maximum_statistic:
            raise SharedMPCError("RMSNorm calibration interval is invalid")
        if self.epsilon < self.minimum_statistic:
            raise SharedMPCError("RMSNorm epsilon does not enforce calibrated lower bound")
        if not 0 < self.maximum_sampled_relative_error < 1:
            raise SharedMPCError("RMSNorm relative-error limit is invalid")
        samples = np.linspace(self.minimum_statistic, self.maximum_statistic, 65537)
        estimate = self.intercept + samples * self.slope / self.scale
        expected = self.scale / np.sqrt(samples / self.scale)
        if (
            float(np.max(np.abs(estimate - expected) / expected))
            > self.maximum_sampled_relative_error
        ):
            raise SharedMPCError("RMSNorm approximation exceeds its relative-error limit")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class RMSNormBounds:
    square: int
    square_scaled: int
    mean_product: int
    statistic: int
    inverse_product: int
    inverse: int
    normalized_product: int
    weight_product: int


class NetworkSharedRMSNorm:
    def __init__(
        self,
        party: SharedComputeParty,
        approximation: RMSNormApproximation,
        *,
        prefix: str,
    ) -> None:
        if not prefix:
            raise ValueError("RMSNorm operation prefix cannot be empty")
        self.party = party
        self.approximation = approximation
        self.prefix = prefix

    def bounds(self, signed_bound: int) -> RMSNormBounds:
        approximation = self.approximation
        width = int(approximation.weight.size)
        mean_scale = 1 << (approximation.scale.bit_length() - 1 + (width - 1).bit_length())
        mean_coefficient = round(mean_scale / width)
        square = signed_bound * signed_bound
        square_scaled = (square >> (approximation.scale.bit_length() - 1)) + 1
        mean_product = square_scaled * width * mean_coefficient
        statistic = (mean_product >> (mean_scale.bit_length() - 1)) + 1
        statistic += approximation.epsilon
        inverse_product = statistic * abs(approximation.slope)
        inverse = (
            (inverse_product >> (approximation.scale.bit_length() - 1))
            + abs(approximation.intercept)
            + 1
        )
        normalized_product = signed_bound * inverse
        normalized = (normalized_product >> (approximation.scale.bit_length() - 1)) + 1
        weight_product = normalized * int(np.max(np.abs(approximation.weight)))
        return RMSNormBounds(
            square,
            square_scaled,
            mean_product,
            statistic,
            inverse_product,
            inverse,
            normalized_product,
            weight_product,
        )

    async def run(self, value: SharedTensor, *, signed_bound: int) -> SharedTensor:
        approximation = self.approximation
        if value.scale != approximation.scale or value.shape[-1] != approximation.weight.size:
            raise SharedMPCError("RMSNorm tensor does not match approximation")
        runtime = self.party.runtime
        scale = approximation.scale
        fraction_bits = scale.bit_length() - 1
        bounds = self.bounds(signed_bound)
        if bounds.statistic > self.approximation.maximum_statistic:
            raise SharedMPCError("RMSNorm bound exceeds calibrated approximation interval")

        square = await self.party.multiply(value, value, f"{self.prefix}.square")
        square = await self.party.truncate(
            square,
            f"{self.prefix}.square_truncate",
            bits=fraction_bits,
            signed_bound=bounds.square,
        )
        total = runtime.sum_last(square, f"{self.prefix}.sum")
        mean_scale = 1 << (fraction_bits + (approximation.weight.size - 1).bit_length())
        mean_coefficient = round(mean_scale / approximation.weight.size)
        statistic = runtime.mul_public(total, mean_coefficient, f"{self.prefix}.mean_product")
        statistic = runtime.reinterpret_scale(
            statistic,
            scale * mean_scale,
            f"{self.prefix}.mean_scaled",
        )
        statistic = await self.party.truncate(
            statistic,
            f"{self.prefix}.mean_truncate",
            bits=mean_scale.bit_length() - 1,
            signed_bound=bounds.mean_product,
        )
        statistic = runtime.add_public(
            statistic,
            approximation.epsilon,
            f"{self.prefix}.epsilon",
        )

        inverse = runtime.mul_public(
            statistic, approximation.slope, f"{self.prefix}.inverse_product"
        )
        inverse = runtime.reinterpret_scale(inverse, scale * scale, f"{self.prefix}.inverse_scaled")
        inverse = await self.party.truncate(
            inverse,
            f"{self.prefix}.inverse_truncate",
            bits=fraction_bits,
            signed_bound=bounds.inverse_product,
        )
        inverse = runtime.add_public(inverse, approximation.intercept, f"{self.prefix}.inverse")
        inverse = runtime.broadcast_last(
            inverse, value.shape[-1], f"{self.prefix}.inverse_broadcast"
        )

        normalized = await self.party.multiply(value, inverse, f"{self.prefix}.normalize")
        normalized = await self.party.truncate(
            normalized,
            f"{self.prefix}.normalize_truncate",
            bits=fraction_bits,
            signed_bound=bounds.normalized_product,
        )
        output = runtime.mul_public_tensor(
            normalized, approximation.weight, f"{self.prefix}.weight"
        )
        output = runtime.reinterpret_scale(output, scale * scale, f"{self.prefix}.weight_scaled")
        return await self.party.truncate(
            output,
            f"{self.prefix}.weight_truncate",
            bits=fraction_bits,
            signed_bound=bounds.weight_product,
        )
