from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shared_mpc import SharedMPCError, SharedTensor
from .shared_party import SharedComputeParty


@dataclass(slots=True)
class SharedKVCache:
    session_id: str
    party: int
    maximum_tokens: int
    key: SharedTensor | None = None
    value: SharedTensor | None = None

    def append(self, key: SharedTensor, value: SharedTensor) -> None:
        if (
            key.session_id != self.session_id
            or value.session_id != self.session_id
            or key.party != self.party
            or value.party != self.party
        ):
            raise SharedMPCError("KV share belongs to another party or session")
        if key.shape != value.shape or key.scale != value.scale or key.values.ndim != 4:
            raise SharedMPCError("KV shares are incompatible")
        if self.key is None:
            next_key, next_value = key.values.copy(), value.values.copy()
        else:
            assert self.value is not None
            if self.key.scale != key.scale or self.value.scale != value.scale:
                raise SharedMPCError("KV share scale changed within session")
            if self.key.shape[:2] != key.shape[:2] or self.key.shape[3:] != key.shape[3:]:
                raise SharedMPCError("KV share shape changed within session")
            next_key = np.concatenate((self.key.values, key.values), axis=2)
            next_value = np.concatenate((self.value.values, value.values), axis=2)
        if next_key.shape[2] > self.maximum_tokens:
            raise SharedMPCError("shared KV cache token budget exceeded")
        self.key = SharedTensor(self.session_id, "kv.key", self.party, next_key, key.scale)
        self.value = SharedTensor(self.session_id, "kv.value", self.party, next_value, value.scale)

    def tensors(self) -> tuple[SharedTensor, SharedTensor]:
        if self.key is None or self.value is None:
            raise SharedMPCError("shared KV cache is empty")
        return self.key, self.value


def _local_tensor(source: SharedTensor, tensor_id: str, values: np.ndarray) -> SharedTensor:
    return SharedTensor(
        source.session_id,
        tensor_id,
        source.party,
        np.ascontiguousarray(values),
        source.scale,
    )


class NetworkSharedAttention:
    """Causal polynomial attention that never opens scores or KV state."""

    def __init__(
        self,
        party: SharedComputeParty,
        *,
        scale: int,
        prefix: str,
        reciprocal_iterations: int = 4,
        probability_scale: int = 1 << 24,
        key_tile_size: int = 64,
    ) -> None:
        if scale < 2 or scale & (scale - 1):
            raise ValueError("attention scale must be a power of two of at least two")
        if not prefix:
            raise ValueError("attention operation prefix cannot be empty")
        if reciprocal_iterations < 1:
            raise ValueError("attention reciprocal needs at least one refinement")
        if probability_scale < scale or probability_scale & (probability_scale - 1):
            raise ValueError("attention probability scale must be a power of two at least scale")
        if key_tile_size <= 0 or key_tile_size > 4096:
            raise ValueError("attention key tile size is invalid")
        self.party = party
        self.scale = scale
        self.prefix = prefix
        self.reciprocal_iterations = reciprocal_iterations
        self.probability_scale = probability_scale
        self.key_tile_size = key_tile_size

    @staticmethod
    def _causal_mask(query_tokens: int, key_tokens: int) -> np.ndarray:
        if query_tokens <= 0 or key_tokens < query_tokens:
            raise SharedMPCError("causal attention dimensions are invalid")
        past = key_tokens - query_tokens
        query = np.arange(query_tokens)[:, None]
        key = np.arange(key_tokens)[None, :]
        return (key <= past + query).astype(np.int64)

    def _initial_inverse(self, *, past: int, query_tokens: int) -> np.ndarray:
        counts = np.arange(past + 1, past + query_tokens + 1, dtype=np.float64)
        return np.rint(self.probability_scale / counts).astype(np.int64)[None, None, :, None]

    async def run(
        self,
        query: SharedTensor,
        key: SharedTensor,
        value: SharedTensor,
        *,
        query_bound: int,
        key_bound: int,
        value_bound: int,
    ) -> SharedTensor:
        if query.scale != self.scale or key.scale != self.scale or value.scale != self.scale:
            raise SharedMPCError("attention share scale mismatch")
        if query.values.ndim != 4 or key.values.ndim != 4 or value.shape != key.shape:
            raise SharedMPCError("attention shares must be [batch, heads, tokens, width]")
        if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
            raise SharedMPCError("attention query and KV shapes are incompatible")
        if min(query_bound, key_bound, value_bound) <= 0:
            raise SharedMPCError("attention signed bounds must be positive")
        batch, heads, query_tokens, width = query.shape
        key_tokens = key.shape[2]
        if key_tokens > self.probability_scale:
            raise SharedMPCError("attention context exceeds probability precision")
        kv_heads = key.shape[1]
        if heads % kv_heads:
            raise SharedMPCError("query heads must be divisible by KV heads")
        if heads != kv_heads:
            repeats = heads // kv_heads
            key = _local_tensor(
                key, f"{self.prefix}.key_groups", np.repeat(key.values, repeats, axis=1)
            )
            value = _local_tensor(
                value, f"{self.prefix}.value_groups", np.repeat(value.values, repeats, axis=1)
            )
        runtime = self.party.runtime
        bits = self.scale.bit_length() - 1

        score_chunks = []
        for index, start in enumerate(range(0, key_tokens, self.key_tile_size)):
            stop = min(start + self.key_tile_size, key_tokens)
            q_values = np.broadcast_to(
                query.values[:, :, :, None, :],
                (batch, heads, query_tokens, stop - start, width),
            )
            k_values = np.broadcast_to(key.values[:, :, None, start:stop, :], q_values.shape)
            q_expanded = _local_tensor(query, f"{self.prefix}.query_expanded.{index}", q_values)
            k_expanded = _local_tensor(key, f"{self.prefix}.key_expanded.{index}", k_values)
            products = await self.party.multiply(
                q_expanded, k_expanded, f"{self.prefix}.qk.{index}"
            )
            score_chunks.append(runtime.sum_axis(products, -1, f"{self.prefix}.score_sum.{index}"))
        scores = _local_tensor(
            query,
            f"{self.prefix}.score_sum",
            np.concatenate([chunk.values for chunk in score_chunks], axis=-1),
        )
        score_bound = query_bound * key_bound * width
        scores = await self.party.truncate(
            scores,
            f"{self.prefix}.score_truncate",
            bits=bits,
            signed_bound=score_bound,
        )
        scale_coefficient = max(1, round(self.scale / np.sqrt(width)))
        scores = runtime.mul_public(scores, scale_coefficient, f"{self.prefix}.score_scale")
        scores = runtime.reinterpret_scale(
            scores, self.scale * self.scale, f"{self.prefix}.score_scaled"
        )
        score_bound = ((score_bound >> bits) + 1) * scale_coefficient
        scores = await self.party.truncate(
            scores,
            f"{self.prefix}.scale_truncate",
            bits=bits,
            signed_bound=score_bound,
        )
        bounded_score = (score_bound >> bits) + 1
        if bounded_score > self.scale // 2:
            raise SharedMPCError("attention score bound exceeds polynomial approximation range")
        mask = self._causal_mask(query_tokens, key_tokens)[None, None, :, :]
        scores = runtime.mul_public_tensor(scores, mask, f"{self.prefix}.causal_score")

        score_square = await self.party.multiply(scores, scores, f"{self.prefix}.square")
        score_square = await self.party.truncate(
            score_square,
            f"{self.prefix}.square_truncate",
            bits=bits,
            signed_bound=bounded_score * bounded_score,
        )
        half_square = runtime.mul_public(
            score_square, self.scale // 2, f"{self.prefix}.half_square"
        )
        half_square = runtime.reinterpret_scale(
            half_square, self.scale * self.scale, f"{self.prefix}.half_scaled"
        )
        square_bound = (bounded_score * bounded_score >> bits) + 1
        half_square_bound = square_bound * (self.scale // 2)
        half_square = await self.party.truncate(
            half_square,
            f"{self.prefix}.half_truncate",
            bits=bits,
            signed_bound=half_square_bound,
        )
        exponential = runtime.add(scores, half_square, f"{self.prefix}.exponential_terms")
        exponential = runtime.add_public(
            exponential, mask * self.scale, f"{self.prefix}.exponential"
        )

        denominator = runtime.sum_axis(exponential, -1, f"{self.prefix}.denominator", keepdims=True)
        initial_inverse = self._initial_inverse(
            past=key_tokens - query_tokens, query_tokens=query_tokens
        )
        inverse = runtime.mul_public(denominator, 0, f"{self.prefix}.inverse_zero")
        inverse = runtime.reinterpret_scale(
            inverse,
            self.probability_scale,
            f"{self.prefix}.inverse_scaled",
        )
        exponential_bound = self.scale + bounded_score + (half_square_bound >> bits) + 1
        denominator_bound = exponential_bound * key_tokens
        inverse = runtime.add_public(inverse, initial_inverse, f"{self.prefix}.inverse")
        # On the enforced score interval, 1 + x + x^2/2 is in [5/8, 13/8].
        # Public 1/count is therefore inside Newton's convergence interval.
        inverse_bound = int(np.max(initial_inverse)) + 1
        for iteration in range(self.reciprocal_iterations):
            product = await self.party.multiply(
                denominator,
                inverse,
                f"{self.prefix}.inverse_refine_{iteration}.product",
            )
            product_bound = denominator_bound * inverse_bound
            product = await self.party.truncate(
                product,
                f"{self.prefix}.inverse_refine_{iteration}.truncate",
                bits=bits,
                signed_bound=product_bound,
            )
            residual = runtime.mul_public(
                product, -1, f"{self.prefix}.inverse_refine_{iteration}.negate"
            )
            residual = runtime.add_public(
                residual,
                2 * self.probability_scale,
                f"{self.prefix}.inverse_refine_{iteration}.residual",
            )
            residual_bound = 2 * self.probability_scale + 1
            inverse = await self.party.multiply(
                inverse,
                residual,
                f"{self.prefix}.inverse_refine_{iteration}.update",
            )
            update_bound = inverse_bound * residual_bound
            inverse = await self.party.truncate(
                inverse,
                f"{self.prefix}.inverse_refine_{iteration}.update_truncate",
                bits=self.probability_scale.bit_length() - 1,
                signed_bound=update_bound,
            )
            inverse_bound = 2 * self.probability_scale + 2
        inverse = runtime.broadcast_last(inverse, key_tokens, f"{self.prefix}.inverse_broadcast")
        probability = await self.party.multiply(exponential, inverse, f"{self.prefix}.probability")
        probability = await self.party.truncate(
            probability,
            f"{self.prefix}.probability_truncate",
            bits=bits,
            signed_bound=exponential_bound * inverse_bound,
        )

        weighted = None
        for index, start in enumerate(range(0, key_tokens, self.key_tile_size)):
            stop = min(start + self.key_tile_size, key_tokens)
            probability_values = np.broadcast_to(
                probability.values[..., start:stop, None],
                (*probability.shape[:-1], stop - start, width),
            )
            value_values = np.broadcast_to(
                value.values[:, :, None, start:stop, :], probability_values.shape
            )
            probability_expanded = _local_tensor(
                probability, f"{self.prefix}.probability_expanded.{index}", probability_values
            )
            value_expanded = _local_tensor(
                value, f"{self.prefix}.value_expanded.{index}", value_values
            )
            product = await self.party.multiply(
                probability_expanded, value_expanded, f"{self.prefix}.value.{index}"
            )
            partial = runtime.sum_axis(product, -2, f"{self.prefix}.value_sum.{index}")
            weighted = (
                partial
                if weighted is None
                else runtime.add(weighted, partial, f"{self.prefix}.value_accumulate.{index}")
            )
        if weighted is None:
            raise SharedMPCError("attention value product is empty")
        probability_bound = (exponential_bound * inverse_bound >> bits) + 1
        return await self.party.truncate(
            weighted,
            f"{self.prefix}.value_truncate",
            bits=self.probability_scale.bit_length() - 1,
            signed_bound=probability_bound * value_bound * key_tokens,
        )
