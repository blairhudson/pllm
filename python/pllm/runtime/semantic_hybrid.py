"""Bounded, local-clear reference for stateful depthwise causal convolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


class HybridStateError(ValueError):
    """A declared hybrid-operator tensor or resource bound is invalid."""


def _finite_float32(values: np.ndarray, name: str, rank: int) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.dtype != np.float32
        or array.ndim != rank
        or array.size > MAX_CONVOLUTION_OUTPUT_ELEMENTS
        or not np.all(np.isfinite(array))
    ):
        raise HybridStateError(f"{name} must be finite float32 with rank {rank}")
    return array


MAX_CONVOLUTION_CHANNELS = 8192
MAX_CONVOLUTION_KERNEL = 16
MAX_CONVOLUTION_TOKENS = 256
MAX_CONVOLUTION_STATE_BYTES = 2 << 20
MAX_CONVOLUTION_OUTPUT_ELEMENTS = 1 << 22


class BoundedDepthwiseCausalConvolution:
    """Immutable public weights; pure, fail-before-write state transitions.

    Inputs are ``[batch, sequence, channels]``; state owns the last ``kernel``
    *raw* projected samples in ``[batch, channels, kernel]`` order. This numeric
    reference is not an executable Pipeline component or a privacy guarantee.
    """

    __slots__ = ("_weight", "_bias", "channels", "kernel")

    def __init__(self, weight: np.ndarray, *, bias: np.ndarray | None = None) -> None:
        array = np.asarray(weight)
        if (
            array.dtype != np.float32
            or array.ndim != 2
            or not 1 <= array.shape[0] <= MAX_CONVOLUTION_CHANNELS
            or not 1 <= array.shape[1] <= MAX_CONVOLUTION_KERNEL
            or array.size * np.dtype(np.float32).itemsize > MAX_CONVOLUTION_STATE_BYTES
            or not np.all(np.isfinite(array))
        ):
            raise HybridStateError("convolution weight must be finite, bounded float32 [channels, kernel]")
        if bias is not None:
            offset = np.asarray(bias)
            if (
                offset.dtype != np.float32
                or offset.shape != (array.shape[0],)
                or not np.all(np.isfinite(offset))
            ):
                raise HybridStateError("convolution bias must match the finite float32 channel count")
            self._bias = np.frombuffer(offset.tobytes(), dtype=np.float32)
        else:
            self._bias = None
        self.channels, self.kernel = array.shape
        self._weight = np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)

    def initial_state(self) -> np.ndarray:
        """Create zero-filled state for one batch-one prefill."""
        return np.zeros((1, self.channels, self.kernel), dtype=np.float32)

    def evaluate(
        self,
        values: np.ndarray,
        state: np.ndarray,
        *,
        mode: Literal["prefill", "decode"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute SiLU(depthwise convolution) and return new raw-input state."""
        samples, retained = np.asarray(values), np.asarray(state)
        if (
            samples.dtype != np.float32
            or samples.ndim != 3
            or samples.shape[0] != 1
            or samples.shape[2] != self.channels
            or not 1 <= samples.shape[1] <= MAX_CONVOLUTION_TOKENS
            or samples.size > MAX_CONVOLUTION_OUTPUT_ELEMENTS
            or retained.dtype != np.float32
            or retained.shape != (1, self.channels, self.kernel)
            or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(retained))
        ):
            raise HybridStateError("convolution inputs and retained state exceed the finite float32 contract")
        if mode not in {"prefill", "decode"} or (mode == "decode" and samples.shape[1] != 1):
            raise HybridStateError("convolution mode does not match the sequence length")

        projected = samples.transpose(0, 2, 1)
        history = np.concatenate((retained[:, :, 1:], projected), axis=-1)
        windows = np.lib.stride_tricks.sliding_window_view(history, self.kernel, axis=2)
        with np.errstate(over="ignore", invalid="ignore"):
            linear = np.einsum("bctk,ck->btc", windows, self._weight, optimize=False)
            if self._bias is not None:
                linear += self._bias
            positive = linear >= 0
            result = np.empty_like(linear)
            result[positive] = linear[positive] / (1.0 + np.exp(-linear[positive]))
            negative_exp = np.exp(linear[~positive])
            result[~positive] = linear[~positive] * negative_exp / (1.0 + negative_exp)
        if not np.all(np.isfinite(result)):
            raise HybridStateError("convolution result exceeds the finite float32 domain")
        new_state = np.concatenate((retained, projected), axis=-1)[:, :, -self.kernel :].copy()
        return result, new_state


def bounded_gated_delta_decay(
    projected: np.ndarray, a_log: np.ndarray, dt_bias: np.ndarray
) -> np.ndarray:
    """Public-coefficient float32 decay: ``-exp(A_log) * softplus(a + dt_bias)``."""
    a = _finite_float32(projected, "projected delta decay", 3)
    log_weight = _finite_float32(a_log, "A_log", 1)
    bias = _finite_float32(dt_bias, "dt_bias", 1)
    if (
        a.shape[0] != 1
        or not 1 <= a.shape[1] <= MAX_CONVOLUTION_TOKENS
        or a.shape[2] != log_weight.shape[0]
        or log_weight.shape != bias.shape
        or not 1 <= bias.size <= 64
    ):
        raise HybridStateError("delta decay tensor shape exceeds its declared head/token bound")
    with np.errstate(over="ignore", invalid="ignore"):
        decay = -np.exp(log_weight)[None, None, :] * np.logaddexp(
            np.float32(0), a + bias[None, None, :]
        )
    if not np.all(np.isfinite(decay)) or np.any(decay > 0):
        raise HybridStateError("delta decay must remain finite and nonpositive")
    return np.ascontiguousarray(decay)


@dataclass(frozen=True, slots=True)
class BoundedGatedDeltaRecurrence:
    """One-party bounded float32 recurrent reference; not a Pipeline component."""

    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int

    def __post_init__(self) -> None:
        if (
            type(self.key_heads) is not int
            or type(self.value_heads) is not int
            or type(self.key_head_dim) is not int
            or type(self.value_head_dim) is not int
            or not 1 <= self.key_heads <= 32
            or not 1 <= self.value_heads <= 64
            or self.value_heads % self.key_heads != 0
            or self.value_heads // self.key_heads > 16
            or not 1 <= self.key_head_dim <= 256
            or not 1 <= self.value_head_dim <= 256
            or self.value_heads * self.key_head_dim * self.value_head_dim * 4 > 8 * 1024 * 1024
        ):
            raise HybridStateError("gated-delta head geometry exceeds its numeric/state budget")

    def initial_state(self) -> np.ndarray:
        return np.zeros(
            (1, self.value_heads, self.key_head_dim, self.value_head_dim),
            dtype=np.float32,
        )

    def evaluate(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        decay: np.ndarray,
        beta: np.ndarray,
        state: np.ndarray,
        *,
        mode: Literal["prefill", "decode"],
    ) -> tuple[np.ndarray, np.ndarray]:
        q, k, v, g, b, prior = self._inputs(query, key, value, decay, beta, state, mode=mode)
        tokens = q.shape[1]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
            k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
            q = np.repeat(q, self.value_heads // self.key_heads, axis=2)
            k = np.repeat(k, self.value_heads // self.key_heads, axis=2)
            q *= np.float32(1.0 / math.sqrt(self.key_head_dim))
            retained = np.ascontiguousarray(prior.copy())
            output = np.empty_like(v)
            for token in range(tokens):
                key_at = k[0, token]
                query_at = q[0, token]
                retained[0] *= np.exp(g[0, token])[:, None, None]
                remembered = np.einsum("hkv,hk->hv", retained[0], key_at)
                delta = (v[0, token] - remembered) * b[0, token, :, None]
                retained[0] += key_at[:, :, None] * delta[:, None, :]
                output[0, token] = np.einsum("hkv,hk->hv", retained[0], query_at)
                if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(output[0, token])):
                    raise HybridStateError("gated-delta execution exceeded its finite float32 domain")
        return np.ascontiguousarray(output), retained

    def _inputs(self, query, key, value, decay, beta, state, *, mode):
        q = _finite_float32(query, "gated-delta query", 4)
        k = _finite_float32(key, "gated-delta key", 4)
        v = _finite_float32(value, "gated-delta value", 4)
        g = _finite_float32(decay, "gated-delta decay", 3)
        b = _finite_float32(beta, "gated-delta beta", 3)
        prior = _finite_float32(state, "gated-delta state", 4)
        tokens = q.shape[1]
        if (
            mode not in ("prefill", "decode")
            or not 1 <= tokens <= MAX_CONVOLUTION_TOKENS
            or (mode == "decode" and tokens != 1)
            or q.shape != (1, tokens, self.key_heads, self.key_head_dim)
            or k.shape != q.shape
            or v.shape != (1, tokens, self.value_heads, self.value_head_dim)
            or g.shape != (1, tokens, self.value_heads)
            or b.shape != g.shape
            or prior.shape != (1, self.value_heads, self.key_head_dim, self.value_head_dim)
            or np.any(g > 0)
            or np.any(b < 0)
            or np.any(b > 1)
        ):
            raise HybridStateError("gated-delta tensor, state, or phase contract is invalid")
        return q, k, v, g, b, prior


class _GatedDeltaWorkspace:
    """One bounded scratch slab, reusable serially across decoder layers."""

    def __init__(self, geometry: BoundedGatedDeltaRecurrence):
        self.geometry = geometry
        h = geometry
        self.q = np.empty((h.value_heads, h.key_head_dim), np.float32)
        self.k = np.empty_like(self.q)
        self.delta = np.empty((h.value_heads, h.value_head_dim), np.float32)
        self.outer = np.empty((h.value_heads, h.key_head_dim, h.value_head_dim), np.float32)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.q, self.k, self.delta, self.outer))

    def clear(self):
        for array in (self.q, self.k, self.delta, self.outer):
            array.fill(0)

    def evaluate(self, query, key, value, decay, beta, state, *, mode):
        h = self.geometry
        q, k, v, g, b, prior = h._inputs(query, key, value, decay, beta, state, mode=mode)
        if (not prior.flags.c_contiguous or not prior.flags.writeable
                or any(np.shares_memory(prior, x) for x in (q, k, v, g, b))):
            raise HybridStateError("owned recurrence requires exclusive contiguous writable state")
        output = np.empty_like(v, order="C")
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                # Keep the reference's complete tensor layout for reductions.
                # Token slicing can change NumPy's reduction order for strided
                # checkpoint projections, changing retained state by a few ULPs.
                q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
                k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
                for token in range(q.shape[1]):
                    for dst, src in ((self.q, q), (self.k, k)):
                        dst.reshape(h.key_heads, h.value_heads // h.key_heads, h.key_head_dim)[:] = src[0, token, :, None, :]
                    self.q *= np.float32(1.0 / math.sqrt(h.key_head_dim))
                    prior[0] *= np.exp(g[0, token])[:, None, None]
                    np.einsum("hkv,hk->hv", prior[0], self.k, out=self.delta)
                    np.subtract(v[0, token], self.delta, out=self.delta)
                    self.delta *= b[0, token, :, None]
                    np.multiply(self.k[:, :, None], self.delta[:, None, :], out=self.outer)
                    prior[0] += self.outer
                    np.einsum("hkv,hk->hv", prior[0], self.q, out=output[0, token])
                    if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(output[0, token])):
                        raise HybridStateError("gated-delta execution exceeded its finite float32 domain")
        except BaseException:
            prior.fill(0)
            output.fill(0)
            self.clear()
            raise
        return output, prior


class OwnedGatedDeltaRecurrence:
    """Local float32 recurrence with explicit state ownership and retirement.

    Returned outputs and snapshots are independent arrays. Invalid inputs leave
    state intact; a failure during arithmetic burns the owner. Instances are
    single-threaded and cannot be copied. No private shares or role transport.
    """

    def __init__(self, key_heads: int, value_heads: int, key_head_dim: int,
                 value_head_dim: int, *, maximum_tokens: int = 4096):
        if type(maximum_tokens) is not int or not 1 <= maximum_tokens <= 1 << 20:
            raise HybridStateError("owned recurrence token bound must be in [1, 1048576]")
        self.geometry = BoundedGatedDeltaRecurrence(key_heads, value_heads, key_head_dim, value_head_dim)
        self.maximum_tokens = maximum_tokens
        self.position = 0
        self._state = self.geometry.initial_state()
        self._workspace = _GatedDeltaWorkspace(self.geometry)

    @property
    def state_bytes(self) -> int:
        return 0 if self._state is None else self._state.nbytes

    @property
    def workspace_bytes(self) -> int:
        return 0 if self._workspace is None else self._workspace.nbytes

    def snapshot(self) -> np.ndarray:
        if self._state is None:
            raise HybridStateError("owned recurrence is closed")
        return self._state.copy()

    def evaluate(self, query, key, value, decay, beta, *, mode="decode") -> np.ndarray:
        if self._state is None:
            raise HybridStateError("owned recurrence is closed")
        arrays = self.geometry._inputs(query, key, value, decay, beta, self._state, mode=mode)
        tokens = arrays[0].shape[1]
        if self.position + tokens > self.maximum_tokens:
            raise HybridStateError("owned recurrence exceeds its token bound")
        try:
            output, _ = self._workspace.evaluate(*arrays, mode=mode)
        except BaseException:
            self.close()
            raise
        self.position += tokens
        return output

    def close(self) -> None:
        if self._state is not None:
            self._state.fill(0)
            self._workspace.clear()
            self._state = self._workspace = None

    def __copy__(self):
        raise TypeError("owned recurrence cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("owned recurrence cannot be copied")

    def __reduce__(self):
        raise TypeError("owned recurrence cannot be serialized")

    def __enter__(self):
        if self._state is None:
            raise HybridStateError("owned recurrence is closed")
        return self

    def __exit__(self, *_):
        self.close()


def execute_declared_causal_convolution(
    operation: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    weight: np.ndarray,
) -> np.ndarray:
    """Execute one exact semantic convolution/state step using client-bound weights."""
    kind = operation.get("operator")
    attrs = operation.get("attributes")
    inputs = operation.get("inputs")
    shape = operation.get("output_shape")
    if (
        kind not in {"causal_convolution", "convolution_state_update"}
        or not isinstance(attrs, dict)
        or set(attrs) != {"weight", "bias", "kernel_size", "groups", "activation", "mode"}
        or not isinstance(attrs.get("weight"), str)
        or not attrs["weight"]
        or attrs["bias"] is not None
        or attrs["activation"] != "silu"
        or type(attrs["kernel_size"]) is not int
        or type(attrs["groups"]) is not int
        or attrs["mode"] not in {"chunk", "recurrent"}
        or not isinstance(inputs, list)
        or len(inputs) != 2
        or inputs[0] not in values
        or inputs[1] not in values
        or not isinstance(shape, list)
    ):
        raise HybridStateError("semantic convolution declaration is invalid")
    source, state, coefficients = (np.asarray(values[inputs[0]]), np.asarray(values[inputs[1]]), np.asarray(weight))
    if source.ndim == 2 and source.shape[1] == attrs["groups"]:
        source = source[None, :, :]
    if (
        coefficients.ndim != 3
        or coefficients.shape != (attrs["groups"], 1, attrs["kernel_size"])
        or source.ndim != 3
        or source.shape[0] != 1
        or source.shape[2] != attrs["groups"]
        or (kind == "causal_convolution" and (
            len(shape) != 3 or shape[0] != 1 or shape[2] != attrs["groups"]
            or not 0 < source.shape[1] <= shape[1]
        ))
        or (kind == "convolution_state_update" and (
            shape != [1, attrs["groups"], attrs["kernel_size"]]
            or operation.get("state_kind") != "convolution"
        ))
    ):
        raise HybridStateError("semantic convolution tensor shapes do not match the declaration")
    kernel = BoundedDepthwiseCausalConvolution(coefficients[:, 0, :])
    convolved, updated = kernel.evaluate(
        source,
        state,
        mode="prefill" if attrs["mode"] == "chunk" else "decode",
    )
    return convolved if kind == "causal_convolution" else updated


def execute_declared_gated_delta(
    operation: Mapping[str, Any], values: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Bind an exact paired recurrent operator to the bounded float32 reference."""
    kernel, arrays, mode = _bind_declared_gated_delta(operation, values)
    output, retained = kernel.evaluate(*arrays, mode=mode)
    return output if operation["operator"] == "gated_delta_rule" else retained


def _bind_declared_gated_delta(operation, values):
    kind = operation.get("operator")
    attrs = operation.get("attributes")
    inputs = operation.get("inputs")
    shape = operation.get("output_shape")
    if (
        kind not in {"gated_delta_rule", "gated_delta_state_update"}
        or not isinstance(attrs, dict)
        or set(attrs) != {
            "qk_l2_normalize", "normalization_epsilon", "query_scale", "key_head_repeats",
            "mode", "chunk_size", "state_update",
        }
        or attrs["qk_l2_normalize"] is not True
        or attrs["normalization_epsilon"] != "0.000001"
        or not isinstance(attrs["query_scale"], dict)
        or set(attrs["query_scale"]) != {"numerator", "sqrt_denominator"}
        or type(attrs["query_scale"]["numerator"]) is not int
        or attrs["query_scale"]["numerator"] != 1
        or type(attrs["query_scale"]["sqrt_denominator"]) is not int
        or type(attrs["key_head_repeats"]) is not int
        or attrs["mode"] not in {"chunked", "recurrent"}
        or type(attrs["chunk_size"]) is not int
        or attrs["chunk_size"] != 64
        or attrs["state_update"] != "S'=exp(g)*S+k*(v-S^T*k)*beta"
        or not isinstance(inputs, list)
        or len(inputs) != 6
        or any(not isinstance(name, str) or name not in values for name in inputs)
        or not isinstance(shape, list)
    ):
        raise HybridStateError("semantic gated-delta declaration is invalid")
    q, k, v, decay, beta, prior = (np.asarray(values[name]) for name in inputs)
    if prior.ndim != 4 or len(prior.shape) != 4 or prior.shape[0] != 1:
        raise HybridStateError("gated-delta state has no declared head geometry")
    _, value_heads, key_dim, value_dim = prior.shape
    repeats = attrs["key_head_repeats"]
    if repeats < 1 or value_heads % repeats != 0:
        raise HybridStateError("gated-delta repeated key heads do not divide value heads")
    key_heads = value_heads // repeats
    tokens = q.shape[1] if q.ndim == 3 else 0
    if (
        attrs["query_scale"]["sqrt_denominator"] != key_dim
        or q.shape != (1, tokens, key_heads * key_dim)
        or k.shape != q.shape
        or v.shape != (1, tokens, value_heads * value_dim)
        or (kind == "gated_delta_rule" and (
            len(shape) != 4 or shape[0] != 1 or shape[2:] != [value_heads, value_dim]
            or type(shape[1]) is not int or not 0 < tokens <= shape[1] <= MAX_CONVOLUTION_TOKENS
            or operation.get("state_kind") is not None
        ))
        or (kind == "gated_delta_state_update" and (
            shape != [1, value_heads, key_dim, value_dim]
            or operation.get("state_kind") != "recurrent"
        ))
    ):
        raise HybridStateError("semantic gated-delta shapes or state ownership differ")
    kernel = BoundedGatedDeltaRecurrence(key_heads, value_heads, key_dim, value_dim)
    mode = "prefill" if attrs["mode"] == "chunked" else "decode"
    arrays = kernel._inputs(
        q.reshape(1, tokens, key_heads, key_dim),
        k.reshape(1, tokens, key_heads, key_dim),
        v.reshape(1, tokens, value_heads, value_dim),
        decay, beta, prior,
        mode=mode,
    )
    return kernel, arrays, mode


def execute_declared_gated_rms_norm(
    operation: Mapping[str, Any], values: Mapping[str, np.ndarray], weight: np.ndarray
) -> np.ndarray:
    """Float32 RMSNorm before SiLU gate; input/state dimensions are explicit."""
    attrs = operation.get("attributes")
    inputs = operation.get("inputs")
    shape = operation.get("output_shape")
    raw_epsilon = attrs.get("epsilon") if isinstance(attrs, dict) else None
    try:
        epsilon = float(raw_epsilon) if isinstance(raw_epsilon, (str, int, float)) and not isinstance(raw_epsilon, bool) else math.nan
    except ValueError:
        epsilon = math.nan
    if (
        operation.get("operator") != "rms_norm_gated"
        or not isinstance(attrs, dict)
        or set(attrs) != {"epsilon", "weight", "activation", "weight_offset", "norm_before_gate"}
        or not math.isfinite(epsilon) or not 0 < epsilon <= 0.01
        or not isinstance(attrs["weight"], str) or not attrs["weight"]
        or attrs["activation"] != "silu"
        or type(attrs["weight_offset"]) is not int or attrs["weight_offset"] != 0
        or attrs["norm_before_gate"] is not True
        or not isinstance(inputs, list) or len(inputs) != 2
        or any(not isinstance(name, str) or name not in values for name in inputs)
        or not isinstance(shape, list) or len(shape) != 4
    ):
        raise HybridStateError("gated RMSNorm declaration is invalid")
    source = _finite_float32(values[inputs[0]], "gated RMSNorm input", 4)
    gate = _finite_float32(values[inputs[1]], "gated RMSNorm gate", 4)
    coefficients = _finite_float32(weight, "gated RMSNorm coefficient", 1)
    if (
        source.shape != gate.shape
        or source.shape[0] != 1
        or not 1 <= source.shape[1] <= MAX_CONVOLUTION_TOKENS
        or not 1 <= source.shape[2] <= 64
        or not 1 <= source.shape[3] <= 256
        or source.shape[2:] != tuple(shape[2:])
        or not 0 < source.shape[1] <= shape[1]
        or shape[0] != 1
        or coefficients.shape != (source.shape[-1],)
    ):
        raise HybridStateError("gated RMSNorm tensor and coefficient shapes differ")
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.mean(source * source, axis=-1, keepdims=True)
        normalized = source / np.sqrt(variance + np.float32(epsilon))
        positive = gate >= 0
        activated = np.empty_like(gate)
        activated[positive] = gate[positive] / (1 + np.exp(-gate[positive]))
        negative_exp = np.exp(gate[~positive])
        activated[~positive] = gate[~positive] * negative_exp / (1 + negative_exp)
        result = normalized * coefficients * activated
    if not np.all(np.isfinite(result)):
        raise HybridStateError("gated RMSNorm exceeded its finite float32 domain")
    return np.ascontiguousarray(result)


__all__ = [
    "BoundedDepthwiseCausalConvolution", "BoundedGatedDeltaRecurrence", "OwnedGatedDeltaRecurrence", "HybridStateError",
    "bounded_gated_delta_decay", "execute_declared_causal_convolution", "execute_declared_gated_delta",
    "execute_declared_gated_rms_norm",
]
