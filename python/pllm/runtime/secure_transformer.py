from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .authenticated_mpc import (
    AuthenticatedMPC,
    AuthenticatedValue,
    InputMask,
    LinearCorrelation,
    TrustedPreprocessor,
)
from .secure_selection import SecureSelectionResult, secure_argmax, verify_one_hot


@dataclass(frozen=True, slots=True)
class SecureDecoderConfig:
    vocabulary_size: int = 8
    hidden_size: int = 8
    head_count: int = 2
    head_size: int = 4
    intermediate_size: int = 12
    maximum_context: int = 4
    field_modulus: int = 65537
    attention_offset: int = 4

    def __post_init__(self) -> None:
        if self.head_count * self.head_size != self.hidden_size:
            raise ValueError("head count times head size must equal hidden size")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary size must be at least two")


@dataclass(frozen=True, slots=True)
class SecureDecoderWeights:
    embedding: np.ndarray
    qkv: np.ndarray
    attention_output: np.ndarray
    gate_up: np.ndarray
    down: np.ndarray
    vocabulary_head: np.ndarray
    input_scale: np.ndarray
    mlp_scale: np.ndarray

    @classmethod
    def random(cls, config: SecureDecoderConfig, *, seed: int = 1) -> "SecureDecoderWeights":
        rng = np.random.default_rng(seed)

        def matrix(rows: int, columns: int) -> np.ndarray:
            return rng.integers(-1, 2, size=(rows, columns), dtype=np.int64)

        return cls(
            embedding=matrix(config.hidden_size, config.vocabulary_size),
            qkv=matrix(config.hidden_size * 3, config.hidden_size),
            attention_output=matrix(config.hidden_size, config.hidden_size),
            gate_up=matrix(config.intermediate_size * 2, config.hidden_size),
            down=matrix(config.hidden_size, config.intermediate_size),
            vocabulary_head=matrix(config.vocabulary_size, config.hidden_size),
            input_scale=np.ones(config.hidden_size, dtype=np.int64),
            mlp_scale=np.ones(config.hidden_size, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class SecureTokenMaterial:
    input_mask: InputMask
    correlations: dict[str, LinearCorrelation]
    bit_mask: object | None


@dataclass(slots=True)
class SecureKVCache:
    keys: list[AuthenticatedValue] = field(default_factory=list)
    values: list[AuthenticatedValue] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.keys)


@dataclass(slots=True)
class ClearKVCache:
    keys: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.keys)


@dataclass(frozen=True, slots=True)
class SecureTokenResult:
    token_ids: np.ndarray
    logits: AuthenticatedValue
    selection: SecureSelectionResult | None
    disclosed_logits: np.ndarray | None
    online_rounds: int
    uploaded_bytes: int
    downloaded_bytes: int


class SecureDecoder:
    """A small complete decoder used to measure the secure execution path.

    The graph keeps every intermediate value authenticated and split between
    the client and server. It uses a public calibrated normalization scale and
    polynomial attention instead of dynamic RMS normalization and softmax. The
    graph is intended to measure protocol cost and tamper resistance rather
    than to replace a production language model.
    """

    def __init__(
        self,
        config: SecureDecoderConfig,
        weights: SecureDecoderWeights,
        preprocessor: TrustedPreprocessor,
    ) -> None:
        self.config = config
        self.weights = weights
        self.preprocessor = preprocessor
        if preprocessor.modulus != config.field_modulus:
            raise ValueError("preprocessor field does not match decoder field")

    def prepare_token(self, batch_size: int, *, output_disclosure: str = "token") -> SecureTokenMaterial:
        config = self.config
        correlations = {
            "embedding": self.preprocessor.linear_correlation(
                self.weights.embedding, (batch_size, config.vocabulary_size)
            ),
            "qkv": self.preprocessor.linear_correlation(
                self.weights.qkv, (batch_size, config.hidden_size)
            ),
            "attention_output": self.preprocessor.linear_correlation(
                self.weights.attention_output, (batch_size, config.hidden_size)
            ),
            "gate_up": self.preprocessor.linear_correlation(
                self.weights.gate_up, (batch_size, config.hidden_size)
            ),
            "down": self.preprocessor.linear_correlation(
                self.weights.down, (batch_size, config.intermediate_size)
            ),
            "vocabulary_head": self.preprocessor.linear_correlation(
                self.weights.vocabulary_head, (batch_size, config.hidden_size)
            ),
        }
        return SecureTokenMaterial(
            input_mask=self.preprocessor.input_mask((batch_size, config.vocabulary_size)),
            correlations=correlations,
            bit_mask=(
                self.preprocessor.bit_mask((batch_size, config.vocabulary_size))
                if output_disclosure == "token"
                else None
            ),
        )

    def _rotation(self, position: int) -> np.ndarray:
        size = self.config.hidden_size
        result = np.zeros((size, size), dtype=np.int64)
        phase = position % 4
        for start in range(0, size, 2):
            if phase == 0:
                block = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
            elif phase == 1:
                block = np.asarray([[0, -1], [1, 0]], dtype=np.int64)
            elif phase == 2:
                block = np.asarray([[-1, 0], [0, -1]], dtype=np.int64)
            else:
                block = np.asarray([[0, 1], [-1, 0]], dtype=np.int64)
            result[start : start + 2, start : start + 2] = block
        return result

    def forward_token(
        self,
        token_ids: np.ndarray | Sequence[int],
        *,
        cache: SecureKVCache | None = None,
        material: SecureTokenMaterial | None = None,
        tamper_cache: bool = False,
        output_disclosure: str = "token",
    ) -> SecureTokenResult:
        token_array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        batch_size = token_array.shape[0]
        if np.any(token_array < 0) or np.any(token_array >= self.config.vocabulary_size):
            raise ValueError("token is outside the vocabulary")
        one_hot = np.zeros((batch_size, self.config.vocabulary_size), dtype=np.int64)
        one_hot[np.arange(batch_size), token_array] = 1
        return self.forward_one_hot(
            one_hot,
            cache=cache,
            material=material,
            tamper_cache=tamper_cache,
            output_disclosure=output_disclosure,
        )

    def forward_one_hot(
        self,
        one_hot: np.ndarray,
        *,
        cache: SecureKVCache | None = None,
        material: SecureTokenMaterial | None = None,
        tamper_cache: bool = False,
        output_disclosure: str = "token",
    ) -> SecureTokenResult:
        config = self.config
        clear_one_hot = np.asarray(one_hot, dtype=np.int64)
        if clear_one_hot.ndim != 2 or clear_one_hot.shape[1] != config.vocabulary_size:
            raise ValueError("one hot input shape does not match vocabulary")
        batch_size = clear_one_hot.shape[0]
        token_material = material or self.prepare_token(
            batch_size, output_disclosure=output_disclosure
        )
        runtime = AuthenticatedMPC(self.preprocessor)
        shared_one_hot = runtime.input(clear_one_hot, token_material.input_mask)
        verify_one_hot(runtime, shared_one_hot)

        hidden = runtime.linear(
            shared_one_hot,
            self.weights.embedding,
            token_material.correlations["embedding"],
        )
        hidden = runtime.mul_public(hidden, self.weights.input_scale)
        residual = hidden

        qkv = runtime.linear(hidden, self.weights.qkv, token_material.correlations["qkv"])
        hidden_size = config.hidden_size
        q = runtime.take(qkv, (..., slice(0, hidden_size)))
        k = runtime.take(qkv, (..., slice(hidden_size, hidden_size * 2)))
        v = runtime.take(qkv, (..., slice(hidden_size * 2, hidden_size * 3)))
        position = 0 if cache is None else cache.length
        rotation = self._rotation(position)
        q = runtime.linear_public(q, rotation)
        k = runtime.linear_public(k, rotation)
        q = runtime.reshape(q, (batch_size, config.head_count, config.head_size))
        k = runtime.reshape(k, (batch_size, config.head_count, config.head_size))
        v = runtime.reshape(v, (batch_size, config.head_count, config.head_size))

        current_cache = cache if cache is not None else SecureKVCache()
        if current_cache.length >= config.maximum_context:
            current_cache.keys.pop(0)
            current_cache.values.pop(0)
        current_cache.keys.append(k)
        current_cache.values.append(v)
        if tamper_cache:
            current_cache.keys[-1] = current_cache.keys[-1].tamper_client(value_delta=1)

        key_stack = runtime.concatenate(
            [runtime.reshape(item, (batch_size, config.head_count, 1, config.head_size)) for item in current_cache.keys],
            axis=2,
        )
        value_stack = runtime.concatenate(
            [runtime.reshape(item, (batch_size, config.head_count, 1, config.head_size)) for item in current_cache.values],
            axis=2,
        )
        query_stack = runtime.broadcast_to(
            runtime.reshape(q, (batch_size, config.head_count, 1, config.head_size)),
            key_stack.shape,
        )
        score_products = runtime.multiply(
            query_stack,
            key_stack,
            self.preprocessor.multiplication_triple(key_stack.shape),
        )
        scores = runtime.sum(score_products, axis=-1)
        positive_scores = runtime.add_public(scores, config.attention_offset)
        attention_weights = runtime.square(
            positive_scores,
            self.preprocessor.multiplication_triple(positive_scores.shape),
        )
        weight_stack = runtime.broadcast_to(
            runtime.reshape(attention_weights, attention_weights.shape + (1,)),
            value_stack.shape,
        )
        weighted_values = runtime.multiply(
            weight_stack,
            value_stack,
            self.preprocessor.multiplication_triple(value_stack.shape),
        )
        attention = runtime.sum(weighted_values, axis=2)
        attention = runtime.reshape(attention, (batch_size, hidden_size))
        attention = runtime.linear(
            attention,
            self.weights.attention_output,
            token_material.correlations["attention_output"],
        )
        hidden = runtime.add(residual, attention)

        mlp_input = runtime.mul_public(hidden, self.weights.mlp_scale)
        gate_up = runtime.linear(
            mlp_input,
            self.weights.gate_up,
            token_material.correlations["gate_up"],
        )
        gate = runtime.take(gate_up, (..., slice(0, config.intermediate_size)))
        up = runtime.take(gate_up, (..., slice(config.intermediate_size, config.intermediate_size * 2)))
        gate_squared = runtime.square(
            gate,
            self.preprocessor.multiplication_triple(gate.shape),
        )
        activated_gate = runtime.add(gate, gate_squared)
        mlp_product = runtime.multiply(
            activated_gate,
            up,
            self.preprocessor.multiplication_triple(up.shape),
        )
        mlp_output = runtime.linear(
            mlp_product,
            self.weights.down,
            token_material.correlations["down"],
        )
        hidden = runtime.add(hidden, mlp_output)
        logits = runtime.linear(
            hidden,
            self.weights.vocabulary_head,
            token_material.correlations["vocabulary_head"],
        )
        if output_disclosure == "token":
            if token_material.bit_mask is None:
                raise ValueError("token disclosure requires prepared comparison material")
            selection = secure_argmax(runtime, logits, bit_mask=token_material.bit_mask)
            token_ids = selection.indices
            disclosed_logits = None
        elif output_disclosure == "logits":
            selection = None
            disclosed_logits = runtime.open(logits, to_client=True)
            token_ids = np.argmax(disclosed_logits, axis=-1)
        else:
            raise ValueError("output disclosure must be token or logits")
        return SecureTokenResult(
            token_ids=token_ids,
            logits=logits,
            selection=selection,
            disclosed_logits=disclosed_logits,
            online_rounds=runtime.stats.online_rounds,
            uploaded_bytes=runtime.stats.uploaded_bytes,
            downloaded_bytes=runtime.stats.downloaded_bytes,
        )

    def clear_forward_token(
        self,
        token_ids: np.ndarray | Sequence[int],
        *,
        cache: ClearKVCache | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        token_array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        one_hot = np.zeros((token_array.shape[0], self.config.vocabulary_size), dtype=np.int64)
        one_hot[np.arange(token_array.shape[0]), token_array] = 1
        return self.clear_forward_one_hot(one_hot, cache=cache)

    def clear_forward_one_hot(
        self,
        one_hot: np.ndarray,
        *,
        cache: ClearKVCache | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.config.field_modulus
        config = self.config
        x = np.asarray(one_hot, dtype=np.int64) % p
        hidden = (x @ (self.weights.embedding % p).T) % p
        hidden = hidden * self.weights.input_scale % p
        residual = hidden.copy()
        qkv = hidden @ (self.weights.qkv % p).T % p
        q, k, v = np.split(qkv, 3, axis=-1)
        current_cache = cache if cache is not None else ClearKVCache()
        position = current_cache.length
        rotation = self._rotation(position) % p
        q = q @ rotation.T % p
        k = k @ rotation.T % p
        batch_size = x.shape[0]
        q = q.reshape(batch_size, config.head_count, config.head_size)
        k = k.reshape(batch_size, config.head_count, config.head_size)
        v = v.reshape(batch_size, config.head_count, config.head_size)
        if current_cache.length >= config.maximum_context:
            current_cache.keys.pop(0)
            current_cache.values.pop(0)
        current_cache.keys.append(k)
        current_cache.values.append(v)
        key_stack = np.stack(current_cache.keys, axis=2)
        value_stack = np.stack(current_cache.values, axis=2)
        scores = np.sum(q[:, :, None, :] * key_stack, axis=-1, dtype=np.int64) % p
        weights = (scores + config.attention_offset) ** 2 % p
        attention = np.sum(weights[..., None] * value_stack, axis=2, dtype=np.int64) % p
        attention = attention.reshape(batch_size, config.hidden_size)
        attention = attention @ (self.weights.attention_output % p).T % p
        hidden = (residual + attention) % p
        mlp_input = hidden * self.weights.mlp_scale % p
        gate_up = mlp_input @ (self.weights.gate_up % p).T % p
        gate, up = np.split(gate_up, 2, axis=-1)
        activated = (gate + gate * gate) % p
        mlp = activated * up % p
        mlp = mlp @ (self.weights.down % p).T % p
        hidden = (hidden + mlp) % p
        logits = hidden @ (self.weights.vocabulary_head % p).T % p
        return np.argmax(logits, axis=-1), logits
