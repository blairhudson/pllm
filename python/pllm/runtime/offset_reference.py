"""Bounded in-process two-worker offset comparator for compiled public decoders.

Each worker evaluates one fresh modular input share with the existing native
stage kernel. Both workers run in this process: these are serialized stage-body
counts and kernel times, not network measurements or independent operators.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from math import prod
from typing import Callable

import numpy as np

from pllm.runtime.model_binding import CompiledRuntimeModel
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.transformer_client import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_engine import MaskedTransformerEngine

_MAX_TENSOR_ELEMENTS = 4_000_000


class OffsetReferenceError(ValueError):
    """A comparator role, stage, or response differs from its compiled contract."""


@dataclass(frozen=True, slots=True)
class OffsetReferenceCosts:
    stages: int
    rows: int
    client_to_worker_a_bytes: int
    worker_a_to_client_bytes: int
    client_to_worker_b_bytes: int
    worker_b_to_client_bytes: int
    worker_a_stage_ns: int
    worker_b_stage_ns: int
    worker_a_integer_macs: int
    worker_b_integer_macs: int

    @property
    def total_stage_body_bytes(self) -> int:
        return (
            self.client_to_worker_a_bytes
            + self.worker_a_to_client_bytes
            + self.client_to_worker_b_bytes
            + self.worker_b_to_client_bytes
        )

    @property
    def total_integer_macs(self) -> int:
        return self.worker_a_integer_macs + self.worker_b_integer_macs


class TwoOnlineOffsetReference:
    """A non-deployable reference with two separately loaded provider engines."""

    def __init__(
        self, compiled: CompiledRuntimeModel, worker_a: MaskedTransformerEngine,
        worker_b: MaskedTransformerEngine,
        *, model_id: str,
        exchange_a: Callable[[str, list[bytes]], list[bytes]],
        exchange_b: Callable[[str, list[bytes]], list[bytes]],
    ) -> None:
        if (
            type(compiled) is not CompiledRuntimeModel
            or type(worker_a) is not MaskedTransformerEngine
            or type(worker_b) is not MaskedTransformerEngine
            or worker_a is worker_b
            or exchange_a is exchange_b
            or not callable(exchange_a)
            or not callable(exchange_b)
        ):
            raise OffsetReferenceError("two distinct loaded offset workers are required")
        compiled.validate()
        manifest = compiled._bundle.manifest
        fingerprint = manifest["metadata"]["body_fingerprint"]
        bindings = {stage.stage_id: stage for stage in compiled._stages}
        for worker in (worker_a, worker_b):
            model = worker._model(model_id)
            if model.manifest.metadata.get("body_fingerprint") != fingerprint:
                raise OffsetReferenceError("offset worker body differs from the compiled decoder")
            for stage_id, binding in bindings.items():
                if binding.client_weight_layout is not None:
                    continue
                stage = compiled._bundle.stages[stage_id]
                runtime = model.stages.get(stage_id)
                if runtime is None:
                    raise OffsetReferenceError("offset worker stage is absent")
                if (
                    runtime.spec.in_features != binding.in_features
                    or runtime.spec.out_features != binding.out_features
                    or runtime.spec.weight_bits != binding.weight_bits
                    or runtime.spec.activation_bits != binding.activation_bits
                ):
                    raise OffsetReferenceError("offset worker stage geometry differs")
                if runtime.weight_digest != binding.weight_digest:
                    raise OffsetReferenceError("offset worker stage weight differs")
                if runtime.seeded_profile != stage.seeded_profile:
                    raise OffsetReferenceError("offset worker stage ring differs")
                if not np.array_equal(runtime.weight.scales, stage.weight_scales):
                    raise OffsetReferenceError("offset worker stage scales differ")
                if not np.array_equal(runtime.bias, stage.bias):
                    raise OffsetReferenceError("offset worker stage bias differs")
        self._compiled = compiled
        self._model_id = model_id
        self._fingerprint = fingerprint
        self._bindings = bindings
        self._exchange_a = exchange_a
        self._exchange_b = exchange_b
        self._costs = OffsetReferenceCosts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._maximum_rows = int(compiled._plan.to_dict()["prefill"]["query_sequence"])

    @property
    def costs(self) -> OffsetReferenceCosts:
        return self._costs

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self._compiled._bundle.stages.get(stage_id)
        binding = self._bindings.get(stage_id)
        if (
            stage is None
            or binding is None
            or binding.client_weight_layout is not None
            or stage.seeded_profile is None
        ):
            raise OffsetReferenceError("stage is not a compiled remote offset stage")
        value = np.asarray(activation)
        if (
            value.dtype != np.float32
            or value.ndim not in (2, 3)
            or value.shape[-1] != stage.in_features
        ):
            raise OffsetReferenceError("offset stage input is malformed")
        rows = prod(value.shape[:-1])
        if (
            rows < 1
            or rows > self._maximum_rows
            or rows * max(stage.in_features, stage.out_features) > _MAX_TENSOR_ELEMENTS
        ):
            raise OffsetReferenceError("offset stage exceeds the bounded tensor policy")
        if not np.all(np.isfinite(value)):
            raise OffsetReferenceError("offset stage input is not finite")
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        profile = stage.seeded_profile
        if profile.ring not in {"u16", "u24", "u32"}:
            raise OffsetReferenceError("offset stages require an exact wrapping ring")
        modulus = profile.modulus
        clear = quantized.values.reshape(rows, stage.in_features).astype(np.int64) % modulus
        raw_mask = bytearray(secrets.token_bytes(rows * stage.in_features * 4))
        mask = np.frombuffer(raw_mask, dtype="<u4").reshape(rows, stage.in_features)
        left = np.asarray((clear - mask.astype(np.int64)) % modulus, dtype=np.uint32)
        try:
            mask %= modulus
            def request(share: np.ndarray) -> tuple[str, bytes]:
                ticket = secrets.token_hex(16)
                return ticket, MaskedStageRequest(
                    model=self._model_id,
                    stage_id=stage_id,
                    correlation_id=ticket,
                    masked_input=share,
                    activation_scales=np.ones(1, dtype=np.float32),
                    modulus=modulus,
                    wire_bits=profile.wire_bits,
                    ring=profile.ring,
                    body_fingerprint=self._fingerprint,
                    weight_digest=stage.weight_digest,
                    weight_bits=stage.weight_bits,
                    activation_bits=stage.activation_bits,
                    session_id=secrets.token_hex(16),
                    out_features=stage.out_features,
                    signed_output_bound=profile.signed_output_bound,
                ).pack()

            left_ticket, left_request = request(left)
            right_ticket, right_request = request(mask)
            left_result = self._exchange_a(stage_id, [left_request])
            right_result = self._exchange_b(stage_id, [right_request])
            if (
                type(left_result) is not list
                or type(right_result) is not list
                or len(left_result) != 1
                or len(right_result) != 1
                or type(left_result[0]) is not bytes
                or type(right_result[0]) is not bytes
            ):
                raise OffsetReferenceError("offset worker returned an invalid response batch")
            first = MaskedStageResponse.unpack(left_result[0])
            second = MaskedStageResponse.unpack(right_result[0])
            for result, ticket in ((first, left_ticket), (second, right_ticket)):
                if (
                    result.correlation_id != ticket
                    or result.stage_id != stage_id
                    or result.modulus != modulus
                    or result.wire_bits != profile.wire_bits
                    or result.ring != profile.ring
                    or result.masked_output.shape != (rows, stage.out_features)
                    or result.server_ns < 0
                ):
                    raise OffsetReferenceError("offset worker result differs from its stage contract")
            combined = (
                first.masked_output.astype(np.int64)
                + second.masked_output.astype(np.int64)
            ) % modulus
            centered = np.where(combined >= modulus // 2, combined - modulus, combined)
            if np.any(np.abs(centered) > profile.signed_output_bound):
                raise OffsetReferenceError("offset result exceeds the declared signed stage bound")
            output = dequantize_matmul(
                centered, quantized.scales, stage.weight_scales,
                output_shape=quantized.original_shape[:-1] + (stage.out_features,),
            )
            if stage.bias is not None:
                output = output + stage.bias
            previous = self._costs
            self._costs = OffsetReferenceCosts(
                previous.stages + 1,
                previous.rows + rows,
                previous.client_to_worker_a_bytes + len(left_request),
                previous.worker_a_to_client_bytes + len(left_result[0]),
                previous.client_to_worker_b_bytes + len(right_request),
                previous.worker_b_to_client_bytes + len(right_result[0]),
                previous.worker_a_stage_ns + first.server_ns,
                previous.worker_b_stage_ns + second.server_ns,
                previous.worker_a_integer_macs + rows * stage.in_features * stage.out_features,
                previous.worker_b_integer_macs + rows * stage.in_features * stage.out_features,
            )
            return np.ascontiguousarray(output, dtype=np.float32)
        finally:
            left.fill(0)
            mask.fill(0)
