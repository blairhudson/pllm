from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .shared_mlp import SharedMLPWeights
from .shared_mpc import (
    BeaverTripleShare,
    OpeningFrame,
    PartyRuntime,
    PreprocessingSource,
    PublicQuantizedMatrix,
    SharedMPCError,
    SharedTensor,
    TruncationMaskShare,
    ValueOpeningFrame,
)
from .shared_session import SharedPeerConnection


class LocalPreprocessingInventory(PreprocessingSource):
    """One party's opaque, one-use preprocessing shares."""

    def __init__(self, session_id: str, party: int, *, capacity: int) -> None:
        if party not in (0, 1) or capacity <= 0:
            raise ValueError("invalid preprocessing inventory configuration")
        self.session_id = session_id
        self.party = party
        self.capacity = capacity
        self._triples: dict[str, BeaverTripleShare] = {}
        self._masks: dict[str, TruncationMaskShare] = {}

    def add_triple(self, operation_id: str, triple: BeaverTripleShare) -> None:
        self._require_space(operation_id)
        if triple.session_id != self.session_id or triple.party != self.party:
            raise SharedMPCError("preprocessing triple belongs to another party or session")
        self._triples[operation_id] = triple

    def add_truncation(self, operation_id: str, mask: TruncationMaskShare) -> None:
        self._require_space(operation_id)
        if mask.session_id != self.session_id or mask.party != self.party:
            raise SharedMPCError("truncation mask belongs to another party or session")
        self._masks[operation_id] = mask

    def take_multiplication(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
    ) -> BeaverTripleShare:
        self._require_identity(session_id, party)
        triple = self._triples.pop(operation_id, None)
        if triple is None:
            raise SharedMPCError("multiplication preprocessing is unavailable or consumed")
        if triple.a.shape != tuple(shape):
            raise SharedMPCError("multiplication preprocessing shape mismatch")
        return triple

    def take_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> TruncationMaskShare:
        self._require_identity(session_id, party)
        mask = self._masks.pop(operation_id, None)
        if mask is None:
            raise SharedMPCError("truncation preprocessing is unavailable or consumed")
        if mask.value.shape != tuple(shape) or mask.bits != bits:
            raise SharedMPCError("truncation preprocessing parameters mismatch")
        return mask

    @property
    def remaining(self) -> int:
        return len(self._triples) + len(self._masks)

    def _require_space(self, operation_id: str) -> None:
        if not operation_id or operation_id in self._triples or operation_id in self._masks:
            raise SharedMPCError("preprocessing operation identifier is invalid or duplicated")
        if self.remaining >= self.capacity:
            raise SharedMPCError("preprocessing inventory capacity exhausted")

    def _require_identity(self, session_id: str, party: int) -> None:
        if session_id != self.session_id or party != self.party:
            raise SharedMPCError("preprocessing request belongs to another party or session")


class SharedComputeParty:
    def __init__(
        self,
        runtime: PartyRuntime,
        peer: SharedPeerConnection,
        preprocessing: PreprocessingSource,
        *,
        max_opening_elements: int,
    ) -> None:
        if max_opening_elements <= 0:
            raise ValueError("opening element limit must be positive")
        if (
            runtime.session_id != peer.state.commitment.session_id
            or runtime.party != peer.state.party
            or preprocessing.session_id != runtime.session_id
            or preprocessing.party != runtime.party
        ):
            raise SharedMPCError("shared compute party bindings do not match")
        self.runtime = runtime
        self.peer = peer
        self.preprocessing = preprocessing
        self.max_opening_elements = max_opening_elements

    async def multiply(
        self,
        left: SharedTensor,
        right: SharedTensor,
        operation_id: str,
    ) -> SharedTensor:
        triple = self.preprocessing.take_multiplication(
            self.runtime.session_id,
            operation_id,
            left.shape,
            self.runtime.party,
        )
        try:
            state, opening = self.runtime.begin_multiply(left, right, triple, operation_id)
            peer_payload = await self.peer.exchange("multiply.open", operation_id, opening.pack())
            peer_opening = OpeningFrame.unpack(
                peer_payload,
                max_elements=self.max_opening_elements,
            )
            return self.runtime.finish_multiply(
                state,
                opening,
                peer_opening,
                f"{operation_id}.result",
            )
        except BaseException as exc:
            self.peer.state.abort(f"shared multiplication failed: {type(exc).__name__}")
            raise

    async def truncate(
        self,
        value: SharedTensor,
        operation_id: str,
        *,
        bits: int,
        signed_bound: int,
    ) -> SharedTensor:
        mask = self.preprocessing.take_truncation(
            self.runtime.session_id,
            operation_id,
            value.shape,
            self.runtime.party,
            bits,
        )
        try:
            state, opening = self.runtime.begin_truncate(
                value,
                mask,
                operation_id,
                signed_bound=signed_bound,
            )
            peer_payload = await self.peer.exchange("truncate.open", operation_id, opening.pack())
            peer_opening = ValueOpeningFrame.unpack(
                peer_payload,
                max_elements=self.max_opening_elements,
            )
            return self.runtime.finish_truncate(
                state,
                opening,
                peer_opening,
                f"{operation_id}.result",
            )
        except BaseException as exc:
            self.peer.state.abort(f"shared truncation failed: {type(exc).__name__}")
            raise

    async def quantized_linear(
        self,
        value: SharedTensor,
        matrix: PublicQuantizedMatrix,
        operation_id: str,
        *,
        signed_bound: int,
    ) -> tuple[SharedTensor, int]:
        scaled_bound = matrix.scaled_bound(signed_bound)
        result = self.runtime.linear_public(
            value,
            matrix.coefficients,
            f"{operation_id}.coefficients",
            signed_bound=signed_bound,
        )
        result = self.runtime.mul_public_tensor(
            result,
            matrix.row_multipliers,
            f"{operation_id}.row_scale",
        )
        if matrix.scale == 1:
            return result, scaled_bound
        result = self.runtime.reinterpret_scale(
            result,
            value.scale * matrix.scale,
            f"{operation_id}.scaled",
        )
        result = await self.truncate(
            result,
            f"{operation_id}.truncate",
            bits=matrix.scale.bit_length() - 1,
            signed_bound=scaled_bound,
        )
        return result, matrix.output_bound(signed_bound)


@dataclass(frozen=True, slots=True)
class SharedMLPBounds:
    hidden: int
    gate: int
    up: int
    activation_numerator: int
    activation: int
    product: int
    output: int


class NetworkSharedMLP:
    """Static-scale MLP executed by one party over a live peer connection."""

    def __init__(
        self, party: SharedComputeParty, weights: SharedMLPWeights, *, prefix: str
    ) -> None:
        if not prefix:
            raise SharedMPCError("network shared MLP prefix cannot be empty")
        self.party = party
        self.weights = weights
        self.prefix = prefix

    def bounds(self, hidden_bound: int, scale: int) -> SharedMLPBounds:
        gate = self.weights.gate_matrix.output_bound(hidden_bound)
        up = self.weights.up_matrix.output_bound(hidden_bound)
        numerator = 2 * scale * gate + gate * gate
        activation = (numerator >> (scale.bit_length() + 1)) + 1
        product = activation * up
        return SharedMLPBounds(
            hidden_bound,
            gate,
            up,
            numerator,
            activation,
            product,
            self.weights.down_matrix.output_bound((product >> (scale.bit_length() - 1)) + 1),
        )

    async def run(self, hidden: SharedTensor, *, hidden_bound: int) -> SharedTensor:
        scale = hidden.scale
        if scale < 2 or scale & (scale - 1):
            raise SharedMPCError("network shared MLP scale must be a power of two of at least two")
        bounds = self.bounds(hidden_bound, scale)
        if bounds.gate > scale:
            raise SharedMPCError("quadratic SiLU is calibrated only for gate values in [-1, 1]")
        runtime = self.party.runtime
        gate, gate_bound = await self.party.quantized_linear(
            hidden,
            self.weights.gate_matrix,
            f"{self.prefix}.gate_quantize",
            signed_bound=hidden_bound,
        )
        up, up_bound = await self.party.quantized_linear(
            hidden,
            self.weights.up_matrix,
            f"{self.prefix}.up_quantize",
            signed_bound=hidden_bound,
        )
        if gate_bound != bounds.gate or up_bound != bounds.up:
            raise SharedMPCError("network shared MLP quantization bound mismatch")
        square = await self.party.multiply(gate, gate, f"{self.prefix}.gate_square")
        linear = runtime.mul_public(gate, 2 * scale, f"{self.prefix}.gate_linear")
        linear = runtime.reinterpret_scale(linear, scale * scale, f"{self.prefix}.gate_aligned")
        numerator = runtime.add(linear, square, f"{self.prefix}.activation_numerator")
        numerator = runtime.reinterpret_scale(
            numerator,
            4 * scale * scale,
            f"{self.prefix}.activation_scaled",
        )
        activation = await self.party.truncate(
            numerator,
            f"{self.prefix}.activation_truncate",
            bits=scale.bit_length() + 1,
            signed_bound=bounds.activation_numerator,
        )
        product = await self.party.multiply(activation, up, f"{self.prefix}.gate_product")
        product = await self.party.truncate(
            product,
            f"{self.prefix}.product_truncate",
            bits=scale.bit_length() - 1,
            signed_bound=bounds.product,
        )
        output, output_bound = await self.party.quantized_linear(
            product,
            self.weights.down_matrix,
            f"{self.prefix}.down_quantize",
            signed_bound=(bounds.product >> (scale.bit_length() - 1)) + 1,
        )
        if output_bound != bounds.output:
            raise SharedMPCError("network shared MLP output bound mismatch")
        return output
