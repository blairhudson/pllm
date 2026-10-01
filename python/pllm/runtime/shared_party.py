from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .shared_dpf import DpfReferenceError, pack_bit_shares, unpack_bit_shares
from .shared_mlp import SharedMLPWeights
from .shared_mpc import (
    BeaverTripleShare,
    ExactSessionAdmission,
    ExactTruncationMaskShare,
    FssTruncationMaskShare,
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

    def __init__(
        self, session_id: str, party: int, *, capacity: int,
        max_fss_key_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if party not in (0, 1) or capacity <= 0 or type(max_fss_key_bytes) is not int or max_fss_key_bytes <= 0:
            raise ValueError("invalid preprocessing inventory configuration")
        self.session_id = session_id
        self.party = party
        self.capacity = capacity
        self._triples: dict[str, BeaverTripleShare] = {}
        self._masks: dict[str, TruncationMaskShare] = {}
        self._exact_masks: dict[str, ExactTruncationMaskShare] = {}
        self._fss_masks: dict[str, FssTruncationMaskShare] = {}
        self._fss_key_bytes = 0
        self._max_fss_key_bytes = max_fss_key_bytes
        self._cancelled = False
        self.exact_admission: ExactSessionAdmission | None = None

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

    def add_exact_truncation(self, operation_id: str, mask: ExactTruncationMaskShare) -> None:
        self._require_space(operation_id)
        if mask.session_id != self.session_id or mask.party != self.party:
            raise SharedMPCError("exact truncation mask belongs to another party or session")
        self._exact_masks[operation_id] = mask

    def add_fss_truncation(self, operation_id: str, mask: FssTruncationMaskShare) -> None:
        self._require_space(operation_id)
        if mask.session_id != self.session_id or mask.party != self.party or mask.mask_id != operation_id:
            raise SharedMPCError("FSS truncation mask belongs to another party or session")
        if self._fss_key_bytes + mask.key_bytes > self._max_fss_key_bytes:
            raise SharedMPCError("FSS preprocessing key budget exceeded")
        try:
            for key in mask.comparison_keys:
                key.prepare()
        except DpfReferenceError:
            for key in mask.comparison_keys:
                key.cancel()
            raise SharedMPCError("FSS key expansion failed") from None
        self._fss_masks[operation_id] = mask
        self._fss_key_bytes += mask.key_bytes

    def bind_exact_admission(self, admission: ExactSessionAdmission) -> None:
        if (
            self._cancelled or self.exact_admission is not None
            or admission.session_id != self.session_id
            or set(self._exact_masks) | set(self._fss_masks)
            != {slot.operation_id for slot in admission.slots}
            or self._masks
        ):
            raise SharedMPCError("exact session material does not match admission")
        for slot in admission.slots:
            if slot.method in ("fss", "hybrid"):
                mask = self._fss_masks.get(slot.operation_id)
                if mask is None or mask.comparison_bits != (
                    slot.bits if slot.method == "fss" else slot.low_bits
                ):
                    raise SharedMPCError("FSS comparison width differs from session admission")
                self.preflight_fss_truncation(
                    self.session_id, slot.operation_id, slot.shape, self.party, slot.bits,
                )
            else:
                self.preflight_exact_truncation(
                    self.session_id, slot.operation_id, slot.shape, self.party, slot.bits,
                )
        self.exact_admission = admission

    def preflight_fss_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> None:
        self._require_identity(session_id, party)
        if self._cancelled:
            raise SharedMPCError("preprocessing inventory was cancelled")
        mask = self._fss_masks.get(operation_id)
        if (
            mask is None or mask.value.shape != tuple(shape) or mask.bits != bits
            or any(key.expanded_bytes == 0 for key in mask.comparison_keys)
        ):
            raise SharedMPCError("FSS truncation comparison material is incomplete")
        for bit in range(mask.comparison_bits, bits):
            triple = self._triples.get(f"{operation_id}.carry.{bit}")
            if triple is None or triple.shape != tuple(shape):
                raise SharedMPCError("FSS high-bit carry material is incomplete")

    def take_fss_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> FssTruncationMaskShare:
        self.preflight_fss_truncation(session_id, operation_id, shape, party, bits)
        mask = self._fss_masks.pop(operation_id)
        self._fss_key_bytes -= mask.key_bytes
        return mask

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

    def take_exact_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> ExactTruncationMaskShare:
        self._require_identity(session_id, party)
        if self._cancelled:
            raise SharedMPCError("preprocessing inventory was cancelled")
        mask = self._exact_masks.pop(operation_id, None)
        if mask is None:
            raise SharedMPCError("exact truncation preprocessing is unavailable or consumed")
        if mask.value.shape != tuple(shape) or mask.bits != bits:
            raise SharedMPCError("exact truncation preprocessing parameters mismatch")
        return mask

    def preflight_exact_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> None:
        self._require_identity(session_id, party)
        if self._cancelled:
            raise SharedMPCError("preprocessing inventory was cancelled")
        mask = self._exact_masks.get(operation_id)
        if mask is None or mask.value.shape != tuple(shape) or mask.bits != bits:
            raise SharedMPCError("matching exact truncation mask is unavailable")
        for bit in range(1, bits):
            triple = self._triples.get(f"{operation_id}.carry.{bit}")
            if triple is None or triple.shape != tuple(shape):
                raise SharedMPCError("exact truncation comparison material is incomplete")

    def cancel(self) -> None:
        self._triples.clear()
        self._masks.clear()
        self._exact_masks.clear()
        for mask in self._fss_masks.values():
            for key in mask.comparison_keys:
                key.cancel()
        self._fss_masks.clear()
        self._fss_key_bytes = 0
        self._cancelled = True

    @property
    def remaining(self) -> int:
        return len(self._triples) + len(self._masks) + len(self._exact_masks) + len(self._fss_masks)

    def _require_space(self, operation_id: str) -> None:
        if self._cancelled:
            raise SharedMPCError("preprocessing inventory was cancelled")
        if (
            not operation_id
            or operation_id in self._triples
            or operation_id in self._masks
            or operation_id in self._exact_masks
            or operation_id in self._fss_masks
        ):
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
        truncation: str = "probabilistic",
    ) -> None:
        if max_opening_elements <= 0:
            raise ValueError("opening element limit must be positive")
        if truncation not in ("probabilistic", "exact", "fss", "fss_lowbits_exact_highbits"):
            raise SharedMPCError("shared truncation method is unsupported")
        if (
            runtime.session_id != peer.state.commitment.session_id
            or runtime.party != peer.state.party
            or preprocessing.session_id != runtime.session_id
            or preprocessing.party != runtime.party
        ):
            raise SharedMPCError("shared compute party bindings do not match")
        if truncation != "probabilistic":
            admission = preprocessing.exact_admission
            if (
                admission is None or admission.session_id != runtime.session_id
                or admission.graph_digest != peer.state.commitment.graph_fingerprint
            ):
                raise SharedMPCError("exact truncation lacks a peer-bound session admission")
        self.runtime = runtime
        self.peer = peer
        self.preprocessing = preprocessing
        self.max_opening_elements = max_opening_elements
        self.truncation_method = truncation

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
        if self.truncation_method == "exact":
            return await self.truncate_exact(
                value, operation_id, bits=bits, signed_bound=signed_bound
            )
        if self.truncation_method in ("fss", "fss_lowbits_exact_highbits"):
            return await self.truncate_fss(
                value, operation_id, bits=bits, signed_bound=signed_bound
            )
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

    async def truncate_exact(
        self,
        value: SharedTensor,
        operation_id: str,
        *,
        bits: int,
        signed_bound: int,
    ) -> SharedTensor:
        """Separate-party exact shift using one masked opening and b−1 ANDs.

        This research path accepts only dealer-issued masks and triples. It is
        not an executable decoder placement until a distributed issuer and
        whole-session material budget are compiled and admitted.
        """
        try:
            if self.truncation_method not in ("exact", "fss_lowbits_exact_highbits"):
                raise SharedMPCError("exact Beaver truncation is not selected for this party")
            self.preprocessing.preflight_exact_truncation(
                self.runtime.session_id, operation_id, value.shape, self.runtime.party, bits
            )
            mask = self.preprocessing.take_exact_truncation(
                self.runtime.session_id, operation_id, value.shape, self.runtime.party, bits
            )
            state, opening = self.runtime.begin_exact_truncate(
                value, mask, operation_id, signed_bound=signed_bound
            )
            peer_payload = await self.peer.exchange(
                "truncate.exact.open", operation_id, opening.pack()
            )
            peer_opening = ValueOpeningFrame.unpack(
                peer_payload, max_elements=self.max_opening_elements
            )
            opened, carry = self.runtime.exact_carry_start(state, opening, peer_opening)
            for bit in range(1, bits):
                mask_bit = SharedTensor(
                    self.runtime.session_id, f"{operation_id}.mask_bit.{bit}", self.runtime.party,
                    np.ascontiguousarray(mask.low_bit_shares[..., bit]), 1,
                )
                product = await self.multiply(
                    carry, mask_bit, f"{operation_id}.carry.{bit}"
                )
                carry = self.runtime.exact_carry_bit(state, opened, carry, product, bit)
            return self.runtime.finish_exact_truncate(
                state, opened, carry, f"{operation_id}.result"
            )
        except BaseException as exc:
            self.preprocessing.cancel()
            self.peer.state.abort(f"shared exact truncation failed: {type(exc).__name__}")
            raise

    async def truncate_fss(
        self,
        value: SharedTensor,
        operation_id: str,
        *,
        bits: int,
        signed_bound: int,
    ) -> SharedTensor:
        """One-use FSS comparison and bit-to-ring conversion over a peer channel."""
        mask: FssTruncationMaskShare | None = None
        try:
            if (
                self.truncation_method not in ("fss", "fss_lowbits_exact_highbits")
                or (self.truncation_method == "fss" and bits > 10)
            ):
                raise SharedMPCError("FSS truncation is not selected for this shift width")
            self.preprocessing.preflight_fss_truncation(
                self.runtime.session_id, operation_id, value.shape, self.runtime.party, bits
            )
            mask = self.preprocessing.take_fss_truncation(
                self.runtime.session_id, operation_id, value.shape, self.runtime.party, bits
            )
            state, opening = self.runtime.begin_exact_truncate(
                value, mask, operation_id, signed_bound=signed_bound
            )
            peer_opening = ValueOpeningFrame.unpack(
                await self.peer.exchange("truncate.fss.open", operation_id, opening.pack()),
                max_elements=self.max_opening_elements,
            )
            opened = self.runtime._open_exact(state, opening, peer_opening)
            comparison = tuple(
                key.less_than_share(int(word & np.uint64((1 << mask.comparison_bits) - 1))) ^ int(secret)
                for key, word, secret in zip(
                    mask.comparison_keys, opened.flat, mask.xor_mask_share.flat, strict=True,
                )
            )
            packed = pack_bit_shares(comparison)
            peer_bits = unpack_bit_shares(
                await self.peer.exchange("truncate.fss.bit", f"{operation_id}.bit", packed),
                int(opened.size),
            )
            self.runtime.stats.uploaded_bytes += len(packed)
            self.runtime.stats.downloaded_bytes += len(packed)
            public_bit = np.asarray(
                tuple(first ^ second for first, second in zip(comparison, peer_bits, strict=True)),
                dtype=np.uint64,
            ).reshape(opened.shape)
            carry = self.runtime.fss_carry_share(state, opened, public_bit)
            for bit in range(mask.comparison_bits, bits):
                mask_bit = SharedTensor(
                    self.runtime.session_id, f"{operation_id}.mask_bit.{bit}", self.runtime.party,
                    np.ascontiguousarray(mask.high_bit_shares[..., bit - mask.comparison_bits]), 1,
                )
                product = await self.multiply(carry, mask_bit, f"{operation_id}.carry.{bit}")
                carry = self.runtime.exact_carry_bit(state, opened, carry, product, bit)
            return self.runtime.finish_exact_truncate(
                state, opened, carry, f"{operation_id}.result"
            )
        except BaseException as exc:
            if mask is not None:
                for key in mask.comparison_keys:
                    key.cancel()
            self.preprocessing.cancel()
            self.peer.state.abort(f"shared FSS truncation failed: {type(exc).__name__}")
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
