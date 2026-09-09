from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shared_mpc import (
    BeaverTripleShare,
    MultiplicationState,
    OpeningFrame,
    PartyRuntime,
    PublicQuantizedMatrix,
    SharedMPCError,
    SharedTensor,
    TruncationMaskShare,
    TruncationState,
    ValueOpeningFrame,
)


@dataclass(frozen=True, slots=True)
class SharedMLPWeights:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gate_row_multipliers: np.ndarray | None = None
    up_row_multipliers: np.ndarray | None = None
    down_row_multipliers: np.ndarray | None = None
    gate_scale: int = 1
    up_scale: int = 1
    down_scale: int = 1
    gate_matrix: PublicQuantizedMatrix = field(init=False)
    up_matrix: PublicQuantizedMatrix = field(init=False)
    down_matrix: PublicQuantizedMatrix = field(init=False)

    def __post_init__(self) -> None:
        gate = np.asarray(self.gate)
        up = np.asarray(self.up)
        down = np.asarray(self.down)
        if gate.ndim != 2 or up.shape != gate.shape:
            raise SharedMPCError("gate and up weights must be equal-shape matrices")
        if down.ndim != 2 or down.shape[1] != gate.shape[0]:
            raise SharedMPCError("down weight does not match intermediate width")
        if down.shape[0] != gate.shape[1]:
            raise SharedMPCError("down weight output does not match hidden width")
        for value in (gate, up, down):
            if np.any(value < -128) or np.any(value > 127):
                raise SharedMPCError("shared MLP weights must fit int8")
        matrices = []
        for value, multipliers, scale in (
            (gate, self.gate_row_multipliers, self.gate_scale),
            (up, self.up_row_multipliers, self.up_scale),
            (down, self.down_row_multipliers, self.down_scale),
        ):
            if multipliers is None:
                multipliers = np.ones(value.shape[0], dtype=np.int64)
            matrices.append(PublicQuantizedMatrix(value, multipliers, scale))
        object.__setattr__(self, "gate_matrix", matrices[0])
        object.__setattr__(self, "up_matrix", matrices[1])
        object.__setattr__(self, "down_matrix", matrices[2])


@dataclass(frozen=True, slots=True)
class SharedMLPProjection:
    gate: SharedTensor
    up: SharedTensor
    gate_bound: int
    up_bound: int


@dataclass(frozen=True, slots=True)
class GateSquareState:
    projection: SharedMLPProjection
    multiplication: MultiplicationState
    opening: OpeningFrame


@dataclass(frozen=True, slots=True)
class GateProductState:
    multiplication: MultiplicationState
    opening: OpeningFrame
    output_bound: int


@dataclass(frozen=True, slots=True)
class ActivationTruncationState:
    projection: SharedMLPProjection
    truncation: TruncationState
    opening: ValueOpeningFrame
    output_bound: int


@dataclass(frozen=True, slots=True)
class StaticGateProductState:
    multiplication: MultiplicationState
    opening: OpeningFrame
    output_bound: int


@dataclass(frozen=True, slots=True)
class ProductTruncationState:
    truncation: TruncationState
    opening: ValueOpeningFrame
    output_bound: int


class SharedMLPParty:
    """Party-local quadratic-SiLU MLP vertical slice.

    Quadratic activation is ``x / 2 + x^2 / 4``. Scale growth is retained
    exactly instead of using an insecure local truncation.
    """

    def __init__(self, runtime: PartyRuntime, weights: SharedMLPWeights) -> None:
        if any(
            matrix.scale != 1 or np.any(matrix.row_multipliers != 1)
            for matrix in (weights.gate_matrix, weights.up_matrix, weights.down_matrix)
        ):
            raise SharedMPCError("legacy local MLP does not support quantized weight scales")
        self.runtime = runtime
        self.weights = weights

    def project(self, hidden: SharedTensor, *, signed_bound: int) -> SharedMLPProjection:
        gate_norm = int(np.max(np.sum(np.abs(self.weights.gate.astype(np.int64)), axis=1)))
        up_norm = int(np.max(np.sum(np.abs(self.weights.up.astype(np.int64)), axis=1)))
        return SharedMLPProjection(
            gate=self.runtime.linear_public(
                hidden, self.weights.gate, "mlp.gate", signed_bound=signed_bound
            ),
            up=self.runtime.linear_public(
                hidden, self.weights.up, "mlp.up", signed_bound=signed_bound
            ),
            gate_bound=signed_bound * gate_norm,
            up_bound=signed_bound * up_norm,
        )

    def begin_gate_square(
        self,
        projection: SharedMLPProjection,
        triple: BeaverTripleShare,
    ) -> GateSquareState:
        state, opening = self.runtime.begin_multiply(
            projection.gate,
            projection.gate,
            triple,
            "mlp.gate_square",
        )
        return GateSquareState(projection, state, opening)

    def finish_gate_square(
        self,
        state: GateSquareState,
        peer: OpeningFrame,
        product_triple: BeaverTripleShare,
    ) -> GateProductState:
        gate_square = self.runtime.finish_multiply(
            state.multiplication,
            state.opening,
            peer,
            "mlp.gate_squared",
        )
        input_scale = state.projection.gate.scale
        linear = self.runtime.mul_public(
            state.projection.gate,
            2 * input_scale,
            "mlp.gate_linear_numerator",
        )
        linear = self.runtime.reinterpret_scale(
            linear,
            input_scale * input_scale,
            "mlp.gate_linear_aligned",
        )
        numerator = self.runtime.add(linear, gate_square, "mlp.activation_numerator")
        activated = self.runtime.reinterpret_scale(
            numerator,
            4 * input_scale * input_scale,
            "mlp.activated_gate",
        )
        multiplication, opening = self.runtime.begin_multiply(
            activated,
            state.projection.up,
            product_triple,
            "mlp.gate_product",
        )
        numerator_bound = (
            2 * input_scale * state.projection.gate_bound
            + state.projection.gate_bound * state.projection.gate_bound
        )
        return GateProductState(
            multiplication,
            opening,
            numerator_bound * state.projection.up_bound,
        )

    def finish(
        self,
        state: GateProductState,
        peer: OpeningFrame,
    ) -> SharedTensor:
        product = self.runtime.finish_multiply(
            state.multiplication,
            state.opening,
            peer,
            "mlp.product",
        )
        return self.runtime.linear_public(
            product,
            self.weights.down,
            "mlp.output",
            signed_bound=state.output_bound,
        )


class StaticSharedMLPParty:
    """Party-local quadratic-SiLU MLP with bounded probabilistic truncation."""

    def __init__(self, runtime: PartyRuntime, weights: SharedMLPWeights) -> None:
        if any(
            matrix.scale != 1 or np.any(matrix.row_multipliers != 1)
            for matrix in (weights.gate_matrix, weights.up_matrix, weights.down_matrix)
        ):
            raise SharedMPCError("static local MLP does not support quantized weight scales")
        self.runtime = runtime
        self.weights = weights

    def project(
        self,
        hidden: SharedTensor,
        *,
        signed_bound: int,
    ) -> tuple[SharedMLPProjection, int, int]:
        if signed_bound <= 0:
            raise SharedMPCError("shared MLP input bound must be positive")
        gate_norm = int(np.max(np.sum(np.abs(self.weights.gate.astype(np.int64)), axis=1)))
        up_norm = int(np.max(np.sum(np.abs(self.weights.up.astype(np.int64)), axis=1)))
        projection = SharedMLPProjection(
            gate=self.runtime.linear_public(
                hidden,
                self.weights.gate,
                "mlp.static.gate",
                signed_bound=signed_bound,
            ),
            up=self.runtime.linear_public(
                hidden,
                self.weights.up,
                "mlp.static.up",
                signed_bound=signed_bound,
            ),
            gate_bound=signed_bound * gate_norm,
            up_bound=signed_bound * up_norm,
        )
        return projection, projection.gate_bound, projection.up_bound

    def begin_gate_square(
        self,
        projection: SharedMLPProjection,
        triple: BeaverTripleShare,
    ) -> GateSquareState:
        state, opening = self.runtime.begin_multiply(
            projection.gate,
            projection.gate,
            triple,
            "mlp.static.gate_square",
        )
        return GateSquareState(projection, state, opening)

    def begin_activation_truncation(
        self,
        state: GateSquareState,
        peer: OpeningFrame,
        mask: TruncationMaskShare,
        *,
        gate_bound: int,
    ) -> ActivationTruncationState:
        gate_square = self.runtime.finish_multiply(
            state.multiplication,
            state.opening,
            peer,
            "mlp.static.gate_squared",
        )
        scale = state.projection.gate.scale
        if scale < 2 or scale & (scale - 1):
            raise SharedMPCError("static shared MLP scale must be a power of two of at least two")
        if gate_bound > scale:
            raise SharedMPCError("quadratic SiLU is calibrated only for gate values in [-1, 1]")
        linear = self.runtime.mul_public(
            state.projection.gate,
            2 * scale,
            "mlp.static.gate_linear_numerator",
        )
        linear = self.runtime.reinterpret_scale(
            linear,
            scale * scale,
            "mlp.static.gate_linear_aligned",
        )
        numerator = self.runtime.add(linear, gate_square, "mlp.static.activation_numerator")
        numerator = self.runtime.reinterpret_scale(
            numerator,
            4 * scale * scale,
            "mlp.static.activation_scaled",
        )
        numerator_bound = 2 * scale * gate_bound + gate_bound * gate_bound
        truncate_bits = scale.bit_length() + 1
        truncation, opening = self.runtime.begin_truncate(
            numerator,
            mask,
            "mlp.static.activation_truncate",
            signed_bound=numerator_bound,
        )
        output_bound = (numerator_bound >> truncate_bits) + 1
        return ActivationTruncationState(
            state.projection,
            truncation,
            opening,
            output_bound,
        )

    def begin_gate_product(
        self,
        state: ActivationTruncationState,
        peer: ValueOpeningFrame,
        triple: BeaverTripleShare,
        *,
        up_bound: int,
    ) -> StaticGateProductState:
        activated = self.runtime.finish_truncate(
            state.truncation,
            state.opening,
            peer,
            "mlp.static.activated_gate",
        )
        multiplication, opening = self.runtime.begin_multiply(
            activated,
            state.projection.up,
            triple,
            "mlp.static.gate_product",
        )
        return StaticGateProductState(
            multiplication,
            opening,
            state.output_bound * up_bound,
        )

    def begin_product_truncation(
        self,
        state: StaticGateProductState,
        peer: OpeningFrame,
        mask: TruncationMaskShare,
    ) -> ProductTruncationState:
        product = self.runtime.finish_multiply(
            state.multiplication,
            state.opening,
            peer,
            "mlp.static.product_scaled",
        )
        scale = int(np.sqrt(product.scale))
        if scale * scale != product.scale or scale & (scale - 1):
            raise SharedMPCError("static shared MLP product scale is invalid")
        truncation, opening = self.runtime.begin_truncate(
            product,
            mask,
            "mlp.static.product_truncate",
            signed_bound=state.output_bound,
        )
        return ProductTruncationState(
            truncation,
            opening,
            (state.output_bound >> (scale.bit_length() - 1)) + 1,
        )

    def finish(
        self,
        state: ProductTruncationState,
        peer: ValueOpeningFrame,
    ) -> SharedTensor:
        product = self.runtime.finish_truncate(
            state.truncation,
            state.opening,
            peer,
            "mlp.static.product",
        )
        return self.runtime.linear_public(
            product,
            self.weights.down,
            "mlp.static.output",
            signed_bound=state.output_bound,
        )


def clear_mlp_reference(
    hidden: np.ndarray,
    weights: SharedMLPWeights,
    *,
    scale: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(hidden, dtype=np.uint64)
    gate_weight = np.asarray(weights.gate, dtype=np.int64).view(np.uint64)
    up_weight = np.asarray(weights.up, dtype=np.int64).view(np.uint64)
    down_weight = np.asarray(weights.down, dtype=np.int64).view(np.uint64)
    gate = np.matmul(values, gate_weight.T, dtype=np.uint64)
    up = np.matmul(values, up_weight.T, dtype=np.uint64)
    activated_numerator = np.uint64(2 * scale) * gate + gate * gate
    product = activated_numerator * up
    output = np.matmul(product, down_weight.T, dtype=np.uint64)
    return output, 4 * scale * scale * scale
