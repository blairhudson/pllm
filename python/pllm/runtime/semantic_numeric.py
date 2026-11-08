"""Bounded local numeric contracts shared by semantic decoder operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import ml_dtypes
import numpy as np


class SemanticNumericError(ValueError):
    pass


def _contract_mapping(value):
    """Only JSON objects and the native facade's frozen object representation."""
    if (type(value) is not dict and type(value) is not MappingProxyType
            or any(type(key) is not str for key in value)):
        raise SemanticNumericError("semantic contract requires an exact object container")
    return value


def _contract_dimensions(value):
    """Normalize small shape declarations, never arbitrary sequence objects."""
    if (type(value) is not list and type(value) is not tuple
            or not value or any(type(n) is not int or n < 1 for n in value)):
        return None
    return tuple(value)


def layer_norm_contract(attributes: dict[str, Any], width: int) -> np.float32:
    """Validate the existing centered affine FP32 declaration, without weights."""
    attributes = _contract_mapping(attributes)
    allowed = {
        "epsilon", "weight", "bias", "weight_shape", "bias_shape", "axis",
        "variance", "epsilon_placement", "affine_order", "compute_dtype", "output_dtype",
    }
    if (
        set(attributes) - allowed or type(width) is not int or width < 1
        or type(attributes.get("axis")) is not int or attributes["axis"] != -1
        or any(type(attributes.get(key)) is not str or attributes[key] != expected
               for key, expected in (("variance", "centered_mean_square"),
                                     ("epsilon_placement", "inside_sqrt"),
                                     ("affine_order", "normalize_multiply_add"),
                                     ("compute_dtype", "float32")))
        or ("output_dtype" in attributes and (type(attributes["output_dtype"]) is not str
                                             or attributes["output_dtype"] != "float32"))
        or type(attributes.get("weight")) is not str or not attributes["weight"]
        or _contract_dimensions(attributes.get("weight_shape")) != (width,)
    ):
        raise SemanticNumericError("unsupported centered affine LayerNorm contract")
    bias = attributes.get("bias")
    if (
        type(bias) is not str or not bias
        or _contract_dimensions(attributes.get("bias_shape")) != (width,)
    ):
        raise SemanticNumericError("LayerNorm bias declaration is invalid")
    epsilon = attributes.get("epsilon")
    if type(epsilon) is not str:
        raise SemanticNumericError("LayerNorm epsilon is invalid")
    try:
        from pllm import _native as core
        result = np.float32(core.parse_positive_f32(epsilon))
    except (ValueError, OverflowError) as exc:
        raise SemanticNumericError("LayerNorm epsilon is invalid") from exc
    if not np.isfinite(result) or result <= 0:
        raise SemanticNumericError("LayerNorm epsilon is invalid")
    return result


@dataclass(frozen=True, slots=True)
class LocalNumericWorkspace:
    """Logical array storage estimate, not physical allocator or work-ledger proof."""

    max_rows: int
    width: int
    output_bytes: int
    row_scratch_bytes: int
    finite_check_bytes: int
    scalar_bytes: int
    estimated_scratch_bytes: int

    @property
    def temporary_bytes(self) -> int:
        return self.output_bytes + self.estimated_scratch_bytes


def _compiled_local_geometry(plan, phase: str, operation_id: str):
    from pllm.modeling import ModelPlan, DecoderContinuationSchedule

    if type(plan) is ModelPlan and phase in ("prefill", "decode"):
        graph = plan.to_dict()[phase]
    elif type(plan) is DecoderContinuationSchedule and phase == "continuation":
        graph = plan.to_dict()["graph"]
    else:
        raise SemanticNumericError("local numeric geometry requires a compiled phase")
    operations = {op["id"]: op for op in graph["operations"]}
    operation = operations.get(operation_id)
    if operation is None or operation["operator"] not in ("layer_norm", "token_lookup", "gelu_tanh"):
        raise SemanticNumericError("operation has no embedding/normalization geometry")
    shape = operation["output_shape"]
    query, maximum = graph["query_sequence"], graph["maximum_key_sequence"]
    if (
        not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1
        or any(type(n) is not int or n < 1 for n in (*shape, query, maximum))
        or shape[1] != query or query > maximum
    ):
        raise SemanticNumericError("local numeric geometry differs from its compiled phase")
    if operation["operator"] in ("layer_norm", "gelu_tanh"):
        inputs = operation["inputs"]
        if len(inputs) != 1 or inputs[0] not in operations or operations[inputs[0]]["output_shape"] != shape:
            raise SemanticNumericError("LayerNorm geometry differs from its producer")
    return operation, query, shape[-1]


def local_numeric_workspace(plan, phase: str, operation_id: str) -> LocalNumericWorkspace:
    """Maximum geometry comes solely from the native compiled phase declaration."""
    operation, rows, width = _compiled_local_geometry(plan, phase, operation_id)
    from .bundle_storage import CHUNK_BYTES
    from .semantic_workspace import semantic_local_workspace_bytes
    norm = operation["operator"] in ("layer_norm", "gelu_tanh")
    return LocalNumericWorkspace(rows, width, 4 * rows * width,
                                 4 * width if norm else 0,
                                 min(width, CHUNK_BYTES // 4) if norm else rows,
                                 16 if norm else 0,
                                 semantic_local_workspace_bytes(operation, (1, rows, width)))


def _finite_row(row):
    from .bundle_storage import CHUNK_BYTES
    elements = CHUNK_BYTES // 4
    return all(np.all(np.isfinite(row[start:start + elements]))
               for start in range(0, row.size, elements))


def source_float32_layer_norm(source, weight, bias, epsilon):
    """Plan-neutral centered population LayerNorm; bounded row scratch."""
    if (type(source) is not np.ndarray or source.dtype != np.float32
            or source.ndim < 2 or source.shape[-1] < 1):
        raise SemanticNumericError("LayerNorm requires a nonempty FP32 tensor")
    width = source.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,):
        raise SemanticNumericError("LayerNorm coefficient width differs")
    for index in np.ndindex(source.shape[:-1]):
        if not _finite_row(source[index]):
            raise SemanticNumericError("LayerNorm input must be finite FP32")
    output = np.empty(source.shape, dtype=np.float32)
    squared = np.empty(width, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for index in np.ndindex(source.shape[:-1]):
            row, target = source[index], output[index]
            mean = np.mean(row, dtype=np.float32)
            np.subtract(row, mean, out=target)
            np.multiply(target, target, out=squared)
            variance = np.mean(squared, dtype=np.float32)
            adjusted = np.float32(variance + epsilon)
            denominator = np.sqrt(adjusted)
            if not all(np.isfinite(item) for item in (mean, variance, adjusted, denominator)):
                raise SemanticNumericError("LayerNorm arithmetic exceeds the finite float32 domain")
            np.divide(target, denominator, out=target)
            np.multiply(target, weight, out=target)
            if not _finite_row(target):
                raise SemanticNumericError("LayerNorm affine multiply exceeds the finite float32 domain")
            np.add(target, bias, out=target)
            if not _finite_row(target):
                raise SemanticNumericError("LayerNorm result exceeds the finite float32 domain")
    return output


def source_float32_lookup(ids, weight):
    """Exact immutable F32 table lookup, preserving input physical dimensions."""
    from .bundle_storage import immutable_f32_backing
    immutable_f32_backing(weight)
    if type(ids) is not np.ndarray or ids.dtype != np.int64 or ids.size < 1:
        raise SemanticNumericError("lookup requires exact int64 IDs")
    if any(not 0 <= int(item) < weight.shape[0] for item in ids.flat):
        raise SemanticNumericError("embedding IDs exceed their compiled domain")
    return weight[ids]


def source_float32_softmax(source):
    """Stable last-axis FP32 softmax; no tensor-sized reduction masks."""
    if type(source) is not np.ndarray or source.dtype != np.float32 or min(source.shape) < 1:
        raise SemanticNumericError("softmax requires nonempty FP32 scores")
    output = np.empty(source.shape, dtype=np.float32)
    for index in np.ndindex(source.shape[:-1]):
        row, target = source[index], output[index]
        if not _finite_row(row):
            raise SemanticNumericError("softmax scores must be finite")
        np.subtract(row, np.max(row), out=target)
        np.exp(target, out=target)
        denominator = np.sum(target, dtype=np.float32)
        if not np.isfinite(denominator) or denominator <= 0:
            raise SemanticNumericError("softmax normalization is invalid")
        np.divide(target, denominator, out=target)
    return output


def source_float32_activation(source, kind):
    """Native erff/tanhf source arithmetic, serialized in bounded F32 chunks."""
    from pllm import _native
    from .bundle_storage import CHUNK_BYTES
    if (type(source) is not np.ndarray or source.dtype != np.dtype("<f4")
            or kind not in {"gelu_erf", "tanh"}):
        raise SemanticNumericError("unsupported source FP32 activation")
    output = np.empty(source.shape, dtype=np.float32)
    # nditer buffers noncontiguous inputs in bounded chunks; no full reshape copy.
    iterator = np.nditer([source, output], flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=[["readonly"], ["writeonly"]], order="C",
                         buffersize=CHUNK_BYTES // 4)
    with iterator:
        for value, target in iterator:
            if not _finite_row(value):
                raise SemanticNumericError("activation input must be finite")
            raw = _native.source_float32_activation(kind, value.tobytes())
            if type(raw) is not bytes or len(raw) != value.size * 4:
                raise SemanticNumericError("native activation returned invalid storage")
            decoded = np.frombuffer(raw, dtype="<f4")
            if not _finite_row(decoded):
                raise SemanticNumericError("activation result must be finite")
            target[:] = decoded
    return output


class _BoundLocalNumeric:
    """Private runtime kernel. Tensor authority belongs to CompiledRuntimeModel."""

    def __init__(self, plan, phase: str, operation_id: str, owner):
        from .model_binding import CompiledRuntimeModel
        from pllm.modeling import ModelPlan, DecoderContinuationSchedule

        if (type(owner) is not CompiledRuntimeModel
                or not ((type(plan) is ModelPlan and owner._plan is plan)
                        or (type(plan) is DecoderContinuationSchedule
                            and plan._source_plan == owner._plan.canonical_bytes()))):
            raise SemanticNumericError("local numeric kernel requires its private compiled tensor owner")
        tensor = owner.local_tensor
        operation, self.max_rows, self.width = _compiled_local_geometry(plan, phase, operation_id)
        self.operation = operation
        self.workspace = local_numeric_workspace(plan, phase, operation_id)
        attrs = operation["attributes"]
        if operation["operator"] == "gelu_tanh":
            if attrs != {
                "formula": "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))",
                "compute_dtype": "float32",
            }:
                raise SemanticNumericError("unsupported compiled FP32 GELU declaration")
            self.gelu_cubic = np.float32(0.044715)
            self.gelu_coefficient = np.float32(math.sqrt(2.0 / math.pi))
        elif operation["operator"] == "layer_norm":
            self.epsilon = layer_norm_contract(attrs, self.width)
            self.weight = tensor(attrs["weight"])
            self.bias = tensor(attrs["bias"])
            self._coefficient(self.weight, (self.width,))
            self._coefficient(self.bias, (self.width,))
        else:
            from .semantic_tensors import lookup_contract
            _, _, self.table_shape = lookup_contract(operation)
            self.weight = tensor(attrs["weight"])
            self._coefficient(self.weight, self.table_shape)

    @staticmethod
    def _coefficient(value, shape):
        if (type(value) is not np.ndarray or value.dtype != np.float32
                or value.shape != shape or value.flags.writeable or not value.flags.c_contiguous):
            raise SemanticNumericError("local numeric coefficients require admitted immutable FP32 backing")
        backing = value
        while isinstance(backing, np.ndarray):
            if backing.flags.writeable:
                raise SemanticNumericError("local numeric coefficient has mutable backing")
            backing = backing.base
        if type(backing) is not bytes:
            raise SemanticNumericError("local numeric coefficient lacks its immutable admitted snapshot")

    def _rows(self, source):
        if (type(source) is not np.ndarray or source.dtype != np.float32
                or source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[0] != 1)
                or source.shape[-1] != self.width or not 1 <= source.shape[-2] <= self.max_rows):
            raise SemanticNumericError("local FP32 operand exceeds its compiled geometry")
        # Do not reshape a noncontiguous matrix: that could allocate another NxW.
        return source if source.ndim == 2 else source[0]

    def normalize(self, source: np.ndarray) -> np.ndarray:
        rows = self._rows(source)
        from .semantic_workspace import semantic_local_workspace_bytes
        semantic_local_workspace_bytes(self.operation, (1, rows.shape[0], self.width))
        return source_float32_layer_norm(source, self.weight, self.bias, self.epsilon)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        if (type(ids) is not np.ndarray or ids.dtype != np.int64 or ids.ndim != 1
                or not 1 <= ids.size <= self.max_rows
                or np.any(ids < 0) or np.any(ids >= self.table_shape[0])):
            raise SemanticNumericError("embedding IDs exceed their compiled domain or row geometry")
        from .semantic_workspace import semantic_local_workspace_bytes
        semantic_local_workspace_bytes(self.operation, (1, ids.size, self.width))
        # Import owner already checked and retained the entire immutable table.
        return source_float32_lookup(ids, self.weight)

    def activate(self, source: np.ndarray) -> np.ndarray:
        rows = self._rows(source)
        from .semantic_workspace import semantic_local_workspace_bytes
        semantic_local_workspace_bytes(self.operation, (1, rows.shape[0], self.width))
        for row in rows:
            if not _finite_row(row):
                raise SemanticNumericError("float32 GELU input must be finite float32")
        output = np.empty(source.shape, dtype=np.float32)
        target = output if output.ndim == 2 else output[0]
        scratch = np.empty(self.width, dtype=np.float32)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for index, row in enumerate(rows):
                np.multiply(self.gelu_cubic, row, out=scratch)
                np.multiply(scratch, row, out=scratch)
                np.multiply(scratch, row, out=scratch)
                np.add(row, scratch, out=scratch)
                np.multiply(self.gelu_coefficient, scratch, out=scratch)
                np.tanh(scratch, out=scratch)
                np.add(np.float32(1), scratch, out=scratch)
                np.multiply(np.float32(0.5), row, out=target[index])
                np.multiply(target[index], scratch, out=target[index])
                if not _finite_row(target[index]):
                    raise SemanticNumericError("float32 GELU result exceeds the finite domain")
        return output


def float32_layer_norm(value: np.ndarray, contract: _BoundLocalNumeric) -> np.ndarray:
    if type(contract) is not _BoundLocalNumeric or contract.operation["operator"] != "layer_norm":
        raise SemanticNumericError("LayerNorm requires its private compiled runtime contract")
    return contract.normalize(value)


def float32_embedding_lookup(indices: np.ndarray, contract: _BoundLocalNumeric) -> np.ndarray:
    if type(contract) is not _BoundLocalNumeric or contract.operation["operator"] != "token_lookup":
        raise SemanticNumericError("lookup requires its private compiled runtime contract")
    return contract.lookup(indices)


def float32_gelu_tanh(value: np.ndarray, contract: _BoundLocalNumeric) -> np.ndarray:
    """Existing tanh GELU formula with FP32 arithmetic and no BF16 boundary."""
    if type(contract) is not _BoundLocalNumeric or contract.operation["operator"] != "gelu_tanh":
        raise SemanticNumericError("GELU requires its private compiled runtime contract")
    return contract.activate(value)


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
        try:
            coefficient = math.sqrt(radicand)
        except OverflowError as exc:
            raise SemanticNumericError("semantic scale factor exceeds the numeric domain") from exc
        if kind == "inverse_sqrt":
            coefficient = 1.0 / coefficient
    elif kind == "rational":
        if factor.get("factor_source_dtype") != "exact_integer":
            raise SemanticNumericError("unsupported rational scale factor source")
        numerator, denominator = factor.get("numerator"), factor.get("denominator")
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise SemanticNumericError("semantic rational scale requires bounded integers")
        try:
            coefficient = numerator / denominator
        except OverflowError as exc:
            raise SemanticNumericError("semantic scale factor exceeds the numeric domain") from exc
    elif kind == "checkpoint_scalar":
        if (
            factor.get("factor_source_dtype") != "bfloat16"
            or factor.get("weight_shape") != [1]
            or weight is None
            or np.asarray(weight).shape != (1,)
        ):
            raise SemanticNumericError("semantic checkpoint scale requires one bound BF16 value")
        coefficient = weight
    else:
        raise SemanticNumericError("unsupported semantic scale factor")
    input_value = round_bfloat16(value)
    if input_value.ndim == 0:
        raise SemanticNumericError("semantic scale requires a tensor input")
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


def bfloat16_rms_norm(
    value: np.ndarray, *, epsilon: str, weight: np.ndarray | None, offset: int
) -> np.ndarray:
    """Float32 variance and learned scale with a BF16 activation boundary."""
    if type(epsilon) is not str or not 1 <= len(epsilon) <= 40 or type(offset) is not int:
        raise SemanticNumericError("unsupported BF16 normalization contract")
    try:
        epsilon_value = np.float32(float(Fraction(epsilon)))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise SemanticNumericError("BF16 normalization epsilon is invalid") from exc
    if not np.isfinite(epsilon_value) or epsilon_value <= 0:
        raise SemanticNumericError("BF16 normalization epsilon is invalid")
    source = round_bfloat16(value)
    if source.ndim < 1 or source.shape[-1] < 1:
        raise SemanticNumericError("BF16 normalization requires nonempty feature width")
    with np.errstate(over="ignore"):
        squared = source * source
    if not np.all(np.isfinite(squared)):
        raise SemanticNumericError("BF16 normalization variance exceeds the numeric domain")
    with np.errstate(over="ignore"):
        variance = np.mean(squared, axis=-1, keepdims=True, dtype=np.float32)
    if not np.all(np.isfinite(variance)):
        raise SemanticNumericError("BF16 normalization variance exceeds the numeric domain")
    with np.errstate(over="ignore"):
        adjusted_variance = variance + epsilon_value
    if not np.all(np.isfinite(adjusted_variance)):
        raise SemanticNumericError("BF16 normalization variance exceeds the numeric domain")
    normalized = source / np.sqrt(adjusted_variance)
    if weight is None:
        if offset != 0:
            raise SemanticNumericError("unweighted BF16 normalization requires zero weight offset")
        return round_bfloat16(normalized)
    learned = _finite_float32(weight)
    if learned.shape != (source.shape[-1],) or offset not in {0, 1}:
        raise SemanticNumericError("BF16 normalization weight has the wrong shape or offset")
    return round_bfloat16(normalized * (learned + np.float32(offset)))


def bfloat16_softmax(value: np.ndarray) -> np.ndarray:
    """Float32 last-axis normalization with one BF16 output boundary."""
    source = np.asarray(value, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] < 1 or not np.all(np.isfinite(source) | np.isneginf(source)):
        raise SemanticNumericError("BF16 softmax scores are invalid")
    maximum = np.max(source, axis=-1, keepdims=True)
    if not np.all(np.isfinite(maximum)):
        raise SemanticNumericError("BF16 softmax has a fully masked query")
    exponential = np.exp(source - maximum).astype(np.float32)
    return round_bfloat16(exponential / exponential.sum(axis=-1, keepdims=True))


def bfloat16_rotary(
    value: np.ndarray, positions: np.ndarray, attributes: dict[str, object]
) -> np.ndarray:
    """Full-head or zero-frequency proportional RoPE with BF16 step boundaries."""
    source = np.asarray(value)
    indices = np.asarray(positions)
    head = attributes.get("head_dim")
    theta = attributes.get("theta")
    partial = attributes.get("partial_rotary_factor")
    scaling = attributes.get("attention_scaling")
    if (
        source.ndim != 4
        or source.dtype != np.float32
        or not np.all(np.isfinite(source))
        or indices.ndim != 1
        or indices.dtype != np.int64
        or not np.all((indices >= 0) & (indices < (1 << 24)))
        or attributes.get("input_layout") != "batch_sequence_heads_feature"
        or attributes.get("output_layout") != "batch_sequence_heads_feature"
        or attributes.get("pairing") != "split_half"
        or attributes.get("position_policy") != "sequential_absolute"
        or attributes.get("coefficient_profile") != "pllm.numeric.rope.bfloat16_stepwise.v1"
        or attributes.get("tail_policy") != "zero_frequency_pass_through"
        or attributes.get("frequency_compute_dtype") != "float32"
        or attributes.get("numeric_semantics") != "bfloat16_stepwise"
        or attributes.get("output_dtype") != "bfloat16"
        or type(head) is not int
        or head < 2
        or head % 2
        or source.shape[-1] != head
        or source.shape[1] != indices.size
        or type(theta) is not int
        or theta < 1
        or not isinstance(partial, dict)
        or not isinstance(scaling, dict)
        or scaling != {"numerator": 1, "denominator": 1}
        or np.any((source.view(np.uint32) & 0xFFFF) != 0)
    ):
        raise SemanticNumericError("BF16 rotary shape, source or policy is invalid")
    numerator, denominator = partial.get("numerator"), partial.get("denominator")
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 1
        or denominator < numerator
    ):
        raise SemanticNumericError("BF16 rotary fraction is invalid")
    rope_type = attributes.get("rope_type")
    if rope_type == "default" and numerator == denominator:
        frequencies = np.arange(0, head, 2, dtype=np.float32) / np.float32(head)
    elif rope_type == "proportional" and numerator < denominator:
        angles = (head * numerator) // (2 * denominator)
        if angles < 1 or angles > head // 2:
            raise SemanticNumericError("proportional rotary angle count is invalid")
        frequencies = np.arange(0, 2 * angles, 2, dtype=np.float32) / np.float32(head)
    else:
        raise SemanticNumericError("BF16 rotary variant is unsupported")
    inverse = np.float32(1.0) / (np.float32(theta) ** frequencies)
    if inverse.size < head // 2:
        inverse = np.pad(inverse, (0, head // 2 - inverse.size))
    angles = indices.astype(np.float32)[:, None] * inverse[None, :]
    repeated = np.concatenate((angles, angles), axis=-1)
    cosine = round_bfloat16(np.cos(repeated).astype(np.float32))[None, :, None, :]
    sine = round_bfloat16(np.sin(repeated).astype(np.float32))[None, :, None, :]
    half = head // 2
    rotated = np.concatenate((-source[..., half:], source[..., :half]), axis=-1)
    return round_bfloat16(
        round_bfloat16(source * cosine) + round_bfloat16(rotated * sine)
    )


def float32_rotary_wavelength(
    value: np.ndarray, positions: np.ndarray, attributes: dict[str, object]
) -> np.ndarray:
    """Bounded split-half RoPE with a public wavelength-transition frequency profile."""
    source = np.asarray(value)
    indices = np.asarray(positions)
    scaling = attributes.get("frequency_scaling")
    theta = attributes.get("theta")
    dimensions = attributes.get("rotary_dimensions")
    if (
        set(attributes) != {
            "theta", "rotary_dimensions", "pairing", "position_policy",
            "coefficient_profile", "input_layout", "output_layout", "tail_policy",
            "frequency_scaling",
        }
        or attributes.get("pairing") != "split_half"
        or attributes.get("position_policy") != "sequential_absolute"
        or attributes.get("coefficient_profile") != "pllm.numeric.rope.float32.wavelength.v1"
        or attributes.get("input_layout") != "batch_heads_sequence_feature"
        or attributes.get("output_layout") != "batch_heads_sequence_feature"
        or attributes.get("tail_policy") != "unchanged"
        or source.ndim != 4
        or source.dtype != np.float32
        or not np.all(np.isfinite(source))
        or indices.ndim != 1
        or indices.dtype != np.int64
        or not np.all((indices >= 0) & (indices < (1 << 24)))
        or source.shape[2] != indices.size
        or type(dimensions) is not int
        or not (2 <= dimensions <= source.shape[-1])
        or dimensions % 2
        or type(theta) is not int
        or theta < 1
        or not isinstance(scaling, dict)
        or set(scaling) != {
            "kind", "factor", "low_freq_factor", "high_freq_factor",
            "original_max_position_embeddings",
        }
        or scaling.get("kind") != "wavelength_transition"
    ):
        raise SemanticNumericError("float32 wavelength rotary shape or policy is invalid")
    factor = scaling["factor"]
    low = scaling["low_freq_factor"]
    high = scaling["high_freq_factor"]
    original = scaling["original_max_position_embeddings"]
    if (
        any(type(item) not in (float, int) or not math.isfinite(item)
            for item in (factor, low, high))
        or not (1 < factor <= 256 and 1 / 256 <= low < high <= 256)
        or np.float32(factor) <= 1
        or np.float32(high) - np.float32(low) < 1 / 256
        or type(original) is not int
        or not 1 <= original <= (1 << 24)
    ):
        raise SemanticNumericError("float32 wavelength rotary scaling is invalid")
    power = np.arange(0, dimensions, 2, dtype=np.float32) / np.float32(dimensions)
    inverse = np.float32(1.0) / (np.float32(theta) ** power)
    wavelength = np.float32(2.0 * math.pi) / inverse
    original32 = np.float32(original)
    factor32, low32, high32 = (np.float32(item) for item in (factor, low, high))
    smooth = (original32 / wavelength - low32) / (high32 - low32)
    scaled = np.where(
        wavelength > original32 / low32,
        inverse / factor32,
        np.where(
            wavelength < original32 / high32,
            inverse,
            (np.float32(1.0) - smooth) * (inverse / factor32) + smooth * inverse,
        ),
    )
    angles = indices.astype(np.float32)[:, None] * scaled[None, :]
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)[None, None]
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)[None, None]
    current = source[..., :dimensions]
    half = dimensions // 2
    rotated = np.concatenate((-current[..., half:], current[..., :half]), axis=-1)
    result = source.copy()
    result[..., :dimensions] = current * cosine + rotated * sine
    return result


def float32_rotary_per_frequency(
    value: np.ndarray, positions: np.ndarray, attributes: dict[str, object]
) -> np.ndarray:
    """Short-context, public per-frequency split-half RoPE; no KV frequency switch."""
    source = np.asarray(value)
    indices = np.asarray(positions)
    descriptor = attributes.get("frequency_scaling")
    dimensions = attributes.get("rotary_dimensions")
    theta = attributes.get("theta")
    if (
        set(attributes) != {
            "theta", "rotary_dimensions", "rope_type", "pairing", "position_policy",
            "coefficient_profile", "input_layout", "output_layout", "tail_policy",
            "frequency_scaling",
        }
        or attributes.get("rope_type") != "longrope"
        or attributes.get("pairing") != "split_half"
        or attributes.get("position_policy") != "sequential_absolute"
        or attributes.get("coefficient_profile") != "pllm.numeric.rope.float32.per_frequency.v1"
        or attributes.get("input_layout") != "batch_heads_sequence_feature"
        or attributes.get("output_layout") != "batch_heads_sequence_feature"
        or attributes.get("tail_policy") != "unchanged"
        or source.ndim != 4
        or source.dtype != np.float32
        or not np.all(np.isfinite(source))
        or indices.ndim != 1
        or indices.dtype != np.int64
        or source.shape[2] != indices.size
        or type(dimensions) is not int
        or not (2 <= dimensions <= source.shape[-1])
        or dimensions % 2
        or type(theta) is not int
        or theta < 1
        or not isinstance(descriptor, dict)
        or set(descriptor) != {
            "kind", "factor", "original_max_position_embeddings", "short_factor", "long_factor",
        }
        or descriptor.get("kind") != "per_frequency_context"
    ):
        raise SemanticNumericError("per-frequency rotary shape or policy is invalid")
    original = descriptor["original_max_position_embeddings"]
    factor = descriptor["factor"]
    short, long = descriptor["short_factor"], descriptor["long_factor"]
    if (
        type(original) is not int
        or not (2 <= original <= (1 << 24))
        or type(factor) not in (float, int)
        or not math.isfinite(factor)
        or not (1 <= factor <= 256)
        or not isinstance(short, list)
        or not isinstance(long, list)
        or len(short) != dimensions // 2
        or len(long) != dimensions // 2
        or any(
            type(item) not in (float, int)
            or not math.isfinite(item)
            or not (1 / 256 <= item <= 256)
            for item in (*short, *long)
        )
        or np.any((indices < 0) | (indices >= original))
    ):
        raise SemanticNumericError("per-frequency rotary exceeds bounded short context")
    powers = np.arange(0, dimensions, 2, dtype=np.float32) / np.float32(dimensions)
    inverse = np.float32(1.0) / (
        np.asarray(short, dtype=np.float32) * (np.float32(theta) ** powers)
    )
    angles = indices.astype(np.float32)[:, None] * inverse[None, :]
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)[None, None]
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)[None, None]
    attention_factor = np.float32(
        math.sqrt(1 + math.log(factor) / math.log(original)) if factor > 1 else 1
    )
    cosine *= attention_factor
    sine *= attention_factor
    current = source[..., :dimensions]
    half = dimensions // 2
    rotated = np.concatenate((-current[..., half:], current[..., :half]), axis=-1)
    result = source.copy()
    result[..., :dimensions] = current * cosine + rotated * sine
    return result


def float32_rotary_text_mrope(
    value: np.ndarray, positions: np.ndarray, attributes: dict[str, object]
) -> np.ndarray:
    """Text-only partial RoPE: the three MRoPE axes share the same absolute positions."""
    source = np.asarray(value)
    indices = np.asarray(positions)
    dimensions = attributes.get("rotary_dimensions")
    theta = attributes.get("theta")
    sections = attributes.get("mrope_section")
    partial = attributes.get("partial_rotary_factor")
    theta_float = float(theta) if isinstance(theta, (float, int)) and not isinstance(theta, bool) else math.nan
    if (
        set(attributes) != {
            "theta", "rope_type", "rotary_dimensions", "partial_rotary_factor",
            "mrope_interleaved", "mrope_section", "position_policy",
        }
        or attributes.get("rope_type") != "default"
        or attributes.get("position_policy") != "text_replicated_axes"
        or attributes.get("mrope_interleaved") is not True
        or source.dtype != np.float32 or source.ndim != 4
        or source.shape[0] != 1 or not 1 <= source.shape[1] <= 64
        or not 1 <= source.shape[2] <= 256 or source.size > 1 << 22
        or indices.dtype != np.int64 or indices.ndim != 1
        or indices.size != source.shape[2]
        or type(dimensions) is not int or not 2 <= dimensions <= source.shape[-1]
        or dimensions % 2
        or not math.isfinite(theta_float) or not 1 <= theta_float <= 1e12
        or partial not in ("0.25", "1")
        or dimensions != source.shape[-1] * (0.25 if partial == "0.25" else 1)
        or not isinstance(sections, (list, tuple)) or len(sections) != 3
        or any(type(part) is not int or part < 0 for part in sections)
        or sum(sections) != dimensions // 2
    ):
        raise SemanticNumericError("text-only multimodal rotary descriptor is invalid")
    if np.any(indices < 0) or np.any(indices >= 1 << 18) or not np.all(np.isfinite(source)):
        raise SemanticNumericError("text-only rotary input exceeds its finite position domain")
    powers = np.arange(0, dimensions, 2, dtype=np.float32) / np.float32(dimensions)
    inverse = np.float32(1.0) / (np.float32(theta_float) ** powers)
    angles = indices.astype(np.float32)[:, None] * inverse[None, :]
    # Interleaving selects the time, height, and width frequency slots. All
    # three equal these text positions, so the selection is value-preserving.
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)[None, None]
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)[None, None]
    current = source[..., :dimensions]
    half = dimensions // 2
    rotated = np.concatenate((-current[..., half:], current[..., :half]), axis=-1)
    result = source.copy()
    result[..., :dimensions] = current * cosine + rotated * sine
    if not np.all(np.isfinite(result)):
        raise SemanticNumericError("text-only rotary output is not finite")
    return result


__all__ = [
    "SemanticNumericError",
    "LocalNumericWorkspace",
    "local_numeric_workspace",
    "layer_norm_contract",
    "float32_layer_norm",
    "float32_embedding_lookup",
    "float32_gelu_tanh",
    "bfloat16_gelu_tanh",
    "bfloat16_scale",
    "bfloat16_softcap",
    "bfloat16_rms_norm",
    "bfloat16_rotary",
    "bfloat16_softmax",
    "float32_rotary_wavelength",
    "float32_rotary_per_frequency",
    "float32_rotary_text_mrope",
    "round_bfloat16",
]
