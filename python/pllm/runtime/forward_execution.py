"""Typed native-plan dispatch for batch-one, public source-FP32 forward calls."""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from .bundle_storage import CHUNK_BYTES, immutable_f32_array
from .forward_inputs import inspect_forward_inputs, snapshot_forward_inputs
from .semantic_attention import (
    mask_key_padding_scores, semantic_attention_scores, semantic_attention_values,
)
from .semantic_numeric import (
    layer_norm_contract, source_float32_activation, source_float32_layer_norm,
    source_float32_lookup, source_float32_softmax,
)
from .semantic_tensors import attention_layout
from .source_io import _json_frames_supported


def _shape(spec, sequence):
    dimensions = spec["shape"]
    if not dimensions or any(not (type(n) is int and n > 0 or n == "sequence")
                             for n in dimensions):
        raise ValueError("invalid native forward tensor dimensions")
    return tuple(sequence if n == "sequence" else n for n in dimensions)


def _finite(array):
    return all(np.all(np.isfinite(chunk)) for chunk in np.nditer(
        array, flags=["external_loop", "buffered", "zerosize_ok"], op_flags=["readonly"],
        buffersize=CHUNK_BYTES // 4))


def _python_workspace(plan, sequence, metadata_bytes):
    """Supplement native quote with actual Python/FFI storage overlaps.

    Conservative retained-all-values execution. Native logical quote is not RSS.
    Extra allowance prices bounded native activation serialization/Vec/result
    buffers, finite masks, row reductions, reshape copies, input
    list snapshots, and immutable output publication alongside live operators.
    """
    from .forward_binding import _QUOTE_BOOTSTRAP_BYTES
    sizes = [4 * math.prod(_shape(op["output"], sequence)) for op in plan["operations"]]
    published = sum(4 * math.prod(_shape(spec["tensor"], sequence))
                    for spec in plan["outputs"].values())
    return (8 * CHUNK_BYTES + 2 * max(sizes, default=0) + published
            + 256 * sequence + metadata_bytes + _QUOTE_BOOTSTRAP_BYTES)


class _BoundForwardOperation:
    """Private plan-neutral kernel context; exact native owner supplies tensors."""

    __slots__ = ("operation", "sequence", "shape", "weights")

    def __init__(self, owner, operation_id, sequence):
        from pllm import _native
        from .forward_binding import CompiledForwardModel, ForwardBindingError
        if (type(owner) is not CompiledForwardModel
                or type(owner._contract) is not _native.ForwardContract or not owner._active):
            raise ForwardBindingError("forward kernel requires its native-bound active owner")
        operations = owner._plan["operations"]
        self.operation = next(op for op in operations if op["id"] == operation_id)
        self.sequence = sequence
        self.shape = _shape(self.operation["output"], sequence)
        self.weights = {}
        for name in ("weight", "bias"):
            key = self.operation["attributes"].get(name)
            if key is not None:
                raw, shape = owner._weights[key]
                self.weights[name] = immutable_f32_array(raw, shape)

    def __call__(self, values):
        operation, weights = self.operation, self.weights
        kind, attrs = operation["operator"], operation["attributes"]
        args = [values[name] for name in operation["inputs"]]
        if kind == "token_lookup":
            output = source_float32_lookup(args[0], weights["weight"])
        elif kind == "linear":
            source, weight = args[0], weights["weight"]
            if weight.ndim != 2 or source.shape[-1] != weight.shape[1]:
                raise ValueError("source-oriented FP32 linear dimensions differ")
            output = np.empty(source.shape[:-1] + (weight.shape[0],), dtype=np.float32)
            np.einsum("...i,oi->...o", source, weight, out=output,
                      dtype=np.float32, casting="no", optimize=False)
            if "bias" in weights:
                if weights["bias"].shape != (weight.shape[0],):
                    raise ValueError("FP32 linear bias shape differs")
                np.add(output, weights["bias"], out=output)
        elif kind == "residual_add":
            if len(args) != 2 or args[0].shape != args[1].shape:
                raise ValueError("residual operands must have identical physical shapes")
            output = np.empty(args[0].shape, dtype=np.float32)
            np.add(args[0], args[1], out=output)
        elif kind == "layer_norm":
            epsilon = layer_norm_contract(attrs, args[0].shape[-1])
            output = source_float32_layer_norm(args[0], weights["weight"], weights["bias"], epsilon)
        elif kind == "reshape":
            output = attention_layout(args[0], input_layout=attrs["input_layout"],
                                      output_layout=attrs["layout"], output_shape=self.shape)
        elif kind == "permute":
            axes = attrs["axes"]
            if sorted(axes) != list(range(args[0].ndim)):
                raise ValueError("permute requires a complete axis permutation")
            output = args[0].transpose(axes)
        elif kind == "attention_scores":
            output = semantic_attention_scores(args[0], args[1], group=1,
                                               layout="batch_kv_heads_sequence_feature")
        elif kind == "attention_scale":
            dimension = attrs["head_dim"]
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("attention scale requires positive native head dimension")
            denominator = np.sqrt(np.float32(dimension))
            output = np.empty(args[0].shape, dtype=np.float32)
            np.divide(args[0], denominator, out=output)
        elif kind == "attention_mask":
            if len(args) != 2 or operation["inputs"][1] != "input.attention_mask":
                raise ValueError("key-padding mask requires explicit boolean input")
            output = mask_key_padding_scores(args[0], args[1], attrs)
        elif kind == "softmax":
            output = source_float32_softmax(args[0])
        elif kind == "attention_values":
            output = semantic_attention_values(args[0], args[1], group=1,
                                               layout="batch_kv_heads_sequence_feature")
        elif kind in {"gelu_erf", "tanh"}:
            output = source_float32_activation(args[0], kind)
        elif kind == "slice":
            axis, start, end = attrs["axis"], attrs["start"], attrs["end"]
            if (type(axis) is not int or not 0 <= axis < args[0].ndim
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= args[0].shape[axis]):
                raise ValueError("slice attributes exceed operand extents")
            selection = [slice(None)] * args[0].ndim
            selection[axis] = slice(start, end)
            output = args[0][tuple(selection)]
            if attrs.get("squeeze") is True:
                if end - start != 1:
                    raise ValueError("slice squeeze requires singleton axis")
                output = np.squeeze(output, axis=axis)
        else:
            raise ValueError(f"unsupported native forward operator: {kind}")
        if (type(output) is not np.ndarray or output.dtype != np.float32
                or output.shape != self.shape or not _finite(output)):
            raise ValueError(f"forward operation {operation['id']} violated its F32 output contract")
        return output


def execute_forward(owner, inputs):
    from pllm import _native
    from .forward_binding import (
        CompiledForwardModel, ForwardBindingError, ForwardResult, _admit, _quote,
    )
    if (type(owner) is not CompiledForwardModel
            or type(owner._contract) is not _native.ForwardContract):
        raise ForwardBindingError("forward requires an exact native-bound model")
    if not _EXECUTION_JSON_FRAMES_SUPPORTED:
        raise ForwardBindingError("forward JSON frame layout unsupported")
    # Lease covers preflight too: concurrent input mutation cannot race ownership
    # or closing. Invalid preflight releases lease without spending an attempt.
    with owner._condition:
        if owner._closed:
            raise ForwardBindingError("forward model is closed")
        if owner._active:
            raise ForwardBindingError("forward model already has an in-flight call")
        if owner._requests <= 0:
            raise ForwardBindingError("forward request budget exhausted")
        owner._active = True
    values = storage = kernel = array = None
    try:
        sequence = inspect_forward_inputs(inputs, maximum=owner._maximum, domains=owner._domains)
        retained = sum(len(raw) for raw, _ in owner._weights.values())
        # Lifetime reservation includes the same bounded quote parse envelope
        # used below. Admit before getter/decode/hook, not only after quote.
        def admit_quote(scratch):
            _admit(max(owner._lifetime_reservation["required_bytes"],
                       retained + owner._metadata_bytes + scratch),
                   owner._memory_bytes, retained=retained)
        quote = _quote(owner._contract, sequence, admit=admit_quote)
        required = quote["required_bytes"] + _python_workspace(owner._plan, sequence, owner._metadata_bytes)
        _admit(max(required, owner._lifetime_reservation["required_bytes"]),
               owner._memory_bytes, retained=retained)
        values = snapshot_forward_inputs(inputs, maximum=owner._maximum,
                                         domains=owner._domains, sequence=sequence)
        with owner._condition:
            if owner._closed:
                raise ForwardBindingError("forward model closed during preflight")
            owner._requests -= 1
        for operation in owner._plan["operations"]:
            kernel = _BoundForwardOperation(owner, operation["id"], sequence)
            values[operation["id"]] = kernel(values)
        storage = {}
        names = set(owner._plan["outputs"])
        if names not in ({"last_hidden_state", "pooler_output"},
                         {"last_hidden_state", "pooler_output", "logits"}):
            raise ForwardBindingError("native plan has unsupported augmented output names")
        for name, spec in owner._plan["outputs"].items():
            array = values[spec["value"]]
            shape = _shape(spec["tensor"], sequence)
            if array.shape != shape or array.dtype != np.float32 or not _finite(array):
                raise ForwardBindingError("forward output publication contract differs")
            storage[name] = (array.tobytes(order="C"), shape)
        result = ForwardResult(MappingProxyType(storage), owner._identities)
        storage = None  # Immutable result now owns these independent bytes.
        return result
    finally:
        if values is not None:
            values.clear()
        if storage is not None:
            storage.clear()
        kernel = array = None
        with owner._condition:
            owner._active = False
            owner._condition.notify_all()


_EXECUTION_JSON_FRAME_FUNCTIONS = (_python_workspace, execute_forward)
_EXECUTION_JSON_FRAMES_SUPPORTED = _json_frames_supported(_EXECUTION_JSON_FRAME_FUNCTIONS)


__all__ = []
