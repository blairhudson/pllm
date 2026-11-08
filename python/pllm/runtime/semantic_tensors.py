"""Client-owned tensor requirements derived from semantic operator contracts."""

from __future__ import annotations

from pllm.modeling import ModelPlan

from .safetensors_store import SafeTensorStore, TensorStoreError
from .semantic_numeric import (
    SemanticNumericError, _contract_dimensions, _contract_mapping, layer_norm_contract,
)


class SemanticTensorError(ValueError):
    pass


def attention_layout(source, *, input_layout, output_layout, output_shape):
    """Neutral head split/merge adapter with explicit physical output rank.

    Decoder callers may retain [S,H]; forward callers retain [1,S,H]. This
    helper never infers semantic geometry from a source-family spelling.
    """
    import math
    import numpy as np
    shape = tuple(output_shape)
    if (type(source) is not np.ndarray or source.dtype != np.float32
            or any(type(n) is not int or n < 1 for n in shape)
            or source.size != math.prod(shape)):
        raise SemanticTensorError("attention layout has invalid physical geometry")
    if input_layout == "batch_sequence_hidden" and output_layout == "batch_heads_sequence_feature":
        if (len(shape) != 4 or shape[0] != 1 or source.ndim not in (2, 3)
                or source.shape[-2:] != (shape[2], shape[1] * shape[3])
                or source.ndim == 3 and source.shape[0] != 1):
            raise SemanticTensorError("attention head split differs from its typed output")
        return np.ascontiguousarray(source).reshape(1, shape[2], shape[1], shape[3]).transpose(0, 2, 1, 3)
    if input_layout == "batch_heads_sequence_feature" and output_layout == "batch_sequence_hidden":
        if (source.ndim != 4 or source.shape[0] != 1
                or shape not in ((source.shape[2], source.shape[1] * source.shape[3]),
                                 (1, source.shape[2], source.shape[1] * source.shape[3]))):
            raise SemanticTensorError("attention head merge differs from its typed output")
        return np.ascontiguousarray(source.transpose(0, 2, 1, 3)).reshape(shape)
    raise SemanticTensorError("unsupported explicit attention layout adapter")


def _operation_attributes(operation):
    try:
        operation = _contract_mapping(operation)
        return _contract_mapping(operation.get("attributes"))
    except SemanticNumericError as exc:
        raise SemanticTensorError(str(exc)) from exc


def lookup_contract(
    operation: dict, *, legacy_vocabulary: int | None = None,
) -> tuple[str, str, tuple[int, int]]:
    """Resolve table geometry from its integer source domain, never from a head."""
    attrs = _operation_attributes(operation)
    inputs = operation.get("inputs")
    if (type(inputs) is not list and type(inputs) is not tuple
            or len(inputs) != 1 or type(inputs[0]) is not str):
        raise SemanticTensorError("lookup has an ambiguous or mismatched index domain")
    domains = {
        "tokens": "input.tokens",
        "absolute_positions": "input.positions",
        "segments": "input.segments",
    }
    domain = attrs.get("index_domain")
    if "index_domain" not in attrs and inputs[0] == "input.tokens":
        domain = "tokens"
    if type(domain) is not str or domain not in domains or inputs[0] != domains[domain]:
        raise SemanticTensorError("lookup has an ambiguous or mismatched index domain")
    source = domains[domain]
    if type(attrs.get("source", source)) is not str or attrs.get("source", source) not in (source, domain):
        raise SemanticTensorError("lookup source differs from its index domain")
    if attrs.get("bias") is not None:
        raise SemanticTensorError("lookup cannot declare a projection bias")
    if "weight_layout" in attrs and (type(attrs["weight_layout"]) is not str
                                     or attrs["weight_layout"] != "output_input"):
        raise SemanticTensorError("lookup table requires output_input checkpoint layout")
    if domain == "absolute_positions" and any(
        type(attrs.get(key)) is not str or attrs[key] != expected
        for key, expected in (("position_policy", "sequential_absolute"),
                              ("out_of_bounds", "reject"), ("compute_dtype", "float32"))
    ):
        raise SemanticTensorError("learned position lookup policy is unsupported")
    shape = attrs.get("weight_shape")
    output = _contract_dimensions(operation.get("output_shape"))
    if output is None:
        raise SemanticTensorError("lookup output width is invalid")
    # Only the actual old token boundary may infer a missing table declaration.
    if "weight_shape" not in attrs and domain == "tokens" and "index_domain" not in attrs:
        shape = (legacy_vocabulary, output[-1])
    shape = _contract_dimensions(shape)
    if (
        shape is None or len(shape) != 2
        or shape[1] != output[-1]
        or type(attrs.get("weight")) is not str or not attrs["weight"]
    ):
        raise SemanticTensorError("lookup requires a declared domain-sized table")
    return domain, source, (shape[0], shape[1])


def projection_weight_shape(operation: dict, input_width: int) -> tuple[int, int]:
    """Checkpoint orientation for an existing IO or OI projection declaration."""
    attrs = _operation_attributes(operation)
    output = _contract_dimensions(operation.get("output_shape"))
    if output is None or type(input_width) is not int or input_width < 1:
        raise SemanticTensorError("semantic projection has an invalid shape")
    output_width = output[-1]
    layout = attrs.get("weight_layout", "output_input")
    if type(layout) is not str or layout not in ("input_output", "output_input"):
        raise SemanticTensorError("semantic projection weight layout is unsupported")
    shape = (input_width, output_width) if layout == "input_output" else (output_width, input_width)
    if "weight_shape" in attrs and _contract_dimensions(attrs["weight_shape"]) != shape:
        raise SemanticTensorError("semantic projection declared shape differs from its layout")
    if "bias_shape" in attrs and _contract_dimensions(attrs["bias_shape"]) != (output_width,):
        raise SemanticTensorError("semantic projection bias shape is invalid")
    return shape


def required_client_tensors(plan: ModelPlan) -> dict[str, tuple[int, ...]]:
    """Reject ambiguous or unbound local weights before any session material is issued."""
    required: dict[str, tuple[int, ...]] = {}
    def bind(weight: object, shape: tuple[int, ...]) -> None:
        if not isinstance(weight, str) or not weight or any(
            type(width) is not int or width < 1 for width in shape
        ):
            raise SemanticTensorError("local tensor has an invalid weight or shape")
        if weight in required and required[weight] != shape:
            raise SemanticTensorError("local tensor shape differs between semantic uses")
        required[weight] = shape

    document = plan.to_dict()
    for phase in ("prefill", "decode"):
        heads = [op for op in document[phase]["operations"] if op["operator"] == "output_head"]
        legacy_vocabulary = heads[0]["output_shape"][-1] if len(heads) == 1 else None
        for operation in document[phase]["operations"]:
            attrs = operation["attributes"]
            kind = operation["operator"]
            shape: tuple[int, ...]
            if kind == "token_lookup":
                domain, _, table_shape = lookup_contract(operation, legacy_vocabulary=legacy_vocabulary)
                if domain == "segments":
                    raise SemanticTensorError("semantic runtime does not supply input.segments")
                if domain != "tokens":
                    bind(attrs.get("weight"), table_shape)
                continue
            elif kind == "layer_norm":
                shape = (operation["output_shape"][-1],)
                try:
                    layer_norm_contract(attrs, shape[0])
                except SemanticNumericError as exc:
                    raise SemanticTensorError(str(exc)) from exc
                bind(attrs.get("weight"), shape)
                if attrs.get("bias") is not None:
                    bind(attrs["bias"], shape)
                continue
            elif kind == "rms_norm":
                weight = attrs.get("weight")
                if weight is None:
                    if attrs.get("with_scale") is not False or attrs.get("output_dtype") != "bfloat16":
                        raise SemanticTensorError("unweighted normalization needs an explicit contract")
                    continue
                shape = (operation["output_shape"][-1],)
            elif kind == "rms_norm_gated":
                weight = attrs.get("weight")
                if (
                    attrs.get("activation") != "silu" or attrs.get("weight_offset") != 0
                    or attrs.get("norm_before_gate") is not True
                ):
                    raise SemanticTensorError("gated normalization tensor contract is unsupported")
                shape = (operation["output_shape"][-1],)
            elif kind == "gated_delta_decay":
                if (
                    attrs.get("formula") != "-exp(A_log)*softplus(a+dt_bias)"
                    or attrs.get("compute_dtype") != "float32"
                ):
                    raise SemanticTensorError("gated-delta coefficient contract is unsupported")
                shape = (operation["output_shape"][-1],)
                for name in ("a_log", "dt_bias"):
                    bind(attrs.get(name), shape)
                continue
            elif kind == "scale" and isinstance(attrs.get("factor"), dict) and attrs["factor"].get("kind") == "checkpoint_scalar":
                factor = attrs["factor"]
                weight = factor.get("weight")
                if factor.get("weight_shape") != [1]:
                    raise SemanticTensorError("checkpoint scalar must have a declared unit shape")
                shape = (1,)
            elif kind in {"causal_convolution", "convolution_state_update"}:
                width, kernel = attrs.get("groups"), attrs.get("kernel_size")
                weight = attrs.get("weight")
                if (
                    type(width) is not int
                    or not 1 <= width <= 8192
                    or type(kernel) is not int
                    or not 1 <= kernel <= 16
                    or attrs.get("bias") is not None
                    or not isinstance(weight, str)
                    or not weight
                ):
                    raise SemanticTensorError("local convolution has an invalid tensor declaration")
                shape = (width, 1, kernel)
            else:
                continue
            bind(weight, shape)
    return required


def required_checkpoint_tensors(plan: ModelPlan) -> dict[str, tuple[int, ...]]:
    """Shape commitments for every local and remote artifact in both phases."""
    required = required_client_tensors(plan)
    document = plan.to_dict()
    for phase in ("prefill", "decode"):
        operations = {row["id"]: row for row in document[phase]["operations"]}
        heads = [row for row in operations.values() if row["operator"] == "output_head"]
        if len(heads) != 1:
            raise SemanticTensorError("semantic checkpoint needs one output head per phase")
        vocabulary = heads[0]["output_shape"][-1]
        if type(vocabulary) is not int or vocabulary < 1:
            raise SemanticTensorError("semantic vocabulary has an invalid width")
        for operation in operations.values():
            kind = operation["operator"]
            if kind not in {"token_lookup", "linear", "output_head"}:
                continue
            output_width = operation["output_shape"][-1]
            if kind == "token_lookup":
                _, _, weight_shape = lookup_contract(operation, legacy_vocabulary=vocabulary)
            else:
                sources = operation["inputs"]
                if len(sources) != 1 or sources[0] not in operations:
                    raise SemanticTensorError("projection has no single declared input")
                input_width = operations[sources[0]]["output_shape"][-1]
                weight_shape = projection_weight_shape(operation, input_width)
            if not isinstance(operation["attributes"].get("weight"), str) or not operation["attributes"]["weight"]:
                raise SemanticTensorError("semantic artifact has no weight identity")
            for key, shape in (
                (operation["attributes"].get("weight"), weight_shape),
                (operation["attributes"].get("bias"), (output_width,)),
            ):
                if key is None:
                    continue
                if not isinstance(key, str) or not key:
                    raise SemanticTensorError("semantic artifact has no weight identity")
                if key in required and required[key] != shape:
                    raise SemanticTensorError(
                        f"semantic artifact {key!r} shape differs: {required[key]} != {shape}"
                    )
                required[key] = shape
    return required


def preflight_semantic_checkpoint(plan: ModelPlan, store: SafeTensorStore) -> int:
    """Reject missing/mismatched source tensors without materializing their bytes."""
    required = required_checkpoint_tensors(plan)
    for key, shape in required.items():
        try:
            actual_shape = store.tensor_shape(key)
            actual_dtype = store.tensor_dtype(key)
        except TensorStoreError as exc:
            raise SemanticTensorError(f"semantic checkpoint is missing {key!r}") from exc
        if actual_shape != shape or actual_dtype not in {"F32", "F16", "BF16"}:
            raise SemanticTensorError(f"semantic checkpoint artifact {key!r} has an invalid shape or dtype")
    return len(required)


__all__ = [
    "SemanticTensorError",
    "lookup_contract",
    "projection_weight_shape",
    "preflight_semantic_checkpoint",
    "required_checkpoint_tensors",
    "required_client_tensors",
]
