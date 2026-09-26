"""Client-owned tensor requirements derived from semantic operator contracts."""

from __future__ import annotations

from pllm.modeling import ModelPlan

from .safetensors_store import SafeTensorStore, TensorStoreError


class SemanticTensorError(ValueError):
    pass


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
        for operation in document[phase]["operations"]:
            attrs = operation["attributes"]
            kind = operation["operator"]
            shape: tuple[int, ...]
            if kind == "rms_norm":
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
                if operation["inputs"] != ["input.tokens"]:
                    raise SemanticTensorError("token table has an unsupported source")
                input_width = vocabulary
            else:
                sources = operation["inputs"]
                if len(sources) != 1 or sources[0] not in operations:
                    raise SemanticTensorError("projection has no single declared input")
                input_width = operations[sources[0]]["output_shape"][-1]
            if any(type(width) is not int or width < 1 for width in (output_width, input_width)):
                raise SemanticTensorError("semantic projection has an invalid shape")
            weight_shape = (
                (input_width, output_width)
                if kind == "token_lookup"
                else (output_width, input_width)
            )
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
    "preflight_semantic_checkpoint",
    "required_checkpoint_tensors",
    "required_client_tensors",
]
