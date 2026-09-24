"""Client-owned tensor requirements derived from semantic operator contracts."""

from __future__ import annotations

from pllm.modeling import ModelPlan


class SemanticTensorError(ValueError):
    pass


def required_client_tensors(plan: ModelPlan) -> dict[str, tuple[int, ...]]:
    """Reject ambiguous or unbound local weights before any session material is issued."""
    required: dict[str, tuple[int, ...]] = {}
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
            elif kind == "scale" and isinstance(attrs.get("factor"), dict) and attrs["factor"].get("kind") == "checkpoint_scalar":
                factor = attrs["factor"]
                weight = factor.get("weight")
                if factor.get("weight_shape") != [1]:
                    raise SemanticTensorError("checkpoint scalar must have a declared unit shape")
                shape = (1,)
            else:
                continue
            if not isinstance(weight, str) or not weight or any(
                type(width) is not int or width < 1 for width in shape
            ):
                raise SemanticTensorError("local tensor has an invalid weight or shape")
            if weight in required and required[weight] != shape:
                raise SemanticTensorError("local tensor shape differs between semantic uses")
            required[weight] = shape
    return required


__all__ = ["SemanticTensorError", "required_client_tensors"]
