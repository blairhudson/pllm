"""Source-only heterogeneous nonlinear declarations, not encrypted executors.

References are logical labels, not resolved capabilities or provisioned keys.
Construction performs no graph lookup, provisioning, or authority issuance.
The compiler must later bind the full conversion/arithmetic graph, numeric
contract, trust, keys, and state. The numeric 53-bit port remains unresolved;
these declarations qualify neither numerical behavior nor security/admission.
Encrypted HomComp requires its encrypted graph: scalar source-only execution
is not a bypass. Conversion labels and arithmetic dependencies stay distinct.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _reference(value: str, name: str) -> str:
    if type(value) is not str or not value.strip() or "__" in value:
        raise ConfigurationError(f"{name} must be a non-empty logical reference without '__'")
    return value


class LWEHighPrecisionComparison(ComponentRef):
    """Declare encrypted high-precision comparison bound to keys and numeric rules.

    No comparison, key provisioning, or scalar-source fallback is performed.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/lwe-high-precision-comparison/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "key_graph": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": ["key_graph", "numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, key_graph: str, numeric: str) -> None:
        super().__init__(self.descriptor.component, {
            "key_graph": _reference(key_graph, "key_graph"),
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class RLWEEncryptedAccumulatorSegmentedAffine(ComponentRef):
    """Declare encrypted-accumulator segmented affine with explicit comparison.

    Comparison, key graph, and numeric references require later binding; this
    declaration supplies no coefficients, accumulator, keys, or execution.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/rlwe-encrypted-accumulator-segmented-affine/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "comparison": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "key_graph": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": ["comparison", "key_graph", "numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, comparison: str, key_graph: str, numeric: str) -> None:
        super().__init__(self.descriptor.component, {
            "comparison": _reference(comparison, "comparison"),
            "key_graph": _reference(key_graph, "key_graph"),
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEEvalMod(ComponentRef):
    """Declare the numeric-bound HE EvalMod stage without evaluating ciphertexts.

    This stage is not an entire conversion bridge or a numeric qualification.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-evalmod/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": ["numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str) -> None:
        super().__init__(self.descriptor.component, {
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HENormalization(ComponentRef):
    """Declare HE normalization with separate arithmetic and conversion stages.

    Square reduction, inverse square root, and multiplication are distinct
    dependencies, not a conversion bundle. Every stage needs compiler binding
    to the full graph and its numeric, trust, key, and state contracts.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-normalization/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "square_reduce": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "inverse_sqrt": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "multiply": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "to_coefficients": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "ring_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "extract": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "compare": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evaluate": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "key_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "modulus_raise": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "to_slots": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evalmod": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": [
                "square_reduce", "inverse_sqrt", "multiply", "to_coefficients",
                "ring_switch", "extract", "compare", "evaluate", "key_switch",
                "modulus_raise", "to_slots", "evalmod", "numeric",
            ], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, square_reduce: str, inverse_sqrt: str, multiply: str,
        to_coefficients: str, ring_switch: str, extract: str, compare: str,
        evaluate: str, key_switch: str, modulus_raise: str, to_slots: str,
        evalmod: str, numeric: str,
    ) -> None:
        super().__init__(self.descriptor.component, {
            "square_reduce": _reference(square_reduce, "square_reduce"),
            "inverse_sqrt": _reference(inverse_sqrt, "inverse_sqrt"),
            "multiply": _reference(multiply, "multiply"),
            "to_coefficients": _reference(to_coefficients, "to_coefficients"),
            "ring_switch": _reference(ring_switch, "ring_switch"),
            "extract": _reference(extract, "extract"),
            "compare": _reference(compare, "compare"),
            "evaluate": _reference(evaluate, "evaluate"),
            "key_switch": _reference(key_switch, "key_switch"),
            "modulus_raise": _reference(modulus_raise, "modulus_raise"),
            "to_slots": _reference(to_slots, "to_slots"),
            "evalmod": _reference(evalmod, "evalmod"),
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HESoftmaxSquareNormalize(ComponentRef):
    """Declare heterogeneous HE softmax square-normalize dependencies.

    This is not THOR's pllm/he-square-normalize-softmax/v1 contract and not
    an alias for bounded packed encrypted softmax. Square reduction,
    reciprocal, and multiplication remain separate from conversion stages.
    No encrypted arithmetic or graph/numeric qualification occurs here.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-softmax-square-normalize/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "square_reduce": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "reciprocal": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "multiply": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "to_coefficients": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "ring_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "extract": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "compare": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evaluate": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "key_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "modulus_raise": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "to_slots": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evalmod": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": [
                "square_reduce", "reciprocal", "multiply", "to_coefficients",
                "ring_switch", "extract", "compare", "evaluate", "key_switch",
                "modulus_raise", "to_slots", "evalmod", "numeric",
            ], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, square_reduce: str, reciprocal: str, multiply: str,
        to_coefficients: str, ring_switch: str, extract: str, compare: str,
        evaluate: str, key_switch: str, modulus_raise: str, to_slots: str,
        evalmod: str, numeric: str,
    ) -> None:
        super().__init__(self.descriptor.component, {
            "square_reduce": _reference(square_reduce, "square_reduce"),
            "reciprocal": _reference(reciprocal, "reciprocal"),
            "multiply": _reference(multiply, "multiply"),
            "to_coefficients": _reference(to_coefficients, "to_coefficients"),
            "ring_switch": _reference(ring_switch, "ring_switch"),
            "extract": _reference(extract, "extract"),
            "compare": _reference(compare, "compare"),
            "evaluate": _reference(evaluate, "evaluate"),
            "key_switch": _reference(key_switch, "key_switch"),
            "modulus_raise": _reference(modulus_raise, "modulus_raise"),
            "to_slots": _reference(to_slots, "to_slots"),
            "evalmod": _reference(evalmod, "evalmod"),
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEActivationBridge(ComponentRef):
    """Declare every conversion and encrypted activation stage explicitly.

    Labels alone supply no encrypted HomComp implementation or authority.
    Compiler binding must preserve stage distinctions and bind full numeric,
    trust, key, and state contracts; no scalar-source-only bypass is provided.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-activation-bridge/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "to_coefficients": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "ring_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "extract": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "compare": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evaluate": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "key_switch": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "modulus_raise": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "to_slots": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "evalmod": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}},
            }, "required": [
                "to_coefficients", "ring_switch", "extract", "compare", "evaluate",
                "key_switch", "modulus_raise", "to_slots", "evalmod", "numeric",
            ], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, to_coefficients: str, ring_switch: str, extract: str,
        compare: str, evaluate: str, key_switch: str, modulus_raise: str,
        to_slots: str, evalmod: str, numeric: str,
    ) -> None:
        super().__init__(self.descriptor.component, {
            "to_coefficients": _reference(to_coefficients, "to_coefficients"),
            "ring_switch": _reference(ring_switch, "ring_switch"),
            "extract": _reference(extract, "extract"),
            "compare": _reference(compare, "compare"),
            "evaluate": _reference(evaluate, "evaluate"),
            "key_switch": _reference(key_switch, "key_switch"),
            "modulus_raise": _reference(modulus_raise, "modulus_raise"),
            "to_slots": _reference(to_slots, "to_slots"),
            "evalmod": _reference(evalmod, "evalmod"),
            "numeric": _reference(numeric, "numeric"),
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "LWEHighPrecisionComparison",
    "RLWEEncryptedAccumulatorSegmentedAffine",
    "HEEvalMod",
    "HENormalization",
    "HESoftmaxSquareNormalize",
    "HEActivationBridge",
]
