"""HE nonlinear configuration declarations, not executable approximations.

Null parameters explicitly leave source calibration or authority unresolved.
Construction does not resolve them, choose coefficients, or activate execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class CalibratedPolynomialExp(ComponentRef):
    """Declare polynomial exponential with unresolved calibration and domain."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/calibrated-polynomial-exp/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "coefficients": {"type": "null"}, "domain": {"type": "null"},
            }, "required": ["coefficients", "domain"], "additionalProperties": False,
        },
    )

    def __init__(self, *, coefficients: None, domain: None) -> None:
        if coefficients is not None or domain is not None:
            raise ConfigurationError("coefficients and domain must be null")
        super().__init__(self.descriptor.component, {"coefficients": coefficients, "domain": domain})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class StandardGoldschmidtReciprocal(ComponentRef):
    """Declare standard Goldschmidt, not an alternative compiler algorithm."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/standard-goldschmidt-reciprocal/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "iterations": {"type": "null"}, "domain": {"type": "null"},
            }, "required": ["iterations", "domain"], "additionalProperties": False,
        },
    )

    def __init__(self, *, iterations: None, domain: None) -> None:
        if iterations is not None or domain is not None:
            raise ConfigurationError("iterations and domain must be null")
        super().__init__(self.descriptor.component, {"iterations": iterations, "domain": domain})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class CompositePolynomialActivation(ComponentRef):
    """Declare composite polynomial activation without invented calibration."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/composite-polynomial-activation/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "coefficients": {"type": "null"}, "domain": {"type": "null"},
            }, "required": ["coefficients", "domain"], "additionalProperties": False,
        },
    )

    def __init__(self, *, coefficients: None, domain: None) -> None:
        if coefficients is not None or domain is not None:
            raise ConfigurationError("coefficients and domain must be null")
        super().__init__(self.descriptor.component, {"coefficients": coefficients, "domain": domain})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class InverseRoot(ComponentRef):
    """Declare the certified-inverse-root identity, not certification evidence."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/certified-inverse-root/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "iterations": {"type": "null"}, "domain": {"type": "null"},
            }, "required": ["iterations", "domain"], "additionalProperties": False,
        },
    )

    def __init__(self, *, iterations: None, domain: None) -> None:
        if iterations is not None or domain is not None:
            raise ConfigurationError("iterations and domain must be null")
        super().__init__(self.descriptor.component, {"iterations": iterations, "domain": domain})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEMaskedSoftmax(ComponentRef):
    """Declare HE masked softmax; not an alias for PackedEncryptedSoftmax."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-masked-softmax/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SourceNormalization(ComponentRef):
    """Declare source normalization without resolving or executing source work."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/source-normalization/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HERepeatedSquaringSoftmax(ComponentRef):
    """Declare eight source squarings with explicitly unresolved authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-repeated-squaring-softmax/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "source_squarings": {"type": "integer", "enum": [8]},
                "parameter_authority": {"type": "null"},
            }, "required": ["source_squarings", "parameter_authority"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, source_squarings: Literal[8], parameter_authority: None) -> None:
        if type(source_squarings) is not int or source_squarings != 8:
            raise ConfigurationError("source_squarings must be integer 8")
        if parameter_authority is not None:
            raise ConfigurationError("parameter_authority must be null")
        super().__init__(self.descriptor.component, {
            "source_squarings": source_squarings, "parameter_authority": parameter_authority,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEPiecewisePolynomialActivation(ComponentRef):
    """Declare HE piecewise polynomial activation without supplying coefficients."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-piecewise-polynomial-activation/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HERowNormalization(ComponentRef):
    """Declare HE row normalization without performing normalization."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-row-normalization/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HESignComposition(ComponentRef):
    """Declare HE sign composition without choosing an approximation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-sign-composition/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HELayerNorm(ComponentRef):
    """Declare HE layer normalization without evaluating it."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-layernorm/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HESquareNormalizeSoftmax(ComponentRef):
    """Declare HE square-normalize softmax without executing its stages."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-square-normalize-softmax/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEPolynomialActivation(ComponentRef):
    """Declare an explicit source activation choice without resolving coefficients."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-polynomial-activation/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "function": {"type": "string", "enum": ["source-gelu", "source-tanh"]},
            }, "required": ["function"], "additionalProperties": False,
        },
    )

    def __init__(self, *, function: Literal["source-gelu", "source-tanh"]) -> None:
        if type(function) is not str or function not in ("source-gelu", "source-tanh"):
            raise ConfigurationError("function must be 'source-gelu' or 'source-tanh'")
        super().__init__(self.descriptor.component, {"function": function})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "CalibratedPolynomialExp", "StandardGoldschmidtReciprocal",
    "CompositePolynomialActivation", "InverseRoot", "HEMaskedSoftmax",
    "SourceNormalization", "HERepeatedSquaringSoftmax",
    "HEPiecewisePolynomialActivation", "HERowNormalization", "HESignComposition",
    "HELayerNorm", "HESquareNormalizeSoftmax", "HEPolynomialActivation",
]
