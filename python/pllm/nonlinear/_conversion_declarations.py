"""Source-only conversion nonlinear configuration, not execution authority.

These declarations preserve existing empty wire parameter maps. Paper/artifact
disagreement (Q18, -14, six squarings versus Q16, four squarings) remains
unresolved; no source is reconciled or adopted here. The numeric declaration's
B-N1 remains None elsewhere, and NearestFullU remains unadopted. Construction
performs no numerical or protocol execution and grants no qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef


class SharedRingTruncation(ComponentRef):
    """Declare the existing shared-ring truncation identity, not SharedRescale.

    This declaration selects no rounding rule and performs no truncation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-ring-truncation/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


class SharedComparisonMux(ComponentRef):
    """Declare the existing shared comparison/mux identity without executing it."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-comparison-mux/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


class ShiftedClipRepeatedSquareSoftmax(ComponentRef):
    """Declare the existing shifted/clip/repeated-square softmax identity.

    Scale, clip, and squaring-count disagreements remain unresolved. No
    numerical variant is adopted and no softmax evaluation is performed.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shifted-clip-repeated-square-softmax/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


class SharedReciprocalInverseRootNormalization(ComponentRef):
    """Declare the existing shared reciprocal/inverse-root normalization identity.

    No domain, coefficients, numerical revision, or execution is supplied.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-reciprocal-inverse-root-normalization/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


class PiecewisePolynomialActivation(ComponentRef):
    """Declare the existing piecewise-polynomial activation identity.

    This declaration supplies no pieces, domains, or coefficients and performs
    no activation evaluation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/piecewise-polynomial-activation/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


class SharedRingResidualBias(ComponentRef):
    """Declare the existing shared-ring residual/bias identity without execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-ring-residual-bias/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
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


__all__ = [
    "SharedRingTruncation",
    "SharedComparisonMux",
    "ShiftedClipRepeatedSquareSoftmax",
    "SharedReciprocalInverseRootNormalization",
    "PiecewisePolynomialActivation",
    "SharedRingResidualBias",
]
