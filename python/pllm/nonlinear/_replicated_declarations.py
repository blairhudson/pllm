"""Replicated Q18 configuration declarations, not executable implementations.

Source ordering and numerical contracts below are binding requirements, not
claims of independent qualification. Construction does not evaluate source
functions, generate randomness, introduce coefficients, or activate a runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef


class ReplicatedTruncAprQuarterGap(ComponentRef):
    """Require source TruncApr quarter-gap semantics, not SharedRescale/SigmaGapARS.

    Preserve the full Q36 sum before fresh stochastic truncation to Q18.
    This declaration neither samples randomness nor performs truncation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-truncapr-quarter-gap/v1",
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


class Q18PiecewiseSiluSourceOrder(ComponentRef):
    """Require Q18 piecewise SiLU with source coefficients and operation order.

    This declaration supplies no coefficients and performs no evaluation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-piecewise-silu-source-order/v1",
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


class Q18SourceReciprocal(ComponentRef):
    """Require the source Q18 reciprocal and its source operation ordering.

    This declaration supplies no coefficients and performs no evaluation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-source-reciprocal/v1",
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


class Q18SourceRsqrt(ComponentRef):
    """Require the source Q18 reciprocal square root and operation ordering.

    This declaration supplies no coefficients and performs no evaluation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-source-rsqrt/v1",
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


class Q18RmsNormEpsilon1(ComponentRef):
    """Require source-ordered Q18 RMSNorm with integer epsilon 1, not float epsilon.

    This declaration performs no normalization or reciprocal-square-root work.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-rmsnorm-epsilon1/v1",
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


class Q18SourceRotary(ComponentRef):
    """Require source Q18 rotary semantics and source operation ordering.

    This declaration supplies no coefficients and performs no rotation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-source-rotary/v1",
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


class Q18ShiftedClipExp5Reciprocal(ComponentRef):
    """Require source-ordered shifted Q18 softmax: strict -14 clip, five squares.

    Preserve source exponential and reciprocal ordering; this declaration
    supplies no coefficients and performs no softmax evaluation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q18-shifted-clip-exp5-reciprocal/v1",
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
    "ReplicatedTruncAprQuarterGap",
    "Q18PiecewiseSiluSourceOrder",
    "Q18SourceReciprocal",
    "Q18SourceRsqrt",
    "Q18RmsNormEpsilon1",
    "Q18SourceRotary",
    "Q18ShiftedClipExp5Reciprocal",
]
