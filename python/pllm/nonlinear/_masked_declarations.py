"""Source-only masked nonlinear configuration, never execution authority.

These declarations preserve distinct wire contracts without evaluating arithmetic,
loading source or tables, creating plans or tapes, or qualifying implementations.
Null semantic revisions remain unresolved compiler/runtime admission gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ReducedRingTruncation(ComponentRef):
    """Declare reduced-ring truncation, not same-ring GapARS.

    This contract is distinct from replicated Q18 fresh stochastic truncation;
    construction performs no truncation and generates no randomness.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/reduced-ring-truncation/v1",
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


class SameRingGapARS(ComponentRef):
    """Declare ring-preserving GapARS, not reduced-ring truncation.

    This is not an alias for replicated Q18 fresh stochastic truncation.
    Construction performs no shift and generates no randomness.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/same-ring-gapars/v1",
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


class XorMaskedDReLU(ComponentRef):
    """Declare source XOR-masked DReLU without generating masks or evaluating it."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/xor-masked-drelu/v1",
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


class EarlyFloorResidualActivation(ComponentRef):
    """Declare source early-floor residual activation, preserving operation order.

    This declaration supplies no coefficients and performs no activation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/early-floor-residual-activation/v1",
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


class IntegerRsqrtTableNorm(ComponentRef):
    """Declare source-kind, source-epsilon integer reciprocal-square-root tables.

    All fields are required, including the unresolved null semantic revision.
    Null does not bind a table revision or authorize execution. Ring50/Q12 edge
    metadata elsewhere does not establish arithmetic equivalence to other norms.
    This declaration loads no tables and performs no normalization.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/integer-rsqrt-table-norm/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["source"]},
                "epsilon": {"type": "string", "enum": ["source"]},
                "semantic_revision": {"type": "null"},
            },
            "required": ["kind", "epsilon", "semantic_revision"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self, *, kind: Literal["source"], epsilon: Literal["source"],
        semantic_revision: None,
    ) -> None:
        if type(kind) is not str or kind != "source":
            raise ConfigurationError("IntegerRsqrtTableNorm.kind must be 'source'")
        if type(epsilon) is not str or epsilon != "source":
            raise ConfigurationError("IntegerRsqrtTableNorm.epsilon must be 'source'")
        if semantic_revision is not None:
            raise ConfigurationError("IntegerRsqrtTableNorm.semantic_revision must be null")
        super().__init__(self.descriptor.component, {
            "kind": kind, "epsilon": epsilon, "semantic_revision": semantic_revision,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SourceRing50Q12Rotary(ComponentRef):
    """Declare source ring50/Q12 rotary with a required unresolved null revision.

    Shared edge metadata does not establish arithmetic equivalence to another
    rotary contract. Null supplies no source binding or execution authority;
    compiler/runtime admission remains unresolved. No coefficients are loaded
    and no rotation is performed.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/source-ring50-q12-rotary/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object",
            "properties": {"semantic_revision": {"type": "null"}},
            "required": ["semantic_revision"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, semantic_revision: None) -> None:
        if semantic_revision is not None:
            raise ConfigurationError("SourceRing50Q12Rotary.semantic_revision must be null")
        super().__init__(self.descriptor.component, {"semantic_revision": semantic_revision})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "ReducedRingTruncation",
    "SameRingGapARS",
    "XorMaskedDReLU",
    "EarlyFloorResidualActivation",
    "IntegerRsqrtTableNorm",
    "SourceRing50Q12Rotary",
]
