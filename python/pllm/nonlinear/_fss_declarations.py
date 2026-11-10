"""Source-fixed ring50/Q12 nonlinear configuration, not execution authority.

These closed declarations preserve existing wire identities and source equations.
They do not qualify numerical accuracy, security, input ranges, or admission.
Compiler/runtime source and table bindings remain required; constructing a
declaration performs no source I/O, native activation, plan construction,
material issuance, or tape creation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class FSSSignedFloorRescale(ComponentRef):
    """Declare signed floor rescaling in the source-fixed ring50/Q12 profile.

    Rescaling is floor(X / 2**shift), not truncation toward zero; Q24 products
    return to Q12 with shift 12. The operation's shift and no-wrap premises
    require existing compiler/runtime binding, not declaration parameters.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-signed-trunc-floor/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if params:
            raise ConfigurationError("FSSSignedFloorRescale accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSSigned38InputSiLU(ComponentRef):
    """Declare source-table SiLU with a signed38 input corridor only.

    The source ring50/Q12 input X must satisfy -2**37 <= X < 2**37.
    With y = floor(X / 64), output is (0 if y < 0 else X) minus
    T[min(abs(y), 1023)], using the bound source correction table T.
    Signed38 describes this input corridor, not the whole graph's identity
    or its intermediate/output widths. This declaration proves no membership.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-silu/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if params:
            raise ConfigurationError("FSSSigned38InputSiLU accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSSumSquaresTableRMSNorm(ComponentRef):
    """Declare source RMSNorm's unaveraged sum of squares and table inverse.

    In ring50/Q12, S = 1 + sum(X_i**2) retains Q24 squares and adds integer
    code one, not a substituted checkpoint epsilon or a pre-averaged energy.
    For e = floor(log2(S)), m = floor(128*S / 2**e) - 128, the row-length-k
    source table at 64*m + e supplies the inverse R. Its payload is
    isqrt((2**55*k) // ((128+m)*2**e)). Each output is
    floor(floor(X_i*R / 4096)*gamma_i / 4096) + beta_i: normalization and
    affine products have separate floors. Source/table, affine, and no-wrap
    bindings remain required; this declaration certifies no input energy.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-rmsnorm/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if params:
            raise ConfigurationError("FSSSumSquaresTableRMSNorm accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSProductFloorRotary(ComponentRef):
    """Declare source ring50/Q12 rotary with product floors before addition.

    Half-split pair (l, r) becomes
    (floor(l*c/4096) + floor((-r)*s/4096),
    floor(r*c/4096) + floor(l*s/4096)). The sign belongs inside the product;
    neither flooring the combined sum nor negating a floored positive term
    preserves this equation. Source positions, coefficients, layout, and
    no-wrap premises require compiler/runtime binding.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-rotary/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if params:
            raise ConfigurationError("FSSProductFloorRotary accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSSourceTableSoftmax(ComponentRef):
    """Declare source-table softmax, not arbitrary floating-point softmax.

    The ring50/Q12 source equation takes the maximum over public-mask active
    scores only. For active i, C_i = min(maximum-X_i, 65535), and
    E_i = floor(T0[C_i % 256]*T1[C_i // 256] / 4096).
    With D = floor(sum(E_i) / 64), the active-count-k source reciprocal
    table supplies R = Treciprocal_k[D]; output is floor(E_i*R / 4096).
    Inactive entries are zero and excluded from max, exponentiation, and sum;
    outputs are not renormalized. Compiler/runtime binding must establish
    source/table revision, score scale, row layout, public causal mask, active
    count, and existing range/no-wrap requirements. This declaration supplies
    none of those proofs and admits no new ranges.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-softmax/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if params:
            raise ConfigurationError("FSSSourceTableSoftmax accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "FSSSignedFloorRescale",
    "FSSSigned38InputSiLU",
    "FSSSumSquaresTableRMSNorm",
    "FSSProductFloorRotary",
    "FSSSourceTableSoftmax",
]
