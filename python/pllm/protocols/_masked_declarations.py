"""Masked-ring configuration declarations, not executors or security guarantees.

Source numeric and ownership contracts remain independently checked compiler and
runtime requirements. These identities do not alias PUMA or FSS contracts, choose
backend tapes, issue authority, read material, or change xhat PRG/AES behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class RingPublicCoefficientDot(ComponentRef):
    """Declare ring dot product with public coefficients, not execution.

    Source ring, accumulation, and rescaling semantics remain distinct from
    PUMA/FSS numeric contracts. Capacity is a requirement, not admission.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/ring-public-coefficient-dot/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "capacity": {"type": "string", "enum": ["full-vocabulary"]},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, capacity: Literal["full-vocabulary"] | None = None) -> None:
        params: dict[str, object] = {}
        if capacity is not None:
            if type(capacity) is not str or capacity != "full-vocabulary":
                raise ConfigurationError("capacity must be 'full-vocabulary'")
            params["capacity"] = capacity
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if "capacity" in params and params["capacity"] is None:
            raise ConfigurationError("capacity must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class NegativeMaskMatrixProduct(ComponentRef):
    """Declare negative-mask matrix product requirements without executing them.

    Source signs, ring arithmetic, and matrix boundaries require runtime binding;
    this is not a PUMA/FSS multiplication or an implicit rescale.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/negative-mask-matrix-product/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("NegativeMaskMatrixProduct accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class NegativeMaskElementwiseProduct(ComponentRef):
    """Declare negative-mask elementwise product requirements, not execution.

    Source ring, sign, shape, and scale contracts remain independently checked;
    no gated-multiply alias or implicit rescale is introduced.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/negative-mask-elementwise-product/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("NegativeMaskElementwiseProduct accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class MaskedRingAddSubProduct(ComponentRef):
    """Declare masked-ring add/sub product requirements, not an executor.

    Source arithmetic and boundary contracts stay distinct from PUMA/FSS;
    declaring this component does not supply masks or perform rescaling.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/masked-ring-add-sub-product/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("MaskedRingAddSubProduct accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class BoundedDpfPublicTable(ComponentRef):
    """Declare bounded DPF public-table requirements without lookup execution.

    Source domain bounds, index semantics, and DPF material ownership remain
    runtime obligations; this does not alias FSS public-table lookup.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/bounded-dpf-public-table/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("BoundedDpfPublicTable accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class XorPredicateFiveWordSelect(ComponentRef):
    """Declare XOR-predicate five-word selection, not an executable selector.

    Source predicate encoding, word layout, and selection semantics require
    independent binding; this is not a generic gated-multiply declaration.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/xor-predicate-five-word-select/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("XorPredicateFiveWordSelect accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class CausalMaxNexpReciprocalAttention(ComponentRef):
    """Declare causal max/nexp/reciprocal attention requirements, not execution.

    Source causality, maximum, nexp, reciprocal, and numeric boundaries remain
    explicit runtime contracts; no whole-decoder or PUMA/FSS alias is created.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/causal-max-nexp-reciprocal-attention/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("CausalMaxNexpReciprocalAttention accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class StablePrivateArgmaxEmbedding(ComponentRef):
    """Declare stable private argmax-to-embedding requirements, not execution.

    Source tie stability, comparison, index, and embedding boundaries require
    independent checking; no FSS/PUMA argmax identity or privacy is implied.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/stable-private-argmax-embedding/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("StablePrivateArgmaxEmbedding accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class AuthenticatedNativeInput(ComponentRef):
    """Declare native-input authentication and ownership requirements only.

    This configuration grants no authenticated-sender capability. A local xhat
    encoder supplies no authority; source authentication and ownership must be
    checked independently. No credentials, tapes, or material are supplied.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-native-input/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("AuthenticatedNativeInput accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ConsumingShareToCommonMask(ComponentRef):
    """Declare consuming share-to-common-mask handoff ownership requirements.

    Consumption and ownership are runtime obligations, not effects of this
    constructor. No authenticated-sender capability is granted; a local xhat
    encoder has no source authority. PRG/AES and material lifecycles are unchanged.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/consuming-share-to-common-mask/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("ConsumingShareToCommonMask accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "RingPublicCoefficientDot", "NegativeMaskMatrixProduct",
    "NegativeMaskElementwiseProduct", "MaskedRingAddSubProduct",
    "BoundedDpfPublicTable", "XorPredicateFiveWordSelect",
    "CausalMaxNexpReciprocalAttention", "StablePrivateArgmaxEmbedding",
    "AuthenticatedNativeInput", "ConsumingShareToCommonMask",
]
