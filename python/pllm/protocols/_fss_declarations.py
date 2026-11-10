"""FSS configuration declarations, not executors or security guarantees.

Compiler/runtime binding must independently satisfy FSS requirements and Source
edge contracts. The current numeric contract uses a fixed 50-bit ring and Q12;
these declarations neither select an implementation nor validate execution,
admission, key material, role separation, or transport security.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class FSSPublicTableLookup(ComponentRef):
    """Declare FSS private lookup into a public table.

    Source table, index, and boundary semantics remain runtime requirements;
    declaring this component does not perform lookup or guarantee privacy.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-private-public-table-lookup/v1", provider="pllm",
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
            raise ConfigurationError("FSSPublicTableLookup accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSPublicDot(ComponentRef):
    """Declare FSS dot product with a public operand, not execution.

    Source numeric requirements accumulate Q24 products, floor once to Q12,
    then add bias in the fixed 50-bit ring. Optional full-vocabulary capacity
    is configuration only, not an admission or execution guarantee.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-public-dot/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
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


class FSSPrivateDot(ComponentRef):
    """Declare FSS private dot product without executing it.

    Source numeric requirements accumulate Q24 products, floor once to Q12
    before bias, and use the fixed 50-bit ring. FSS privacy and edge contracts
    require independent compiler/runtime binding.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-private-dot/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("FSSPrivateDot accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSSecretMultiply(ComponentRef):
    """Declare FSS secret-gated multiplication, not an executor.

    The current Source contract produces Q24 products in the fixed 50-bit
    ring; rescaling to Q12 is separate, not implicit in this declaration.
    FSS and operand edge requirements remain independently checked.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-secret-gated-multiply/v1", provider="pllm",
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
            raise ConfigurationError("FSSSecretMultiply accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSAdd(ComponentRef):
    """Declare FSS addition under Source fixed-50-bit-ring/Q12 requirements.

    Scale alignment and Source edge contracts remain compiler/runtime duties;
    this configuration neither executes addition nor guarantees security.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-add/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("FSSAdd accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSArgmax(ComponentRef):
    """Declare FSS argmax using Source comparison, tie, and index semantics.

    Fixed-50-bit-ring/Q12 inputs and Source edge contracts require independent
    binding; this declaration neither computes argmax nor guarantees privacy.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-argmax/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("FSSArgmax accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FSSAuthenticatedRoleTransport(ComponentRef):
    """Declare the distinct Source FSS authenticated-role-transport identity.

    Source role, authentication, and channel requirements remain runtime
    obligations. This is not the shared authenticated-confidential transport
    alias and creates no channels, credentials, or security guarantees.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-authenticated-role-transport/v1", provider="pllm",
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
            raise ConfigurationError("FSSAuthenticatedRoleTransport accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "FSSPublicTableLookup", "FSSPublicDot", "FSSPrivateDot", "FSSSecretMultiply",
    "FSSAdd", "FSSArgmax", "FSSAuthenticatedRoleTransport",
]
