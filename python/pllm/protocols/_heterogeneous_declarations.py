"""Model- and method-neutral heterogeneous protocol configuration, not executors.

Logical slot names are references only, never authenticated keys, birth proofs,
or input-ownership capabilities. Constructors perform no reference lookup;
later admission must bind compatibility. These declarations adopt no upstream
executable dependency, numerical adaptation, security claim, or 53-bit adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _slot(name: str, value: object) -> None:
    if type(value) is not str or not value.strip() or "__" in value:
        raise ConfigurationError(f"{name} must be a nonblank logical slot reference without '__'")


class AuthenticatedClientInputExport(ComponentRef):
    """Declare client-input export references, not authentication or input export.

    This is distinct from source authentication and server-confidential weight
    handling. Role, transport, and birth names grant no input-ownership authority.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-client-input-export/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "roles": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
                "transport": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
                "birth": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["roles", "transport", "birth"], "additionalProperties": False,
        },
    )

    def __init__(self, *, roles: str, transport: str, birth: str) -> None:
        _slot("roles", roles)
        _slot("transport", transport)
        _slot("birth", birth)
        super().__init__(self.descriptor.component, {"roles": roles, "transport": transport, "birth": birth})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEEncryptedMultiplyReduce(ComponentRef):
    """Declare an encrypted multiply/reduce operation, not its execution.

    Encrypted HomComp and private accumulator semantics require independent
    binding and validation; neither follows from naming a numeric slot.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-encrypted-multiply-reduce/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "operation": {"type": "string", "minLength": 1, "enum": [
                    "query-key-dot-product", "probability-value-reduction",
                    "normalization-square-reduction", "normalization-elementwise-product",
                    "softmax-square-and-row-sum", "softmax-row-normalization", "activation-gated-product",
                ]},
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["operation", "numeric"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, operation: Literal[
            "query-key-dot-product", "probability-value-reduction",
            "normalization-square-reduction", "normalization-elementwise-product",
            "softmax-square-and-row-sum", "softmax-row-normalization", "activation-gated-product",
        ], numeric: str,
    ) -> None:
        if type(operation) is not str or operation not in (
            "query-key-dot-product", "probability-value-reduction",
            "normalization-square-reduction", "normalization-elementwise-product",
            "softmax-square-and-row-sum", "softmax-row-normalization", "activation-gated-product",
        ):
            raise ConfigurationError("operation must be a supported encrypted multiply/reduce operation")
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"operation": operation, "numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HESlotCoefficientTransform(ComponentRef):
    """Declare slot-to-coefficient transform configuration, not encrypted execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-slot-coefficient-transform/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str) -> None:
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HERingSwitch(ComponentRef):
    """Declare ring-switch references without resolving keys or executing a switch."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-ring-switch/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "key_graph": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["key_graph", "numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, key_graph: str, numeric: str) -> None:
        _slot("key_graph", key_graph)
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"key_graph": key_graph, "numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HECoefficientExtraction(ComponentRef):
    """Declare coefficient extraction configuration, not encrypted HomComp.

    Extraction does not declare private accumulator computation, decryption,
    or output release. Those semantics need separate binding and validation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-coefficient-extraction/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str) -> None:
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEKeySwitch(ComponentRef):
    """Declare key-switch references, not authenticated keys or switching execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-key-switch/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "key_graph": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["key_graph", "numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, key_graph: str, numeric: str) -> None:
        _slot("key_graph", key_graph)
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"key_graph": key_graph, "numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEModulusRaise(ComponentRef):
    """Declare modulus-raising configuration without executing numerical adaptation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-modulus-raise/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str) -> None:
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HECoefficientSlotTransform(ComponentRef):
    """Declare coefficient-to-slot transform configuration, not encrypted execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-coefficient-slot-transform/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": "\\S", "not": {"pattern": "__"}},
            }, "required": ["numeric"], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str) -> None:
        _slot("numeric", numeric)
        super().__init__(self.descriptor.component, {"numeric": numeric})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "AuthenticatedClientInputExport", "HEEncryptedMultiplyReduce", "HESlotCoefficientTransform",
    "HERingSwitch", "HECoefficientExtraction", "HEKeySwitch", "HEModulusRaise",
    "HECoefficientSlotTransform",
]
