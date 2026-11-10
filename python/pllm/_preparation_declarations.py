"""Typed preparation requirements, not material generation or funding authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class TrustedDealerOneUseFSSMaterial(ComponentRef):
    """Declare trusted-dealer one-use FSS material without issuing seeds or tapes."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-one-use-material/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
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


__all__ = ["TrustedDealerOneUseFSSMaterial"]


class OneUseRoleLocalMaterial(ComponentRef):
    """Declare role-local one-use material requirements, not issuance or burn authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/one-use-role-local-material/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
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


__all__ += ["OneUseRoleLocalMaterial"]


class _Omitted:
    __slots__ = ()


_OMITTED = _Omitted()


class HEKeyContext(ComponentRef):
    """Declare key-context requirements without generating or exporting keys."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-key-context/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "prime_chain": {"type": "null"}, "key_parameters": {"type": "null"},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, prime_chain: None | _Omitted = _OMITTED,
        key_parameters: None | _Omitted = _OMITTED,
    ) -> None:
        params: dict[str, object] = {}
        for name, value in (("prime_chain", prime_chain), ("key_parameters", key_parameters)):
            if value is _OMITTED:
                continue
            if value is not None:
                raise ConfigurationError(f"{name} must be null or omitted")
            params[name] = None
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        for name in ("prime_chain", "key_parameters"):
            if name in params and params[name] is not None:
                raise ConfigurationError(f"{name} must be null or omitted")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEDualKeyContext(ComponentRef):
    """Declare dual-key context requirements without establishing key custody."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-dual-key-context/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
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


class HECiphertextBirth(ComponentRef):
    """Declare ciphertext-birth links without encrypting or provisioning material."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-ciphertext-birth/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "key_graph": {"type": "string", "minLength": 1, "pattern": r"\S",
                              "not": {"pattern": "__"}},
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S",
                            "not": {"pattern": "__"}},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, key_graph: str | None = None, numeric: str | None = None) -> None:
        params: dict[str, object] = {}
        for name, value in (("key_graph", key_graph), ("numeric", numeric)):
            if value is not None:
                if type(value) is not str or not value.strip() or "__" in value:
                    raise ConfigurationError(f"{name} must be a nonblank slot string without '__'")
                params[name] = value
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        for name in ("key_graph", "numeric"):
            if name in params and params[name] is None:
                raise ConfigurationError(f"{name} must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEKeyGraph(ComponentRef):
    """Declare key ownership requirements without granting secrets or custody."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-key-graph/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "secret_key_owner": {"type": "string", "enum": ["client"]},
                "server_secret_keys": {"type": "boolean", "enum": [False]},
            }, "required": ["secret_key_owner", "server_secret_keys"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self, *, secret_key_owner: Literal["client"], server_secret_keys: Literal[False],
    ) -> None:
        if type(secret_key_owner) is not str or secret_key_owner != "client":
            raise ConfigurationError("secret_key_owner must be 'client'")
        if type(server_secret_keys) is not bool or server_secret_keys is not False:
            raise ConfigurationError("server_secret_keys must be false")
        super().__init__(self.descriptor.component, {
            "secret_key_owner": secret_key_owner, "server_secret_keys": server_secret_keys,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class OneUsePartyLocalConversion(ComponentRef):
    """Declare party-local conversion requirements without issuing material."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/one-use-party-local-conversion/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/preparation-provider", category_version="1", lifecycle_phase="compile",
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


__all__ += [
    "HEKeyContext", "HEDualKeyContext", "HECiphertextBirth", "HEKeyGraph",
    "OneUsePartyLocalConversion",
]
