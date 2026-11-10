"""Model- and method-neutral role configuration declarations, not executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ClientMaskSingleServerRoles(ComponentRef):
    """Declare client-mask/single-server role configuration, not role execution.

    No credentials, trust guarantees, or runtime ownership are established by
    this model- and method-neutral declaration; compiler/runtime binding is required.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-mask-single-server-roles/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/role-topology",
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
            raise ConfigurationError("ClientMaskSingleServerRoles accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["ClientMaskSingleServerRoles"]


class ClientKeyServerEvaluationRoles(ComponentRef):
    """Declare client-key/server-evaluation roles without establishing key custody."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-key-server-evaluation-roles/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/role-topology", category_version="1", lifecycle_phase="compile",
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


class ClientSecretServerWeightSecretRoles(ComponentRef):
    """Declare client-secret/server-weight-secret roles without owning secrets."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-secret-server-weight-secret-roles/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/role-topology", category_version="1", lifecycle_phase="compile",
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


class ClientKeyServerConfidentialWeightRoles(ComponentRef):
    """Declare key/weight owner requirements without granting custody or authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-key-server-confidential-weight-roles/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/role-topology", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "secret_key_owner": {"type": "string", "enum": ["client"]},
                "weight_owner": {"type": "string", "enum": ["server"]},
            }, "required": ["secret_key_owner", "weight_owner"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, secret_key_owner: Literal["client"], weight_owner: Literal["server"],
    ) -> None:
        if type(secret_key_owner) is not str or secret_key_owner != "client":
            raise ConfigurationError("secret_key_owner must be 'client'")
        if type(weight_owner) is not str or weight_owner != "server":
            raise ConfigurationError("weight_owner must be 'server'")
        super().__init__(self.descriptor.component, {
            "secret_key_owner": secret_key_owner, "weight_owner": weight_owner,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class TwoOwnerPrivateInputPrivateModelRoles(ComponentRef):
    """Declare two-owner private-input/private-model roles, not ownership proofs."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/two-owner-private-input-private-model-roles/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/role-topology", category_version="1", lifecycle_phase="compile",
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
    "ClientKeyServerEvaluationRoles", "ClientSecretServerWeightSecretRoles",
    "ClientKeyServerConfidentialWeightRoles", "TwoOwnerPrivateInputPrivateModelRoles",
]
