"""Model- and method-neutral protocol configuration declarations, not executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class PublicFieldMatvec(ComponentRef):
    """Declare public-field matvec configuration, not an execution implementation.

    Compiler/runtime binding remains required; no model or method is selected.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/public-field-matvec/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("PublicFieldMatvec accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class AuthenticatedConfidentialRoleTransport(ComponentRef):
    """Declare role-transport configuration without creating channels or credentials.

    This model- and method-neutral declaration provides no authentication,
    confidentiality, or execution authority; runtime binding remains required.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-confidential-role-transport/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "roles": {"type": "string", "enum": ["roles"]},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, roles: Literal["roles"] | None = None) -> None:
        params: dict[str, object] = {}
        if roles is not None:
            if type(roles) is not str or roles != "roles":
                raise ConfigurationError("roles must be 'roles'")
            params["roles"] = roles
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if "roles" in params and params["roles"] is None:
            raise ConfigurationError("roles must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["PublicFieldMatvec", "AuthenticatedConfidentialRoleTransport"]
