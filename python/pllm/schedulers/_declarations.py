"""Typed online scheduling requirements, not executable schedulers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class FSSOnlineSchedule(ComponentRef):
    """Declare FSS online scheduling without allocating or consuming material."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-online-schedule/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protected-scheduler", category_version="1", lifecycle_phase="compile",
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


__all__ = ["FSSOnlineSchedule"]


class BoundedConsumingProtectedDag(ComponentRef):
    """Declare bounded consuming DAG requirements, not paid-work or cancellation proof."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/bounded-consuming-protected-dag/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protected-scheduler", category_version="1", lifecycle_phase="compile",
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


__all__ += ["BoundedConsumingProtectedDag"]


class HELevelErrorSchedule(ComponentRef):
    """Declare level/error scheduling links, not ciphertext lifetime guarantees."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-level-error-schedule/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protected-scheduler", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "numeric": {"type": "string", "minLength": 1, "pattern": r"\S",
                            "not": {"pattern": "__"}},
                "key_graph": {"type": "string", "minLength": 1, "pattern": r"\S",
                              "not": {"pattern": "__"}},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, numeric: str | None = None, key_graph: str | None = None) -> None:
        params: dict[str, object] = {}
        for name, value in (("numeric", numeric), ("key_graph", key_graph)):
            if value is not None:
                if type(value) is not str or not value.strip() or "__" in value:
                    raise ConfigurationError(f"{name} must be a nonblank slot string without '__'")
                params[name] = value
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        for name in ("numeric", "key_graph"):
            if name in params and params[name] is None:
                raise ConfigurationError(f"{name} must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HELevelSchemePlacement(ComponentRef):
    """Declare level/scheme placement links without executing a schedule."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-level-scheme-placement/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protected-scheduler", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "schedule": {"type": "string", "minLength": 1, "pattern": r"\S",
                             "not": {"pattern": "__"}},
                "roles": {"type": "string", "minLength": 1, "pattern": r"\S",
                          "not": {"pattern": "__"}},
            }, "required": ["schedule", "roles"], "additionalProperties": False,
        },
    )

    def __init__(self, *, schedule: str, roles: str) -> None:
        for name, value in (("schedule", schedule), ("roles", roles)):
            if type(value) is not str or not value.strip() or "__" in value:
                raise ConfigurationError(f"{name} must be a nonblank slot string without '__'")
        super().__init__(self.descriptor.component, {"schedule": schedule, "roles": roles})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SetupOnlineRetirementSchedule(ComponentRef):
    """Declare setup/online/retirement requirements without proving retirement."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/setup-online-retirement-schedule/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protected-scheduler", category_version="1", lifecycle_phase="compile",
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


__all__ += ["HELevelErrorSchedule", "HELevelSchemePlacement", "SetupOnlineRetirementSchedule"]
