"""Model- and method-neutral state configuration declarations, not executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class TentativeStateCommit(ComponentRef):
    """Declare tentative-state commit configuration without mutating runtime state.

    This model- and method-neutral declaration implements no commit or rollback;
    compiler/runtime binding remains required.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/tentative-state-commit/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/state-protocol", category_version="1", lifecycle_phase="compile",
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
            raise ConfigurationError("TentativeStateCommit accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["TentativeStateCommit"]


class InterleavedKVTransaction(ComponentRef):
    """Declare interleaved KV transaction requirements without retaining state."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/interleaved-kv-transaction/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/state-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {"cache_retention": {"type": "null"}},
            "required": ["cache_retention"], "additionalProperties": False,
        },
    )

    def __init__(self, *, cache_retention: None) -> None:
        if cache_retention is not None:
            raise ConfigurationError("cache_retention must be null")
        super().__init__(self.descriptor.component, {"cache_retention": cache_retention})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class EncryptedKVTransaction(ComponentRef):
    """Declare encrypted KV transaction links without mutating ciphertext state."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/encrypted-kv-transaction/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/state-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "birth": {"type": "string", "minLength": 1, "pattern": r"\S",
                          "not": {"pattern": "__"}},
                "schedule": {"type": "string", "minLength": 1, "pattern": r"\S",
                             "not": {"pattern": "__"}},
            }, "required": ["birth", "schedule"], "additionalProperties": False,
        },
    )

    def __init__(self, *, birth: str, schedule: str) -> None:
        for name, value in (("birth", birth), ("schedule", schedule)):
            if type(value) is not str or not value.strip() or "__" in value:
                raise ConfigurationError(f"{name} must be a nonblank slot string without '__'")
        super().__init__(self.descriptor.component, {"birth": birth, "schedule": schedule})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SharedFullPrefixState(ComponentRef):
    """Declare shared full-prefix state requirements without acquiring state."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-full-prefix-state/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/state-protocol", category_version="1", lifecycle_phase="compile",
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


__all__ += ["InterleavedKVTransaction", "EncryptedKVTransaction", "SharedFullPrefixState"]
