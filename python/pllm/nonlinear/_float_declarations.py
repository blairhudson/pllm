"""Typed source activation declaration; compiler/runtime binding remains required."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class F32Activation(ComponentRef):
    """Declare the source activation, not arbitrary activation execution.

    Compiler/runtime binding must resolve the source function (for example SiLU).
    This typed declaration does not implement that execution.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-activation/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/nonlinear-protocol", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "function": {"type": "string", "enum": ["source"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["function"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, function: Literal["source"] = "source",
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        if type(function) is not str or function != "source":
            raise ConfigurationError("function must be 'source'")
        params: dict[str, object] = {"function": function}
        if execution_owner is not None:
            if type(execution_owner) is not str or execution_owner != "client":
                raise ConfigurationError("execution_owner must be 'client'")
            params["execution_owner"] = execution_owner
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        if "execution_owner" in params and params["execution_owner"] is None:
            raise ConfigurationError("execution_owner must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["F32Activation"]
