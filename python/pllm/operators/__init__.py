"""Typed operator declarations requiring compiler/runtime binding, not executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _closed_params(params: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(params, Mapping):
        raise ConfigurationError("component params must be a mapping")
    for name in ("execution_owner", "rows", "logits"):
        if name in params and params[name] is None:
            raise ConfigurationError(f"{name} must be omitted rather than null")
    return params


def _literal(name: str, value: object, expected: str) -> None:
    if type(value) is not str or value != expected:
        raise ConfigurationError(f"{name} must be {expected!r}")


def _owner(params: dict[str, object], execution_owner: object) -> dict[str, object]:
    if execution_owner is not None:
        _literal("execution_owner", execution_owner, "client")
        params["execution_owner"] = execution_owner
    return params


class F32RmsNorm(ComponentRef):
    """Source FP32 RMS norm declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-rmsnorm/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "epsilon": {"type": "string", "enum": ["source"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["epsilon"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, epsilon: Literal["source"] = "source",
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("epsilon", epsilon, "source")
        super().__init__(self.descriptor.component, _owner({"epsilon": epsilon}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class F32Rotary(ComponentRef):
    """Source FP32 rotary declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-rotary/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "configuration": {"type": "string", "enum": ["source"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["configuration"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, configuration: Literal["source"] = "source",
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("configuration", configuration, "source")
        super().__init__(self.descriptor.component, _owner({"configuration": configuration}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class F32CausalAttention(ComponentRef):
    """Source FP32 causal attention declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-causal-attention/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "heads": {"type": "string", "enum": ["source"]},
                "kv": {"type": "string", "enum": ["state"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["heads", "kv"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, heads: Literal["source"] = "source", kv: Literal["state"] = "state",
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("heads", heads, "source")
        _literal("kv", kv, "state")
        super().__init__(self.descriptor.component, _owner({"heads": heads, "kv": kv}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class F32Multiply(ComponentRef):
    """FP32 multiplication declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-multiply/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, execution_owner: Literal["client"] | None = None) -> None:
        super().__init__(self.descriptor.component, _owner({}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class F32Add(ComponentRef):
    """FP32 addition declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-add/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, execution_owner: Literal["client"] | None = None) -> None:
        super().__init__(self.descriptor.component, _owner({}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class F32AddBias(ComponentRef):
    """FP32 reconstructed-bias declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/f32-add-bias/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "placement": {"type": "string", "enum": ["after-reconstruction"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["placement"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, placement: Literal["after-reconstruction"] = "after-reconstruction",
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("placement", placement, "after-reconstruction")
        super().__init__(self.descriptor.component, _owner({"placement": placement}, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FullVocabularyHead(ComponentRef):
    """Full-vocabulary head declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/full-vocabulary-head/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "linear": {"type": "string", "enum": ["linear"]},
                "rows": {"type": "string", "enum": ["last-prefill-and-each-decode"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["linear"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, linear: Literal["linear"] = "linear",
        rows: Literal["last-prefill-and-each-decode"] | None = None,
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("linear", linear, "linear")
        params: dict[str, object] = {"linear": linear}
        if rows is not None:
            _literal("rows", rows, "last-prefill-and-each-decode")
            params["rows"] = rows
        super().__init__(self.descriptor.component, _owner(params, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class GreedySelection(ComponentRef):
    """Greedy selection declaration requiring compiler/runtime binding."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/greedy-selection/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "eos": {"type": "string", "enum": ["continue"]},
                "logits": {"type": "string", "enum": ["complete"]},
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["eos"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, eos: Literal["continue"] = "continue",
        logits: Literal["complete"] | None = None,
        execution_owner: Literal["client"] | None = None,
    ) -> None:
        _literal("eos", eos, "continue")
        params: dict[str, object] = {"eos": eos}
        if logits is not None:
            _literal("logits", logits, "complete")
            params["logits"] = logits
        super().__init__(self.descriptor.component, _owner(params, execution_owner))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(_closed_params(params)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PublicF32Lookup(ComponentRef):
    """Client-owned FP32 lookup configuration declaration, not execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/public-f32-lookup/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/operators", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "execution_owner": {"type": "string", "enum": ["client"]},
            }, "required": ["execution_owner"], "additionalProperties": False,
        },
    )

    def __init__(self, *, execution_owner: Literal["client"]) -> None:
        _literal("execution_owner", execution_owner, "client")
        super().__init__(self.descriptor.component, {"execution_owner": execution_owner})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "F32RmsNorm", "F32Rotary", "F32CausalAttention", "F32Multiply", "F32Add",
    "F32AddBias", "FullVocabularyHead", "GreedySelection",
    "PublicF32Lookup",
]
