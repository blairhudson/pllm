"""Typed source-input declarations, not model sources or authentication proofs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class AuthenticatedSourceInputProducer(ComponentRef):
    """Declare authenticated source-input production without owning credentials."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-source-input-producer/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PinnedSemanticSourceBindings(ComponentRef):
    """Declare source-configuration requirements, not metadata authentication."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/pinned-semantic-source-bindings/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "required_model_type": {"type": "string", "enum": ["bert"]},
                "required_architectures": {
                    "type": "array", "minItems": 1, "maxItems": 1,
                    "items": {
                        "type": "string", "enum": ["BertForNextSentencePrediction"],
                    },
                },
                "required_output_logits": {"type": "integer", "enum": [2]},
            }, "additionalProperties": False,
        },
    )

    def __init__(
        self, *, required_model_type: Literal["bert"] | None = None,
        required_architectures: tuple[Literal["BertForNextSentencePrediction"]] | None = None,
        required_output_logits: Literal[2] | None = None,
    ) -> None:
        params: dict[str, object] = {}
        if required_model_type is not None:
            if type(required_model_type) is not str or required_model_type != "bert":
                raise ConfigurationError("required_model_type must be 'bert'")
            params["required_model_type"] = required_model_type
        if required_architectures is not None:
            if (type(required_architectures) is not tuple
                    or len(required_architectures) != 1
                    or type(required_architectures[0]) is not str
                    or required_architectures[0] != "BertForNextSentencePrediction"):
                raise ConfigurationError(
                    "required_architectures must be ('BertForNextSentencePrediction',)"
                )
            params["required_architectures"] = required_architectures
        if required_output_logits is not None:
            if type(required_output_logits) is not int or required_output_logits != 2:
                raise ConfigurationError("required_output_logits must be the integer 2")
            params["required_output_logits"] = required_output_logits
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        names = ("required_model_type", "required_architectures", "required_output_logits")
        for name in params:
            if type(name) is not str or name not in names:
                raise ConfigurationError(f"unknown parameter {name!r} for {cls.__name__}")
            if params[name] is None:
                raise ConfigurationError(f"{name} must be omitted rather than null")
        normalized = dict(params)
        architectures = normalized.get("required_architectures")
        if type(architectures) is list:
            normalized["required_architectures"] = tuple(architectures)
        return cast(Self, super().from_params(normalized))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["AuthenticatedSourceInputProducer", "PinnedSemanticSourceBindings"]


class PublicCommittedQ18Source(ComponentRef):
    """Declare a public committed Q18 source requirement without source I/O."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/public-committed-q18-source/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
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


class FSSSourceBinding(ComponentRef):
    """Declare FSS source binding without acquiring source authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/fss-source-binding/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
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


__all__ += ["PublicCommittedQ18Source", "FSSSourceBinding"]


class AuthenticatedSourceBindings(ComponentRef):
    """Declare source-binding requirements without authenticating a checkpoint."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-source-bindings/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "weight_owner": {"type": "string", "enum": ["server"]},
                "bind_checkpoint_and_numeric_domain": {"type": "boolean", "enum": [True]},
            }, "required": ["weight_owner", "bind_checkpoint_and_numeric_domain"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self, *, weight_owner: Literal["server"], bind_checkpoint_and_numeric_domain: Literal[True],
    ) -> None:
        if type(weight_owner) is not str or weight_owner != "server":
            raise ConfigurationError("weight_owner must be 'server'")
        if (type(bind_checkpoint_and_numeric_domain) is not bool
                or bind_checkpoint_and_numeric_domain is not True):
            raise ConfigurationError("bind_checkpoint_and_numeric_domain must be true")
        super().__init__(self.descriptor.component, {
            "weight_owner": weight_owner,
            "bind_checkpoint_and_numeric_domain": bind_checkpoint_and_numeric_domain,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class AuthenticatedOriginalProducer(ComponentRef):
    """Declare an authenticated original producer without creating source authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-original-producer/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/sources", category_version="1", lifecycle_phase="compile",
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


__all__ += ["AuthenticatedSourceBindings", "AuthenticatedOriginalProducer"]
