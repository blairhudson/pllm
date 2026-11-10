"""Typed exact source/field boundary declarations without source execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ExactSourceFieldBoundaries(ComponentRef):
    """Declare explicit source/field boundaries; no future certificates implied."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/exact-source-field-boundaries/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "parameter_authority": {"type": "null"},
                "client_arithmetic": {"type": "string", "enum": ["f32-extension"]},
                "source_precision": {"type": "string", "enum": ["source"]},
                "source_to_field_rounding": {"type": "null"},
                "field_to_f32_rounding": {"type": "null"},
            },
            "required": [
                "parameter_authority", "client_arithmetic", "source_precision",
                "source_to_field_rounding", "field_to_f32_rounding",
            ],
            "additionalProperties": False,
        },
    )

    def __init__(
        self, *, parameter_authority: None,
        client_arithmetic: Literal["f32-extension"],
        source_precision: Literal["source"],
        source_to_field_rounding: None, field_to_f32_rounding: None,
    ) -> None:
        if parameter_authority is not None:
            raise ConfigurationError("parameter_authority must be null")
        if type(client_arithmetic) is not str or client_arithmetic != "f32-extension":
            raise ConfigurationError("client_arithmetic must be 'f32-extension'")
        if type(source_precision) is not str or source_precision != "source":
            raise ConfigurationError("source_precision must be 'source'")
        if source_to_field_rounding is not None:
            raise ConfigurationError("source_to_field_rounding must be null")
        if field_to_f32_rounding is not None:
            raise ConfigurationError("field_to_f32_rounding must be null")
        super().__init__(self.descriptor.component, {
            "parameter_authority": parameter_authority,
            "client_arithmetic": client_arithmetic,
            "source_precision": source_precision,
            "source_to_field_rounding": source_to_field_rounding,
            "field_to_f32_rounding": field_to_f32_rounding,
        })

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["ExactSourceFieldBoundaries"]


class ReplicatedRing64Q18Edges(ComponentRef):
    """Declare source-ordered replicated ring64 Q18 edge requirements, not proof."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-ring64-q18-edges/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "revision": {"type": "string", "enum": ["source-order-v1"]},
            }, "required": ["revision"], "additionalProperties": False,
        },
    )

    def __init__(self, *, revision: Literal["source-order-v1"]) -> None:
        if type(revision) is not str or revision != "source-order-v1":
            raise ConfigurationError("revision must be 'source-order-v1'")
        super().__init__(self.descriptor.component, {"revision": revision})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Ring50Q12Edges(ComponentRef):
    """Declare ring50 Q12 edges, not whole-graph proof or interchangeable arithmetic."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/ring50-q12-edges/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "revision": {"type": "string", "enum": [
                    "sigma-source-order-v1", "fusefss.dense-rope-ring50-q12.v1",
                ]},
                "ranges": {"type": "string", "enum": ["source-derived"]},
            }, "required": ["revision", "ranges"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, revision: Literal[
            "sigma-source-order-v1", "fusefss.dense-rope-ring50-q12.v1",
        ], ranges: Literal["source-derived"],
    ) -> None:
        if type(revision) is not str or revision not in (
            "sigma-source-order-v1", "fusefss.dense-rope-ring50-q12.v1",
        ):
            raise ConfigurationError(
                "revision must be 'sigma-source-order-v1' or 'fusefss.dense-rope-ring50-q12.v1'"
            )
        if type(ranges) is not str or ranges != "source-derived":
            raise ConfigurationError("ranges must be 'source-derived'")
        super().__init__(self.descriptor.component, {"revision": revision, "ranges": ranges})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ += ["ReplicatedRing64Q18Edges", "Ring50Q12Edges"]


class SourceCalibration(ComponentRef):
    """Declare source-authenticated calibration requirements, not authentication."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/source-authenticated-calibration/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
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


class NumericDomain(ComponentRef):
    """Declare a numeric-domain requirement without granting parameter authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/numeric-domain-certificate/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {"parameter_authority": {"type": "null"}},
            "required": ["parameter_authority"], "additionalProperties": False,
        },
    )

    def __init__(self, *, parameter_authority: None) -> None:
        if parameter_authority is not None:
            raise ConfigurationError("parameter_authority must be null")
        super().__init__(self.descriptor.component, {"parameter_authority": parameter_authority})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SourceNumericDomainCertificates(ComponentRef):
    """Declare source numeric-domain requirements without issuing certificates."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/source-numeric-domain-certificates/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "parameter_authority": {"type": "null"},
                "profile": {"type": "string", "enum": ["paper-ring64-q18"]},
                "extension": {"type": "string", "enum": ["ring32-q12-where-bound"]},
                "source": {"type": "string", "minLength": 1, "pattern": r"\S"},
            }, "required": ["parameter_authority"], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, parameter_authority: None,
        profile: Literal["paper-ring64-q18"] | None = None,
        extension: Literal["ring32-q12-where-bound"] | None = None,
        source: str | None = None,
    ) -> None:
        if parameter_authority is not None:
            raise ConfigurationError("parameter_authority must be null")
        params: dict[str, object] = {"parameter_authority": parameter_authority}
        if profile is not None:
            if type(profile) is not str or profile != "paper-ring64-q18":
                raise ConfigurationError("profile must be 'paper-ring64-q18'")
            params["profile"] = profile
        if extension is not None:
            if type(extension) is not str or extension != "ring32-q12-where-bound":
                raise ConfigurationError("extension must be 'ring32-q12-where-bound'")
            params["extension"] = extension
        if source is not None:
            if type(source) is not str or not source.strip():
                raise ConfigurationError("source must be a nonblank string")
            params["source"] = source
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        for name in ("profile", "extension", "source"):
            if name in params and params[name] is None:
                raise ConfigurationError(f"{name} must be omitted rather than null")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ += ["SourceCalibration", "NumericDomain", "SourceNumericDomainCertificates"]
