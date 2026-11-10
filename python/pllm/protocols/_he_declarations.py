"""HE protocol configuration requirements, not execution implementations.

These inert declarations grant no HE execution, key access, private-weight
access, native binding, numeric authority, or security guarantee. Compiler and
runtime bindings must independently satisfy arithmetic and trust requirements.
They are distinct from bounded PackedEncryptedLinear/Softmax interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _slot_link(value: str, name: str) -> str:
    if type(value) is not str or not value.strip() or "__" in value:
        raise ConfigurationError(f"{name} must be a nonblank string without '__'")
    return value


def _reject_nulls(params: Mapping[str, object], names: tuple[str, ...]) -> None:
    if not isinstance(params, Mapping):
        raise ConfigurationError("component params must be a mapping")
    for name in names:
        if name in params and params[name] is None:
            raise ConfigurationError(f"{name} must be omitted rather than null")


class CkksRectangularLinear(ComponentRef):
    """Declare CKKS rectangular-linear requirements without numeric authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/ckks-rectangular-linear/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("CkksRectangularLinear accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class InterleavedQK(ComponentRef):
    """Declare interleaved QK arithmetic requirements, not HE execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/interleaved-qk/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("InterleavedQK accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class InterleavedWeightedV(ComponentRef):
    """Declare interleaved weighted-V requirements without execution authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/interleaved-weighted-v/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("InterleavedWeightedV accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class RoutedGQA(ComponentRef):
    """Declare routed grouped-query-attention requirements, not execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/routed-gqa/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("RoutedGQA accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HERotaryTransform(ComponentRef):
    """Declare HE rotary-transform requirements without numeric authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-rotary-transform/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HERotaryTransform accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HECoefficientCompression(ComponentRef):
    """Declare HE coefficient-compression requirements, not an implementation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-coefficient-compression/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HECoefficientCompression accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HECommutableLinear(ComponentRef):
    """Declare HE commutable-linear requirements without weight access authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-commutable-linear/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HECommutableLinear accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEColumnProduct(ComponentRef):
    """Declare HE column-product arithmetic requirements, not an executor."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-column-product/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEColumnProduct accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HELayoutTransform(ComponentRef):
    """Declare HE layout-transform requirements without native binding authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-layout-transform/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HELayoutTransform accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEAttentionValue(ComponentRef):
    """Declare HE attention-value arithmetic requirements, not execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-attention-value/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEAttentionValue accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEAttentionProducts(ComponentRef):
    """Declare HE attention-product requirements without numeric authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-attention-products/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEAttentionProducts accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEHeadScaleMask(ComponentRef):
    """Declare HE head-scale-mask requirements, not an arithmetic implementation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-head-scale-mask/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEHeadScaleMask accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEResidualAdd(ComponentRef):
    """Declare HE residual-add requirements without numeric or execution authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-residual-add/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEResidualAdd accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEPrivateEmbedding(ComponentRef):
    """Declare HE private-embedding requirements without private-weight access."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-private-embedding/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEPrivateEmbedding accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HESourceEmbeddingPositionSegment(ComponentRef):
    """Declare Source embedding/position/segment requirements, not HE execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-source-embedding-position-segment/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
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
            raise ConfigurationError("HESourceEmbeddingPositionSegment accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEClsPooler(ComponentRef):
    """Declare HE CLS-pooler requirements without execution or weight authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-cls-pooler/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEClsPooler accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HENspTwoLogitHead(ComponentRef):
    """Declare HE NSP two-logit-head requirements without numeric authority."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-nsp-two-logit-head/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HENspTwoLogitHead accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEFoldSelection(ComponentRef):
    """Declare HE fold-selection requirements without selection execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-fold-selection/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEFoldSelection accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class NspLogitsOnlyOutputPolicy(ComponentRef):
    """Declare NSP-logits-only output requirements, not disclosure enforcement."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/nsp-logits-only-output-policy/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("NspLogitsOnlyOutputPolicy accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEBootstrap(ComponentRef):
    """Declare an HE bootstrap protocol primitive, not preparation or execution.

    Keys, dual preparation, and ciphertext birth belong to separate preparation
    contracts. This declaration creates none and grants no numeric authority.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-bootstrap/v1", provider="pllm", distribution="pllm.run",
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
            raise ConfigurationError("HEBootstrap accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEPackedLinear(ComponentRef):
    """Declare HE packed-linear requirements, distinct from bounded interfaces.

    Optional fields are independent configuration values, not resolved links.
    Ownership labels grant no private-weight access; bindings, output-head,
    vocabulary, and selection requirements grant no execution authority.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-packed-linear/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "weight_owner": {"type": "string", "enum": ["server"]},
                "source_bindings": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
                "operation": {"type": "string", "enum": ["source-output-head"]},
                "vocabulary": {"type": "string", "enum": ["full"]},
                "selection": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, weight_owner: Literal["server"] | None = None,
        source_bindings: str | None = None,
        operation: Literal["source-output-head"] | None = None,
        vocabulary: Literal["full"] | None = None, selection: str | None = None,
    ) -> None:
        params: dict[str, object] = {}
        if weight_owner is not None:
            if type(weight_owner) is not str or weight_owner != "server":
                raise ConfigurationError("weight_owner must be 'server'")
            params["weight_owner"] = weight_owner
        if source_bindings is not None:
            params["source_bindings"] = _slot_link(source_bindings, "source_bindings")
        if operation is not None:
            if type(operation) is not str or operation != "source-output-head":
                raise ConfigurationError("operation must be 'source-output-head'")
            params["operation"] = operation
        if vocabulary is not None:
            if type(vocabulary) is not str or vocabulary != "full":
                raise ConfigurationError("vocabulary must be 'full'")
            params["vocabulary"] = vocabulary
        if selection is not None:
            params["selection"] = _slot_link(selection, "selection")
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        _reject_nulls(params, ("weight_owner", "source_bindings", "operation", "vocabulary", "selection"))
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PrivateTokenSelection(ComponentRef):
    """Declare private-token-selection trust requirements, not privacy guarantees.

    Comparison and role strings remain independent, unresolved configuration;
    declaring them neither runs comparison nor authenticates roles.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/private-token-selection/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "comparison": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
                "roles": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self, *, comparison: str | None = None, roles: str | None = None) -> None:
        params: dict[str, object] = {}
        if comparison is not None:
            params["comparison"] = _slot_link(comparison, "comparison")
        if roles is not None:
            params["roles"] = _slot_link(roles, "roles")
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        _reject_nulls(params, ("comparison", "roles"))
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PrivateTokenFeedback(ComponentRef):
    """Declare private-token-feedback requirements without state or key authority.

    Selection, input, and state strings remain independent, unresolved
    configuration values. This declaration performs no feedback or disclosure.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/private-token-feedback/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "selection": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
                "input": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
                "state": {
                    "type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"},
                },
            }, "required": [], "additionalProperties": False,
        },
    )

    def __init__(
        self, *, selection: str | None = None, input: str | None = None,
        state: str | None = None,
    ) -> None:
        params: dict[str, object] = {}
        if selection is not None:
            params["selection"] = _slot_link(selection, "selection")
        if input is not None:
            params["input"] = _slot_link(input, "input")
        if state is not None:
            params["state"] = _slot_link(state, "state")
        super().__init__(self.descriptor.component, params)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        _reject_nulls(params, ("selection", "input", "state"))
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ClassifierOutputPolicy(ComponentRef):
    """Declare model-metadata classifier-output requirements, not enforcement.

    The source label does not read metadata or grant disclosure, execution,
    private-weight, native, or numeric authority.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/classifier-output-policy/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {
                "source": {"type": "string", "enum": ["model-metadata"]},
            }, "required": ["source"], "additionalProperties": False,
        },
    )

    def __init__(self, *, source: Literal["model-metadata"]) -> None:
        if type(source) is not str or source != "model-metadata":
            raise ConfigurationError("source must be 'model-metadata'")
        super().__init__(self.descriptor.component, {"source": source})

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
    "CkksRectangularLinear", "InterleavedQK", "InterleavedWeightedV", "RoutedGQA",
    "HERotaryTransform", "HECoefficientCompression", "HECommutableLinear", "HEColumnProduct",
    "HELayoutTransform", "HEAttentionValue", "HEAttentionProducts", "HEHeadScaleMask",
    "HEResidualAdd", "HEPrivateEmbedding", "HESourceEmbeddingPositionSegment", "HEClsPooler",
    "HENspTwoLogitHead", "HEFoldSelection", "NspLogitsOnlyOutputPolicy", "HEBootstrap",
    "HEPackedLinear", "PrivateTokenSelection", "PrivateTokenFeedback", "ClassifierOutputPolicy",
]
