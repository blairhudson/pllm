"""Public preprocessing correlation-source declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending


class CorrelationSource(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class SeededExpansion(CorrelationSource):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/seeded-expansion",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/correlation-source",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("domain-separated-mask-expansion",),
        role_eligibility=("client", "preparation"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["CorrelationSource", "SeededExpansion"]


class SecretReplicationPcf(CorrelationSource):
    """One-use full-field scalar/vector correlations with a bounded local consumer.

    One scalar applies to the complete vector. This does not generate independent
    ring32 elementwise triples. A local dealer supplies both party-local keys.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/secret-replication-pcf/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/correlation-source",
        category_version="1", lifecycle_phase="preparation",
        parameter_schema={"type": "object", "additionalProperties": False,
            "properties": {"field": {"enum": [2, 17, 257]},
                           "encoding": {"enum": ["expanded", "punctured"]}},
            "required": ["field", "encoding"]},
        capabilities=("full-field-scalar-vector-correlation", "bounded-operator-benchmark"),
        required_host_features=("native-core", "local-correlation-dealer"),
        role_eligibility=("client",),
    )

    def __init__(self, *, field: int = 257, encoding: str = "punctured") -> None:
        if type(field) is not int or field not in (2, 17, 257):
            raise ConfigurationError("field must be 2, 17 or 257")
        if type(encoding) is not str or encoding not in {"expanded", "punctured"}:
            raise ConfigurationError("encoding must be expanded or punctured")
        super().__init__(self.descriptor.component, {"field": field, "encoding": encoding})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


@pending("finite-field-pcg")
class FiniteFieldPcg(PendingMethod):
    pass


@pending("ring-pcg")
class RingPcg(PendingMethod):
    pass


@pending("sparse-unit-correlations")
class SparseUnitVectorCorrelation(PendingMethod):
    pass


@pending("mozzarella")
class RingVectorOle(PendingMethod):
    pass


@pending("ring-lpn-pcg")
class RingLpnPcg(PendingMethod):
    pass


@pending("silent-ot")
class SilentOtExtension(PendingMethod):
    pass


@pending("compressing-vole")
class CompressedVectorOle(PendingMethod):
    pass


__all__ += [
    "SecretReplicationPcf", "FiniteFieldPcg", "RingPcg", "SparseUnitVectorCorrelation",
    "RingVectorOle", "RingLpnPcg", "SilentOtExtension", "CompressedVectorOle",
]

install_planned_components(globals())
