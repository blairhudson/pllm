"""Public preprocessing correlation-source declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef
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


@pending("pcf-secret-replication")
class SecretReplicationPcf(PendingMethod):
    pass


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
