"""Masked-linear protocol declarations."""

from pllm.configuration import ComponentDescriptor
from pllm.protocols.base import ProtocolMethod


class MaskedLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/masked-linear",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("masked-linear",),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["MaskedLinear"]
