"""Masked-linear protocol declarations."""

from pllm.configuration import ComponentDescriptor, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, planned as pending
from pllm.protocols.base import ProtocolMethod


class MaskedLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/masked-linear",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "properties": {
            "output_encoding": {"type": "string", "enum": ["raw", "row_residues"]},
        }, "additionalProperties": False},
        capabilities=("masked-linear",),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, *, output_encoding: str = "raw") -> None:
        if type(output_encoding) is not str or output_encoding not in {"raw", "row_residues"}:
            raise ConfigurationError("prepared output encoding must be raw or row_residues")
        super().__init__(self.descriptor.component,
                         {} if output_encoding == "raw" else {"output_encoding": output_encoding})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"output_encoding": self.params.get("output_encoding", "raw")}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


@pending("slalom")
class SlalomTeeVerifiableInference(PendingMethod):
    pass


__all__ = ["MaskedLinear", "SlalomTeeVerifiableInference"]
