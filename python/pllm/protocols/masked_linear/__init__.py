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
            "request_encoding": {"type": "string", "enum": ["raw", "compact"]},
            "prefill_pruning": {"type": "string", "enum": ["none", "terminal"]},
            "prefill_chunk_rows": {"type": "integer", "enum": [0, 4, 8, 16, 32]},
        }, "additionalProperties": False},
        capabilities=("masked-linear",),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, *, output_encoding: str = "raw", prefill_chunk_rows: int = 0,
                 request_encoding: str = "raw", prefill_pruning: str = "none") -> None:
        if type(output_encoding) is not str or output_encoding not in {"raw", "row_residues"}:
            raise ConfigurationError("prepared output encoding must be raw or row_residues")
        if type(prefill_chunk_rows) is not int or prefill_chunk_rows not in {0, 4, 8, 16, 32}:
            raise ConfigurationError("prefill_chunk_rows must be 0, 4, 8, 16 or 32")
        if type(request_encoding) is not str or request_encoding not in {"raw", "compact"}:
            raise ConfigurationError("prepared request encoding must be raw or compact")
        if type(prefill_pruning) is not str or prefill_pruning not in {"none", "terminal"}:
            raise ConfigurationError("prepared prefill pruning must be none or terminal")
        params: dict[str, object] = {} if output_encoding == "raw" else {"output_encoding": output_encoding}
        if prefill_chunk_rows:
            params["prefill_chunk_rows"] = prefill_chunk_rows
        if request_encoding != "raw":
            params["request_encoding"] = request_encoding
        if prefill_pruning != "none":
            params["prefill_pruning"] = prefill_pruning
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"output_encoding": self.params.get("output_encoding", "raw"),
                "request_encoding": self.params.get("request_encoding", "raw"),
                "prefill_pruning": self.params.get("prefill_pruning", "none"),
                "prefill_chunk_rows": self.params.get("prefill_chunk_rows", 0)}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


@pending("slalom")
class SlalomTeeVerifiableInference(PendingMethod):
    pass


__all__ = ["MaskedLinear", "SlalomTeeVerifiableInference"]
