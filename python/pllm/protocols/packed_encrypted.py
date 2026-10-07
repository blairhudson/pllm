"""Bounded CKKS linear-operator configuration."""
from __future__ import annotations

from typing import Literal
from pllm.configuration import ComponentDescriptor, ConfigurationError
from .base import ProtocolMethod


class PackedEncryptedLinear(ProtocolMethod):
    """Public diagonal-matrix/hidden-vector CKKS reference, not decoder execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/packed-encrypted-linear/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocol", category_version="1", lifecycle_phase="runtime",
        parameter_schema={"type": "object", "additionalProperties": False,
            "properties": {"rotation_keys": {"enum": ["single", "full"]}},
            "required": ["rotation_keys"]},
        capabilities=("packed-ckks-linear-reference", "bounded-operator-benchmark"),
        required_host_features=("native-core", "seal"), role_eligibility=("client", "inference"),
    )

    def __init__(self, *, rotation_keys: Literal["single", "full"] = "single") -> None:
        if rotation_keys not in ("single", "full"):
            raise ConfigurationError("rotation_keys must be single or full")
        super().__init__(self.descriptor.component, {"rotation_keys": rotation_keys})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor
