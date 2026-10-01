"""Immutable numeric choices for prepared public linear stages."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._model_capabilities import PendingModelCapability, model_capability_stub
from pllm.components._planned import install_planned_components


class QuantizationScheme(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class SymmetricPerRow(QuantizationScheme):
    """Choose the public stage's symmetric weight and activation bit widths.

    The values bind quantized weights, prepared corrections, online masked
    tensors, and the compiled runtime. This does not promise model fidelity.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/symmetric-per-row-quantization/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/quantization",
        category_version="1",
        lifecycle_phase="compile",
        parameter_schema={
            "type": "object",
            "properties": {
                "weight_bits": {"enum": [4, 8]},
                "activation_bits": {"enum": [4, 8]},
            },
            "required": ["weight_bits", "activation_bits"],
            "additionalProperties": False,
        },
        capabilities=("symmetric", "per-row", "prepared-masked-linear"),
    )

    def __init__(self, *, weight_bits: int = 8, activation_bits: int = 8) -> None:
        if type(weight_bits) is not int or weight_bits not in {4, 8}:
            raise ConfigurationError("weight_bits must be 4 or 8")
        if type(activation_bits) is not int or activation_bits not in {4, 8}:
            raise ConfigurationError("activation_bits must be 4 or 8")
        super().__init__(
            self.descriptor.component,
            {"weight_bits": weight_bits, "activation_bits": activation_bits},
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PublicPerChannelEqualized(QuantizationScheme):
    """Use a source-bound public calibration profile for W8A8 linear stages.

    The profile is generated offline from public token IDs; no request data
    enters calibration. Its digest binds per-stage input scales and the source
    lock. It changes the prepared body and must not reuse baseline inventory.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/public-per-channel-equalized/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/quantization",
        category_version="1",
        lifecycle_phase="compile",
        parameter_schema={
            "type": "object",
            "properties": {
                "profile_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "required": ["profile_digest"],
            "additionalProperties": False,
        },
        capabilities=("symmetric", "public-calibrated", "per-channel-equalized", "prepared-masked-linear"),
    )

    def __init__(self, profile_digest: str) -> None:
        if (
            type(profile_digest) is not str
            or len(profile_digest) != 64
            or any(character not in "0123456789abcdef" for character in profile_digest)
        ):
            raise ConfigurationError("profile_digest must be lowercase SHA-256")
        super().__init__(self.descriptor.component, {"profile_digest": profile_digest})

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


def fit_public_equalization_profile(source, calibration_tokens, *, threads: int = 4):
    """Build an immutable profile from a locked source and public token cohort."""
    from pllm.runtime.public_equalization import fit_public_equalization_profile as fit

    return fit(source, tuple(tuple(row) for row in calibration_tokens), threads=threads)


@model_capability_stub("mxfp4-import")
class Mxfp4CheckpointImport(PendingModelCapability):
    pass


@model_capability_stub("fp8-import")
class Fp8BlockScaleImport(PendingModelCapability):
    pass


__all__ = [
    "QuantizationScheme", "SymmetricPerRow", "PublicPerChannelEqualized",
    "fit_public_equalization_profile", "Mxfp4CheckpointImport", "Fp8BlockScaleImport",
]

install_planned_components(globals())
