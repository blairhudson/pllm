"""Immutable numeric choices for prepared public linear stages."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


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


__all__ = ["QuantizationScheme", "SymmetricPerRow"]
