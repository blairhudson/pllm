"""Public nonlinear protocol component declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.components._planned import PendingComponent as PendingMethod, planned as pending

if TYPE_CHECKING:
    from pllm._native import CompactQ7Reference


class NonlinearProtocol(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class ArithmeticGarblingSiluQ7(NonlinearProtocol):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/arithmetic-garbling-silu-q7/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="runtime",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-silu-q7",),
        required_host_features=("native-core",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class BinaryTableGatedMultiplyQ7(NonlinearProtocol):
    """Binary-table implementation of protected Q7 ``SiLU(gate) * up``."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/binary-table/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-gated-multiply-q7",),
        required_host_features=("decoder-plan-v1", "signed-q7"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class R03CrtGatedMultiplyQ7(NonlinearProtocol):
    """R03 CRT implementation of protected Q7 ``SiLU(gate) * up``."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/r03-crt/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-gated-multiply-q7",),
        required_host_features=("decoder-plan-v1", "signed-q7"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "ArithmeticGarblingSiluQ7",
    "BinaryTableGatedMultiplyQ7",
    "NonlinearProtocol",
    "R03CrtGatedMultiplyQ7",
]


@pending("shaft")
class ShaftFixedPointTransformerOps(PendingMethod):
    pass


@pending("compact")
class CompactPiecewiseActivation(PendingMethod):
    pass


@pending("curl")
class CurlWaveletEncodedTable(PendingMethod):
    pass


@pending("llama-math")
class LlamaSecureMath(PendingMethod):
    pass


@pending("sirnn")
class SirnnSecureMath(PendingMethod):
    pass


@pending("ripple")
class WaveletHomomorphicLookup(PendingMethod):
    pass


__all__ += [
    "ShaftFixedPointTransformerOps", "CompactPiecewiseActivation", "CurlWaveletEncodedTable",
    "LlamaSecureMath", "SirnnSecureMath", "WaveletHomomorphicLookup",
]


def fit_compact_silu_q7_reference(
    public_calibration_counts: bytes, max_pieces: int = 4
) -> CompactQ7Reference:
    """Fit a bounded public-calibration Q7 SiLU numeric reference.

    Supply exactly 257 little-endian unsigned-32 counts for Q7 inputs
    -128..128. Inputs must come from public *offline* calibration, never a
    private prompt. The returned profile selects intervals in plaintext;
    it is not a private protocol or an executable pipeline component.
    """
    if not isinstance(public_calibration_counts, bytes):
        raise TypeError("public calibration counts must be immutable bytes")
    from pllm import _native

    return _native.fit_compact_silu_q7_reference(public_calibration_counts, max_pieces)


__all__.append("fit_compact_silu_q7_reference")
