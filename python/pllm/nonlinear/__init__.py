"""Public nonlinear protocol component declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending

if TYPE_CHECKING:
    from pllm._native import CompactQ7Reference, ScaledSiluQ7Reference


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


class MaskedWireGatedMultiplyQ12(NonlinearProtocol):
    """SIGMA-derived masked-wire ``SiLU(gate) * up`` with signed Q24 inputs.

    Gate and up ranges are [-48, 48] and [-128, 128]. The fixed residual
    table and floor shifts produce Q12 output. ``layout="fused"`` computes the
    input gate shifts and predicates through public masked views, preserving
    the numeric contract. Operator benchmarks use a local dealer. The CPU
    two-offset Experiment uses client-issued material and authenticated
    party-local workers, with peer frames relayed through the trusted client.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/masked-wire-gated-multiply-q12/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="runtime",
        parameter_schema={
            "type": "object", "properties": {"layout": {"enum": ["unfused", "fused"]}},
            "additionalProperties": False,
        },
        capabilities=("protected-gated-multiply-q12", "bounded-operator-benchmark", "compiled-gated-regions"),
        required_host_features=("native-core", "trusted-client-fss-dealer"),
        role_eligibility=("client", "worker_a", "worker_b"),
    )

    def __init__(self, *, layout: str = "unfused") -> None:
        if type(layout) is not str or layout not in {"unfused", "fused"}:
            raise ConfigurationError("layout must be unfused or fused")
        # Preserve the original default component and historical configurations.
        super().__init__(self.descriptor.component, {} if layout == "unfused" else {"layout": layout})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"layout": self.params.get("layout", "unfused")}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SharedRescale(NonlinearProtocol):
    """Signed shared rescaling and the original input's nonnegative predicate.

    The fused/unfused layouts share one numeric contract and comparison
    backend. Floor and the ties-even extension have distinct numeric identities.
    Bounded operator execution uses a local dealer and two one-use evaluators.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/shared-rescale/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="runtime",
        parameter_schema={
            "type": "object",
            "properties": {
                "bits": {"type": "integer", "minimum": 2, "maximum": 32},
                "shift": {"type": "integer", "minimum": 1, "maximum": 31},
                "rounding": {"enum": ["floor", "ties_even"]},
                "layout": {"enum": ["fused", "unfused"]},
                "backend": {"enum": ["prefix_dpf", "compact_dcf", "interval_dcf"]},
            },
            "required": ["bits", "shift", "rounding", "layout", "backend"],
            "additionalProperties": False,
        },
        capabilities=("protected-signed-rescale", "original-input-nonnegative", "bounded-operator-benchmark"),
        required_host_features=("native-core", "local-fss-dealer"),
        role_eligibility=("client",),
    )

    def __init__(
        self,
        *,
        bits: int = 16,
        shift: int = 7,
        rounding: str = "floor",
        layout: str = "fused",
        backend: str = "interval_dcf",
    ) -> None:
        if type(bits) is not int or not 2 <= bits <= 32:
            raise ConfigurationError("bits must be an integer in [2, 32]")
        if type(shift) is not int or not 1 <= shift < bits:
            raise ConfigurationError("shift must be an integer in [1, bits)")
        for name, value, choices in (
            ("rounding", rounding, {"floor", "ties_even"}),
            ("layout", layout, {"fused", "unfused"}),
            ("backend", backend, {"prefix_dpf", "compact_dcf", "interval_dcf"}),
        ):
            if type(value) is not str or value not in choices:
                raise ConfigurationError(f"{name} must be one of {sorted(choices)}")
        super().__init__(
            self.descriptor.component,
            {"bits": bits, "shift": shift, "rounding": rounding, "layout": layout, "backend": backend},
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "ArithmeticGarblingSiluQ7",
    "BinaryTableGatedMultiplyQ7",
    "NonlinearProtocol",
    "R03CrtGatedMultiplyQ7",
    "MaskedWireGatedMultiplyQ12",
    "SharedRescale",
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

install_planned_components(globals())


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


def create_logrow_scaled_silu_q7_reference(max_abs: int) -> ScaledSiluQ7Reference:
    """Fix a public integer SiLU range for one-use LogRow research sessions.

    ``max_abs`` is a public/offline range in 1..=16, not a statistic derived
    from a private request. No clipping occurs; a gate outside the selected
    range fails and burns the remaining session material. The profile is not
    an executable Pipeline component.
    """
    from pllm import _native

    return _native.create_logrow_scaled_silu_q7_reference(max_abs)


__all__.append("create_logrow_scaled_silu_q7_reference")
