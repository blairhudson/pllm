"""Public protocol component declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pllm.components._planned import (
    PendingComponent as PendingMethod,
    install_planned_components,
    planned as pending,
)
from pllm.protocols.base import ProtocolMethod
from pllm.protocols.bundle_transport import ClientBundleTransport
from pllm.protocols.masked_linear import MaskedLinear
from pllm.protocols.runtime_arms import (
    BlindedLinear,
    CleartextLinear,
    DirectFHE,
    GuardedLinear,
    SecureLinear,
    TwoOnlineOffsetLinear,
)

if TYPE_CHECKING:
    from pllm._native import (
        CompactQ7Reference,
        LogRowQ7SessionReference,
        LogRowQ7TensorReference,
        ScaledSiluQ7Reference,
    )
    from pllm.modeling import ModelPlan

__all__ = [
    "BlindedLinear",
    "CleartextLinear",
    "DirectFHE",
    "GuardedLinear",
    "MaskedLinear",
    "ProtocolMethod",
    "ClientBundleTransport",
    "SecureLinear",
    "TwoOnlineOffsetLinear",
    "LogRowQ7SessionEstimate",
    "estimate_logrow_q7_session_reference",
    "prepare_logrow_q7_session_reference",
    "prepare_logrow_q7_tensor_reference",
]


@dataclass(frozen=True, slots=True)
class LogRowQ7SessionEstimate:
    """Worst-case SiLU evaluator-body bytes; no material or execution rights."""

    plan_digest: str
    profile_id: str
    profile_digest: bytes
    max_decode_steps: int
    prefill_elements: int
    decode_elements_per_step: int
    reserved_evaluator_material_bytes: int
    largest_tensor_material_bytes: int
    estimate_digest: str


def estimate_logrow_q7_session_reference(
    plan: ModelPlan,
    profile: CompactQ7Reference | ScaledSiluQ7Reference,
    *,
    max_elements: int,
    max_evaluator_material_bytes: int,
    max_decode_steps: int,
    max_session_evaluator_material_bytes: int,
) -> LogRowQ7SessionEstimate:
    """Preflight total Q7 SiLU material across prefill and bounded decode.

    This estimate includes only evaluator bodies, not labels, transport, or
    peak memory. It does not issue material or make a Pipeline executable.
    """
    from pllm import _native
    from pllm.modeling import ModelPlan

    if type(plan) is not ModelPlan:
        raise TypeError("plan must be an immutable ModelPlan")
    document = json.loads(
        _native.estimate_logrow_q7_session_reference(
            plan.canonical_bytes(),
            profile,
            max_elements,
            max_evaluator_material_bytes,
            max_decode_steps,
            max_session_evaluator_material_bytes,
        )
    )
    if document.pop("schema_version", None) != "pllm.logrow_q7_session_estimate.v1":
        raise ValueError("unsupported LogRow Q7 session estimate schema")
    digest = document.pop("profile_digest")
    if not isinstance(digest, list) or len(digest) != 32:
        raise ValueError("invalid LogRow Q7 profile digest")
    return LogRowQ7SessionEstimate(profile_digest=bytes(digest), **document)


def prepare_logrow_q7_session_reference(
    plan: ModelPlan,
    profile: CompactQ7Reference | ScaledSiluQ7Reference,
    *,
    max_elements: int,
    max_evaluator_material_bytes: int,
    max_decode_steps: int,
    max_session_evaluator_material_bytes: int,
) -> LogRowQ7SessionReference:
    """Preissue all bounded Q7 SiLU material offline in semantic order.

    The opaque in-process handle consumes each tensor once, and abort/drop
    burns the remainder. It is not selectable by Pipeline or Experiment.
    """
    from pllm import _native
    from pllm.modeling import ModelPlan

    if type(plan) is not ModelPlan:
        raise TypeError("plan must be an immutable ModelPlan")
    return _native.prepare_logrow_q7_session_reference(
        plan.canonical_bytes(),
        profile,
        max_elements,
        max_evaluator_material_bytes,
        max_decode_steps,
        max_session_evaluator_material_bytes,
    )


def prepare_logrow_q7_tensor_reference(
    plan: ModelPlan,
    profile: CompactQ7Reference,
    *,
    mode: Literal["prefill", "decode"],
    operation_id: str,
    max_elements: int,
    max_evaluator_material_bytes: int,
) -> LogRowQ7TensorReference:
    """Issue one bounded Q7 SiLU tensor; not an Experiment or deployed 2PC method.

    The profile must come from public offline calibration. The handle consumes
    exactly one little-endian signed-i16 Q7 tensor (one value per plan element).
    Its float32 bridge rejects values outside [-1, 1]; it has no provider transport.
    """
    from pllm import _native
    from pllm.modeling import ModelPlan

    if type(plan) is not ModelPlan:
        raise TypeError("plan must be an immutable ModelPlan")
    return _native.prepare_logrow_q7_tensor_reference(
        plan.canonical_bytes(),
        mode,
        operation_id,
        profile,
        max_elements,
        max_evaluator_material_bytes,
    )


@pending("duty-free-bits")
class ProjectiveLabelConversion(PendingMethod):
    pass


@pending("mosaic")
class MosaicMaskedGpuOutsourcing(PendingMethod):
    pass


@pending("maverick")
class MaverickDelegatedLinear(PendingMethod):
    pass


@pending("oblivious-compression")
class EncryptedSparseVectorCompression(PendingMethod):
    pass


@pending("open-weight-verifiable")
class OpenWeightVerifiableOutsourcing(PendingMethod):
    pass


@pending("sort-sweep-mirror")
class BatchedIntervalLookup(PendingMethod):
    pass


@pending("bumblebee")
class BumblebeeTwoPartyTransformer(PendingMethod):
    pass


@pending("dash")
class DashPreparedArithmeticGarbling(PendingMethod):
    pass


@pending("emvp")
class SecretDualCodeMatVec(PendingMethod):
    pass


@pending("moai")
class MoaiNoninteractiveTransformer(PendingMethod):
    pass


@pending("nexus")
class NexusNoninteractiveTransformer(PendingMethod):
    pass


@pending("trapdoored-matrices")
class TrapdooredMatrixDelegation(PendingMethod):
    pass


@pending("shark")
class ActiveFssProtocol(PendingMethod):
    pass


@pending("thor")
class ThorHomomorphicTransformer(PendingMethod):
    pass


@pending("bolt")
class BoltHybridTransformer(PendingMethod):
    pass


@pending("fastquery")
class PrivateEmbeddingLookup(PendingMethod):
    pass


@pending("logrow")
class LogRowGarbledLookup(PendingMethod):
    pass


@pending("nimbus")
class NimbusTwoPartyTransformer(PendingMethod):
    pass


@pending("orca")
class OrcaFssGpuExecution(PendingMethod):
    pass


@pending("sigma")
class SigmaFssDecoder(PendingMethod):
    pass


@pending("ciphergpt")
class CipherGptPrivateTokenSelection(PendingMethod):
    pass


@pending("flute")
class FlutePrivateLookup(PendingMethod):
    pass


@pending("grotto")
class GrottoRingDpfProtocol(PendingMethod):
    pass


@pending("puma")
class PumaThreePartyDecoder(PendingMethod):
    pass


@pending("primer")
class PrimerHomomorphicTransformer(PendingMethod):
    pass


@pending("cheetah")
class CheetahTwoPartyNeural(PendingMethod):
    pass


@pending("iron")
class IronTwoPartyTransformer(PendingMethod):
    pass


@pending("pika")
class PikaRingFssProtocol(PendingMethod):
    pass


@pending("the-x")
class TheXHomomorphicTransformer(PendingMethod):
    pass


@pending("aby2")
class Aby2ArithmeticBooleanConversion(PendingMethod):
    pass


@pending("fss-mixed")
class FssMixedFixedPointConversion(PendingMethod):
    pass


@pending("cryptflow2")
class CryptFlowTwoPartyNeural(PendingMethod):
    pass


@pending("mixed-arithmetic-primitives")
class MixedArithmeticEdabitConversion(PendingMethod):
    pass


@pending("silent-nisc")
class SilentNiscCircuitSetup(PendingMethod):
    pass


@pending("garbled-nn")
class GarbledNeuralProtocol(PendingMethod):
    pass


@pending("fss-preprocessing")
class FssOfflinePreprocessing(PendingMethod):
    pass


@pending("gazelle")
class GazelleHybridHeGarbling(PendingMethod):
    pass


@pending("cryptonets")
class CryptoNetsHomomorphicNeural(PendingMethod):
    pass


@pending("fss-extensions")
class ExtendedFssKeys(PendingMethod):
    pass


@pending("garbling-gadgets")
class CrtGarblingGadgetProtocol(PendingMethod):
    pass


@pending("fss-original")
class FunctionSecretSharingKeys(PendingMethod):
    pass


@pending("half-gates")
class HalfGatesCircuit(PendingMethod):
    pass


@pending("oblivious-decision-programs")
class ProtectedDecisionProgram(PendingMethod):
    pass


__all__ += [
    "ProjectiveLabelConversion",
    "MosaicMaskedGpuOutsourcing",
    "MaverickDelegatedLinear",
    "EncryptedSparseVectorCompression",
    "OpenWeightVerifiableOutsourcing",
    "BatchedIntervalLookup",
    "BumblebeeTwoPartyTransformer",
    "DashPreparedArithmeticGarbling",
    "SecretDualCodeMatVec",
    "MoaiNoninteractiveTransformer",
    "NexusNoninteractiveTransformer",
    "TrapdooredMatrixDelegation",
    "ActiveFssProtocol",
    "ThorHomomorphicTransformer",
    "BoltHybridTransformer",
    "PrivateEmbeddingLookup",
    "LogRowGarbledLookup",
    "NimbusTwoPartyTransformer",
    "OrcaFssGpuExecution",
    "SigmaFssDecoder",
    "CipherGptPrivateTokenSelection",
    "FlutePrivateLookup",
    "GrottoRingDpfProtocol",
    "PumaThreePartyDecoder",
    "PrimerHomomorphicTransformer",
    "CheetahTwoPartyNeural",
    "IronTwoPartyTransformer",
    "PikaRingFssProtocol",
    "TheXHomomorphicTransformer",
    "Aby2ArithmeticBooleanConversion",
    "FssMixedFixedPointConversion",
    "CryptFlowTwoPartyNeural",
    "MixedArithmeticEdabitConversion",
    "SilentNiscCircuitSetup",
    "GarbledNeuralProtocol",
    "FssOfflinePreprocessing",
    "GazelleHybridHeGarbling",
    "CryptoNetsHomomorphicNeural",
    "ExtendedFssKeys",
    "CrtGarblingGadgetProtocol",
    "FunctionSecretSharingKeys",
    "HalfGatesCircuit",
    "ProtectedDecisionProgram",
]

install_planned_components(globals())
