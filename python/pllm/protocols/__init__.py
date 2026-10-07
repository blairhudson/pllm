"""Public protocol component declarations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pllm.components._planned import (
    PendingComponent as PendingMethod,
    install_planned_components,
    planned as pending,
)
from pllm.protocols.base import ProtocolMethod
from pllm.configuration import ComponentDescriptor, ConfigurationError
from pllm.components._q7_activation import parameter_schema as _q7_schema, parameters as _q7_parameters
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


class MaverickDelegatedLinear(ProtocolMethod):
    """Bounded RAA masked-matvec benchmark with an uncertified coded check.

    Public preprocessing stays client-local. Code distance and dual-LPN privacy
    remain unproved; this component cannot select decoder execution.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/maverick-delegated-linear/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocol",
        category_version="1", lifecycle_phase="runtime",
        parameter_schema={"type": "object", "additionalProperties": False,
            "properties": {"public_code_seed": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
            "required": ["public_code_seed"]},
        capabilities=("raa-delegation-reference", "bounded-operator-benchmark"),
        required_host_features=("native-core",), role_eligibility=("client", "inference"),
    )

    def __init__(self, *, public_code_seed: str = "0d" * 32) -> None:
        if (type(public_code_seed) is not str or len(public_code_seed) != 64
                or any(c not in "0123456789abcdef" for c in public_code_seed)):
            raise ConfigurationError("public_code_seed must be 64 lowercase hexadecimal characters")
        super().__init__(self.descriptor.component, {"public_code_seed": public_code_seed})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


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


class PrivateEmbeddingLookup(ProtocolMethod):
    """Bounded BFV coefficient lookup and protected packed-share conversion.

    Benchmarks caller-owned signed integer tables. Optional interleaving and
    party-local conversion retain explicit local-dealer/adaptation contracts.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/private-embedding-lookup/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocol",
        category_version="1", lifecycle_phase="runtime",
        parameter_schema={"type": "object", "additionalProperties": False,
            "properties": {"vocabulary_size": {"type": "integer", "minimum": 1, "maximum": 4096},
                 "precision": {"enum": ["int8", "int4", "mixed_4_3_3"]},
                  "repacking": {"enum": ["none", "interleave", "pack_lwes"]},
                 "conversion": {"enum": ["local", "party_local"]}},
            "required": ["vocabulary_size", "precision"]},
        capabilities=("private-packed-embedding-reference", "bounded-operator-benchmark"),
        required_host_features=("native-core", "seal-bfv", "local-garbler"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, *, vocabulary_size: int = 64,
                  precision: Literal["int8", "int4", "mixed_4_3_3"] = "mixed_4_3_3",
                  repacking: Literal["none", "interleave", "pack_lwes"] = "none",
                  conversion: Literal["local", "party_local"] = "local") -> None:
        if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= 4096:
            raise ConfigurationError("embedding vocabulary_size must be an integer in [1, 4096]")
        if precision not in ("int8", "int4", "mixed_4_3_3"):
            raise ConfigurationError("unsupported embedding coefficient precision")
        if repacking not in ("none", "interleave", "pack_lwes") or conversion not in ("local", "party_local"):
            raise ConfigurationError("unsupported embedding repacking or conversion")
        params = {"vocabulary_size": vocabulary_size, "precision": precision}
        if repacking != "none": params["repacking"] = repacking
        if conversion != "local": params["conversion"] = conversion
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"repacking": "none", "conversion": "local", **self.to_spec()["params"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class LogRowGarbledLookup(ProtocolMethod):
    """One-use LogRow lookup of a public-calibrated signed-Q7 SiLU profile.

    Uses the bounded 512-row reference and earlier two-row one-hot setup.
    Complete evaluator material includes masked table bits. Input/output
    labels stay opaque until the trusted client decodes the output.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/logrow-garbled-lookup-q7/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocol",
        category_version="1", lifecycle_phase="runtime", parameter_schema=_q7_schema(),
        capabilities=("protected-table-lookup-q7", "bounded-operator-benchmark"),
        required_host_features=("native-core", "local-garbler"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, *, max_pieces: int = 4,
                 public_calibration_counts: Sequence[int] | None = None) -> None:
        super().__init__(self.descriptor.component, _q7_parameters(max_pieces, public_calibration_counts))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


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
