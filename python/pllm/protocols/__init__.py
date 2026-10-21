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
from pllm.protocols.packed_encrypted import PackedEncryptedLinear
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
    "PackedEncryptedLinear",
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


class OpenWeightVerifiableOutsourcing(ProtocolMethod):
    """Bounded experimental GKM delegation with source arithmetic.

    Linear weights use nearest-even i16/Q11 encoding and ring64 products;
    encoded inputs have absolute bound 2**31. Private lookup returns the
    original float32 token row. Greedy execution
    has a fixed request length, including EOS, with full client-owned KV.
    Native source/workload admission and request-end verification are required.
    Seeded sharing gives worker One fresh, purpose-separated AES window streams;
    worker Zero receives complementary shares. Its privacy assumes the AES PRG.
    Optional source_stage_batch_v1 execution requires separate native stage,
    aggregate-frame and outer-copy admission; omission retains scalar execution.
    This declaration does not certify full-size execution or cryptographic security.
    """

    __slots__ = ()
    NUMERIC_PROFILE = (
        "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22."
        "le-finite-f32.sha256.secp256k1.v1"
    )
    LOCAL_NUMERIC_PROFILE = "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
    descriptor = ComponentDescriptor(
        component="pllm/open-weight-verifiable-outsourcing/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/protocol-method", category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={
            "type": "object", "additionalProperties": False,
            "properties": {
                "sharing": {"type": "string", "enum": ["full", "seeded"]},
                "generation": {"type": "string", "const": "fixed_length_greedy"},
                "capacity": {"type": "string", "const": "decoder"},
                "execution": {"type": "string", "const": "source_stage_batch_v1"},
            },
            "required": ["sharing", "generation"],
        },
        capabilities=(
            "bounded-experimental-gkm", "source-i16-q11-ring64",
            "private-original-float32-lookup", "full-client-kv",
            "fixed-length-greedy", "full-input-shares", "seeded-input-shares",
        ),
        required_host_features=("native-gkm-decoder", "authenticated-confidential-channels"),
        role_eligibility=("client", "worker_a", "worker_b"),
    )

    def __init__(self, *, sharing: Literal["full", "seeded"] = "full",
                   generation: Literal["fixed_length_greedy"] = "fixed_length_greedy",
                   capacity: Literal["capsule", "decoder"] = "capsule",
                   execution: Literal["source_stage_batch_v1"] | None = None) -> None:
        if type(sharing) is not str or sharing not in {"full", "seeded"}:
            raise ConfigurationError("GKM remote sharing must be full or seeded")
        if type(generation) is not str or generation != "fixed_length_greedy":
            raise ConfigurationError("GKM generation must be fixed_length_greedy")
        if type(capacity) is not str or capacity not in {"capsule", "decoder"}:
            raise ConfigurationError("GKM capacity must be capsule or decoder")
        if execution is not None and (
                type(execution) is not str or execution != "source_stage_batch_v1"):
            raise ConfigurationError("GKM execution must be source_stage_batch_v1 or omitted")
        super().__init__(self.descriptor.component, {"sharing": sharing, "generation": generation,
            **({"capacity": capacity} if capacity != "capsule" else {}),
            **({"execution": execution} if execution is not None else {})})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {**self.params, "capacity": self.params.get("capacity", "capsule"),
            "execution": self.params.get("execution")}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


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


class FuseFssSourceDecoder(ProtocolMethod):
    """Public-weight ring50/Q12 contract, not executable capability admission.

    Original learned-position and adapted dense-RoPE arithmetic have distinct
    identities. The caller must select one; neither is floating-point parity.
    The opt-in checked-range identity retains D13 arithmetic on accepted requests;
    native membership rejection is whole-request, after the complete schedule and
    final mask, without a failure location.
    Material delivery is an independent transport choice; batch-v1 retains the
    same material and numerical contract and requires native queue admission.
    """

    __slots__ = ()
    GPT2_NUMERIC_PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"
    DENSE_ROPE_NUMERIC_PROFILE = "fusefss.dense-rope-ring50-q12.v1"
    DENSE_ROPE_CHECKED_NUMERIC_PROFILE = "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
    NUMERIC_PROFILES = (GPT2_NUMERIC_PROFILE, DENSE_ROPE_NUMERIC_PROFILE, DENSE_ROPE_CHECKED_NUMERIC_PROFILE)
    MATERIAL_DELIVERIES = ("single", "batch-v1")
    ONLINE_SCHEDULES = ("serial", "wavefront-v1", "tensor-v1")
    RING_BITS = 50
    FRACTIONAL_BITS = 12
    descriptor = ComponentDescriptor(
        component="pllm/fusefss-source-decoder/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocol-method", category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False,
            "required": ["numeric_profile"], "properties": {
                 "numeric_profile": {"type": "string", "enum": list(NUMERIC_PROFILES)},
                 "weights": {"type": "string", "const": "public"},
                 "capacity": {"type": "string", "enum": ["decoder"]},
                  "material_delivery": {"type": "string", "enum": ["batch-v1"]},
                  "online_schedule": {"type": "string", "enum": ["wavefront-v1", "tensor-v1"]}}},
        capabilities=("public-weight-source-decoder", "ring50-q12", "external-nearest-even",
                      "internal-signed-floor", "protected-full-prefix-recomputation",
                       "private-greedy-feedback", "final-only-token-ids", "opt-in-checked-range-whole-request-outcome"),
        required_host_features=("native-fss-decoder-admission", "independent-trusted-dealer",
                                "distinct-authenticated-confidential-role-channels"),
        role_eligibility=("client", "dealer", "worker_a", "worker_b"),
    )

    def __init__(self, *, numeric_profile: Literal[
            "fusefss.gpt2-complete-ring50-q12.v1", "fusefss.dense-rope-ring50-q12.v1",
            "fusefss.dense-rope-checked-ranges-ring50-q12.v1"],
             weights: Literal["public"] = "public", capacity: Literal["capsule", "decoder"] = "capsule",
              material_delivery: Literal["single", "batch-v1"] = "single",
              online_schedule: Literal["serial", "wavefront-v1", "tensor-v1"] = "serial") -> None:
        if type(numeric_profile) is not str or numeric_profile not in self.NUMERIC_PROFILES:
            raise ConfigurationError("FuseFSS numeric_profile must explicitly select a complete-decoder identity")
        if type(weights) is not str or weights != "public":
            raise ConfigurationError("FuseFSS source decoder supports public weights only")
        if type(capacity) is not str or capacity not in {"capsule", "decoder"}:
            raise ConfigurationError("FuseFSS capacity must be capsule or decoder")
        if type(material_delivery) is not str or material_delivery not in self.MATERIAL_DELIVERIES:
            raise ConfigurationError("FuseFSS material_delivery must be single or batch-v1")
        if type(online_schedule) is not str or online_schedule not in self.ONLINE_SCHEDULES:
            raise ConfigurationError("FuseFSS online_schedule must be serial, wavefront-v1 or tensor-v1")
        params = {"numeric_profile": numeric_profile}
        if capacity != "capsule":
            params["capacity"] = capacity
        if material_delivery != "single":
            params["material_delivery"] = material_delivery
        if online_schedule != "serial":
            params["online_schedule"] = online_schedule
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"numeric_profile": self.params["numeric_profile"], "weights": "public",
                "capacity": self.params.get("capacity", "capsule"),
                "material_delivery": self.params.get("material_delivery", "single"),
                "online_schedule": self.params.get("online_schedule", "serial")}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PumaThreePartyDecoder(ProtocolMethod):
    """Bounded source-derived Q18 decoder with private feedback.

    Public weights optionally use checked ranges, tiled lookup, a bounded complete
    stable-argmax selection template, and persistent peer streams. Tiled selection
    never prunes vocabulary entries and is independent of lookup tiling.
    """

    __slots__ = ()
    NUMERIC_PROFILE = "puma.source-q18.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1"
    PUBLIC_NUMERIC_PROFILE = "puma.public-q18.rmsnorm-epsilon1.full-prefix.batched-final-greedy-ids.v1"
    CHECKED_NUMERIC_PROFILE = "puma.public-q18.checked-range.rmsnorm-epsilon1.full-prefix.final-greedy-ids.v1"
    descriptor = ComponentDescriptor(
        component="pllm/puma-three-party-decoder/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocol-method", category_version="1",
        lifecycle_phase="offline+online", parameter_schema={"type": "object", "additionalProperties": False,
            "properties": {"weights": {"type": "string", "const": "public"},
                "range_policy": {"type": "string", "enum": ["checked"]},
                "max_abs_activation": {"type": "integer", "minimum": 1, "maximum": 1024},
                "lookup": {"type": "string", "enum": ["tiled", "trie"]},
                "selection": {"type": "string", "enum": ["tiled"]},
                "peer_transport": {"type": "string", "enum": ["streams"]}}},
        capabilities=("bounded-replicated-q18", "protected-full-prefix-recomputation", "private-greedy-feedback",
                      "public-checked-range-policy", "public-tiled-lookup", "public-trie-lookup", "public-tiled-stable-argmax", "bounded-peer-streams"),
        required_host_features=("native-replicated-decoder", "authenticated-confidential-channels"),
        role_eligibility=("client", "worker_a", "worker_b", "worker_c"),
    )

    def __init__(self, *, weights: Literal["private", "public"] = "private",
                 range_policy: Literal["static", "checked"] = "static", max_abs_activation: int = 256,
                 lookup: Literal["per_token", "tiled", "trie"] = "per_token",
                 selection: Literal["expanded", "tiled"] = "expanded",
                 peer_transport: Literal["http", "streams"] = "http"):
        if type(weights) is not str or weights not in {"private", "public"}:
            raise ConfigurationError("PUMA weights must be private or public")
        if (type(range_policy) is not str or range_policy not in {"static", "checked"}
                or type(max_abs_activation) is not int or not 1 <= max_abs_activation <= 1024
                or range_policy == "static" and max_abs_activation != 256
                or type(lookup) is not str or lookup not in {"per_token", "tiled", "trie"}
                or type(selection) is not str or selection not in {"expanded", "tiled"}
                or type(peer_transport) is not str or peer_transport not in {"http", "streams"}
                or weights == "private" and (range_policy != "static" or lookup != "per_token"
                                             or selection != "expanded" or peer_transport != "http")):
            raise ConfigurationError("invalid PUMA public execution policy")
        params = {"weights": weights} if weights == "public" else {}
        if range_policy == "checked":
            params.update(range_policy=range_policy, max_abs_activation=max_abs_activation)
        if lookup != "per_token":
            params["lookup"] = lookup
        if selection != "expanded":
            params["selection"] = selection
        if peer_transport != "http":
            params["peer_transport"] = peer_transport
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep=True):
        return {"weights": self.params.get("weights", "private"),
                "range_policy": self.params.get("range_policy", "static"),
                "max_abs_activation": self.params.get("max_abs_activation", 256),
                "lookup": self.params.get("lookup", "per_token"),
                "selection": self.params.get("selection", "expanded"),
                "peer_transport": self.params.get("peer_transport", "http")}

    @classmethod
    def describe(cls):
        return cls.descriptor


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
    "FuseFssSourceDecoder",
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
