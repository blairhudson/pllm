"""Public protocol component declarations."""

from pllm.components._planned import PendingComponent as PendingMethod, planned as pending
from pllm.protocols.base import ProtocolMethod
from pllm.protocols.masked_linear import MaskedLinear
from pllm.protocols.runtime_arms import (
    BlindedLinear,
    CleartextLinear,
    DirectFHE,
    GuardedLinear,
    SecureLinear,
)

__all__ = [
    "BlindedLinear",
    "CleartextLinear",
    "DirectFHE",
    "GuardedLinear",
    "MaskedLinear",
    "ProtocolMethod",
    "SecureLinear",
]


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
    "ProjectiveLabelConversion", "MosaicMaskedGpuOutsourcing", "MaverickDelegatedLinear",
    "EncryptedSparseVectorCompression", "OpenWeightVerifiableOutsourcing", "BatchedIntervalLookup",
    "BumblebeeTwoPartyTransformer", "DashPreparedArithmeticGarbling", "SecretDualCodeMatVec",
    "MoaiNoninteractiveTransformer", "NexusNoninteractiveTransformer", "TrapdooredMatrixDelegation",
    "ActiveFssProtocol", "ThorHomomorphicTransformer", "BoltHybridTransformer",
    "PrivateEmbeddingLookup", "LogRowGarbledLookup", "NimbusTwoPartyTransformer",
    "OrcaFssGpuExecution", "SigmaFssDecoder", "CipherGptPrivateTokenSelection",
    "FlutePrivateLookup", "GrottoRingDpfProtocol", "PumaThreePartyDecoder",
    "PrimerHomomorphicTransformer", "CheetahTwoPartyNeural", "IronTwoPartyTransformer",
    "PikaRingFssProtocol", "TheXHomomorphicTransformer", "Aby2ArithmeticBooleanConversion",
    "FssMixedFixedPointConversion", "CryptFlowTwoPartyNeural", "MixedArithmeticEdabitConversion",
    "SilentNiscCircuitSetup", "GarbledNeuralProtocol", "FssOfflinePreprocessing",
    "GazelleHybridHeGarbling", "CryptoNetsHomomorphicNeural", "ExtendedFssKeys",
    "CrtGarblingGadgetProtocol", "FunctionSecretSharingKeys", "HalfGatesCircuit",
    "ProtectedDecisionProgram",
]
