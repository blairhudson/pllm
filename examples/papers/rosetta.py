"""Rosetta benchmark-first declaration of desired reusable HE primitives.

This is a target composition, not an implemented or qualified HE deployment.
The common pinned offline Qwen source is a benchmark target; Rosetta compatibility
is not assumed. Existing compilation/admission guards must refuse unsupported
components, source operations, numeric domains, and role placements.

Source context only: R=65536, LWE dimension 1024, ring-switch dimension 2048.
These are not adopted defaults. Exact primes, scales, noise bounds, comparison
floor/endpoints, and the 53-bit port remain unresolved. No parameter authority is
invented here; None authority must remain blocked. The 32-/64-bit torus reference
profiles are reference-only, not authority for Rosetta's unresolved 53-bit port.
Coefficient extraction or a clear torus oracle is not encrypted
HomComp; encrypted comparison and accumulator evaluation require their own
implementations and evidence.
"""

from pllm import (
    ConfigurationError,
    Deployment,
    ExecutionBudget,
    Experiment,
    Model,
    Pipeline,
    benchmark,
)
from pllm.kernels import Cpu
from pllm.nonlinear import (
    LWEHighPrecisionComparison, RLWEEncryptedAccumulatorSegmentedAffine, HEEvalMod,
    HENormalization, HESoftmaxSquareNormalize, HEActivationBridge,
)
from pllm.numeric import SourceNumericDomainCertificates
from pllm.preparation import HEKeyGraph, HECiphertextBirth
from pllm.protocols import (
    AuthenticatedConfidentialRoleTransport, AuthenticatedClientInputExport,
    HEPackedLinear, HEResidualAdd, HEEncryptedMultiplyReduce,
    HESlotCoefficientTransform, HERingSwitch, HECoefficientExtraction,
    HEKeySwitch, HEModulusRaise, HECoefficientSlotTransform,
    PrivateTokenSelection, PrivateTokenFeedback,
)
from pllm.roles import ClientKeyServerConfidentialWeightRoles
from pllm.schedulers import HELevelErrorSchedule, HELevelSchemePlacement
from pllm.sources import AuthenticatedSourceBindings
from pllm.state import EncryptedKVTransaction
from pllm.sources import ModelSource

if __package__:
    from .common import MODEL, benchmark_arguments
else:
    from common import MODEL, benchmark_arguments


def benchmark_experiment(*, model=None, budget=None):
    """Expose primitive requirements backwards from the public benchmark API."""
    model = MODEL if model is None else model
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    return Experiment(
        "rosetta",
        Pipeline(
            model=normalized_model,
            components={
                "kernels": Cpu(threads=1),
                # Server-owned weights remain confidential; client holds HE secrets.
                "roles": ClientKeyServerConfidentialWeightRoles(
                    secret_key_owner="client", weight_owner="server",
                ),
                "transport": AuthenticatedConfidentialRoleTransport(roles="roles"),
                "input": AuthenticatedClientInputExport(
                    roles="roles", transport="transport", birth="ciphertext_birth",
                ),
                "source_bindings": AuthenticatedSourceBindings(
                    weight_owner="server", bind_checkpoint_and_numeric_domain=True,
                ),
                "key_graph": HEKeyGraph(
                    secret_key_owner="client", server_secret_keys=False,
                ),
                "ciphertext_birth": HECiphertextBirth(
                    key_graph="key_graph", numeric="numeric",
                ),
                "numeric": SourceNumericDomainCertificates(
                    parameter_authority=None, source="rosetta-53-bit-port-unresolved",
                ),
                "level_error_schedule": HELevelErrorSchedule(
                    numeric="numeric", key_graph="key_graph",
                ),
                "placement": HELevelSchemePlacement(
                    schedule="level_error_schedule", roles="roles",
                ),
                # Separate operation bindings, not an opaque decoder/attention region.
                "linear": HEPackedLinear(
                    weight_owner="server", source_bindings="source_bindings",
                ),
                "output_head": HEPackedLinear(
                    operation="source-output-head", vocabulary="full",
                    weight_owner="server", source_bindings="source_bindings",
                    selection="selection",
                ),
                "residual": HEResidualAdd(),
                "attention_scores": HEEncryptedMultiplyReduce(
                    operation="query-key-dot-product", numeric="numeric",
                ),
                "attention_values": HEEncryptedMultiplyReduce(
                    operation="probability-value-reduction", numeric="numeric",
                ),
                "normalization_square_reduce": HEEncryptedMultiplyReduce(
                    operation="normalization-square-reduction", numeric="numeric",
                ),
                "normalization": HENormalization(
                    square_reduce="normalization_square_reduce", inverse_sqrt="segmented_affine",
                    to_coefficients="slot_coefficient_transform", ring_switch="ring_switch",
                    extract="coefficient_extraction", compare="comparison", evaluate="segmented_affine",
                    key_switch="key_switch", modulus_raise="modulus_raise",
                    to_slots="coefficient_slot_transform", evalmod="evalmod",
                    multiply="normalization_multiply", numeric="numeric",
                ),
                "normalization_multiply": HEEncryptedMultiplyReduce(
                    operation="normalization-elementwise-product", numeric="numeric",
                ),
                "softmax": HESoftmaxSquareNormalize(
                    square_reduce="softmax_square_reduce", reciprocal="segmented_affine",
                    to_coefficients="slot_coefficient_transform", ring_switch="ring_switch",
                    extract="coefficient_extraction", compare="comparison", evaluate="segmented_affine",
                    key_switch="key_switch", modulus_raise="modulus_raise",
                    to_slots="coefficient_slot_transform", evalmod="evalmod",
                    multiply="softmax_multiply", numeric="numeric",
                ),
                "softmax_square_reduce": HEEncryptedMultiplyReduce(
                    operation="softmax-square-and-row-sum", numeric="numeric",
                ),
                "softmax_multiply": HEEncryptedMultiplyReduce(
                    operation="softmax-row-normalization", numeric="numeric",
                ),
                "activation": HEActivationBridge(
                    to_coefficients="slot_coefficient_transform", ring_switch="ring_switch",
                    extract="coefficient_extraction", compare="comparison", evaluate="segmented_affine",
                    key_switch="key_switch", modulus_raise="modulus_raise",
                    to_slots="coefficient_slot_transform", evalmod="evalmod", numeric="numeric",
                ),
                "activation_gate": HEEncryptedMultiplyReduce(
                    operation="activation-gated-product", numeric="numeric",
                ),
                "slot_coefficient_transform": HESlotCoefficientTransform(numeric="numeric"),
                "ring_switch": HERingSwitch(
                    key_graph="key_graph", numeric="numeric",
                ),
                "coefficient_extraction": HECoefficientExtraction(numeric="numeric"),
                "comparison": LWEHighPrecisionComparison(
                    key_graph="key_graph", numeric="numeric",
                ),
                "segmented_affine": RLWEEncryptedAccumulatorSegmentedAffine(
                    comparison="comparison", key_graph="key_graph", numeric="numeric",
                ),
                "key_switch": HEKeySwitch(key_graph="key_graph", numeric="numeric"),
                "modulus_raise": HEModulusRaise(numeric="numeric"),
                "coefficient_slot_transform": HECoefficientSlotTransform(numeric="numeric"),
                "evalmod": HEEvalMod(numeric="numeric"),
                "state": EncryptedKVTransaction(
                    birth="ciphertext_birth", schedule="level_error_schedule",
                ),
                "selection": PrivateTokenSelection(
                    comparison="comparison", roles="roles",
                ),
                "feedback": PrivateTokenFeedback(
                    selection="selection", input="input", state="state",
                ),
                # Input/checkpoint authentication is not result verification.
                # No output-integrity claim or substitute verification component.
            },
        ),
        Deployment.local(root="target/paper-benchmarks/rosetta-deployment"),
        ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
        if budget is None else budget,
    )


def main(argv=None):
    args = benchmark_arguments(
        "rosetta", argv, task="response", max_input_tokens=32, max_new_tokens=2,
    )
    return benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


if __name__ == "__main__":
    main()
