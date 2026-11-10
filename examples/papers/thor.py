"""Desired THOR BERT-NSP benchmark composition, not an implemented HE executor.

The proposed primitive identities below still need shared protected bindings,
source ownership, numeric-domain certificates, CKKS scheduling, role transport,
and NSP-only output-policy admission. Existing component/provider guards remain
authoritative: declaring a ComponentRef does not implement its primitive.

The intended shared public API is BenchmarkTarget.forward(experiment), followed
by pllm.benchmark(target, inputs=..., sampling=..., output=...). The current
public benchmark facade accepts response Experiments, not this target overload;
that shared overload remains pending. No paper-specific runner substitutes for
it here. Existing source-FP32 forward support is only a foothold, not protected
HE execution or qualification of the NSP output policy.

Source semantics: word + position + segment embeddings -> LayerNorm -> repeated
bidirectional QK / scaled masked softmax / AV -> attention projection, residual,
LayerNorm -> FFN, source GELU, projection, residual, LayerNorm -> CLS -> pooler
dense, source tanh -> cls.seq_relationship -> two NSP logits. No autoregressive
decoding, token selection, opaque decoder, or whole-encoder component is used.

Run directly or as a module with --model-path pointing to an offline checkpoint
whose config explicitly declares model_type='bert' and architecture
'BertForNextSentencePrediction'. --forward-inputs supplies a mapping JSON file
containing source input tensors (including attention mask and segment IDs).
Checkpoint semantics must never be replaced by architecture overrides. Source
parameters, approximation coefficients, scales, and epsilon have no invented
defaults here; protected admission must bind and verify their source authority.
"""

from __future__ import annotations


def benchmark_experiment(*, model=None, budget=None):
    """Declare THOR primitives; current shared admission may reject pending IDs.

    ForwardBudget bounds accepted attempts and sequence length, not batch size.
    Default workload is one attempt, batch-one inputs, at most 32 input tokens;
    batch-one tensor geometry belongs to the supplied forward input mapping.
    """
    from pllm import (
        ConfigurationError,
        Deployment,
        Experiment,
        ForwardBudget,
        Model,
        Pipeline,
    )
    from pllm.kernels import Cpu
    from pllm.nonlinear import HELayerNorm, HESquareNormalizeSoftmax, HEPolynomialActivation
    from pllm.numeric import SourceNumericDomainCertificates
    from pllm.preparation import HEKeyContext, HECiphertextBirth
    from pllm.protocols import (
        HESourceEmbeddingPositionSegment, HEAttentionProducts, HEHeadScaleMask,
        HEAttentionValue, HEPackedLinear, HEResidualAdd, HEClsPooler,
        HENspTwoLogitHead, HEBootstrap, NspLogitsOnlyOutputPolicy,
        AuthenticatedConfidentialRoleTransport,
    )
    from pllm.roles import ClientKeyServerEvaluationRoles
    from pllm.schedulers import HELevelErrorSchedule
    from pllm.sources import PinnedSemanticSourceBindings
    from pllm.sources import ModelSource

    if model is None:
        raise ConfigurationError("THOR requires an offline BERT NSP checkpoint")
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    if not model.local_files_only or model.kind not in {"huggingface", "safetensors"}:
        raise ConfigurationError("THOR requires an offline BERT NSP checkpoint via Model.path")
    if budget is None:
        budget = ForwardBudget(requests=1, max_input_tokens=32)
    if not isinstance(budget, ForwardBudget):
        raise ConfigurationError("THOR requires ForwardBudget, not a generation budget")

    return Experiment(
        "thor",
        Pipeline(
            model=model,
            components={
                "embedding": HESourceEmbeddingPositionSegment(),
                "layernorm": HELayerNorm(),
                "attention_products": HEAttentionProducts(),
                "head_scale_mask": HEHeadScaleMask(),
                "softmax": HESquareNormalizeSoftmax(),
                "attention_value": HEAttentionValue(),
                "linear": HEPackedLinear(),
                "residual": HEResidualAdd(),
                "activation": HEPolynomialActivation(function="source-gelu"),
                "pooler": HEClsPooler(),
                "pooler_activation": HEPolynomialActivation(function="source-tanh"),
                "head": HENspTwoLogitHead(),
                "bootstrap": HEBootstrap(),
                "key_context": HEKeyContext(),
                "ciphertext_birth": HECiphertextBirth(),
                "schedule": HELevelErrorSchedule(),
                "roles": ClientKeyServerEvaluationRoles(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                # Future requirements, not authoritative metadata or capabilities.
                # The shared source/import owner must validate under admission.
                "source_bindings": PinnedSemanticSourceBindings(
                    required_model_type="bert",
                    required_architectures=("BertForNextSentencePrediction",),
                    required_output_logits=2,
                ),
                "numeric_certificates": SourceNumericDomainCertificates(
                    parameter_authority=None,
                ),
                "output_policy": NspLogitsOnlyOutputPolicy(),
                "kernels": Cpu(threads=1),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/thor-deployment"),
        budget,
    )


def main(argv=None):
    """Use the desired shared forward-target overload once available."""
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "thor", argv, task="forward", max_input_tokens=32, max_new_tokens=0
    )

    import pllm
    from pllm.metrics import BenchmarkTarget

    target = BenchmarkTarget.forward(benchmark_experiment(model=args.model, budget=args.budget))
    return pllm.benchmark(target, inputs=args.inputs, sampling=args.sampling, output=args.output)


if __name__ == "__main__":
    main()
