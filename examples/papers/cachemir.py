"""CacheMir benchmark declaration, not an admitted encrypted decoder.

The reusable component identities below are proposed contracts. Unknown
identities and unsupported parameters must fail normal SDK/native validation;
this example does not register providers or substitute a cleartext executor.
"""


def benchmark_experiment(*, model=None, budget=None):
    """Declare bounded response work; source and numeric admission remain gates."""
    from pllm import (
        ConfigurationError,
        Deployment,
        ExecutionBudget,
        Experiment,
        Model,
        Pipeline,
    )
    from pllm.kernels import Cpu
    from pllm.nonlinear import (
        CalibratedPolynomialExp, StandardGoldschmidtReciprocal, CompositePolynomialActivation,
        InverseRoot, HEMaskedSoftmax, SourceNormalization,
    )
    from pllm.numeric import SourceCalibration, NumericDomain
    from pllm.preparation import HEKeyContext, HECiphertextBirth
    from pllm.protocols import (
        CkksRectangularLinear, InterleavedQK, InterleavedWeightedV, RoutedGQA, HERotaryTransform,
        PrivateTokenSelection, PrivateTokenFeedback, AuthenticatedConfidentialRoleTransport,
    )
    from pllm.roles import ClientKeyServerEvaluationRoles
    from pllm.schedulers import HELevelErrorSchedule
    from pllm.sources import PinnedSemanticSourceBindings
    from pllm.state import InterleavedKVTransaction
    from pllm.sources import ModelSource

    if __package__:
        from .common import MODEL
    else:
        from common import MODEL

    # Shared offline, revision-pinned Qwen is a target, not paper qualification.
    # A Llama proposal does not implicitly qualify Qwen: require pinned semantic
    # bindings and source-qualified GQA/rotary/normalization contracts first.
    model = MODEL if model is None else model
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())

    return Experiment(
        "cachemir",
        Pipeline(
            model=model,
            components={
                "kernels": Cpu(threads=1),
                # Existing CKKS public-matvec references and clear calibration
                # are not an encrypted whole-decoder Executor.
                "linear": CkksRectangularLinear(),
                "attention_qk": InterleavedQK(),
                "attention_value": InterleavedWeightedV(),
                "attention_routing": RoutedGQA(),
                "rotary": HERotaryTransform(),
                "exponential": CalibratedPolynomialExp(
                    coefficients=None, domain=None,
                ),
                # Appendix E specifies standard Goldschmidt. The released
                # adaptive variant conflicts; do not silently substitute it.
                "reciprocal": StandardGoldschmidtReciprocal(
                    iterations=None, domain=None,
                ),
                "softmax": HEMaskedSoftmax(),
                "activation": CompositePolynomialActivation(
                    coefficients=None, domain=None,
                ),
                "inverse_root": InverseRoot(
                    iterations=None, domain=None,
                ),
                "normalization": SourceNormalization(),
                "state": InterleavedKVTransaction(cache_retention=None),
                "selection": PrivateTokenSelection(),
                "feedback": PrivateTokenFeedback(),
                # Calibration extrema are observations, not certificates.
                "calibration": SourceCalibration(),
                "numeric": NumericDomain(parameter_authority=None),
                # None records unresolved coefficients/iterations/domains,
                # prime/key choices and retention; None is NOT authority.
                "keys": HEKeyContext(prime_chain=None, key_parameters=None),
                "ciphertext_birth": HECiphertextBirth(),
                "topology": ClientKeyServerEvaluationRoles(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                "schedule": HELevelErrorSchedule(),
                "source_bindings": PinnedSemanticSourceBindings(),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/cachemir-deployment"),
        ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
        if budget is None else budget,
    )


def main(argv=None):
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "cachemir", argv, task="response", max_input_tokens=32, max_new_tokens=2
    )
    # Parsing (including --help) precedes actual SDK imports and provisioning.
    from pllm import benchmark

    return benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


if __name__ == "__main__":
    main()
