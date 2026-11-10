"""Inert benchmark wiring regressions; capture is NOT binding/execution evidence.

Installed pllm is authoritative. Only the repository root is added for examples.
All component constructors are genuine; only Pipeline/Experiment are captured.
They do not validate composition admission, source binding, or native execution.
"""

import builtins
import importlib
from importlib.abc import MetaPathFinder
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPERS = (
    "gkm", "puma", "fusefss", "sigma", "maverick", "bumblebee",
    "nexus", "thor", "cachemir", "rosetta",
)
FORWARD = ("nexus", "thor")
RESPONSE = tuple(paper for paper in PAPERS if paper not in FORWARD)
TOKEN_DEFAULTS = {"fusefss": (15, 1), "maverick": (8, 8)}
PRIMITIVES = {
    "gkm": {
        "linear": "pllm/open-weight-verifiable-outsourcing/v1",
        "rmsnorm": "pllm/f32-rmsnorm/v1",
        "attention": "pllm/f32-causal-attention/v1",
        "selection": "pllm/greedy-selection/v1",
    },
    "puma": {
        "linear": "pllm/replicated-public-matmul-full-sum/v1",
        "activation": "pllm/q18-piecewise-silu-source-order/v1",
        "normalization": "pllm/q18-rmsnorm-epsilon1/v1",
        "selection": "pllm/replicated-complete-stable-argmax/v1",
    },
    "fusefss": {
        "linear": "pllm/fss-public-dot/v1",
        "rescale": "pllm/fss-signed-trunc-floor/v1",
        "activation": "pllm/fss-silu/v1",
        "selection": "pllm/fss-argmax/v1",
    },
    "sigma": {
        "public_dot": "pllm/ring-public-coefficient-dot/v1",
        "gapars": "pllm/same-ring-gapars/v1",
        "attention": "pllm/causal-max-nexp-reciprocal-attention/v1",
        "private_feedback": "pllm/stable-private-argmax-embedding/v1",
    },
    "maverick": {
        "linear": "pllm/public-field-matvec/v1",
        "verification": "pllm/two-level-coded-batch-verification/v1",
        "activation": "pllm/f32-activation/v1",
        "selection": "pllm/greedy-selection/v1",
    },
    "bumblebee": {
        "linear": "pllm/coefficient-olt/v1",
        "softmax": "pllm/shifted-clip-repeated-square-softmax/v1",
        "selection": "pllm/shared-stable-argmax/v1",
        "state": "pllm/shared-full-prefix-state/v1",
    },
    "nexus": {
        "linear": "pllm/he-commutable-linear/v1",
        "softmax": "pllm/he-repeated-squaring-softmax/v1",
        "selection": "pllm/he-fold-selection/v1",
        "output": "pllm/classifier-output-policy/v1",
    },
    "thor": {
        "linear": "pllm/he-packed-linear/v1",
        "activation": "pllm/he-polynomial-activation/v1",
        "head": "pllm/he-nsp-two-logit-head/v1",
        "output_policy": "pllm/nsp-logits-only-output-policy/v1",
    },
    "cachemir": {
        "linear": "pllm/ckks-rectangular-linear/v1",
        "activation": "pllm/composite-polynomial-activation/v1",
        "state": "pllm/interleaved-kv-transaction/v1",
        "selection": "pllm/private-token-selection/v1",
    },
    "rosetta": {
        "linear": "pllm/he-packed-linear/v1",
        "comparison": "pllm/lwe-high-precision-comparison/v1",
        "state": "pllm/encrypted-kv-transaction/v1",
        "selection": "pllm/private-token-selection/v1",
    },
}
LEGACY_WHOLE_COMPONENTS = {
    "gkm": {"pllm/client-f32-decoder-regions/v1"},
    "puma": {"pllm/puma-three-party-decoder/v1"},
    "fusefss": {"pllm/fusefss-source-decoder/v1"},
}


@pytest.fixture
def sdk(monkeypatch):
    class NoExecutionImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            blocked = (
                "pllm._native", "pllm_native", "pllm.model_loader",
                "pllm.runtime.delegated_source", "pllm.runtime.fss_source",
                "pllm.runtime.replicated_source", "pllm.runtime.forward_binding",
                "torch", "transformers", "huggingface_hub",
            )
            if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                raise AssertionError(f"inert benchmark test attempted execution import: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [NoExecutionImports(), *sys.meta_path])
    monkeypatch.syspath_prepend(str(ROOT))
    import pllm
    from pllm import configuration
    from pllm import metrics
    from pllm.components import list_component_classes, planned_components

    # Initialize genuine SDK metadata before factory source-I/O sentinels close.
    # Otherwise an isolated node can mistake lazy catalog/module reads for
    # factory model access. This does not resolve, provision, or run components.
    planned_components()
    list_component_classes()

    forbidden = Mock(side_effect=AssertionError("model/source/native/provision execution forbidden"))
    # Set lazy exports without importing their implementation modules.
    for name in ("benchmark", "compile", "load_model", "resolve_model"):
        monkeypatch.setitem(pllm.__dict__, name, forbidden)
    monkeypatch.setattr(configuration.Experiment, "resolve", forbidden)
    for name in ("pllm.model_loader", "pllm.runtime.forward_binding"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            for entry in ("resolve_model", "load_model", "compile_forward_model"):
                if entry in vars(loaded):
                    monkeypatch.setattr(loaded, entry, forbidden)
    yield SimpleNamespace(
        pllm=pllm, configuration=configuration, metrics=metrics,
        Model=configuration.Model, ComponentRef=configuration.ComponentRef,
        ExecutionBudget=configuration.ExecutionBudget,
        ForwardBudget=configuration.ForwardBudget, forbidden=forbidden,
    )
    # Sentinel exceptions are failures, never successful SDK refusals.
    forbidden.assert_not_called()


@pytest.fixture
def papers(sdk):
    return {name: importlib.import_module(f"examples.papers.{name}") for name in PAPERS}


@pytest.fixture
def capture(monkeypatch, sdk, papers):
    def pipeline_capture(model, *, components, profile=None):
        return SimpleNamespace(model=model, components=components, profile=profile)

    def experiment_capture(name, pipeline, deployment, budget):
        return SimpleNamespace(name=name, pipeline=pipeline, deployment=deployment, budget=budget)

    pipeline = Mock(side_effect=pipeline_capture)
    experiment = Mock(side_effect=experiment_capture)
    for owner in (sdk.pllm, sdk.configuration, *papers.values()):
        for name, replacement in (
            ("Pipeline", pipeline), ("Experiment", experiment),
        ):
            # No ComponentRef or family constructor is ever replaced.
            # Covers both imported aliases and factory-local lazy imports.
            if owner in (sdk.pllm, sdk.configuration) or name in vars(owner):
                monkeypatch.setattr(owner, name, replacement)
    return SimpleNamespace(
        pipeline=pipeline, experiment=experiment,
    )


@pytest.fixture
def no_source_io(monkeypatch, sdk, papers):
    forbidden = Mock(side_effect=AssertionError("factory attempted source IO"))
    for name in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    yield forbidden
    forbidden.assert_not_called()


@pytest.fixture
def main_spies(monkeypatch, sdk, papers):
    common = importlib.import_module("examples.papers.common")

    def install(paper):
        module = papers[paper]
        args = SimpleNamespace(**{name: object() for name in (
            "model", "budget", "inputs", "sampling", "scenario", "output",
        )})
        parser = Mock(return_value=args)
        factory = Mock(return_value=object())
        benchmark = Mock(return_value=object())
        forward = Mock(return_value=object())
        monkeypatch.setattr(common, "benchmark_arguments", parser)
        if "benchmark_arguments" in vars(module):
            monkeypatch.setattr(module, "benchmark_arguments", parser)
        monkeypatch.setattr(module, "benchmark_experiment", factory)
        monkeypatch.setattr(sdk.pllm, "benchmark", benchmark)
        if "benchmark" in vars(module):
            monkeypatch.setattr(module, "benchmark", benchmark)
        monkeypatch.setattr(sdk.metrics.BenchmarkTarget, "forward", forward)
        return SimpleNamespace(
            module=module, args=args, parser=parser, factory=factory,
            benchmark=benchmark, forward=forward,
        )

    return install


@pytest.mark.parametrize("paper", PAPERS)
def test_public_factory_and_main_signatures(papers, paper):
    factory = inspect.signature(papers[paper].benchmark_experiment)
    assert tuple(factory.parameters) == ("model", "budget")
    for parameter in factory.parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    main = inspect.signature(papers[paper].main)
    assert tuple(main.parameters) == ("argv",)
    assert main.parameters["argv"].default is None


@pytest.mark.parametrize("paper", PAPERS)
def test_factory_explicit_primitive_capture_not_admission(sdk, papers, capture, tmp_path, paper, no_source_io):
    assert sdk.configuration.ComponentRef is sdk.ComponentRef
    model = sdk.Model.path(str(tmp_path))  # Empty directory: declaration only.
    declaration = papers[paper].benchmark_experiment(model=model)
    assert declaration.pipeline.model is model
    assert declaration.pipeline.profile is None
    components = declaration.pipeline.components
    assert all(isinstance(ref, sdk.ComponentRef) and type(ref) is not sdk.ComponentRef
               for ref in components.values())
    if paper in ("gkm", "maverick"):
        assert len(components) == (17 if paper == "gkm" else 23)
        assert all(isinstance(ref, sdk.ComponentRef) for ref in components.values())
        from pllm.nonlinear import F32Activation
        from pllm.operators import (
            F32RmsNorm, F32Rotary, F32CausalAttention, F32Multiply,
            F32AddBias, F32Add, FullVocabularyHead, GreedySelection,
        )

        # Real typed declarations, never constructor replacements. These
        # manual maps preserve each factory's exact historical wire parameters.
        expected = {
            "rmsnorm": (F32RmsNorm, "pllm/f32-rmsnorm/v1", {"epsilon": "source"}),
            "rotary": (F32Rotary, "pllm/f32-rotary/v1", {"configuration": "source"}),
            "attention": (F32CausalAttention, "pllm/f32-causal-attention/v1", {"heads": "source", "kv": "state"}),
            "activation": (F32Activation, "pllm/f32-activation/v1", {"function": "source"}),
            "multiply": (F32Multiply, "pllm/f32-multiply/v1", {}),
            "bias": (F32AddBias, "pllm/f32-add-bias/v1", {"placement": "after-reconstruction"}),
            "add": (F32Add, "pllm/f32-add/v1", {}),
            "head": (FullVocabularyHead, "pllm/full-vocabulary-head/v1", {"linear": "linear"}),
            "selection": (GreedySelection, "pllm/greedy-selection/v1", {"eos": "continue"}),
        }
        for slot, (cls, identity, params) in expected.items():
            assert issubclass(cls, sdk.ComponentRef)
            if paper == "maverick":
                params = {**params, "execution_owner": "client"}
            elif slot == "head":
                params = {**params, "rows": "last-prefill-and-each-decode"}
            elif slot == "selection":
                params = {**params, "logits": "complete"}
            assert type(components[slot]) is cls
            assert components[slot].to_spec() == {"component": identity, "params": params}
        from pllm.kernels import Cpu
        from pllm.state import ClientLocalKv

        expected_support = {
            "state": (ClientLocalKv, "pllm/client-local-kv", {}),
            "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
        }
        if paper == "gkm":
            from pllm.numeric import Q11I16Ring64Boundary
            from pllm.protocols import (
                OpenWeightVerifiableOutsourcing, TwoShareInput, VerifiedPublicF32Lookup,
            )
            from pllm.roles import TwoOnlineDelegationRoles
            from pllm.verification import PublishedLinearVerify

            expected_support.update({
                "linear": (OpenWeightVerifiableOutsourcing, "pllm/open-weight-verifiable-outsourcing/v1", {
                    "capacity": "decoder", "sharing": "full", "generation": "fixed_length_greedy",
                }),
                "input": (TwoShareInput, "pllm/two-share-input/v1", {}),
                "lookup": (VerifiedPublicF32Lookup, "pllm/verified-public-f32-lookup/v1", {}),
                "numeric": (Q11I16Ring64Boundary, "pllm/q11-i16-ring64-boundary/v1", {}),
                "verification": (PublishedLinearVerify, "pllm/published-linear-verify/v1", {"acceptance": "request"}),
                "topology": (TwoOnlineDelegationRoles, "pllm/two-online-delegation-workers/v1", {}),
            })
            assert set(components) == set(expected) | set(expected_support)
        for slot, (cls, identity, params) in expected_support.items():
            assert issubclass(cls, sdk.ComponentRef)
            assert type(components[slot]) is cls
            assert components[slot].to_spec() == {"component": identity, "params": params}
        if paper == "maverick":
            from pllm.correlation import RaaMask
            from pllm.numeric import ExactSourceFieldBoundaries
            from pllm.operators import PublicF32Lookup
            from pllm.protocols import PublicFieldMatvec, AuthenticatedConfidentialRoleTransport
            from pllm.roles import ClientMaskSingleServerRoles
            from pllm.sources import AuthenticatedSourceInputProducer, PinnedSemanticSourceBindings
            from pllm.state import TentativeStateCommit
            from pllm.verification import (
                TwoLevelCodedBatchVerification, ValidatedMatrixCodePublication, OriginalDotWindow,
            )

            expected_protocols = {
                "linear": (PublicFieldMatvec, "pllm/public-field-matvec/v1", {}),
                "mask": (RaaMask, "pllm/raa-mask/v1", {}),
                "verification": (TwoLevelCodedBatchVerification, "pllm/two-level-coded-batch-verification/v1", {}),
                "publication": (ValidatedMatrixCodePublication, "pllm/validated-matrix-code-publication/v1", {}),
                "original_dot": (OriginalDotWindow, "pllm/original-dot-window/v1", {}),
                "commit": (TentativeStateCommit, "pllm/tentative-state-commit/v1", {}),
                "producer": (AuthenticatedSourceInputProducer, "pllm/authenticated-source-input-producer/v1", {}),
                "numeric": (ExactSourceFieldBoundaries, "pllm/exact-source-field-boundaries/v1", {
                    "parameter_authority": None, "client_arithmetic": "f32-extension",
                    "source_precision": "source", "source_to_field_rounding": None,
                    "field_to_f32_rounding": None,
                }),
                "embedding": (PublicF32Lookup, "pllm/public-f32-lookup/v1", {"execution_owner": "client"}),
                "topology": (ClientMaskSingleServerRoles, "pllm/client-mask-single-server-roles/v1", {}),
                "transport": (AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "bindings": (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
            }
            assert set(components) == set(expected) | set(expected_protocols) | set(expected_support)
            for slot, (cls, identity, params) in expected_protocols.items():
                assert issubclass(cls, sdk.ComponentRef)
                assert type(components[slot]) is cls
                assert components[slot].to_spec() == {"component": identity, "params": params}
    if paper in ("puma", "fusefss", "sigma"):
        import pllm.correlation as correlation
        import pllm.nonlinear as nonlinear
        import pllm.numeric as numeric
        import pllm.preparation as preparation
        import pllm.protocols as protocols
        import pllm.schedulers as schedulers
        import pllm.sources as sources
        import pllm.verification as verification
        from pllm.kernels import Cpu
        from pllm.roles import ThreePartyReplicatedRoles, TrustedDealerTwoShareRoles
        from pllm.state import ReplicatedPrivatePrefix, FixedCountPrivatePrefix

        # Complete manually pinned primitive maps, including genuine legacy
        # role/state/kernel declarations. CPU defaults serialize threads=1.
        if paper == "puma":
            expected_ring = {
                "numeric": (numeric.ReplicatedRing64Q18Edges, "pllm/replicated-ring64-q18-edges/v1", {"revision": "source-order-v1"}),
                "weights": (sources.PublicCommittedQ18Source, "pllm/public-committed-q18-source/v1", {}),
                "input": (protocols.ReplicatedPrivateTokenInput, "pllm/replicated-private-token-input/v1", {}),
                "embedding": (protocols.ReplicatedCompleteTableLookup, "pllm/replicated-complete-table-lookup/v1", {}),
                "linear": (protocols.ReplicatedPublicMatmulFullSum, "pllm/replicated-public-matmul-full-sum/v1", {}),
                "multiply": (protocols.ReplicatedSecretMultiply, "pllm/replicated-secret-multiply/v1", {}),
                "truncate": (nonlinear.ReplicatedTruncAprQuarterGap, "pllm/replicated-truncapr-quarter-gap/v1", {}),
                "add_bias": (protocols.ReplicatedRingAddBias, "pllm/replicated-ring-add-bias/v1", {}),
                "conversion": (protocols.ReplicatedArithmeticBooleanConversion, "pllm/replicated-arithmetic-boolean-conversion/v1", {}),
                "activation": (nonlinear.Q18PiecewiseSiluSourceOrder, "pllm/q18-piecewise-silu-source-order/v1", {}),
                "reciprocal": (nonlinear.Q18SourceReciprocal, "pllm/q18-source-reciprocal/v1", {}),
                "rsqrt": (nonlinear.Q18SourceRsqrt, "pllm/q18-source-rsqrt/v1", {}),
                "normalization": (nonlinear.Q18RmsNormEpsilon1, "pllm/q18-rmsnorm-epsilon1/v1", {}),
                "rotary": (nonlinear.Q18SourceRotary, "pllm/q18-source-rotary/v1", {}),
                "attention": (protocols.ReplicatedCausalQkSoftmaxAv, "pllm/replicated-causal-qk-softmax-av/v1", {}),
                "softmax": (nonlinear.Q18ShiftedClipExp5Reciprocal, "pllm/q18-shifted-clip-exp5-reciprocal/v1", {}),
                "selection": (protocols.ReplicatedCompleteStableArgmax, "pllm/replicated-complete-stable-argmax/v1", {}),
                "output": (protocols.ReplicatedFixedCountFinalTokenRelease, "pllm/replicated-fixed-count-final-token-release/v1", {}),
                "randomness": (correlation.ReplicatedPairwisePrss, "pllm/replicated-pairwise-prss/v1", {}),
                "transport": (protocols.AuthenticatedDirectedThreePartyTransport, "pllm/authenticated-directed-three-party-transport/v1", {}),
                "admission": (verification.PerEdgeRangeWorkAdmission, "pllm/per-edge-range-work-admission/v1", {}),
                "topology": (ThreePartyReplicatedRoles, "pllm/three-party-replicated/v1", {}),
                "state": (ReplicatedPrivatePrefix, "pllm/replicated-private-prefix/v1", {}),
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
            }
            assert len(components) == len(expected_ring) == 24
        elif paper == "fusefss":
            expected_ring = {
                "embedding": (protocols.FSSPublicTableLookup, "pllm/fss-private-public-table-lookup/v1", {}),
                "linear": (protocols.FSSPublicDot, "pllm/fss-public-dot/v1", {}),
                "rescale": (nonlinear.FSSSignedFloorRescale, "pllm/fss-signed-trunc-floor/v1", {}),
                "activation": (nonlinear.FSSSigned38InputSiLU, "pllm/fss-silu/v1", {}),
                "normalization": (nonlinear.FSSSumSquaresTableRMSNorm, "pllm/fss-rmsnorm/v1", {}),
                "rotary": (nonlinear.FSSProductFloorRotary, "pllm/fss-rotary/v1", {}),
                "private_dot": (protocols.FSSPrivateDot, "pllm/fss-private-dot/v1", {}),
                "softmax": (nonlinear.FSSSourceTableSoftmax, "pllm/fss-softmax/v1", {}),
                "gated_multiply": (protocols.FSSSecretMultiply, "pllm/fss-secret-gated-multiply/v1", {}),
                "residual": (protocols.FSSAdd, "pllm/fss-add/v1", {}),
                "head": (protocols.FSSPublicDot, "pllm/fss-public-dot/v1", {"capacity": "full-vocabulary"}),
                "selection": (protocols.FSSArgmax, "pllm/fss-argmax/v1", {}),
                "material": (preparation.TrustedDealerOneUseFSSMaterial, "pllm/fss-one-use-material/v1", {}),
                "source": (sources.FSSSourceBinding, "pllm/fss-source-binding/v1", {}),
                "numeric": (numeric.Ring50Q12Edges, "pllm/ring50-q12-edges/v1", {
                    "revision": "fusefss.dense-rope-ring50-q12.v1", "ranges": "source-derived",
                }),
                "schedule": (schedulers.FSSOnlineSchedule, "pllm/fss-online-schedule/v1", {}),
                "transport": (protocols.FSSAuthenticatedRoleTransport, "pllm/fss-authenticated-role-transport/v1", {}),
                "topology": (TrustedDealerTwoShareRoles, "pllm/trusted-dealer-two-share/v1", {}),
                "state": (FixedCountPrivatePrefix, "pllm/fixed-count-private-prefix/v1", {}),
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
            }
            assert len(components) == len(expected_ring) == 20
        else:
            expected_ring = {
                "public_dot": (protocols.RingPublicCoefficientDot, "pllm/ring-public-coefficient-dot/v1", {}),
                "full_head": (protocols.RingPublicCoefficientDot, "pllm/ring-public-coefficient-dot/v1", {"capacity": "full-vocabulary"}),
                "matrix_product": (protocols.NegativeMaskMatrixProduct, "pllm/negative-mask-matrix-product/v1", {}),
                "gated_secret_multiply": (protocols.NegativeMaskElementwiseProduct, "pllm/negative-mask-elementwise-product/v1", {}),
                "ring_arithmetic": (protocols.MaskedRingAddSubProduct, "pllm/masked-ring-add-sub-product/v1", {}),
                "truncation": (nonlinear.ReducedRingTruncation, "pllm/reduced-ring-truncation/v1", {}),
                "gapars": (nonlinear.SameRingGapARS, "pllm/same-ring-gapars/v1", {}),
                "drelu": (nonlinear.XorMaskedDReLU, "pllm/xor-masked-drelu/v1", {}),
                "public_table": (protocols.BoundedDpfPublicTable, "pllm/bounded-dpf-public-table/v1", {}),
                "predicate_select": (protocols.XorPredicateFiveWordSelect, "pllm/xor-predicate-five-word-select/v1", {}),
                "residual_activation": (nonlinear.EarlyFloorResidualActivation, "pllm/early-floor-residual-activation/v1", {}),
                "normalization": (nonlinear.IntegerRsqrtTableNorm, "pllm/integer-rsqrt-table-norm/v1", {
                    "kind": "source", "epsilon": "source", "semantic_revision": None,
                }),
                "rotary": (nonlinear.SourceRing50Q12Rotary, "pllm/source-ring50-q12-rotary/v1", {"semantic_revision": None}),
                "attention": (protocols.CausalMaxNexpReciprocalAttention, "pllm/causal-max-nexp-reciprocal-attention/v1", {}),
                "private_feedback": (protocols.StablePrivateArgmaxEmbedding, "pllm/stable-private-argmax-embedding/v1", {}),
                "numeric_edges": (numeric.Ring50Q12Edges, "pllm/ring50-q12-edges/v1", {
                    "revision": "sigma-source-order-v1", "ranges": "source-derived",
                }),
                "material": (preparation.OneUseRoleLocalMaterial, "pllm/one-use-role-local-material/v1", {}),
                "input_authentication": (protocols.AuthenticatedNativeInput, "pllm/authenticated-native-input/v1", {}),
                "handoff": (protocols.ConsumingShareToCommonMask, "pllm/consuming-share-to-common-mask/v1", {}),
                "schedule": (schedulers.BoundedConsumingProtectedDag, "pllm/bounded-consuming-protected-dag/v1", {}),
                "source_bindings": (sources.PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "topology": (TrustedDealerTwoShareRoles, "pllm/trusted-dealer-two-share/v1", {}),
                "state": (FixedCountPrivatePrefix, "pllm/fixed-count-private-prefix/v1", {}),
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
            }
            assert len(components) == len(expected_ring) == 25
        assert set(components) == set(expected_ring)
        assert all(isinstance(ref, sdk.ComponentRef) for ref in components.values())
        for slot, (cls, identity, params) in expected_ring.items():
            assert issubclass(cls, sdk.ComponentRef)
            assert type(components[slot]) is cls
            assert components[slot].to_spec() == {"component": identity, "params": params}
    if paper in ("bumblebee", "cachemir", "nexus", "thor", "rosetta"):
        import pllm.correlation as correlation
        import pllm.nonlinear as nonlinear
        import pllm.numeric as numeric
        import pllm.preparation as preparation
        import pllm.protocols as protocols
        import pllm.roles as roles
        import pllm.schedulers as schedulers
        import pllm.sources as sources
        import pllm.state as state
        from pllm.kernels import Cpu

        # Literal full factory maps: configuration requirements, not admission.
        if paper == "bumblebee":
            expected_encrypted = {
                "input": (protocols.PrivateTokenShare, "pllm/private-token-share/v1", {}),
                "embedding": (protocols.CoefficientPrivateLookup, "pllm/coefficient-private-lookup/v1", {}),
                "linear": (protocols.CoefficientObliviousLinearTransform, "pllm/coefficient-olt/v1", {}),
                "bilinear": (protocols.SimdBilinearOle, "pllm/simd-bole/v1", {}),
                "arithmetic_to_homomorphic": (protocols.ArithmeticToHomomorphic, "pllm/arithmetic-to-homomorphic/v1", {}),
                "repacking": (protocols.TraceInterleaveRepacking, "pllm/trace-interleave-repacking/v1", {}),
                "coefficient_to_arithmetic": (protocols.CoefficientHomomorphicToArithmetic, "pllm/coefficient-homomorphic-to-arithmetic/v1", {}),
                "simd_to_arithmetic": (protocols.FloodedSimdHomomorphicToArithmetic, "pllm/flooded-simd-homomorphic-to-arithmetic/v1", {}),
                "ring_conversion": (protocols.CrtRingConversion, "pllm/crt-ring-conversion/v1", {}),
                "ot": (correlation.DistributedFerretOt, "pllm/distributed-ferret-ot/v1", {}),
                "ring_extension": (protocols.SignedRingExtension, "pllm/signed-ring-extension/v1", {}),
                "truncation": (nonlinear.SharedRingTruncation, "pllm/shared-ring-truncation/v1", {}),
                "comparison": (nonlinear.SharedComparisonMux, "pllm/shared-comparison-mux/v1", {}),
                "softmax": (nonlinear.ShiftedClipRepeatedSquareSoftmax, "pllm/shifted-clip-repeated-square-softmax/v1", {}),
                "normalization": (nonlinear.SharedReciprocalInverseRootNormalization, "pllm/shared-reciprocal-inverse-root-normalization/v1", {}),
                "activation": (nonlinear.PiecewisePolynomialActivation, "pllm/piecewise-polynomial-activation/v1", {}),
                "residual_bias": (nonlinear.SharedRingResidualBias, "pllm/shared-ring-residual-bias/v1", {}),
                "position": (protocols.SourcePositionOperators, "pllm/source-position-operators/v1", {}),
                "selection": (protocols.SharedStableArgmax, "pllm/shared-stable-argmax/v1", {}),
                "feedback": (protocols.PrivateFullPrefixFeedback, "pllm/private-full-prefix-feedback/v1", {}),
                "state": (state.SharedFullPrefixState, "pllm/shared-full-prefix-state/v1", {}),
                "conversion_lifecycle": (preparation.OneUsePartyLocalConversion, "pllm/one-use-party-local-conversion/v1", {}),
                "producer": (sources.AuthenticatedOriginalProducer, "pllm/authenticated-original-producer/v1", {}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "topology": (roles.TwoOwnerPrivateInputPrivateModelRoles, "pllm/two-owner-private-input-private-model-roles/v1", {}),
                "source_bindings": (sources.PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
                "numeric_certificates": (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {
                    "parameter_authority": None, "profile": "paper-ring64-q18", "extension": "ring32-q12-where-bound",
                }),
                "schedule": (schedulers.SetupOnlineRetirementSchedule, "pllm/setup-online-retirement-schedule/v1", {}),
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
            }
            assert len(components) == len(expected_encrypted) == 29
        elif paper == "cachemir":
            expected_encrypted = {
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
                "linear": (protocols.CkksRectangularLinear, "pllm/ckks-rectangular-linear/v1", {}),
                "attention_qk": (protocols.InterleavedQK, "pllm/interleaved-qk/v1", {}),
                "attention_value": (protocols.InterleavedWeightedV, "pllm/interleaved-weighted-v/v1", {}),
                "attention_routing": (protocols.RoutedGQA, "pllm/routed-gqa/v1", {}),
                "rotary": (protocols.HERotaryTransform, "pllm/he-rotary-transform/v1", {}),
                "exponential": (nonlinear.CalibratedPolynomialExp, "pllm/calibrated-polynomial-exp/v1", {"coefficients": None, "domain": None}),
                "reciprocal": (nonlinear.StandardGoldschmidtReciprocal, "pllm/standard-goldschmidt-reciprocal/v1", {"iterations": None, "domain": None}),
                "softmax": (nonlinear.HEMaskedSoftmax, "pllm/he-masked-softmax/v1", {}),
                "activation": (nonlinear.CompositePolynomialActivation, "pllm/composite-polynomial-activation/v1", {"coefficients": None, "domain": None}),
                "inverse_root": (nonlinear.InverseRoot, "pllm/certified-inverse-root/v1", {"iterations": None, "domain": None}),
                "normalization": (nonlinear.SourceNormalization, "pllm/source-normalization/v1", {}),
                "state": (state.InterleavedKVTransaction, "pllm/interleaved-kv-transaction/v1", {"cache_retention": None}),
                "selection": (protocols.PrivateTokenSelection, "pllm/private-token-selection/v1", {}),
                "feedback": (protocols.PrivateTokenFeedback, "pllm/private-token-feedback/v1", {}),
                "calibration": (numeric.SourceCalibration, "pllm/source-authenticated-calibration/v1", {}),
                "numeric": (numeric.NumericDomain, "pllm/numeric-domain-certificate/v1", {"parameter_authority": None}),
                "keys": (preparation.HEKeyContext, "pllm/he-key-context/v1", {"prime_chain": None, "key_parameters": None}),
                "ciphertext_birth": (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {}),
                "topology": (roles.ClientKeyServerEvaluationRoles, "pllm/client-key-server-evaluation-roles/v1", {}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "schedule": (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {}),
                "source_bindings": (sources.PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
            }
            assert len(components) == len(expected_encrypted) == 23
        elif paper == "nexus":
            expected_encrypted = {
                "coefficient_compression": (protocols.HECoefficientCompression, "pllm/he-coefficient-compression/v1", {}),
                "linear": (protocols.HECommutableLinear, "pllm/he-commutable-linear/v1", {}),
                "column_product": (protocols.HEColumnProduct, "pllm/he-column-product/v1", {}),
                "layout": (protocols.HELayoutTransform, "pllm/he-layout-transform/v1", {}),
                "attention_value": (protocols.HEAttentionValue, "pllm/he-attention-value/v1", {}),
                "softmax": (nonlinear.HERepeatedSquaringSoftmax, "pllm/he-repeated-squaring-softmax/v1", {"source_squarings": 8, "parameter_authority": None}),
                "activation": (nonlinear.HEPiecewisePolynomialActivation, "pllm/he-piecewise-polynomial-activation/v1", {}),
                "normalization": (nonlinear.HERowNormalization, "pllm/he-row-normalization/v1", {}),
                "sign": (nonlinear.HESignComposition, "pllm/he-sign-composition/v1", {}),
                "selection": (protocols.HEFoldSelection, "pllm/he-fold-selection/v1", {}),
                "bootstrap": (protocols.HEBootstrap, "pllm/he-bootstrap/v1", {}),
                "residual": (protocols.HEResidualAdd, "pllm/he-residual-add/v1", {}),
                "embedding": (protocols.HEPrivateEmbedding, "pllm/he-private-embedding/v1", {}),
                "key_context": (preparation.HEDualKeyContext, "pllm/he-dual-key-context/v1", {}),
                "ciphertext_birth": (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {}),
                "schedule": (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "source_bindings": (sources.PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
                "roles": (roles.ClientSecretServerWeightSecretRoles, "pllm/client-secret-server-weight-secret-roles/v1", {}),
                "numeric_certificates": (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None}),
                "output": (protocols.ClassifierOutputPolicy, "pllm/classifier-output-policy/v1", {"source": "model-metadata"}),
            }
            assert len(components) == len(expected_encrypted) == 21
        elif paper == "thor":
            expected_encrypted = {
                "embedding": (protocols.HESourceEmbeddingPositionSegment, "pllm/he-source-embedding-position-segment/v1", {}),
                "layernorm": (nonlinear.HELayerNorm, "pllm/he-layernorm/v1", {}),
                "attention_products": (protocols.HEAttentionProducts, "pllm/he-attention-products/v1", {}),
                "head_scale_mask": (protocols.HEHeadScaleMask, "pllm/he-head-scale-mask/v1", {}),
                "softmax": (nonlinear.HESquareNormalizeSoftmax, "pllm/he-square-normalize-softmax/v1", {}),
                "attention_value": (protocols.HEAttentionValue, "pllm/he-attention-value/v1", {}),
                "linear": (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {}),
                "residual": (protocols.HEResidualAdd, "pllm/he-residual-add/v1", {}),
                "activation": (nonlinear.HEPolynomialActivation, "pllm/he-polynomial-activation/v1", {"function": "source-gelu"}),
                "pooler": (protocols.HEClsPooler, "pllm/he-cls-pooler/v1", {}),
                "pooler_activation": (nonlinear.HEPolynomialActivation, "pllm/he-polynomial-activation/v1", {"function": "source-tanh"}),
                "head": (protocols.HENspTwoLogitHead, "pllm/he-nsp-two-logit-head/v1", {}),
                "bootstrap": (protocols.HEBootstrap, "pllm/he-bootstrap/v1", {}),
                "key_context": (preparation.HEKeyContext, "pllm/he-key-context/v1", {}),
                "ciphertext_birth": (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {}),
                "schedule": (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {}),
                "roles": (roles.ClientKeyServerEvaluationRoles, "pllm/client-key-server-evaluation-roles/v1", {}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
                "source_bindings": (sources.PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {
                    "required_model_type": "bert", "required_architectures": ["BertForNextSentencePrediction"], "required_output_logits": 2,
                }),
                "numeric_certificates": (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None}),
                "output_policy": (protocols.NspLogitsOnlyOutputPolicy, "pllm/nsp-logits-only-output-policy/v1", {}),
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
            }
            assert len(components) == len(expected_encrypted) == 22
        else:
            bridge = {
                "to_coefficients": "slot_coefficient_transform", "ring_switch": "ring_switch",
                "extract": "coefficient_extraction", "compare": "comparison", "evaluate": "segmented_affine",
                "key_switch": "key_switch", "modulus_raise": "modulus_raise",
                "to_slots": "coefficient_slot_transform", "evalmod": "evalmod", "numeric": "numeric",
            }
            expected_encrypted = {
                "kernels": (Cpu, "pllm/cpu", {"threads": 1}),
                "roles": (roles.ClientKeyServerConfidentialWeightRoles, "pllm/client-key-server-confidential-weight-roles/v1", {"secret_key_owner": "client", "weight_owner": "server"}),
                "transport": (protocols.AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {"roles": "roles"}),
                "input": (protocols.AuthenticatedClientInputExport, "pllm/authenticated-client-input-export/v1", {"roles": "roles", "transport": "transport", "birth": "ciphertext_birth"}),
                "source_bindings": (sources.AuthenticatedSourceBindings, "pllm/authenticated-source-bindings/v1", {"weight_owner": "server", "bind_checkpoint_and_numeric_domain": True}),
                "key_graph": (preparation.HEKeyGraph, "pllm/he-key-graph/v1", {"secret_key_owner": "client", "server_secret_keys": False}),
                "ciphertext_birth": (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
                "numeric": (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "source": "rosetta-53-bit-port-unresolved"}),
                "level_error_schedule": (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {"numeric": "numeric", "key_graph": "key_graph"}),
                "placement": (schedulers.HELevelSchemePlacement, "pllm/he-level-scheme-placement/v1", {"schedule": "level_error_schedule", "roles": "roles"}),
                "linear": (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {"weight_owner": "server", "source_bindings": "source_bindings"}),
                "output_head": (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {"operation": "source-output-head", "vocabulary": "full", "weight_owner": "server", "source_bindings": "source_bindings", "selection": "selection"}),
                "residual": (protocols.HEResidualAdd, "pllm/he-residual-add/v1", {}),
                "attention_scores": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "query-key-dot-product", "numeric": "numeric"}),
                "attention_values": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "probability-value-reduction", "numeric": "numeric"}),
                "normalization_square_reduce": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "normalization-square-reduction", "numeric": "numeric"}),
                "normalization": (nonlinear.HENormalization, "pllm/he-normalization/v1", {"square_reduce": "normalization_square_reduce", "inverse_sqrt": "segmented_affine", "multiply": "normalization_multiply", **bridge}),
                "normalization_multiply": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "normalization-elementwise-product", "numeric": "numeric"}),
                "softmax": (nonlinear.HESoftmaxSquareNormalize, "pllm/he-softmax-square-normalize/v1", {"square_reduce": "softmax_square_reduce", "reciprocal": "segmented_affine", "multiply": "softmax_multiply", **bridge}),
                "softmax_square_reduce": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "softmax-square-and-row-sum", "numeric": "numeric"}),
                "softmax_multiply": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "softmax-row-normalization", "numeric": "numeric"}),
                "activation": (nonlinear.HEActivationBridge, "pllm/he-activation-bridge/v1", bridge),
                "activation_gate": (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "activation-gated-product", "numeric": "numeric"}),
                "slot_coefficient_transform": (protocols.HESlotCoefficientTransform, "pllm/he-slot-coefficient-transform/v1", {"numeric": "numeric"}),
                "ring_switch": (protocols.HERingSwitch, "pllm/he-ring-switch/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
                "coefficient_extraction": (protocols.HECoefficientExtraction, "pllm/he-coefficient-extraction/v1", {"numeric": "numeric"}),
                "comparison": (nonlinear.LWEHighPrecisionComparison, "pllm/lwe-high-precision-comparison/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
                "segmented_affine": (nonlinear.RLWEEncryptedAccumulatorSegmentedAffine, "pllm/rlwe-encrypted-accumulator-segmented-affine/v1", {"comparison": "comparison", "key_graph": "key_graph", "numeric": "numeric"}),
                "key_switch": (protocols.HEKeySwitch, "pllm/he-key-switch/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
                "modulus_raise": (protocols.HEModulusRaise, "pllm/he-modulus-raise/v1", {"numeric": "numeric"}),
                "coefficient_slot_transform": (protocols.HECoefficientSlotTransform, "pllm/he-coefficient-slot-transform/v1", {"numeric": "numeric"}),
                "evalmod": (nonlinear.HEEvalMod, "pllm/he-evalmod/v1", {"numeric": "numeric"}),
                "state": (state.EncryptedKVTransaction, "pllm/encrypted-kv-transaction/v1", {"birth": "ciphertext_birth", "schedule": "level_error_schedule"}),
                "selection": (protocols.PrivateTokenSelection, "pllm/private-token-selection/v1", {"comparison": "comparison", "roles": "roles"}),
                "feedback": (protocols.PrivateTokenFeedback, "pllm/private-token-feedback/v1", {"selection": "selection", "input": "input", "state": "state"}),
            }
            assert len(components) == len(expected_encrypted) == 35
        assert set(components) == set(expected_encrypted)
        for slot, (cls, identity, params) in expected_encrypted.items():
            assert issubclass(cls, sdk.ComponentRef)
            assert type(components[slot]) is cls
            assert components[slot].to_spec() == {"component": identity, "params": params}
    for key, identity in PRIMITIVES[paper].items():
        assert components[key].component == identity
    assert not ({ref.component for ref in components.values()}
                & LEGACY_WHOLE_COMPONENTS.get(paper, set()))
    # Legacy recipe identities/helper differences stay in the existing GKM,
    # PUMA and FuseFSS SDK declaration tests; this captures benchmark subsets only.
    capture.pipeline.assert_called_once()
    capture.experiment.assert_called_once()


@pytest.mark.parametrize("paper", RESPONSE)
def test_response_main_passes_all_public_benchmark_keywords(main_spies, paper):
    spy = main_spies(paper)
    argv = ["--prompt", "explicit prompt"]
    spy.module.main(argv)
    input_tokens, new_tokens = TOKEN_DEFAULTS.get(paper, (32, 2))
    spy.parser.assert_called_once_with(
        paper, argv, task="response", max_input_tokens=input_tokens, max_new_tokens=new_tokens,
    )
    spy.factory.assert_called_once_with(model=spy.args.model, budget=spy.args.budget)
    spy.benchmark.assert_called_once_with(
        spy.factory.return_value, scenario=spy.args.scenario, inputs=spy.args.inputs,
        sampling=spy.args.sampling, output=spy.args.output,
    )
    spy.forward.assert_not_called()


@pytest.mark.parametrize("paper", FORWARD)
def test_forward_main_constructs_target_without_response_scenario(main_spies, paper):
    spy = main_spies(paper)
    argv = ["--model-path", "offline", "--forward-inputs", "inputs.json"]
    result = spy.module.main(argv)
    spy.parser.assert_called_once_with(
        paper, argv, task="forward", max_input_tokens=32, max_new_tokens=0,
    )
    spy.factory.assert_called_once_with(model=spy.args.model, budget=spy.args.budget)
    spy.forward.assert_called_once_with(spy.factory.return_value)
    spy.benchmark.assert_called_once_with(
        spy.forward.return_value, inputs=spy.args.inputs,
        sampling=spy.args.sampling, output=spy.args.output,
    )
    assert result is spy.benchmark.return_value


@pytest.mark.parametrize("task", ("response", "forward"))
def test_common_parser_contract_without_model_resolution(sdk, tmp_path, task):
    common = importlib.import_module("examples.papers.common")
    inputs = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "custom": {"raw": True}}
    input_file = tmp_path / "inputs.json"
    input_file.write_text(json.dumps(inputs), encoding="utf-8")
    argv = []
    if task == "forward":
        argv = ["--model-path", str(tmp_path), "--forward-inputs", str(input_file)]
        for missing in ([], argv[:2], argv[2:]):
            with pytest.raises(SystemExit) as exc:
                common.benchmark_arguments("nexus", missing, task=task)
            assert exc.value.code == 2
    args = common.benchmark_arguments("contract", argv, task=task)
    assert args.sampling.processes == args.sampling.repetitions == 1
    assert args.sampling.warmups == 0
    assert args.sampling.memory_bytes == 512 << 20
    assert args.output == Path("target/paper-benchmarks/contract.json")
    assert args.budget.requests == 1 and args.budget.max_input_tokens == 32
    if task == "forward":
        assert isinstance(args.budget, sdk.ForwardBudget)
        assert args.model == sdk.Model.path(str(tmp_path))
        assert args.model.local_files_only
        assert args.inputs == inputs  # No SDK tensor-schema interpretation here.
        assert args.scenario is None
    else:
        assert isinstance(args.budget, sdk.ExecutionBudget)
        assert args.budget.max_new_tokens == 2
        assert args.model is common.MODEL
        assert args.model.source == "Qwen/Qwen2.5-0.5B-Instruct"
        assert args.model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
        assert args.model.local_files_only is True
        assert args.inputs == {"prompt": "hi"}
        assert args.scenario.kind == "cold" and args.scenario.count == 1
        assert args.scenario.max_output_tokens == 2


@pytest.mark.parametrize("task", ("response", "forward"))
def test_common_parser_preserves_explicit_overrides(sdk, tmp_path, task):
    common = importlib.import_module("examples.papers.common")
    input_file = tmp_path / "inputs.json"
    input_file.write_text('{"input_ids": [[7]]}', encoding="utf-8")
    output = tmp_path / "requested.json"
    argv = [
        "--model-path", str(tmp_path), "--max-input-tokens", "47",
        "--processes", "2", "--repetitions", "3", "--warmups", "1",
        "--memory-bytes", str(768 << 20), "--output", str(output),
    ]
    argv += (["--forward-inputs", str(input_file)] if task == "forward"
             else ["--max-new-tokens", "11", "--prompt", "unaltered prompt"])
    args = common.benchmark_arguments("override", argv, task=task)
    assert args.model == sdk.Model.path(str(tmp_path))
    assert args.budget.requests == 1 and args.budget.max_input_tokens == 47
    assert (args.sampling.processes, args.sampling.repetitions, args.sampling.warmups) == (2, 3, 1)
    assert args.sampling.memory_bytes == 768 << 20
    assert args.output == output
    if task == "response":
        assert args.budget.max_new_tokens == args.scenario.max_output_tokens == 11
        assert args.inputs == {"prompt": "unaltered prompt"}
    else:
        assert isinstance(args.budget, sdk.ForwardBudget)
        assert args.inputs == {"input_ids": [[7]]} and args.scenario is None


def test_factory_default_budgets_capture_not_execution(sdk, papers, capture, tmp_path, no_source_io):
    common = importlib.import_module("examples.papers.common")
    offline = sdk.Model.path(str(tmp_path))
    for paper in PAPERS:
        declaration = papers[paper].benchmark_experiment(**({"model": offline} if paper in FORWARD else {}))
        budget = declaration.budget
        assert budget.requests == 1
        if paper in FORWARD:
            assert isinstance(budget, sdk.ForwardBudget)
            assert budget.max_input_tokens == 32
        else:
            assert declaration.pipeline.model is common.MODEL
            assert declaration.pipeline.model.source == "Qwen/Qwen2.5-0.5B-Instruct"
            assert declaration.pipeline.model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
            assert declaration.pipeline.model.local_files_only is True
            assert isinstance(budget, sdk.ExecutionBudget)
            assert (budget.max_input_tokens, budget.max_new_tokens) == TOKEN_DEFAULTS.get(paper, (32, 2))


@pytest.mark.parametrize("field,value", (
    ("--max-input-tokens", "0"), ("--max-input-tokens", "-1"),
    ("--max-new-tokens", "0"), ("--max-new-tokens", "-1"),
))
def test_common_parser_rejects_nonpositive_token_bounds(sdk, field, value, capsys):
    common = importlib.import_module("examples.papers.common")
    with pytest.raises(SystemExit) as exc:
        common.benchmark_arguments("bounds", [field, value])
    assert exc.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("warmups", ("0", "-1"))
def test_common_parser_warmup_boundary(sdk, warmups, capsys):
    common = importlib.import_module("examples.papers.common")
    if warmups == "0":
        assert common.benchmark_arguments("warmups", ["--warmups", warmups]).sampling.warmups == 0
    else:
        with pytest.raises(SystemExit) as exc:
            common.benchmark_arguments("warmups", ["--warmups", warmups])
        assert exc.value.code == 2
        assert "must be a nonnegative integer" in capsys.readouterr().err


@pytest.mark.parametrize("payload", ("[]", "1", '"scalar"', "{"))
def test_forward_parser_rejects_nonmapping_or_invalid_json(sdk, tmp_path, payload, capsys):
    common = importlib.import_module("examples.papers.common")
    input_file = tmp_path / "inputs.json"
    input_file.write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        common.benchmark_arguments("nexus", [
            "--model-path", str(tmp_path), "--forward-inputs", str(input_file),
        ], task="forward")
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert ("cannot read --forward-inputs" if payload == "{"
            else "must contain a JSON mapping") in error


@pytest.mark.parametrize("paper", PAPERS)
def test_factory_preserves_explicit_model_and_budget_capture(sdk, papers, capture, tmp_path, paper, no_source_io):
    model = sdk.Model.path(str(tmp_path))
    budget = (sdk.ForwardBudget(3, 47) if paper in FORWARD else sdk.ExecutionBudget(3, 47, 11))
    declaration = papers[paper].benchmark_experiment(model=model, budget=budget)
    assert declaration.pipeline.model is model
    assert declaration.budget is budget
    capture.pipeline.assert_called_once()
    capture.experiment.assert_called_once()


@pytest.mark.parametrize("paper", PAPERS)
def test_factory_rejects_invalid_model_type_before_binding(sdk, papers, paper, no_source_io):
    # Genuine factory validation, not descriptor capture or a resolve-time test.
    # An AssertionError from IO/execution sentinels cannot satisfy this refusal.
    with pytest.raises(sdk.configuration.ConfigurationError, match="model must satisfy the ModelSource contract"):
        papers[paper].benchmark_experiment(model=object())
