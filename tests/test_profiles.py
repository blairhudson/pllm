from __future__ import annotations

import pytest

import pllm
from pllm.kernels import Cpu
from pllm.preparation import BFVCorrelations, ModelAwareCorrections
from pllm.profiles import (
    DirectFHEProfile,
    MaskedLinearCpu,
    VerifiedMaskedLinearCpu,
    ProprietaryBlinded,
    ProprietaryGuarded,
    _runtime_profile_options,
)
from pllm.protocols import BlindedLinear, DirectFHE, GuardedLinear, MaskedLinear, SecureLinear
from pllm.roles import Inference
from pllm.runtime import build_roles
from pllm.sources import TinyModel
from pllm.verification import FreivaldsVerify


def test_masked_linear_cpu_has_typed_default_slots_and_resolves() -> None:
    pipeline = MaskedLinearCpu(pllm.Model("org/model"), kernels=Cpu(threads=4))
    assert pipeline.profile == "baseline.masked_linear_cpu"
    assert isinstance(pipeline.linear, MaskedLinear)
    assert isinstance(pipeline.preparation, ModelAwareCorrections)
    assert isinstance(pipeline.inference, Inference)
    assert isinstance(pipeline.kernels, Cpu)
    assert pipeline.to_spec() == {
        "profile": "baseline.masked_linear_cpu",
        "model": {"source": "org/model"},
        "components": {
            "linear": {"component": "pllm/masked-linear", "params": {}},
            "preparation": {
                "component": "pllm/model-aware-corrections",
                "params": {},
            },
            "inference": {"component": "pllm/inference", "params": {}},
            "kernels": {"component": "pllm/cpu", "params": {"threads": 4}},
        },
    }
    experiment = pllm.Experiment(
        name="typed-profile",
        pipeline=pipeline,
        deployment=pllm.Deployment.local(root="local://typed-profile"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    resolved = experiment.resolve()
    assert resolved.model == "org/model"
    assert resolved.client_runtime == "masked_transformer_v1"
    assert resolved.privacy_mode == "public"
    assert resolved.requires_preparation is True


def test_verified_masked_profile_resolves_exact_verification_slot() -> None:
    pipeline = VerifiedMaskedLinearCpu(
        pllm.Model("org/model"), verification=FreivaldsVerify(target_failure_bits=48)
    )
    assert pipeline.profile == "research.verified_masked_linear_cpu"
    assert pipeline.verification.target_failure_bits == 48
    restored = pllm.Pipeline.from_spec(pipeline.to_spec())
    assert isinstance(restored, VerifiedMaskedLinearCpu)
    options = _runtime_profile_options(restored)
    assert options is not None
    assert options.verification_component == "pllm/freivalds-verify/v1"
    resolved = pllm.Experiment(
        name="verified-profile",
        pipeline=restored,
        deployment=pllm.Deployment.local(root="local://verified-profile"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    ).resolve()
    assert resolved.verification_target_failure_bits == 48
    with pytest.raises(ValueError, match="target_failure_bits"):
        FreivaldsVerify(0)


def test_profile_accepts_structural_model_sources() -> None:
    class Source:
        source = "org/structural"
        kind = "huggingface"
        model_id = "structural"

        @staticmethod
        def to_spec():
            return {"source": "org/structural", "model_id": "structural"}

    pipeline = MaskedLinearCpu(Source())
    assert pipeline.model == pllm.Model("org/structural", model_id="structural")


def test_pipeline_from_spec_promotes_exact_baseline_profile() -> None:
    original = MaskedLinearCpu(pllm.Model("org/model"), kernels=Cpu(threads=2))
    restored = pllm.Pipeline.from_spec(original.to_spec())
    assert isinstance(restored, MaskedLinearCpu)
    assert restored == original

    extended = original.to_spec()
    extended["components"]["candidate"] = {
        "component": "example/candidate",
        "params": {},
    }
    generic = pllm.Pipeline.from_spec(extended)
    assert type(generic) is pllm.Pipeline
    assert generic == pllm.Pipeline.from_profile(
        "baseline.masked_linear_cpu",
        model=pllm.Model("org/model"),
        components={**original.components, "candidate": pllm.ComponentRef("example/candidate")},
    )


def test_profile_with_params_clones_slots_and_nested_parameters() -> None:
    original = MaskedLinearCpu(TinyModel(), kernels=Cpu(threads=2))
    changed = original.with_params(kernels__threads=8, model__model_id="tiny-profile")
    assert isinstance(changed, MaskedLinearCpu)
    assert isinstance(changed.model, TinyModel)
    assert original.kernels.params["threads"] == 2
    assert original.model.model_id is None
    assert changed.kernels.params["threads"] == 8
    assert changed.model.model_id == "tiny-profile"
    assert changed.get_params()["kernels__threads"] == 8
    with pytest.raises(pllm.ConfigurationError, match="overlapping"):
        original.with_params(kernels=Cpu(), kernels__threads=4)
    with pytest.raises(pllm.ConfigurationError, match="unknown parameter path"):
        original.with_params(unknown=True)


def test_profile_slots_reject_wrong_categories() -> None:
    model = pllm.Model("org/model")
    with pytest.raises(pllm.ConfigurationError, match="linear"):
        MaskedLinearCpu(model, linear=Cpu())
    with pytest.raises(pllm.ConfigurationError, match="preparation"):
        MaskedLinearCpu(model, preparation=Inference())
    with pytest.raises(pllm.ConfigurationError, match="inference"):
        MaskedLinearCpu(model, inference=MaskedLinear())
    with pytest.raises(pllm.ConfigurationError, match="kernels"):
        MaskedLinearCpu(model, kernels=ModelAwareCorrections())


def test_uncomposed_same_category_alternatives_fail_at_resolution_and_serving() -> None:
    pipeline = MaskedLinearCpu(
        pllm.Model("org/model"),
        linear=GuardedLinear(),
        preparation=BFVCorrelations(),
    )
    experiment = pllm.Experiment(
        name="uncomposed",
        pipeline=pipeline,
        deployment=pllm.Deployment.local(root="local://uncomposed"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    with pytest.raises(pllm.ConfigurationError):
        experiment.resolve()
    with pytest.raises(ValueError, match="does not support"):
        build_roles(pipeline)


@pytest.mark.parametrize(
    ("profile_type", "profile_name", "linear_id", "protocol"),
    [
        (
            ProprietaryGuarded,
            "runtime.proprietary_guarded",
            "pllm/guarded-linear/v1",
            "guarded",
        ),
        (
            ProprietaryBlinded,
            "runtime.proprietary_blinded",
            "pllm/blinded-linear/v1",
            "blinded",
        ),
        (DirectFHEProfile, "runtime.direct_fhe", "pllm/direct-fhe", "direct"),
    ],
)
def test_proprietary_profiles_round_trip_resolve_and_publish_runtime_contract(
    profile_type, profile_name: str, linear_id: str, protocol: str
) -> None:
    pipeline = profile_type(
        pllm.Model("org/model", model_id="runtime-model"), kernels=Cpu(threads=3)
    )
    assert pipeline.profile == profile_name
    assert pipeline.linear.component == linear_id
    assert set(pipeline.components) == {"linear", "inference", "kernels"}
    restored = pllm.Pipeline.from_spec(pipeline.to_spec())
    assert type(restored) is profile_type
    assert restored == pipeline
    options = _runtime_profile_options(pipeline)
    assert options is not None
    assert options.privacy_mode == "proprietary"
    assert options.proprietary_protocol == protocol
    assert options.requires_preparation is False
    experiment = pllm.Experiment(
        name=profile_name,
        pipeline=pipeline,
        deployment=pllm.Deployment.local(root=f"local://{profile_name}"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    resolved = experiment.resolve()
    assert resolved.model == "runtime-model"
    assert resolved.canonical_profile == pipeline.canonical_bytes()
    assert resolved.configuration_digest == experiment.configuration_digest()
    assert resolved.client_runtime == options.client_runtime
    assert resolved.privacy_protocol == options.privacy_protocol


def test_guarded_profile_binds_and_clones_guard_policy() -> None:
    original = ProprietaryGuarded(pllm.Model("org/model"))
    changed = original.with_params(
        linear__max_rows_per_request=23,
        linear__max_rows_per_owner_stage=47,
        linear__max_requests_per_minute=89,
        linear__output_dither_bound=2,
    )
    assert isinstance(changed, ProprietaryGuarded)
    options = _runtime_profile_options(changed)
    assert options is not None
    assert options.guard_max_rows_per_request == 23
    assert options.guard_max_rows_per_owner_stage == 47
    assert options.guard_max_requests_per_minute == 89
    assert options.output_dither_bound == 2
    assert original.linear.params["max_rows_per_request"] == 4096


def test_proprietary_profiles_reject_wrong_implementations_and_forged_generic_profiles() -> None:
    model = pllm.Model("org/model")
    with pytest.raises(pllm.ConfigurationError, match="pllm/guarded-linear/v1"):
        ProprietaryGuarded(model, linear=BlindedLinear())
    with pytest.raises(pllm.ConfigurationError, match="pllm/blinded-linear/v1"):
        ProprietaryBlinded(model, linear=GuardedLinear())
    with pytest.raises(pllm.ConfigurationError, match="pllm/direct-fhe"):
        DirectFHEProfile(model, linear=SecureLinear())

    valid = ProprietaryGuarded(model)
    forged = pllm.Pipeline.from_profile(
        valid.profile,
        model=model,
        components=valid.components,
    )
    experiment = pllm.Experiment(
        name="forged",
        pipeline=forged,
        deployment=pllm.Deployment.local(root="local://forged"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=1),
    )
    with pytest.raises(pllm.ConfigurationError, match="typed component contract"):
        experiment.resolve()
    with pytest.raises(ValueError, match="does not support"):
        build_roles(forged)


def test_runtime_profile_matrix_keeps_unsupported_arms_out() -> None:
    baseline = _runtime_profile_options(MaskedLinearCpu(pllm.Model("org/model")))
    assert baseline is not None
    assert (
        baseline.privacy_mode,
        baseline.proprietary_protocol,
        baseline.requires_preparation,
    ) == (
        "public",
        "guarded",
        True,
    )
    assert (
        _runtime_profile_options(MaskedLinearCpu(pllm.Model("org/model"), linear=SecureLinear()))
        is None
    )
    assert (
        _runtime_profile_options(
            MaskedLinearCpu(pllm.Model("org/model"), preparation=BFVCorrelations())
        )
        is None
    )
    assert DirectFHE().component == "pllm/direct-fhe"
