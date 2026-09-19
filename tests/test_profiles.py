from __future__ import annotations

import pytest

import pllm
from pllm.kernels import Cpu
from pllm.preparation import BFVCorrelations, ModelAwareCorrections
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import GuardedLinear, MaskedLinear
from pllm.roles import Inference
from pllm.runtime import build_roles
from pllm.sources import TinyModel


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
    assert experiment.resolve().model == "org/model"


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
    with pytest.raises(ValueError, match="supports only"):
        build_roles(pipeline)
