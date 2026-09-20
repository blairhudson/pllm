from __future__ import annotations

import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

import pllm
import pllm.components as components
import pllm.configuration as configuration
from pllm.correlation import CorrelationSource, SeededExpansion
from pllm.kernels import Cpu, KernelBackend
from pllm.metrics import (
    Accuracy,
    Communication,
    Cost,
    Energy,
    Latency,
    Memory,
    Metric,
    Perplexity,
    Throughput,
)
from pllm.nonlinear import (
    BinaryTableGatedMultiplyQ7,
    NonlinearProtocol,
    R03CrtGatedMultiplyQ7,
)
from pllm.passes import KvCacheEviction, PlanPass
from pllm.preparation import (
    BFVCorrelations,
    HEAuthenticatedPreprocessing,
    ModelAwareCorrections,
    PreparationProvider,
)
from pllm.protocols import (
    BlindedLinear,
    CleartextLinear,
    DirectFHE,
    GuardedLinear,
    MaskedLinear,
    ProtocolMethod,
    SecureLinear,
)
from pllm.roles import Inference, InferenceRole
from pllm.schedulers import (
    ChunkedIndependentLanesProtectedTensorSchedule,
    IndependentLanesProtectedTensorSchedule,
    ProtectedScheduler,
    ScalarProtectedTensorSchedule,
)
from pllm.sources import BundleModel, ModelSource, TinyModel
from pllm.state import ClientLocalKv, StateProtocol
from pllm.verification import FreivaldsVerify, LinearIntegrity, VerificationScheme


def _instances():
    return (
        Accuracy(dataset="fixture"),
        BFVCorrelations(),
        BinaryTableGatedMultiplyQ7(),
        BlindedLinear(),
        ChunkedIndependentLanesProtectedTensorSchedule(max_elements=4096),
        CleartextLinear(),
        ClientLocalKv(),
        Communication(),
        Cost(),
        Cpu(threads=2),
        DirectFHE(),
        Energy(),
        FreivaldsVerify(),
        GuardedLinear(),
        HEAuthenticatedPreprocessing(),
        IndependentLanesProtectedTensorSchedule(),
        Inference(),
        KvCacheEviction(),
        Latency(),
        LinearIntegrity(),
        MaskedLinear(),
        Memory(),
        ModelAwareCorrections(),
        Perplexity(dataset="fixture"),
        R03CrtGatedMultiplyQ7(),
        ScalarProtectedTensorSchedule(),
        SeededExpansion(),
        SecureLinear(),
        Throughput(),
    )


def test_builtin_registry_is_derived_from_family_classes() -> None:
    instances = _instances()
    expected = tuple(sorted(instance.component for instance in instances))
    classes = components.list_component_classes()
    descriptors = components.list_components()

    assert tuple(item.describe().component for item in classes) == expected
    assert tuple(item.component for item in descriptors) == expected
    assert not hasattr(configuration, "_BUILTIN_DESCRIPTORS")
    for instance in instances:
        component = components.get(instance.component)
        descriptor = component.describe()
        assert component is type(instance)
        assert components.get_component(instance.component) is descriptor
        assert descriptor.component == instance.component
        assert descriptor.category.startswith("pllm/")
        parameter_schema = descriptor.to_dict()["parameter_schema"]
        Draft202012Validator.check_schema(parameter_schema)
        params = instance.to_spec()["params"]
        Draft202012Validator(parameter_schema).validate(params)
        restored = components.create_component(instance.component, params)
        assert type(restored) is type(instance)
        assert restored == instance


def test_all_registered_classes_round_trip_through_experiment_configuration() -> None:
    instances = _instances()
    experiment = pllm.Experiment.from_spec({
        "schema": "pllm.experiment.v1",
        "name": "all-components",
        "pipeline": {
            "profile": "registry.roundtrip",
            "model": {"source": "org/model"},
            "components": {
                f"slot-{index}": instance.to_spec()
                for index, instance in enumerate(instances)
            },
        },
        "deployment": {"kind": "local", "root": "local://registry"},
        "budget": {"requests": 1, "max_input_tokens": 1, "max_new_tokens": 1},
    })
    restored = tuple(experiment.pipeline.components.values())
    assert tuple(type(item) for item in restored) == tuple(type(item) for item in instances)
    assert restored == instances


def test_category_bases_match_each_profile_slot() -> None:
    assert isinstance(Cpu(), KernelBackend)
    assert isinstance(Latency(), Metric)
    assert isinstance(MaskedLinear(), ProtocolMethod)
    assert isinstance(ModelAwareCorrections(), PreparationProvider)
    assert isinstance(SeededExpansion(), CorrelationSource)
    assert isinstance(Inference(), InferenceRole)
    assert isinstance(KvCacheEviction(), PlanPass)
    assert isinstance(BinaryTableGatedMultiplyQ7(), NonlinearProtocol)
    assert isinstance(ScalarProtectedTensorSchedule(), ProtectedScheduler)
    assert isinstance(ClientLocalKv(), StateProtocol)
    assert isinstance(LinearIntegrity(), VerificationScheme)
    for category in (
        KernelBackend,
        Metric,
        ProtocolMethod,
        PreparationProvider,
        CorrelationSource,
        InferenceRole,
        PlanPass,
        NonlinearProtocol,
        ProtectedScheduler,
        StateProtocol,
        VerificationScheme,
    ):
        with pytest.raises(TypeError):
            category("pllm/invalid")


def test_runtime_arm_components_have_exact_validated_parameters() -> None:
    guarded = GuardedLinear(
        max_rows_per_request=2,
        max_rows_per_owner_stage=3,
        max_requests_per_minute=4,
        output_dither_bound=1,
    )
    assert guarded.get_params() == {
        "max_rows_per_request": 2,
        "max_rows_per_owner_stage": 3,
        "max_requests_per_minute": 4,
        "output_dither_bound": 1,
    }
    assert BFVCorrelations(poly_modulus_degree=8192).params["poly_modulus_degree"] == 8192
    assert HEAuthenticatedPreprocessing(threads=2).params["threads"] == 2
    for invalid in (
        lambda: GuardedLinear(max_rows_per_request=0),
        lambda: GuardedLinear(output_dither_bound=-1),
        lambda: BFVCorrelations(poly_modulus_degree=2048),
        lambda: HEAuthenticatedPreprocessing(threads=True),
    ):
        with pytest.raises(pllm.ConfigurationError):
            invalid()


def test_family_modules_own_classes_and_legacy_exports_are_aliases() -> None:
    assert Cpu.__module__ == "pllm.kernels"
    assert MaskedLinear.__module__ == "pllm.protocols.masked_linear"
    assert ModelAwareCorrections.__module__ == "pllm.preparation"
    assert KvCacheEviction.__module__ == "pllm.passes"
    assert LinearIntegrity.__module__ == "pllm.verification"
    assert pllm.Cpu is configuration.Cpu is Cpu
    assert pllm.MaskedLinear is configuration.MaskedLinear is MaskedLinear
    assert pllm.ModelAwareCorrections is configuration.ModelAwareCorrections is ModelAwareCorrections
    assert pllm.KvCacheEviction is configuration.KvCacheEviction is KvCacheEviction
    for name in (
        "Cpu",
        "MaskedLinear",
        "ModelAwareCorrections",
        "KvCacheEviction",
        "BinaryTableGatedMultiplyQ7",
    ):
        assert not hasattr(components, name)


def test_model_source_specializations_preserve_one_model_contract(tmp_path) -> None:
    tiny = TinyModel("qwen2", model_id="tiny")
    bundle = BundleModel(str(tmp_path), model_id="bundle")
    assert isinstance(pllm.Model("org/model"), ModelSource)
    assert isinstance(tiny, ModelSource)
    assert isinstance(bundle, ModelSource)
    assert pllm.TinyModel is TinyModel
    assert pllm.BundleModel is BundleModel
    assert tiny == pllm.Model.from_spec(tiny.to_spec())
    assert bundle == pllm.Model.from_spec(bundle.to_spec())
    assert tiny.to_spec() == {"source": "qwen2", "kind": "tiny", "model_id": "tiny"}
    assert bundle.local_files_only is True


def test_family_imports_do_not_eagerly_import_runtime_engines() -> None:
    code = """
import json
import sys
import pllm.components
print(json.dumps(sorted(name for name in sys.modules if name.startswith('pllm.runtime.'))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(completed.stdout)
    assert loaded == []
