"""Independent normalization, complete depth, ownership and fail-before-key checks."""
import gc
import math
import weakref

import pytest

from pllm import _native, Model, Deployment, Experiment, ExecutionBudget, Pipeline
from pllm.components import create_component, get
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import PackedEncryptedSoftmax
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import ckks_reference as ckks


@pytest.mark.parametrize("method", ["repeated_squaring", "square_normalize"])
@pytest.mark.parametrize("width", [1, 4, 32])
def test_complete_encrypted_normalization_against_independent_oracles(method, width):
    pytest.importorskip("_sealapi_cpp")
    rows = [0.0]*width + [0.125 if i % 2 else -0.125 for i in range(width)]
    rows += [((i*7) % 17 - 8)/64 for i in range(width)]
    report = benchmark_operator(PackedEncryptedSoftmax(method=method), lanes=3, samples=1,
                                geometry={"width": width}, inputs={"scores": rows})
    assert report.quality.checked_outputs == report.quality.outputs_within_tolerance == 3*width
    assert report.quality.maximum_profile_absolute_error < 0.0002
    assert report.quality.maximum_absolute_error < 0.0002
    assert report.quality.maximum_normalization_error < 0.0002
    assert report.quality.exact_integer_outputs == 0
    assert report.input_source == "caller-owned"
    sample = report.samples[0]
    assert sample.minimum_remaining_depth == 0
    counts = dict(sample.homomorphic_operations)
    assert counts["relinearizations"] > 0
    assert counts.get("rotations", 0) == 3*(width-1)*(1 if method == "repeated_squaring" else 2)
    assert report.resources.peer_frame_bytes == sum(sum(p) for p in sample.peer_frame_bytes_by_phase)
    assert dict((name, a+b) for name,a,b in report.resources.setup_frame_bytes_by_operation)["relinearization-keys"] > 0


@pytest.mark.parametrize("method", ["repeated_squaring", "square_normalize"])
def test_discovery_replay_and_whole_decoder_rejection(method):
    pytest.importorskip("_sealapi_cpp")
    component = PackedEncryptedSoftmax(method=method)
    assert get(component.component) is type(component)
    assert create_component(component.component, component.params) == component
    report = benchmark_operator(component, lanes=1, samples=1, geometry={"width": 2})
    replay = replay_operator_benchmark(report.configuration)
    assert replay.resources == report.resources
    assert replay.numeric_contract_digest == report.numeric_contract_digest
    assert replay.quality.outputs_within_tolerance == 2
    baseline = MaskedLinearCpu(Model.tiny())
    with pytest.raises(ConfigurationError):
        Experiment(name="ckks-softmax-rejection", pipeline=Pipeline(model=baseline.model,
            components={**baseline.components, "nonlinear": component}),
            deployment=Deployment.local(root="local://ckks-softmax-rejection"),
            budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1)).resolve()


def test_invalid_private_range_and_geometry_reject_before_material(monkeypatch):
    pytest.importorskip("_sealapi_cpp")
    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted softmax reached key generation")
    monkeypatch.setattr(ckks, "setup", forbidden)
    for width in (0, True, 3, 33, 64, -1, 1.0):
        with pytest.raises(ConfigurationError):
            benchmark_operator(PackedEncryptedSoftmax(), lanes=1, geometry={"width": width})
    for value in (True, math.nan, math.inf, 0.126, -0.126, 10**500, "secret", None):
        with pytest.raises(ConfigurationError):
            benchmark_operator(PackedEncryptedSoftmax(), lanes=1, geometry={"width": 1}, inputs={"scores": [value]})
    for width in (0, 3, 64):
        with pytest.raises(ValueError):
            _native.ckks_softmax_parameters(width)


@pytest.mark.parametrize("method", ["repeated_squaring", "square_normalize"])
def test_completed_sample_owners_release_before_next_setup(monkeypatch, method):
    pytest.importorskip("_sealapi_cpp")
    setup = ckks.setup
    owners = []
    def tracked(*args, **kwargs):
        assert all(owner() is None for owner in owners)
        result = setup(*args, **kwargs)
        owners.extend(weakref.ref(owner) for owner in (*result[:2], result[1].arithmetic))
        return result
    monkeypatch.setattr(ckks, "setup", tracked)
    enabled = gc.isenabled()
    gc.disable()
    try:
        benchmark_operator(PackedEncryptedSoftmax(method=method), lanes=1, samples=2, geometry={"width": 1})
        assert all(owner() is None for owner in owners)
    finally:
        if enabled:
            gc.enable()
        gc.collect()
