"""Independent ciphertext outputs, public-shape cost conservation and retirement."""
import math

import pytest

from pllm import _native, Model, Deployment, Experiment, ExecutionBudget, Pipeline
from pllm.components import create_component, get
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import PackedEncryptedLinear
from pllm.runtime import ckks_reference as ckks


@pytest.mark.parametrize("rows,columns", [(1, 1), (3, 5), (17, 32)])
@pytest.mark.parametrize("fixture", ["zero", "ones", "signed"])
def test_encrypted_rectangular_products_both_key_layouts_match_independent_oracle(rows, columns, fixture):
    pytest.importorskip("_sealapi_cpp")
    values = (lambda i: 0.0) if fixture == "zero" else ((lambda i: 1.0) if fixture == "ones" else (lambda i: ((i * 17) % 23 - 11) / 16))
    inputs = {"weights": [values(i) for i in range(rows * columns)],
              "vectors": [values(i + 5) for i in range(2 * columns)]}
    reports = [benchmark_operator(PackedEncryptedLinear(rotation_keys=keys), lanes=2, samples=1,
                    geometry={"rows": rows, "columns": columns}, inputs=inputs)
               for keys in ("single", "full")]
    for report in reports:
        assert report.quality.outputs_within_tolerance == report.quality.checked_outputs == 2 * rows
        assert report.quality.exact_integer_outputs == 0
        assert report.quality.maximum_absolute_error < report.quality.absolute_tolerance
        assert report.input_source == "caller-owned"
        assert report.resources.peer_frame_bytes == sum(sum(p) for p in report.samples[0].peer_frame_bytes_by_phase)
        assert report.samples[0].minimum_remaining_depth == 0
        assert set(report.configuration) == {"schema", "component", "geometry", "lanes", "samples"}
    assert reports[0].numeric_contract_digest == reports[1].numeric_contract_digest
    assert reports[0].resources.peer_frame_bytes == reports[1].resources.peer_frame_bytes
    if max(rows, columns) > 2:
        assert sum(v for _,v,_ in reports[0].resources.setup_frame_bytes_by_operation) < sum(v for _,v,_ in reports[1].resources.setup_frame_bytes_by_operation)


def test_sdk_discovery_replay_and_decoder_rejection():
    pytest.importorskip("_sealapi_cpp")
    component = PackedEncryptedLinear()
    assert get(component.component) is type(component)
    assert create_component(component.component, component.params) == component
    report = benchmark_operator(component, lanes=1, samples=1)
    replay = replay_operator_benchmark(report.configuration)
    assert replay.resources == report.resources
    assert replay.numeric_contract_digest == report.numeric_contract_digest
    baseline = MaskedLinearCpu(Model.tiny())
    with pytest.raises(ConfigurationError):
        Experiment(name="ckks-rejection", pipeline=Pipeline(model=baseline.model,
            components={**baseline.components, "linear": component}),
            deployment=Deployment.local(root="local://ckks-rejection"),
            budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1)).resolve()


def test_geometry_input_and_resource_rejection_precedes_key_material(monkeypatch):
    pytest.importorskip("_sealapi_cpp")
    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted inputs reached CKKS key generation")
    monkeypatch.setattr(ckks, "setup", forbidden)
    for geometry in ({"rows": 33, "columns": 1}, {"rows": True, "columns": 1}, {"rows": 1}, {"rows": 0, "columns": 3}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(PackedEncryptedLinear(), lanes=1, geometry=geometry)
    for bad in (True, math.nan, math.inf, 1.001, "secret", None):
        with pytest.raises(ConfigurationError):
            benchmark_operator(PackedEncryptedLinear(), lanes=1, geometry={"rows": 1, "columns": 1},
                               inputs={"weights": [0.5], "vectors": [bad]})
    with pytest.raises(ValueError, match="512 MiB"):
        _native.ckks_parameters(8, 31, True, 64)


@pytest.mark.parametrize("keys", ["single", "full"])
@pytest.mark.parametrize("lanes,batch,rows,columns", [(8, 8, 3, 5), (7, 4, 5, 3), (3, 4, 1, 1)])
def test_same_client_simd_batches_charge_partial_frames_and_preserve_numeric_contract(keys, lanes, batch, rows, columns):
    pytest.importorskip("_sealapi_cpp")
    geometry = {"rows": rows, "columns": columns}
    serial, packed = [benchmark_operator(PackedEncryptedLinear(rotation_keys=keys, batch_size=size),
        lanes=lanes, samples=1, geometry=geometry) for size in (1, batch)]
    assert serial.numeric_contract_digest == packed.numeric_contract_digest
    assert packed.quality.checked_outputs == lanes * rows
    assert packed.quality.maximum_absolute_error < packed.quality.absolute_tolerance
    queries = (lanes + batch - 1) // batch
    assert packed.resources.peer_rounds == queries
    assert packed.resources.peer_frame_bytes * lanes == serial.resources.peer_frame_bytes * queries
    assert packed.resources.setup_frame_bytes_by_operation == serial.resources.setup_frame_bytes_by_operation
    assert packed.resources.protocol_allocation_estimate_bytes > serial.resources.protocol_allocation_estimate_bytes
    assert create_component(packed.configuration["component"]["component"],
                            packed.configuration["component"]["params"]).params["batch_size"] == batch
    assert replay_operator_benchmark(packed.configuration).resources == packed.resources


def test_simd_batch_geometry_rejects_before_setup_and_preserves_default_identity(monkeypatch):
    assert PackedEncryptedLinear(batch_size=1).params == {"rotation_keys": "single"}
    for value in (0, True, 3, 65, 2.0):
        with pytest.raises(ConfigurationError):
            PackedEncryptedLinear(batch_size=value)
    seal = pytest.importorskip("_sealapi_cpp")
    with pytest.raises(ValueError, match="geometry"):
        ckks.setup(seal, _native, None, depth=1, rotation_steps=(32,), relinearization=False,
                   live_ciphertexts=8, queries=1, binding=bytes(32))


def _session(tmp_path, *, depth=1, relin=False):
    seal = pytest.importorskip("_sealapi_cpp")
    return ckks.setup(seal, _native, tmp_path, depth=depth, rotation_steps=(),
                      relinearization=relin, live_ciphertexts=8, queries=2, binding=bytes(32))


@pytest.mark.parametrize("mutation", ["context", "direction", "short", "padding", "replay"])
def test_foreign_malformed_and_replayed_ciphertexts_retire_provider(tmp_path, mutation):
    client, provider, _ = _session(tmp_path)
    request = client.request([0.5])
    if mutation == "replay":
        provider.evaluate(request, lambda a, x: a.multiply_plain(x, 0.5))
    else:
        frame = bytearray(request)
        if mutation == "context": frame[10] ^= 1
        if mutation == "direction": frame[80] = 1
        if mutation == "short": frame.pop()
        if mutation == "padding": frame[-1] = 1
        request = bytes(frame)
    with pytest.raises(ValueError):
        provider.evaluate(request, lambda a, x: a.multiply_plain(x, 0.5))
    with pytest.raises(ValueError, match="retired"):
        provider.evaluate(request, lambda a, x: x)
    client.retire()


def test_client_burns_bad_response_and_explicit_cancellation(tmp_path):
    client, provider, _ = _session(tmp_path)
    request = client.request([0.5])
    reply = provider.evaluate(request, lambda a, x: a.multiply_plain(x, 0.5))
    with pytest.raises(ValueError):
        client.receive(reply[:-1], 1)
    with pytest.raises(ValueError, match="no live response"):
        client.receive(reply, 1)
    provider.retire()
    with pytest.raises(ValueError, match="retired"):
        provider.evaluate(request, lambda a, x: x)


def test_secret_multiplication_and_modulus_alignment_need_real_levels(tmp_path):
    client, provider, metadata = _session(tmp_path, depth=3, relin=True)
    values = [-1.0, -0.25, 0.0, 0.125, 1.0]
    request = client.request(values)
    def polynomial(a, x):
        x2 = a.multiply(x, x)
        x4 = a.multiply(x2, x2)
        return a.multiply_plain(a.add(a.add(x4, x2), x), 0.5)
    reply = provider.evaluate(request, polynomial)
    actual = client.receive(reply, 3)[:len(values)]
    assert max(abs(y - 0.5 * (x**4 + x*x + x)) for x,y in zip(values, actual)) < 0.0002
    assert provider.arithmetic.counts["relinearizations"] == 2
    assert provider.arithmetic.minimum_remaining_depth == 0
    assert "relinearization-keys" in metadata["setup_frame_bytes"]
    client.retire()
    provider.retire()


def test_nexus_contract_is_ckks_not_a_fabricated_garbling_boundary():
    from pllm.protocols import NexusCkksBootstrap
    from pllm.components._planned import NotYetImplementedError
    with pytest.raises(NotYetImplementedError):
        NexusCkksBootstrap()
