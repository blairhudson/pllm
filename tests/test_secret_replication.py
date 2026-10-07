"""Scalar/vector semantics, SDK replay, independent modular arithmetic and bounds."""
import json
from pathlib import Path
import struct

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline, _native
from pllm.components import create_component, get
from pllm.configuration import ConfigurationError
from pllm.correlation import CorrelationSource, SecretReplicationPcf
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.profiles import MaskedLinearCpu


@pytest.mark.parametrize("field", [2, 17, 257])
def test_key_compression_keeps_full_field_numeric_and_opening_contract(field):
    pair = [benchmark_operator(SecretReplicationPcf(field=field, encoding=encoding), lanes=19, samples=2)
            for encoding in ("expanded", "punctured")]
    before, after = pair
    assert before.numeric_contract_digest == after.numeric_contract_digest
    assert before.configuration_digest != after.configuration_digest
    assert before.quality == after.quality
    assert after.quality.exact_integer_outputs == 38
    assert before.resources.peer_frame_bytes == after.resources.peer_frame_bytes == 2 * (78 + 4 * 20)
    assert before.resources.raw_key_payload_bytes_per_party == ((2 * field - 1) * 16 + 4,) * 2
    assert after.resources.raw_key_payload_bytes_per_party == ((1 + (field - 1).bit_length()) * 16 + 4,) * 2
    assert after.resources.expanded_correlation_bytes_per_party == (4 + 8 * 19,) * 2
    assert after.resources.prf_field_evaluations_per_party == (19 * (2 * field - 1),) * 2
    assert all(sample.correlation_expansion_seconds > 0 for report in pair for sample in report.samples)
    replay = replay_operator_benchmark(after.configuration)
    assert replay.resources == after.resources
    assert replay.quality == after.quality
    assert not after.to_dict()["whole_model_executable"]
    assert after.to_dict()["full_wire_bytes"] is None


def test_caller_owned_scalar_is_one_value_and_reports_never_store_private_values():
    p = 257
    scalar, vector = [193], [0, 1, 17, 193, 256]
    report = benchmark_operator(SecretReplicationPcf(), lanes=5, samples=1,
                                inputs={"scalar": scalar, "vector": vector})
    assert report.input_source == "caller-owned"
    assert report.quality.exact_integer_outputs == 5
    assert "caller-owned" in report.to_json()
    assert "\"inputs\"" not in report.to_json()
    assert "\"outputs\"" not in report.to_json()
    for s in range(p):
        output, _ = _native.secret_replication_benchmark(p, "punctured", s, struct.pack("<5I", *vector))
        assert list(struct.unpack("<5I", output)) == [s * x % p for x in vector]


def test_invalid_configuration_and_wrong_elementwise_contract_reject_before_issuance(monkeypatch):
    monkeypatch.setattr(_native, "secret_replication_benchmark", lambda *args: pytest.fail("unexpected issuance"))
    for params in ({"field": True}, {"field": 256}, {"encoding": "seeded"}):
        with pytest.raises(ConfigurationError): SecretReplicationPcf(**params)
    for values in ({"scalar": [1, 2], "vector": [1, 2]}, {"scalar": [True], "vector": [1, 2]},
                   {"scalar": [1], "vector": [-1, 2]}, {"scalar": [1], "vector": [1, 257]},
                   {"scalar": [1], "vector": [1.0, 2]}, {"scalar": None, "vector": [1, 2]}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(SecretReplicationPcf(), lanes=2, inputs=values)
    for lanes in (0, 257, True):
        with pytest.raises(ConfigurationError): benchmark_operator(SecretReplicationPcf(), lanes=lanes)


def test_discovery_and_experiment_roundtrip_do_not_enable_a_decoder_correlation():
    component = SecretReplicationPcf()
    assert isinstance(component, CorrelationSource)
    assert get(component.component) is SecretReplicationPcf
    assert create_component(component.component, component.params) == component
    baseline = MaskedLinearCpu(model=Model.tiny())
    experiment = Experiment(name="pcf-operator", pipeline=Pipeline(
        model=baseline.model, components={**baseline.components, "correlation": component}),
        deployment=Deployment.local(root="local://pcf-operator"),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1))
    assert Experiment.from_spec(json.loads(experiment.canonical_bytes())) == experiment
    with pytest.raises(ConfigurationError): experiment.resolve()


def test_native_frame_field_and_shape_admission():
    for field, encoding, scalar, vector in ((256, "punctured", 0, bytes(4)),
        (257, "bad", 0, bytes(4)), (257, "punctured", 257, bytes(4)),
        (257, "punctured", 0, bytes(3)), (257, "punctured", 0, bytes(4 * 4097)),
        (257, "punctured", 0, b""), (257, "punctured", 0, struct.pack("<I", 257))):
        with pytest.raises(ValueError): _native.secret_replication_benchmark(field, encoding, scalar, vector)


def test_paper_citation_matches_locked_primary_source():
    from pllm.metrics.secret_replication import _SOURCE
    library = json.loads((Path(__file__).resolve().parents[1] / "docs/data/research/paper-library.json").read_text())
    records = library["papers"]
    paper = next(row for row in records if row["id"] == _SOURCE.paper_id)
    assert paper["sha256"] == _SOURCE.source_sha256
