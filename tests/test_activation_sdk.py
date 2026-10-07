"""Installed activation contracts, independent parity and complete material scopes."""

import json
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline, _native
from pllm.components import create_component, get, list_components
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CompactPiecewiseActivation, CurlWaveletEncodedTable
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import LogRowGarbledLookup


@pytest.mark.parametrize("component", [
    CompactPiecewiseActivation(), CurlWaveletEncodedTable(),
    CurlWaveletEncodedTable(levels=4), LogRowGarbledLookup(),
])
def test_component_discovery_configuration_execution_and_replay(component):
    assert get(component.component) is type(component)
    assert component.describe() in list_components()
    assert create_component(component.component, component.params) == component
    counts = component.get_params()["public_calibration_counts"]
    counts[0] = 1
    assert component.get_params()["public_calibration_counts"][0] == 0
    changed = component.with_params(public_calibration_counts=counts)
    counts[0] = 2
    assert changed.get_params()["public_calibration_counts"][0] == 1
    experiment = Experiment(
        name="activation-config", pipeline=Pipeline(model=Model.tiny(), components={"method": component}),
        deployment=Deployment.local(root="."),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1),
    )
    replay = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    assert replay.configuration_digest() == experiment.configuration_digest()
    report = benchmark_operator(replay.pipeline.components["method"], lanes=3, samples=2)
    assert report.quality.checked_outputs == report.quality.exact_integer_outputs == 6
    repeated = replay_operator_benchmark(report.configuration)
    assert repeated.resources == report.resources
    assert repeated.quality == report.quality
    assert repeated.configuration_digest == report.configuration_digest
    assert repeated.numeric_contract_digest == report.numeric_contract_digest
    assert all(sample.public_setup_seconds > 0 for sample in report.samples)
    with pytest.raises(FrozenInstanceError):
        report.resources.peer_rounds = 99
    assert report.to_dict()["peak_memory_bytes"] is None
    assert report.to_dict()["full_wire_bytes"] is None
    assert not report.to_dict()["whole_model_executable"]
    baseline = MaskedLinearCpu(Model.tiny())
    with pytest.raises(ConfigurationError):
        Experiment(
            name="unbound-activation", pipeline=baseline.with_params(components__nonlinear=component),
            deployment=experiment.deployment, budget=experiment.budget,
        ).resolve()


def test_common_numeric_contract_and_garbled_material_owners():
    inputs = {"values_q7": [-128, -1, 0, 1, 128]}
    compact, logrow, curl = [
        benchmark_operator(method, lanes=5, samples=1, inputs=inputs)
        for method in (CompactPiecewiseActivation(), LogRowGarbledLookup(), CurlWaveletEncodedTable())
    ]
    assert compact.numeric_contract_digest == logrow.numeric_contract_digest == curl.numeric_contract_digest
    assert compact.quality == logrow.quality == curl.quality
    c = compact.resources
    assert c.role_names == ("garbler_client", "evaluator")
    assert c.raw_key_payload_bytes_per_party[0] == 5 * 25 * 32
    assert c.raw_key_payload_bytes_per_party[1] == c.evaluator_ciphertext_bytes + c.evaluator_constant_label_bytes
    assert c.evaluator_constant_label_bytes > 0
    assert c.public_metadata_bytes_estimate > 0
    assert c.client_entry_exit_payload_estimate_bytes == 5 * (25 * 16 + 64)
    assert c.peer_frame_bytes == 0
    assert logrow.resources.raw_key_payload_bytes_per_party == (5 * 304, 5 * 2144)
    assert logrow.resources.client_entry_exit_payload_estimate_bytes == 5 * 352
    assert curl.resources.role_names == ("worker_a", "worker_b")
    assert curl.resources.combined_raw_key_payload_bytes == 5 * 4100
    assert curl.resources.peer_frame_bytes_by_direction == (5 * 68, 5 * 68)
    assert curl.resources.peer_payload_bits == 5 * 32
    assert curl.resources.public_table_bytes_per_party == 512 * 4
    sources = json.loads((Path(__file__).resolve().parents[1] / "docs/data/research/paper-library.json").read_text())
    sources = {source["id"]: source for source in sources["papers"]}
    for report in (compact, logrow, curl):
        for citation in report.citations:
            assert citation.source_sha256 == sources[citation.paper_id]["sha256"]
            assert citation.url == sources[citation.paper_id]["source_url"]


@pytest.mark.parametrize("levels", [0, 1, 2, 4, 8])
def test_haar_compression_all_codes_and_changed_numeric_identity(levels):
    baseline = benchmark_operator(CurlWaveletEncodedTable(), lanes=1, samples=1)
    component = CurlWaveletEncodedTable(levels=levels)
    for values in (list(range(-128, 128)), [128]):
        report = benchmark_operator(component, lanes=len(values), samples=1, inputs={"values_q7": values})
        assert report.quality.exact_integer_outputs == len(values)
        assert (report.numeric_contract_digest == baseline.numeric_contract_digest) == (levels == 0)
        assert report.resources.combined_raw_key_payload_bytes == len(values) * ((512 >> levels) * 8 + 4)
        assert report.resources.peer_frame_bytes == len(values) * 136
        assert report.resources.protocol_allocation_estimate_bytes == 1 << 20
        assert report.quality.base_profile_exact_outputs <= len(values)
        if levels == 0:
            assert report.quality.maximum_encoded_profile_error == 0


def test_public_density_changes_bound_profile_without_exposing_custom_values():
    counts = [0] * 257
    counts[28:49] = [1_000_000] * 21
    component = LogRowGarbledLookup(max_pieces=8, public_calibration_counts=counts)
    result = benchmark_operator(component, lanes=256, samples=1,
                                inputs={"values_q7": list(range(-128, 128))})
    assert result.quality.base_profile_exact_outputs == 256
    assert result.quality.maximum_encoded_profile_error == 0
    assert result.numeric_contract_digest != benchmark_operator(LogRowGarbledLookup(), lanes=1, samples=1).numeric_contract_digest
    text = result.to_json()
    assert "values_q7" not in text and "outputs_q7" not in text and "input_digest" not in text
    assert json.loads(text)["configuration"]["component"]["params"]["public_calibration_counts"] == counts


@pytest.mark.parametrize("kwargs", [
    {"max_pieces": True}, {"max_pieces": 0}, {"max_pieces": 9},
    {"public_calibration_counts": [0] * 256}, {"public_calibration_counts": [True] * 257},
    {"public_calibration_counts": [-1] * 257}, {"public_calibration_counts": [1_000_001] * 257},
    {"public_calibration_counts": [1_000_000] * 257},
])
def test_public_configuration_bounds(kwargs):
    for component in (CompactPiecewiseActivation, CurlWaveletEncodedTable, LogRowGarbledLookup):
        with pytest.raises(ConfigurationError):
            component(**kwargs)


def test_invalid_inputs_reject_before_issuance(monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid input reached native issuance")

    monkeypatch.setattr(_native, "activation_benchmark", forbidden)
    for values in ([True], [1.5], [-129], [129], [], [0, 1], "x"):
        with pytest.raises((ValueError, ConfigurationError)):
            benchmark_operator(LogRowGarbledLookup(), lanes=1, inputs={"values_q7": values})
    for inputs in ({}, {"values": [0]}, {"values_q7": None}, {"values_q7": [0], "extra": []}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(CurlWaveletEncodedTable(), lanes=1, inputs=inputs)


def test_native_boundary_rejects_malformed_shapes_and_out_of_range_values():
    counts = bytes(257 * 4)
    for method, pieces, histogram, levels, values in (
        ("unknown", 4, counts, 0, b"\0\0"), ("compact", 0, counts, 0, b"\0\0"),
        ("logrow", 4, counts, 1, b"\0\0"), ("curl", 4, counts, 9, b"\0\0"),
        ("curl", 4, counts[:-1], 0, b"\0\0"), ("curl", 4, counts, 0, b"\0"),
        ("curl", 4, counts, 0, bytes(514)), ("curl", 4, counts, 0, struct.pack("<h", 129)),
        ("curl", 4, struct.pack("<257I", *([1_000_000] * 257)), 0, b"\0\0"),
    ):
        with pytest.raises(ValueError):
            _native.activation_benchmark(method, pieces, histogram, levels, values)
