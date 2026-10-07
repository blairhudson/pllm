from __future__ import annotations

import json
import struct

import pytest

from pllm import _native
from pllm.components import create_component
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import SharedRescale
from pllm.passes import CipherPruneTokenPruning, PlanPass


def test_public_component_geometry_replay_and_complete_material_accounting():
    component = CipherPruneTokenPruning()
    assert isinstance(component, PlanPass)
    assert create_component(component.component, component.params) == component
    report = benchmark_operator(component, lanes=4, samples=2, geometry={"tokens": 4, "heads": 2})
    replay = replay_operator_benchmark(report.configuration)
    assert report.resources == replay.resources
    assert report.configuration["geometry"] == {"tokens": 4, "heads": 2}
    assert "geometry" not in component.get_params()
    assert report.quality.checked_outputs == report.quality.exact_integer_outputs > 0
    parts = report.resources.key_payload_bytes_by_operation
    assert tuple(sum(row[i] for row in parts) for i in (1, 2)) == report.resources.raw_key_payload_bytes_per_party
    assert report.resources.protocol_allocation_estimate_bytes <= 64 << 20
    assert report.resources.raw_key_payload_bytes_per_party[1] == (
        report.resources.evaluator_ciphertext_bytes + report.resources.evaluator_constant_label_bytes)
    assert all(sample.input_sharing_seconds >= 0 for sample in report.samples)


@pytest.mark.parametrize("scores", [[10, 11, 20, 21], [21, 0, 11, 0], [0, 0, 0, 0], [21, 21, 21, 21]])
def test_private_integer_oracle_includes_threshold_ties_empty_and_signed_payloads(scores):
    component = CipherPruneTokenPruning(threshold_q16=10, reduction_threshold_q16=20)
    values = {"attention_q16": scores * 4, "words": [0xDEADBEEF, 0x80000000, 0xFFFFFFFF, 0] * 2}
    report = benchmark_operator(component, lanes=2, samples=1, inputs=values)
    assert report.input_source == "caller-owned"
    kept = sum(v > 10 for v in scores)
    assert report.quality.checked_outputs == 1 + 2 * kept + (kept if kept < 4 else 0)
    # The report archives neither complete token payloads nor degree/keep masks.
    assert str(0xDEADBEEF) not in report.to_json()
    assert '"attention_q16"' not in report.to_json()
    replay = replay_operator_benchmark(report.configuration, inputs=values)
    assert replay.resources == report.resources


def test_native_projection_is_exact_and_full_width_rejects_before_material():
    raw = json.loads(_native.token_pruning_projection(4, 1, 2, 10, 20, 3))
    _, _, measured = _native.token_pruning_benchmark(
        4, 1, 2, 10, 20, struct.pack("<16I", *([10, 11, 20, 21] * 4)), struct.pack("<8I", *range(8)))
    assert raw["phases"] == json.loads(measured)["phases"]
    assert raw["admitted"]
    assert not json.loads(_native.token_pruning_projection(39, 14, 896, 10, 20, 30))["admitted"]
    with pytest.raises(ValueError, match="64 MiB"):
        _native.token_pruning_benchmark(39, 14, 896, 10, 20, b"", b"")


@pytest.mark.parametrize("params", [
    {"threshold_q16": True}, {"threshold_q16": -1}, {"reduction_threshold_q16": 0},
    {"threshold_q16": 10, "reduction_threshold_q16": 10}, {"reduction_threshold_q16": 65537},
])
def test_threshold_contract_rejects_invalid_public_parameters(params):
    with pytest.raises(ConfigurationError):
        CipherPruneTokenPruning(**params)


def test_geometry_and_input_validation_preserve_existing_report_replay():
    c = CipherPruneTokenPruning()
    for geometry in ({"tokens": 4}, {"tokens": True, "heads": 1}, {"tokens": 257, "heads": 1}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(c, lanes=2, samples=1, geometry=geometry)
    with pytest.raises(ConfigurationError):
        benchmark_operator(c, lanes=2, samples=1, inputs={"attention_q16": [65537] * 16, "words": [0] * 8})
    with pytest.raises(ConfigurationError):
        benchmark_operator(SharedRescale(), lanes=1, samples=1, geometry={"tokens": 4, "heads": 1})
    old = benchmark_operator(SharedRescale(), lanes=1, samples=1)
    assert old.configuration["schema"] == "pllm.operator_benchmark.v1"
    assert "geometry" not in old.configuration
    assert replay_operator_benchmark(old.configuration).resources == old.resources


def test_operator_availability_does_not_activate_a_decoder_transform():
    from pllm import lower_model

    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 32, "max_position_embeddings": 128, "hidden_act": "silu",
              "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True}
    plan = lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2)
    with pytest.raises(ValueError):
        plan.apply(CipherPruneTokenPruning())


def test_public_cost_fixtures_cover_count_dependent_phases():
    from pathlib import Path
    import runpy

    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/research/token_pruning.py"))
    for tokens, heads, features, case in benchmark["CASES"]:
        report = benchmark["worker"](tokens, heads, features, case)
        assert report["quality"]["checked_outputs"] == report["quality"]["exact_integer_outputs"]
        if case in {"none", "all"}:
            phases = report["resources"]["key_payload_bytes_by_operation"]
            assert phases[1][1:] == [0, 0]
