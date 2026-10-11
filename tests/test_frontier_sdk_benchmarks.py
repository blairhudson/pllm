"""Installed target contracts for promoted frontier mechanisms."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pllm
from pllm._benchmark_contract import TargetUnavailable
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_targets import execute_target
from test_benchmark_targets import experiment, measurement


def hybrid_config():
    path = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    config = json.loads(path.read_bytes())
    config["text_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=512,
        max_position_embeddings=256, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    config["text_config"]["rope_parameters"]["mrope_section"] = [2, 1, 1]
    return config


def test_recurrence_target_exact_bits_and_admission(experiment, tmp_path):
    config = hybrid_config()
    target = BenchmarkTarget.recurrence(experiment, config=config, tokens=5)
    control = target.with_params(owner="copy_reference")
    records = []
    for choice in (control, target):
        assert BenchmarkTarget.from_spec(choice.to_spec()).digest == choice.digest
        m = measurement(tmp_path)
        execute_target(choice, m)
        assert m.oracle["outcome"] == "passed"
        assert m.oracle["checked_values"] == 5 * 4 * 8 + 4 * 8 * 8
        records.append(m)
    assert records[0].comparison == records[1].comparison
    assert records[1].details["persistent_workspace_bytes"] > 0
    with pytest.raises(TargetUnavailable, match="exclusive_recurrence_required"):
        execute_target(target.with_params(layer=3), measurement(tmp_path))
    with pytest.raises(ValueError):
        target.with_params(owner="unbounded")
    with pytest.raises(ValueError):
        target.with_params(storage="sealed_local")


def test_recurrence_installed_fresh_process_suite(experiment, tmp_path):
    target = BenchmarkTarget.recurrence(experiment, config=hybrid_config(), tokens=4)
    cases = (BenchmarkCase("copy", target.with_params(owner="copy_reference")),
             BenchmarkCase("owned", target))
    result = benchmark_suite(BenchmarkSuite(cases, Sampling(processes=1)),
                             output=tmp_path / "recurrence.json")
    assert [row.to_dict()["status"] for row in result.results] == ["completed", "completed"]
    assert result.compare("copy", "owned", "target.cpu")["status"] == "comparable"
