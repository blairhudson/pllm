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


def test_batch_target_counts_independent_inventories_and_preserves_comparison(experiment, tmp_path):
    target = BenchmarkTarget.preparation_batch(experiment, rows=3, inventories=3, batch_size=3, window_rows=8)
    records = []
    for size in (1, 3):
        candidate = target.with_params(batch={**target.to_spec()["parameters"]["batch"], "batch_size": size})
        assert BenchmarkTarget.from_spec(candidate.to_spec()).digest == candidate.digest
        root = tmp_path / str(size)
        root.mkdir()
        m = measurement(root)
        execute_target(candidate, m)
        assert m.oracle["outcome"] == "passed"
        assert m.details["batch"]["inventories"] == 3
        assert sum(g["rows"] for g in m.details["samples"][0]["groups"]) == 9
        records.append(m)
    assert records[0].comparison == records[1].comparison
    for changes in ({"storage": "paged"}, {"chunk_rows": 1}, {"kernel": "digit4"}):
        with pytest.raises(ValueError):
            target.with_params(**changes)


def test_checkpoint_installed_fresh_process_suite(experiment, tmp_path):
    from pllm.quantization import SymmetricPerRow
    experiment = experiment.with_params(pipeline=MaskedLinearCpu(experiment.pipeline.model,
        quantization=SymmetricPerRow(causal_reduction="prefix_f32")))
    target = BenchmarkTarget.checkpoint(experiment, tokens=3, steps=1, page_rows=2)
    cases = (BenchmarkCase("pages", target.with_params(storage="layer_pages")),
             BenchmarkCase("graph", target))
    result = benchmark_suite(BenchmarkSuite(cases, Sampling(processes=1, memory_bytes=1 << 30)),
                             output=tmp_path / "checkpoint.json")
    assert [row.to_dict()["status"] for row in result.results] == ["completed", "completed"]
    assert result.compare("pages", "graph", "target.cpu")["status"] == "comparable"
    raw = (tmp_path / "checkpoint.json").read_text()
    for private in ('"pending":', '"key":', '"head":', '"frames":'):
        assert private not in raw


def test_preparation_batch_installed_fresh_process_suite(experiment, tmp_path):
    target = BenchmarkTarget.preparation_batch(experiment, rows=3, inventories=3, batch_size=3, window_rows=8)
    control = target.with_params(batch={**target.to_spec()["parameters"]["batch"], "batch_size": 1})
    result = benchmark_suite(BenchmarkSuite((BenchmarkCase("serial", control), BenchmarkCase("batched", target)),
                                           Sampling(processes=1)), output=tmp_path / "batch.json")
    assert [row.to_dict()["status"] for row in result.results] == ["completed", "completed"]
    assert result.compare("serial", "batched", "target.cpu")["status"] == "comparable"
