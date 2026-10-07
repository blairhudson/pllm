import numpy as np
import pytest
import hashlib
import json
from pathlib import Path


@pytest.fixture
def mapping_class(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.research.nonlinear_decoder_quality import NativeMapping
    return NativeMapping


def test_installed_quality_mapping_signed_boundaries_and_tails(mapping_class):
    mapping = mapping_class()
    gate = np.asarray([-48, -16, -1, -1 / 4096, 0, 1 / 4096, 1, 16, 48], dtype=np.float32)
    up = np.asarray([128, -128, 2, -1, 0, 1, -2, 128, -128], dtype=np.float32)
    output = mapping.evaluate(gate, up)
    assert output.dtype == np.float32 and output.shape == gate.shape
    assert output[4] == 0
    assert output[-1] == -6144
    assert mapping.checked == 9


@pytest.mark.parametrize("gate,up", [(48.01, 1), (-48.01, 1), (1, 128.01), (float("nan"), 1)])
def test_quality_mapping_rejects_without_clipping(mapping_class, gate, up):
    with pytest.raises(ValueError, match="profile domain rejection"):
        mapping_class().evaluate(np.asarray([gate], dtype=np.float32), np.asarray([up], dtype=np.float32))


def test_fresh_quality_evidence_conserves_positions_and_locks_reproduction():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence/research-nonlinear-decoder-quality-qwen25.json").read_text())
    assert report["schema"] == "pllm.nonlinear_decoder_quality.v1"
    assert not report["protected_checkpoint_execution"]
    assert report["memory_guard"]["maximum_swap_growth_bytes"] == 0
    for source, digest in report["source_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == digest
    assert report["configuration"]["input_conversion"] == "float32-to-signed-q24-floor-reject"
    for name in ("control", "masked-wire-q12"):
        teacher = report["teacher_forced"]
        rows = [row for row in teacher["trials"] if row["candidate"] == name]
        assert len(rows) == 12 and all(row["rejection"] is None for row in rows)
        positions = [position for row in rows for position in row["positions"]]
        summary = teacher["summary"][name]
        assert len(positions) == summary["completed_positions"] == 48
        assert sum(p["selection_match"] for p in positions if p["position"] == 0) == summary["prefill_matches"]
        assert sum(p["selection_match"] for p in positions if p["position"] > 0) == summary["decode_matches"]
        assert sum(p["all_logits_bit_exact"] for p in positions) == summary["bit_exact_positions"]
        assert sum(p["kv_bit_exact"] for p in positions) == summary["exact_kv_positions"]
        generation = report["greedy_generation"]
        rows = [row for row in generation["trials"] if row["candidate"] == name]
        summary = generation["summary"][name]
        assert len(rows) == summary["attempted_prompts"] == 12
        assert sum(row["sequence_match"] for row in rows) == summary["exact_trajectories"]
        for key in ("generated_tokens", "reference_output_tokens", "matching_positions", "matching_prefix_tokens"):
            assert sum(row[key] for row in rows) == summary[key]
        assert summary["matching_prefix_tokens"] <= summary["matching_positions"] <= summary["generated_tokens"]


def test_larger_decoder_admission_preserves_replay_and_allocation_owners():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence/research-nonlinear-decoders-qwen3-preflight.json").read_text())
    assert report["schema"] == "pllm.nonlinear_decoder_preflight.v1"
    source = root / "benchmarks/research/nonlinear_decoders.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["configuration_source_sha256"]
    assert [case["layout"] for case in report["cases"]] == ["unfused", "fused"]
    for case in report["cases"]:
        spec = case["experiment"]
        assert spec["pipeline"]["model"]["source"] == "Qwen/Qwen3-4B"
        assert spec["pipeline"]["model"]["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
        admission = case["admission"]
        assert admission["schema"] == "pllm.benchmark_memory.v1"
        assert not admission["admitted"] and admission["selected_backend"] is None
        estimate = admission["estimate"]
        assert estimate["max_input_tokens"] == spec["budget"]["max_input_tokens"]
        assert estimate["max_output_tokens"] == spec["budget"]["max_new_tokens"]
        assert set(estimate["provider_peak_bytes"]) == {"worker_a", "worker_b"}
        assert (estimate["client_peak_bytes"] + sum(estimate["provider_peak_bytes"].values())
                == estimate["native_total_peak_bytes"])
        native = admission["candidates"]["native"]
        assert native["required_host_bytes"] > admission["host"]["admission_budget_bytes"]
        assert any("safe headroom" in reason for reason in native["reasons"])
