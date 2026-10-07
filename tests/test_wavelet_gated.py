"""Complete Curl adaptation, public ranges, independent parity and charged costs."""
import json
import hashlib
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

from pllm import _native
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CurlWaveletGatedMultiplyQ9


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/research/wavelet_gated.py"
    spec = importlib.util.spec_from_file_location("wavelet_gated_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("levels", [0, 2, 4, 8])
def test_complete_operator_replays_and_conserves_keys_frames_and_numeric_errors(levels):
    from pllm.metrics.wavelet_gated import _profile
    component = CurlWaveletGatedMultiplyQ9(levels=levels)
    result = benchmark_operator(component, lanes=64, samples=2)
    replay = replay_operator_benchmark(result.configuration)
    assert result.quality.exact_integer_outputs == result.quality.checked_outputs == 128
    assert result.resources == replay.resources
    assert result.quality == replay.quality
    assert result.numeric_contract_digest == _profile(levels)[1]
    resources = result.resources
    parts = resources.key_payload_bytes_by_operation
    assert sum(a for _, a, _ in parts) == resources.raw_key_payload_bytes_per_party[0]
    assert sum(b for _, _, b in parts) == resources.raw_key_payload_bytes_per_party[1]
    assert dict((name, a) for name, a, _ in parts)["haar_lookup"] == 64 * (2 + 4 * (4096 >> levels))
    assert resources.public_table_bytes_per_party == 4 * (4096 >> levels)
    # Entry/exit includes two 32-bit inputs shared to both parties and two output shares.
    assert resources.client_entry_exit_payload_estimate_bytes == 6 * 4 * 64
    assert resources.protocol_allocation_estimate_bytes <= 64 << 20
    assert resources.peer_rounds == 4
    assert resources.peer_frame_bytes == 1248 + 176 * 64
    for sample in result.samples:
        assert sum(sum(frame) for frame in sample.peer_frame_bytes_by_phase) == resources.peer_frame_bytes
    if levels == 0:
        assert result.quality.base_profile_exact_outputs == 128
        assert result.quality.maximum_encoded_profile_error == 0
    record = json.loads(result.to_json())
    assert not record["whole_model_executable"]
    assert not record["independent_role_transport"]
    assert record["full_wire_bytes"] is None
    assert record["peak_memory_bytes"] is None
    assert "gate_q18" not in record
    assert {citation["paper_id"] for citation in record["citations"]} == {"curl", "fss-mixed"}


def test_compression_changes_numeric_identity_but_does_not_remove_conversion_costs():
    raw = benchmark_operator(CurlWaveletGatedMultiplyQ9(), lanes=1, samples=1)
    compressed = benchmark_operator(CurlWaveletGatedMultiplyQ9(levels=4), lanes=1, samples=1)
    assert raw.numeric_contract_digest != compressed.numeric_contract_digest
    assert raw.resources.peer_frame_bytes == compressed.resources.peer_frame_bytes
    for report in (raw, compressed):
        parts = {name: a for name, a, _ in report.resources.key_payload_bytes_by_operation}
        assert parts["beaver_multiply"] == 12
        assert all(parts[name] > 0 for name in ("index_rescale", "up_rescale", "output_rescale"))
        assert report.resources.raw_key_payload_bytes_per_party[0] > parts["haar_lookup"]


def test_inputs_and_invalid_configurations_reject_before_any_material(monkeypatch):
    monkeypatch.setattr(_native, "wavelet_gated_benchmark", lambda *args: pytest.fail("unexpected issuance"))
    for levels in [-1, 9, True, 0.0, "0"]:
        with pytest.raises(ConfigurationError):
            CurlWaveletGatedMultiplyQ9(levels=levels)
    for inputs in ({}, {"gate_q18": [0]}, {"gate_q18": [0], "up_q18": None}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(CurlWaveletGatedMultiplyQ9(), lanes=1, inputs=inputs)
    for gate, up in (([True], [0]), ([0.0], [0]), ([(48 << 18) + 1], [0]), ([0], [-(128 << 18) - 1]), ([], [0])):
        with pytest.raises(ValueError):
            benchmark_operator(CurlWaveletGatedMultiplyQ9(), lanes=1, inputs={"gate_q18": gate, "up_q18": up})
    for lanes in [0, 257, True]:
        with pytest.raises(ConfigurationError):
            benchmark_operator(CurlWaveletGatedMultiplyQ9(), lanes=lanes)


def test_native_boundary_range_frame_and_commitment_validation():
    word = struct.pack("<i", 0)
    for gate, up, plan, session in (
        (b"", word, bytes(32), bytes(32)), (word, word + b"\0", bytes(32), bytes(32)),
        (word, word, bytes(31), bytes(32)), (word, word, bytes(32), bytes(33)),
        (struct.pack("<i", (48 << 18) + 1), word, bytes(32), bytes(32)),
        (word, struct.pack("<i", -(128 << 18) - 1), bytes(32), bytes(32)),
    ):
        with pytest.raises(ValueError):
            _native.wavelet_gated_benchmark(0, gate, up, plan, session)
    with pytest.raises(ValueError):
        _native.wavelet_gated_resources(0, 257)


@pytest.mark.parametrize("levels", [0, 2, 4, 8])
def test_clear_checkpoint_mapping_exhausts_index_edges_against_independent_oracle(levels):
    bridge = benchmark_module().NativeMapping(levels)
    knots = np.arange(-1536, 1537, dtype=np.float32) / 32
    gate = np.concatenate([np.nextafter(knots[1:], np.float32(-np.inf)), knots,
                           np.nextafter(knots[:-1], np.float32(np.inf))]).reshape(1, 1, -1)
    up = np.linspace(-128, 128, gate.size, dtype=np.float32).reshape(gate.shape)
    output = bridge.evaluate(gate, up)
    assert output.dtype == np.float32
    assert bridge.checked == gate.size
    for invalid in [np.float32(48.01), np.float32(-48.01), np.float32(np.nan)]:
        with pytest.raises(ValueError, match="profile domain rejection"):
            bridge.evaluate(np.full((1, 1, 1), invalid, np.float32), np.zeros((1, 1, 1), np.float32))


def test_full_chunk_projection_charges_all_rows_both_parties_and_boundary():
    report = benchmark_operator(CurlWaveletGatedMultiplyQ9(levels=4), lanes=256, samples=1).to_dict()
    cost = benchmark_module().projected_cost(report)
    assert cost["chunks"] * 256 == 24 * 4864 * (39 + 8 - 1)
    assert cost["combined_key_payload_bytes"] == sum(cost["keys_by_operation_bytes"].values())
    assert cost["client_boundary_payload_estimate_bytes"] == 24 * 24 * 4864 * 46
    assert cost["full_wire_bytes"] is None
    assert not cost["executable_decoder"]


def test_saved_cost_and_quality_keep_source_numeric_and_execution_scopes():
    root = Path(__file__).resolve().parents[1]
    cost = json.loads((root / "docs/evidence/research-wavelet-gated-cost.json").read_bytes())
    quality = json.loads((root / "docs/evidence/research-wavelet-gated-quality-qwen25.json").read_bytes())
    sdk = json.loads((root / "docs/evidence/research-wavelet-gated-operators-sdk.json").read_bytes())
    assert cost["fresh_processes"] == len(cost["reports"]) == 9
    assert sum(r["quality"]["exact_integer_outputs"] for r in cost["reports"]) == cost["checked_outputs"] == 2889
    assert not quality["protected_checkpoint_execution"]
    assert quality["memory_guard"]["maximum_swap_growth_bytes"] == 0
    for path, digest in quality["source_sha256"].items():
        assert (root / path).is_file() and len(digest) == 64
        if path.startswith(("benchmarks/", "scripts/")):
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    module = benchmark_module()
    assert hashlib.sha256(json.dumps(module.PUBLIC_HELDOUT).encode()).hexdigest() == quality["public_heldout_sha256"]
    teacher, greedy = quality["teacher_forced"]["summary"], quality["greedy_generation"]["summary"]
    assert teacher["control"]["bit_exact_positions"] == teacher["control"]["exact_kv_positions"] == 48
    assert greedy["control"]["exact_trajectories"] == 12
    for levels, prefill, decode, trajectories in ((0, 10, 35, 10), (2, 7, 33, 5), (4, 8, 33, 4)):
        name = f"haar-{levels}"
        assert teacher[name]["prefill_matches"] == prefill
        assert teacher[name]["decode_matches"] == decode
        assert teacher[name]["bit_exact_positions"] == teacher[name]["exact_kv_positions"] == 0
        assert not teacher[name]["screen_pass"]
        assert greedy[name]["exact_trajectories"] == trajectories
        assert teacher[name]["rejected_prompts"] == greedy[name]["rejected_prompts"] == 0
        assert quality["independent_integer_checked_elements"][name] == 130043904
        reports = [r for r in cost["reports"] + sdk["reports"]
                   if r["configuration"]["component"]["params"]["levels"] == levels]
        assert len(reports) == 4
        assert {r["numeric_contract_digest"] for r in reports} == {quality["numeric_contract_digests"][name]}
        assert len({r["native_implementation_digest"] for r in reports}) == 1
        for report in reports:
            assert report["benchmark_status"] == "completed" and not report["whole_model_executable"]
            assert not report["independent_role_transport"]
            assert report["full_wire_bytes"] is report["peak_memory_bytes"] is None
        full = next(r for r in reports if r["configuration"]["lanes"] == 256)
        assert next(p for p in cost["projections"] if p["levels"] == levels) == {
            "levels": levels, **module.projected_cost(full)}
