from __future__ import annotations

import json
import struct

import pytest

from pllm import _native
from pllm.components import create_component
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.passes import CipherPruneTokenPruning


def test_mixed_pruning_keeps_numeric_identity_and_serialization_but_changes_execution():
    base = CipherPruneTokenPruning(threshold_q16=10, reduction_threshold_q16=20)
    mixed = base.with_params(backend="mixed")
    assert "backend" not in base.to_spec()["params"]
    assert base != mixed
    assert create_component(mixed.component, mixed.params) == mixed
    inputs = {"attention_q16": [10, 11, 20, 21] * 4,
              "words": [0x80000000, 0xDEADBEEF, 0xFFFFFFFF, 0] * 2}
    left, right = [benchmark_operator(c, lanes=2, samples=1, inputs=inputs) for c in (base, mixed)]
    assert left.numeric_contract_digest == right.numeric_contract_digest
    assert left.configuration_digest != right.configuration_digest
    assert right.quality.checked_outputs == right.quality.exact_integer_outputs == 10
    assert right.resources.peer_rounds > 0
    assert replay_operator_benchmark(right.configuration, inputs=inputs).resources == right.resources
    assert str(0xDEADBEEF) not in right.to_json()
    r = right.resources
    parts = r.key_payload_bytes_by_operation
    assert tuple(sum(p[i] for p in parts) for i in (1, 2)) == r.raw_key_payload_bytes_per_party
    phases = right.samples[0].peer_frame_bytes_by_phase
    assert len(phases) == r.peer_rounds
    assert tuple(sum(p[i] for p in phases) for i in (0, 1)) == r.peer_frame_bytes_by_direction


@pytest.mark.parametrize("scores", [[0, 0, 0, 0], [21, 21, 21, 21], [21, 0, 11, 0], [11, 20, 10, 21]])
def test_threshold_ties_count_disclosure_and_full_width_payloads(scores):
    c = CipherPruneTokenPruning(threshold_q16=10, reduction_threshold_q16=20, backend="mixed")
    width = 1025
    inputs = {"attention_q16": scores * 4, "words": [(i * 0xDEADBEEF) & 0xFFFFFFFF for i in range(4 * width)]}
    report = benchmark_operator(c, lanes=width, samples=1, inputs=inputs)
    count = sum(s > 10 for s in scores)
    assert report.quality.checked_outputs == 1 + count * width + (count if count < 4 else 0)
    assert report.resources.protocol_allocation_estimate_bytes <= 64 << 20
    movement = report.resources.key_payload_bytes_by_operation[1]
    assert (movement[1:] == (0, 0)) == (count in (0, 4))


def test_same_public_count_has_same_schedule_despite_changed_private_positions():
    c = CipherPruneTokenPruning(threshold_q16=10, reduction_threshold_q16=20, backend="mixed")
    reports = [benchmark_operator(c, lanes=4, samples=1, inputs={"attention_q16": scores * 4, "words": list(range(16))})
               for scores in ([11, 0, 0, 21], [0, 21, 11, 0])]
    assert reports[0].resources == reports[1].resources
    assert reports[0].samples[0].peer_frame_bytes_by_phase == reports[1].samples[0].peer_frame_bytes_by_phase


@pytest.mark.parametrize("rows,keep,recent", [(1, 1, 0), (7, 3, 1), (16, 8, 2)])
def test_top_k_compares_scores_and_preserves_complete_original_positions(rows, keep, recent):
    width = 257  # one absolute position plus full 128-feature K and V words
    scores = [float(i * 7 % 5) for i in range(rows)]
    words = [(i * 0x87ABCDEF) & 0xFFFFFFFF for i in range(rows * width)]
    for i in range(rows):
        words[i * width] = 2 * i + 11
    selected = sorted(sorted(range(rows), key=lambda i: (-(i >= rows - recent), -scores[i], i))[:keep])
    expected = [v for i in selected for v in words[i * width:(i + 1) * width]]
    spec = json.dumps({"rows": rows, "words": width, "selection": {"method": "top_k", "keep": keep, "recent": recent}})
    output, degrees, raw = _native.shared_selection_benchmark(spec, struct.pack(f"<{rows}f", *scores), struct.pack(f"<{len(words)}I", *words))
    assert output == struct.pack(f"<{len(expected)}I", *expected)
    assert degrees == b""
    measured = json.loads(raw)
    projected = json.loads(_native.shared_selection_projection(spec, keep))
    assert projected["admitted"]
    assert measured["allocation_estimate_bytes"] == projected["allocation_estimate_bytes"]
    phases = measured["resources"].pop("phases")
    assert measured["resources"] == {k: v for k, v in projected["resources"].items() if k != "phases"}
    assert len(phases) == measured["resources"]["rounds"]


def test_worst_public_count_and_bad_spec_reject_before_input_loading():
    spec = {"rows": 256, "words": 8192, "selection": {"method": "threshold", "threshold": 1, "reduction": 2}}
    assert not json.loads(_native.shared_selection_projection(json.dumps(spec), 256))["admitted"]
    with pytest.raises(ValueError):
        _native.shared_selection_benchmark(json.dumps(spec), b"", b"")
    for invalid in ({**spec, "extra": 1}, {**spec, "rows": 2**64 - 1}, {**spec, "selection": {"method": "unknown"}}):
        with pytest.raises(ValueError):
            _native.shared_selection_projection(json.dumps(invalid), 1)
    for backend in (False, "unknown", None):
        with pytest.raises(ConfigurationError):
            CipherPruneTokenPruning(backend=backend)


def test_mixed_pruning_still_rejects_decoder_transform():
    from pllm import lower_model
    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 32, "max_position_embeddings": 128, "hidden_act": "silu",
              "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True}
    plan = lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2)
    with pytest.raises(ValueError):
        plan.apply(CipherPruneTokenPruning(backend="mixed"))


@pytest.mark.parametrize("batch_size", [4, 16])
def test_batched_pruning_preserves_numeric_replay_and_charges_all_frames(batch_size):
    control = CipherPruneTokenPruning(backend="mixed", threshold_q16=10, reduction_threshold_q16=20)
    candidate = control.with_params(batch_size=batch_size)
    inputs = {"attention_q16": [21, 0, 11, 0] * 4,
              "words": [(i * 0x8765abcd) & 0xffffffff for i in range(4 * 1025)]}
    a, b = [benchmark_operator(c, lanes=1025, samples=1, inputs=inputs) for c in (control, candidate)]
    assert a.numeric_contract_digest == b.numeric_contract_digest
    assert a.resources.raw_key_payload_bytes_per_party == b.resources.raw_key_payload_bytes_per_party
    assert a.resources.peer_rounds > b.resources.peer_rounds
    assert a.resources.peer_payload_bits == b.resources.peer_payload_bits
    assert b.resources == replay_operator_benchmark(b.configuration, inputs=inputs).resources
    assert create_component(candidate.component, candidate.params) == candidate
    phases = b.samples[0].peer_frame_bytes_by_phase
    assert len(phases) == b.resources.peer_rounds
    assert tuple(sum(p[i] for p in phases) for i in (0, 1)) == b.resources.peer_frame_bytes_by_direction
    assert b.resources.protocol_allocation_estimate_bytes <= 64 << 20


@pytest.mark.parametrize("batch_size", [False, 0, 2, 17])
def test_bad_batch_rejects_before_issuance(batch_size):
    with pytest.raises(ConfigurationError):
        CipherPruneTokenPruning(backend="mixed", batch_size=batch_size)
