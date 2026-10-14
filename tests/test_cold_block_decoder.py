from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cold_block_probe", ROOT / "scripts/probe_cold_block_decoder.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_shared_affine_global_matches_independent_clear_oracle_and_kv():
    remote = probe.SharedGlobal()
    state = (0, 0)
    for tokens in ((0, 1), (2,), (1, 0, 2), (3,)):
        feedback = probe.encode(tokens)
        state = probe.clear_commit(state, feedback)
        remote.commit(feedback)
        assert remote.client_state() == state
        assert remote.forward() == probe.clear_forward(state)
        assert tuple((remote.kv[0][-1][j] + remote.kv[1][-1][j]) % 257 for j in range(2)) == state
    assert remote.feedbacks == remote.forwards == 4


def test_one_share_is_uniform_for_every_private_coordinate():
    # Exhaustive information-theoretic one-coordinate control, NOT proof for
    # deterministic fixture masks, collusion, malicious peers, or private weights.
    for secret in (0, 1, 73, 256):
        assert sorted((secret - mask) % 257 for mask in range(257)) == list(range(257))


def test_next_global_block_requires_feedback_and_changes_with_selected_tokens():
    remote = probe.SharedGlobal()
    remote.commit(probe.encode((0, 1)))
    stale = remote.forward()
    with pytest.raises(ValueError, match="requires sampled-token feedback"):
        remote.forward()
    remote.commit(probe.encode((2, 1)))
    assert remote.forward() != stale
    other = probe.SharedGlobal()
    other.commit(probe.encode((0, 1)))
    other.forward()
    other.commit(probe.encode((0, 1)))
    assert probe.clear_forward(other.client_state()) != probe.clear_forward(remote.client_state())
    assert probe.encode((0, 1)) != probe.encode((1, 0))
    assert probe.encode((0,)) != probe.encode((0, 0))


def test_local_decoder_is_causal_and_history_sensitive():
    latent = (31, 79)
    assert probe.probabilities(latent, [0], 1) != probe.probabilities(latent, [1], 1)
    short = probe.run_fixture(max_tokens=5)
    long = probe.run_fixture(max_tokens=9)
    assert short["tokens"] == long["tokens"][:5]
    assert short["blocks"][0] == long["blocks"][0]


def test_independent_heads_fail_even_with_exact_token_marginals():
    failure = probe.independent_head_failure()
    assert failure["total_variation"] == failure["impossible_sequence_mass"] == 0.5


@pytest.mark.parametrize("greedy", [False, True])
def test_sampling_replay_and_final_partial_block(greedy):
    run = probe.run_fixture(max_tokens=10, greedy=greedy)
    assert run == probe.run_fixture(max_tokens=10, greedy=greedy)
    assert [len(b["tokens"]) for b in run["blocks"]] == [4, 4, 2]
    assert run["remote_decode_steps"] == run["feedback_commits"] == 3
    assert run["local_decoder_rows"] == run["emitted_original_tokens"] == 10
    for block in run["blocks"]:
        for row in block["rows"]:
            assert probe.select(row["probabilities"], row["uniform"], greedy=greedy) == row["token"]


@pytest.mark.parametrize("eos_at,emitted", [(0, 1), (2, 3), (4, 5), (9, 10)])
def test_eos_stops_locally_and_commits_only_actual_tokens(eos_at, emitted):
    run = probe.run_fixture(max_tokens=10, eos_at=eos_at)
    assert run["stop"] == "eos"
    assert run["tokens"][-1] == 3
    assert run["emitted_original_tokens"] == emitted
    assert run["local_decoder_rows"] == emitted
    assert run["feedback_commits"] == (emitted + 3) // 4
    assert run["provider_kv_rows_per_party"] == run["prefill_global_rows"] + run["feedback_commits"]


@pytest.mark.parametrize("cancel_after,steps", [(0, 1), (2, 1), (4, 2)])
def test_cancellation_eager_overissue_and_unused_preparation_charged(cancel_after, steps):
    run = probe.run_fixture(max_tokens=10, cancel_after=cancel_after, unused_prepared_blocks=2)
    cost = probe.cold_cost(run)
    assert run["stop"] == "cancel"
    assert run["emitted_original_tokens"] == cancel_after
    assert run["remote_decode_steps"] == steps
    assert run["unused_preparation_blocks"] == 5 - steps
    assert cost["unused_preparation_bytes"] == (5 - steps) * 65536
    assert cost["all_link_bytes"] == sum(cost["ledger_bytes"].values())
    if cancel_after in (0, 4):
        assert run["unused_issued_forwards"] == 1
        assert cost["unused_forward_body_bytes"] == 32768


def test_full_cold_ledger_charges_every_artifact_forward_feedback_and_preparation():
    run = probe.run_fixture(max_tokens=10)
    spec = probe.CostSpec()
    cost = probe.cold_cost(run, spec)
    assert cost["global_rows_including_prefill_and_overissue"] == 5
    assert cost["ledger_bytes"]["global_forward_up"] == 5 * spec.global_row_up
    assert cost["ledger_bytes"]["feedback_up_including_prefill_and_final"] == 5 * spec.feedback_payload_per_commit
    assert cost["ledger_bytes"]["preparation_download_including_unused"] == 5 * 65536
    larger = probe.cold_cost(run, probe.CostSpec(decoder_weights=spec.decoder_weights + 123))
    assert larger["all_link_bytes"] - cost["all_link_bytes"] == 123
    extra = probe.cold_cost(probe.run_fixture(max_tokens=10, unused_prepared_blocks=1))
    assert extra["all_link_bytes"] - cost["all_link_bytes"] == 65536 + 2 * spec.frame_header
    assert cost["measured_wire_bytes"] is None
    assert cost["real_masked_transformer_compatibility"] is False


@pytest.mark.parametrize("kwargs", [{"block_size": 17}, {"max_tokens": 65},
                                   {"prompt": [0] * 65}, {"cancel_after": -1},
                                   {"unused_prepared_blocks": 65}])
def test_resource_bounds(kwargs):
    with pytest.raises(ValueError):
        probe.run_fixture(**kwargs)


def test_empty_request_keeps_cold_setup_but_no_generated_token_division():
    run = probe.run_fixture(max_tokens=0)
    cost = probe.cold_cost(run)
    assert run["remote_decode_steps"] == 0
    assert cost["all_link_bytes"] > 0
    assert cost["bytes_per_emitted_original_token"] is None


def test_sampling_and_cost_inputs_reject_invalid_domains():
    with pytest.raises(ValueError, match="uniform"):
        probe.select((0.5, 0.5), 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        probe.cold_cost(probe.run_fixture(), probe.CostSpec(decoder_weights=-1))
    with pytest.raises(ValueError, match="rates"):
        probe.cold_cost(probe.run_fixture(), probe.CostSpec(link_bytes_per_second=0))
    with pytest.raises(ValueError, match="prompt token count"):
        probe.cold_cost(probe.run_fixture(), prompt_tokens=-1)


def test_public_prepared_placement_keeps_global_state_and_latents_at_client():
    screen = probe.public_prepared_screen()
    assert screen["latent_release_permission_required"] is False
    assert "global_patch_kv" in screen["client_owns"]
    assert "global_body_matrices" in screen["provider_owns"]
    assert screen["direct_blt_or_megabyte_support"] is False
    assert screen["global_rows"] == 18
    assert screen["projected_cold_bytes"] == sum(screen["ledger_bytes"].values())
    assert screen["provider_resident_matrix_bytes_not_client_download"] > 0
    extra = probe.public_prepared_screen(unused_prepared_rows=3)
    expected = 3 * 4 * 2 * sum(s["out"] for s in screen["stage_shapes_per_layer"])
    assert extra["unused_preparation_correction_bytes"] == expected
    assert extra["projected_cold_bytes"] - screen["projected_cold_bytes"] == expected + 3 * 14 * 64


def test_client_patch_kv_geometry_is_not_whole_process_memory_measurement():
    screen = probe.public_prepared_screen()
    kv = screen["kv_geometry"]
    assert kv["global_kv_bytes_per_slot"] == 256
    assert kv["full_token_global_kv_bytes"] == 71 * 256
    assert kv["patch_global_kv_bytes"] == 18 * 256
    assert kv["patch_global_plus_local_window_kv_bytes"] == 8704
    assert screen["measured_peak_rss_bytes"] is None
    serial = probe.public_prepared_screen(block_size=1)["kv_geometry"]
    assert serial["patch_global_kv_bytes"] == serial["full_token_global_kv_bytes"]
    assert serial["patch_global_plus_local_window_kv_bytes"] > serial["full_token_global_kv_bytes"]


def test_evidence_replays_with_all_source_hashes():
    saved = json.loads((ROOT / "docs/evidence/cold-block-decoder-synthetic.json").read_text())
    assert json.loads(json.dumps(probe.report())) == saved
    assert len(saved["sources"]) == 4
    assert all(len(row["sha256"]) == 64 for row in saved["sources"])
    assert saved["same_original_checkpoint_execution"] is False
    assert saved["protected_serving_ready"] is False
