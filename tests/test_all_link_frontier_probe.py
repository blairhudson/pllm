"""Independent resource/accounting checks for the all-link hypothesis screen."""
import importlib.util
import copy
from itertools import combinations
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "all_link_screen", Path(__file__).resolve().parents[1] / "scripts/probe_all_link_frontier.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("mode", probe.MODES)
@pytest.mark.parametrize("raw", [bytes(range(256)), b"\x80", b"\xff\x00\x7f" * 7, bytes(range(127))])
def test_public_codec_preserves_all_signed_values_and_unaligned_planes(mode, raw):
    assert probe.transform(probe.transform(raw, mode), mode, inverse=True) == raw


def test_known_bitplane_order_and_frame_bounds():
    assert probe.transform(b"\x80\x00", "bitplanes-zlib1") == b"\x80\x00"
    for raw in (b"", b"x" * (probe.FRAME_LIMIT + 1)):
        with pytest.raises(ValueError):
            probe.transform(raw, "bitplanes-zlib1")


@pytest.mark.parametrize("horizon", [1, 4, 16])
def test_placement_matches_independent_subset_oracle(horizon):
    items = [{"id": str(i), "macs": c, "weight_and_scale_bytes": w, "pair_stage_bytes": b}
             for i, (c, w, b) in enumerate([(3, 14, 3), (4, 15, 12), (7, 2, 10), (2, 10, 1)])]
    expected = max(sum(horizon * x["pair_stage_bytes"] - x["weight_and_scale_bytes"] for x in subset)
                   for n in range(5) for subset in combinations(items, n)
                   if sum(x["macs"] for x in subset) <= 8)
    actual = probe.choose_placement(items, budget=8, horizon=horizon)
    assert actual["saved_body_bytes_estimate"] == expected
    assert len(actual["selected"]) == len(set(actual["selected"]))
    assert actual["client_dense_macs_per_row"] <= 8


def test_restart_projection_charges_pending_token_and_rejects_nonextension():
    result = probe.history_rows([150, 288, 512], 8)
    assert result["replayed_full_rows"] == 971
    assert result["continued_full_rows"] == 519
    assert result["last_executed_position"] == 519
    with pytest.raises(ValueError):
        probe.history_rows([150, 156], 8)


def test_free_corrections_do_not_erase_outputs_or_public_distribution():
    result = probe.correlation_screen({"pair_bytes": 100, "classes": {"corrections": 28}})
    assert result["free_correction_remaining_pair_bytes"] == 72
    witness = result["field_ring_counterexample"]
    assert witness["field_sum"] != witness["ring_sum"]


def test_publisher_prices_both_operators_and_all_weight_scales():
    class Store:
        keys = ("tokens", "body", "norm")

        def tensor_shape(self, key):
            return {"tokens": (11, 4), "body": (6, 4), "norm": (4,)}[key]

    result = probe.publisher_screen(Store(), {"vocab_size": 11, "hidden_size": 4,
                                              "tie_word_embeddings": True}, 152)
    assert result["inference_artifact_payload_bytes"] == 44 + 24 + 44 + 24 + 16
    assert result["preparation_artifact_payload_bytes"] == 24 + 24
    assert result["saving_in_existing_response_counters"] == 0


def test_historical_ledger_conserves_and_does_not_double_count_receipts():
    document = json.loads(probe.CONTROL.read_text())
    result = probe.conserved_control(document)
    assert result["pair_bytes"] == sum(result["classes"].values()) == 493_575_247
    assert result["classes"]["public_bundle"] == 124_905_091
    broken = copy.deepcopy(document)
    record = next(x for x in broken["results"] if x["execution"]["case"]["name"] == "prepared/digit8")
    record["execution"]["details"]["report"]["topology_accounting"]["runs"][0]["body_bytes_by_edge"][0]["serialized_body_bytes"] += 1
    with pytest.raises(ValueError, match="does not conserve"):
        probe.conserved_control(broken)


def test_unreconciled_stages_and_changed_body_cannot_drive_placement():
    document = json.loads(probe.CONTROL.read_text())
    control = probe.conserved_control(document)
    geometry = json.loads(probe.GEOMETRY.read_text())
    geometry["source"]["body_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="measured quantized body"):
        probe.placement_screen(control, geometry)
    record = next(x for x in document["results"] if x["execution"]["case"]["name"] == "prepared/digit8")
    record["execution"]["details"]["report"]["topology_accounting"]["stages"]["runs"][0]["reconciled_with_protocol_bodies"] = False
    with pytest.raises(ValueError, match="not reconciled"):
        probe.conserved_control(document)
