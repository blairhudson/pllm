"""Locked placement forecasts, semantic contracts and matched prepared tiny runs."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "placement_frontier_probe", ROOT / "scripts/probe_placement_frontier.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def evidence():
    return json.loads(probe.ARTIFACT.read_text())


def test_replay_is_exact_and_retains_remote_body_contract(evidence):
    # JSON normalization converts the integer histogram keys to saved JSON keys.
    replay = json.loads(
        json.dumps(probe.build(evidence["retained_inputs"], evidence["tiny_numerical_parity"]))
    )
    assert replay == evidence
    assert len(replay["candidates"]) == 23
    assert len(replay["rejected_candidates"]) == 1
    assert "retain a remote body stage" in replay["rejected_candidates"][0]["reason"]
    attention = next(r for r in replay["candidates"] if r["name"] == probe.ATTENTION)
    assert attention["roles"] == ["attention_output", "qkv_projection"]
    assert attention["remote_stage_count"] == 48
    assert attention["schedule_verification"]["phase_output_head_rows"] == {
        "prefill": 1,
        "decode": 1,
    }
    remote = [s for s in evidence["retained_inputs"]["stages"] if not probe.owns(s, attention)]
    assert {s["role"] for s in remote} == {"mlp_gate_up", "mlp_down"}


@pytest.mark.parametrize("mutation", ["ring", "weight", "edge", "prompt"])
def test_mutated_geometry_or_directed_ledger_fails_closed(evidence, mutation):
    value = deepcopy(evidence["retained_inputs"])
    if mutation == "ring":
        value["stages"][0]["wire_bits"] = 16
    elif mutation == "weight":
        value["stages"][0]["weight_sha256"] = "0" * 64
    elif mutation == "edge":
        edges = value["archives"]["warm"][probe.BASELINE]["edges"]
        # Preserve all-link sum while falsifying direction.
        edges[0]["serialized_body_bytes"] += 1
        edges[1]["serialized_body_bytes"] -= 1
    else:
        value["archives"]["warm"][probe.BASELINE]["prompt_digest"] = "0" * 64
    with pytest.raises(ValueError, match="source lock mismatch"):
        probe.build(value)


def test_archive_calibration_counts_cold_storage_and_recurring_prep(evidence):
    calibrated = [r for r in evidence["candidates"] if "archive_calibration" in r]
    assert len(calibrated) == 4
    for row in calibrated:
        calibration = row["archive_calibration"]
        assert calibration["warm_forecast_minus_archive_bytes"] == 0
        assert calibration["online_forecast_minus_archive_bytes"] == 0
        assert calibration["native_snapshot_measured_bytes"] == row["client_body_i8_weight_bytes"]
        assert abs(calibration["cold_payload_forecast_minus_archive_bytes"]) < 10_000
        assert (
            sum(row["directed_warm_body_bytes_forecast"].values())
            == row["warm_all_link_body_bytes_forecast"]
        )
        assert row["directed_warm_body_bytes_forecast"]["preparation->inference"] > 0
        assert row["whole_cpu_remote_fraction"] is None
        assert not row["whole_cpu_cap_admitted"]
    attention = next(r for r in calibrated if r["name"] == probe.ATTENTION)
    assert attention["archive_calibration"]["warm_measured_body_bytes"] == 148_297_986
    assert attention["archive_calibration"]["cold_measured_body_bytes"] > 324_947_564
    assert attention["amortization_first_response_count_beating_baseline_payload_forecast"] == 2


def test_actual_ring_payload_and_independent_pareto_budget(evidence):
    stages = evidence["retained_inputs"]["stages"]
    assert evidence["source_ring_counts"] == {"24": 72, "32": 24}
    assert {s["wire_bits"] for s in stages if s["role"] == "mlp_down"} == {32}
    candidates = evidence["candidates"]
    for row in candidates:
        payload = row["directed_tensor_and_root_payload_bytes_70_rows"]
        remote = [s for s in stages if not probe.owns(s, row)]
        assert payload["preparation->inference"] == sum(
            70 * s["out_features"] * s["wire_bits"] // 8 for s in remote
        )
        assert payload["inference->client"] == payload["preparation->inference"]
        assert row["client_body_resident_array_bytes_floor"] == (
            2 * row["client_body_i8_weight_bytes"] + row["client_body_scale_bytes"]
        )
    remote80 = next(b for b in evidence["budgets"] if b["name"] == "remote80")
    admitted = [r for r in candidates if r["name"] in remote80["eligible_candidates"]]
    assert all(r["body_remote_mac_fraction"] >= 0.8 for r in admitted)
    assert remote80["lowest_warm_body_forecast_candidate"] == probe.ATTENTION
    assert "roles-mlp_down" not in remote80["eligible_candidates"]
    # Independent dominance check using tuples, not the implementation helper.
    vectors = {
        r["name"]: (
            r["warm_all_link_body_bytes_forecast"],
            r["body_client_macs_per_row"],
            r["client_body_resident_array_bytes_floor"],
        )
        for r in admitted
    }
    expected = {
        name
        for name, vector in vectors.items()
        if not any(
            other != vector and all(x <= y for x, y in zip(other, vector, strict=True))
            for other in vectors.values()
        )
    }
    assert set(remote80["warm_pareto"]) == expected
    all_linear = next(b for b in evidence["budgets"] if b["name"] == "all-linear-remote80")
    assert probe.ATTENTION not in all_linear["eligible_candidates"]
    assert all_linear["lowest_warm_body_forecast_candidate"] == "roles-attention_output"


def test_retained_tiny_logits_and_native_body_snapshots_cover_every_valid_candidate(evidence):
    tiny = evidence["tiny_numerical_parity"]
    assert tiny["all_exact"]
    assert {r["name"] for r in tiny["candidates"]} == {r["name"] for r in evidence["candidates"]}
    assert len({r["logits_sha256"] for r in tiny["candidates"]}) == 1
    for row in tiny["candidates"]:
        assert row["compared_logit_vectors"] == 9
        assert row["max_abs_logit_error"] == 0
        assert row["native_body_snapshot_bytes"] == row["client_body_weight_bytes"]
        assert row["native_head_snapshot_bytes"] > 0
        assert row["remote_stage_calls"] > 0


def test_public_reproduction_targets_resolve_to_existing_request_sized_components(evidence):
    for target in (
        probe.public_baseline,
        probe.public_qkv,
        probe.public_output,
        probe.public_attention,
    ):
        resolved = target.resolve()
        assert resolved.inventory_policy == "request-sized"
        assert resolved.bundle_compression == "none"
        assert resolved.prefix_cache_bytes == 0
        assert target.pipeline.kernels.params["threads"] == 1
    review = json.loads(
        (ROOT / "docs/evidence/placement-frontier-review-2026-10-01.json").read_text()
    )
    assert json.loads(json.dumps(probe.review(evidence))) == review


@pytest.fixture(scope="module")
def prepared_reference(tmp_path_factory):
    root = create_tiny_llama_checkpoint(
        tmp_path_factory.mktemp("placement-protocol") / "tiny", num_hidden_layers=24
    )
    candidate = probe.candidates()[0]
    return root, run_prepared(root, candidate)


def run_prepared(root, candidate):
    experiment = Experiment(
        candidate["name"],
        probe.composition(candidate, Model.path(str(root), model_id="placement-protocol")),
        Deployment.local(root=str(root.parent)),
        ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=3),
    )
    with build_roles(experiment, engine_threads=1) as topology:
        with topology.client(
            prepared_inventory_rows=1, background_inventory_refill=False, bundle_cache_mode="off"
        ) as client:
            result = client.responses.create(
                input="Public parity.", max_output_tokens=3, temperature=0
            )
            audit = client.privacy_audit
            assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
            assert audit.preparation_requests_during_online == 0
            return result.output_text, result.usage, audit.inference_stage_calls


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        "client-prefix-1",
        "roles-attention_output",
        "roles-qkv_projection",
        probe.ATTENTION,
        "client-prefix-7",
        "roles-mlp_down+qkv_projection",
    ],
)
def test_recommended_budget_candidates_preserve_matched_prepared_private_response(
    prepared_reference, name
):
    root, reference = prepared_reference
    candidate = next(c for c in probe.candidates() if c["name"] == name)
    actual = run_prepared(root, candidate)
    assert actual[:2] == reference[:2]
    assert 0 < actual[2] < reference[2]
