"""Serializer evidence and adversarial reconciliation, without checkpoint loads."""
from types import SimpleNamespace as NS

import msgpack
import numpy as np
import pytest

from pllm.runtime.benchmark_communication import benchmark_traffic_decomposition
from pllm.runtime.preparation_protocol import PreparationAck, ProtocolError
from pllm.runtime.topology_accounting import prepared_body_accounting


def audit(*, startup=False, prepaid=False):
    offline = 0 if prepaid else 150
    return dict(session_authorization_upload_bytes=10, session_authorization_download_bytes=5,
        preparation_upload_bytes=offline // 3, preparation_download_bytes=offline // 3,
        correction_push_bytes=offline // 3, correction_ack_bytes=7 if offline else 0,
        correction_ack_accounted_push_bytes=offline // 3, bundle_network_bytes=60 if startup else 0,
        inference_upload_bytes=0 if startup else 30, inference_download_bytes=0 if startup else 70,
        inference_control_upload_bytes=3, inference_control_download_bytes=4,
        preparation_control_upload_bytes=2, preparation_control_download_bytes=1,
        preparation_attributed_body_bytes=offline, preparation_decode_material_body_bytes=40 if offline else 0,
        preparation_prefill_material_body_bytes=20 if offline else 0,
        preparation_prefill_rows_issued=2 if offline else 0,
        preparation_decode_rows_issued=4 if offline else 0, preparation_padding_rows_issued=0,
        prepared_stage_rows_issued=6 if offline else 0,
        prepared_stage_rows_reserved=0 if startup else 6,
        prepared_stage_rows_claimed=0 if startup else 3, prepared_stage_rows_burned=0 if startup else 3,
        prepared_stage_rows_discarded=0, generation_prefill_online_body_bytes=0 if startup else 60,
        generation_decode_online_body_bytes=0 if startup else 40,
        generation_completed_responses=0 if startup else 1, generation_output_tokens=0 if startup else 2,
        generation_decode_output_tokens=0 if startup else 1, generation_executed_decode_steps=0 if startup else 2,
        generation_requested_output_tokens=0 if startup else 5,
        generation_attribution_supported_responses=0 if startup else 1)


def report(*, prepaid=False):
    start = audit(startup=True, prepaid=not prepaid)
    run = {"status": "completed", "privacy": audit(prepaid=prepaid),
           "tokens": {"authoritative": True, "output_tokens": 2}}
    return {"schema_version": "pllm.loopback_benchmark.v1", "runs": [run], "warmup_runs": [],
        "topology_accounting": {"startup": prepared_body_accounting({"privacy": start}, initial_preparation=True),
            "runs": [prepared_body_accounting(run)], "warmups": []}}


def test_prepaid_and_burned_material_charged_once_but_mixed_bodies_remain_residual():
    result = benchmark_traffic_decomposition(report(prepaid=True))
    cold = result["cold_first_request"]
    assert result["startup"]["known_output_dependent_all_link_body_bytes"] == 40
    assert result["runs"][0]["known_output_dependent_all_link_body_bytes"] == 40
    assert cold["known_output_dependent_all_link_body_bytes"] == 80
    assert cold["incremental_all_link_mb_per_generated_output"] is None
    assert cold["base_all_link_mb_per_request"] is None
    assert cold["unattributed_all_link_body_bytes"] > 0
    assert cold["total_all_link_body_bytes"] == cold["known_base_all_link_body_bytes"] + 80 + cold["unattributed_all_link_body_bytes"]
    row = result["runs"][0]["attribution"]
    assert row["output_allowance"] == 5 and row["executed_decode_steps"] == 2
    assert row["inventory_rows"]["burned"] == 3
    assert row["reserved_equals_claimed_plus_burned"] is True


def test_per_edge_mismatch_cannot_hide_behind_unchanged_total():
    data = report()
    ledger = data["topology_accounting"]["runs"][0]
    ledger["body_bytes_by_edge"][0]["serialized_body_bytes"] += 1
    ledger["body_bytes_by_edge"][1]["serialized_body_bytes"] -= 1
    row = benchmark_traffic_decomposition(data)["runs"][0]
    assert row["status"] == "Unattributed"
    assert row["unattributed_all_link_body_bytes"] == row["total_all_link_body_bytes"]
    ledger["body_bytes_by_edge"][0]["serialized_body_bytes"] += 1
    assert benchmark_traffic_decomposition(data)["runs"][0]["total_all_link_body_bytes"] is None


def test_row_conservation_mismatch_rejects_payload_floor():
    data = report()
    run = data["runs"][0]
    run["privacy"]["prepared_stage_rows_burned"] += 1
    data["topology_accounting"]["runs"][0] = prepared_body_accounting(run)
    row = benchmark_traffic_decomposition(data)["runs"][0]["attribution"]
    assert row["known_output_dependent_all_link_body_bytes"] == 40  # measured online only
    assert row["reserved_equals_claimed_plus_burned"] is False


@pytest.mark.parametrize("name", ["decode_epochs_started", "inflight_joins", "public_prefix_artifact_local_bytes"])
def test_explicit_unsupported_paths_never_admit_stale_supported_phase_proof(name):
    data = report()
    data["runs"][0]["privacy"][name] = 1
    row = benchmark_traffic_decomposition(data)["runs"][0]
    assert row["known_output_dependent_all_link_body_bytes"] == 0
    assert row["base_all_link_mb_per_request"] is None


def test_response_target_metrics_keep_unknown_heads_and_transport_separate():
    from pllm._benchmark_evidence import measurement
    from pllm.evidence import _metric
    from pllm.runtime.benchmark_response_target import response_metrics
    data = report()
    data["runs"][0]["durations"] = {"ttft_seconds": 1, "tokens_per_second": 2}
    metrics = []
    meter = NS(details={}, metric=lambda name, value, **kwargs: metrics.append(measurement(name, value, **kwargs)))
    response_metrics(data, meter)
    observed = {metric["id"]: metric for metric in metrics}
    # Validate the actual persisted evidence boundary, not only the meter's
    # permissive intermediate dictionaries.
    for index, metric in enumerate(metrics):
        _metric(metric, index, version=2)
    assert meter.details["traffic_decomposition"]["schema"] == "pllm.benchmark_traffic_decomposition.v1"
    assert observed["network.base_body_bytes_per_request"]["value"] is None
    assert observed["network.output_dependent_body_bytes_per_generated_token"]["value"] is None
    assert observed["network.unattributed_body_bytes"]["value"] > 0
    assert observed["network.all_link_stream_bytes"]["value"] is None


def test_actual_ack_counts_roundtrip_and_historical_compatibility():
    old = PreparationAck("a" * 32, "stage", 100, 0)
    assert PreparationAck.unpack(old.pack()).correction_rows is None
    ack = PreparationAck("a" * 32, "stage", 100, 0, correction_rows=4,
        correction_row_payload_bytes=32, correction_ack_bytes=11, correction_row_payload_bits_per_row=64)
    assert PreparationAck.unpack(ack.pack()) == ack
    document = msgpack.unpackb(ack.pack(), raw=False)
    document["cb"] = 33
    with pytest.raises(ProtocolError, match="body accounting"):
        PreparationAck.unpack(msgpack.packb(document, use_bin_type=True))


@pytest.mark.parametrize("widths", [None, bytes((9, 12, 16))])
def test_actual_serialized_row_payload_not_geometry(widths):
    from pllm.runtime.preparation_protocol import CorrectionPush, seeded_ring_profile
    profile = seeded_ring_profile(100)
    correction = CorrectionPush("a" * 32, "inventory", "model", "b" * 64, "stage", "c" * 64,
        4, 2, 3, 8, 8, profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
        np.arange(12, dtype=np.uint32).reshape(4, 3), output_residue_bits=widths)
    counts = {}
    document = msgpack.unpackb(correction.pack(body_counts=counts), raw=False)
    assert counts["row_payload_bytes"] == len(document["d"])
    assert (counts["row_payload_bits_per_row"] * counts["rows"] + 7) // 8 == len(document["d"])


def test_packed_phase_boundary_and_padding_bytes_remain_unattributed():
    ack = PreparationAck("a" * 32, "stage", 100, 0, correction_rows=4,
        correction_row_payload_bytes=19, correction_ack_bytes=7, correction_row_payload_bits_per_row=37)
    ack = PreparationAck.unpack(ack.pack())
    prefill, decode = ack.material_body_bytes_by_phase(1, 2)
    assert (prefill, decode) == (4, 8)
    assert ack.correction_row_payload_bytes - prefill - decode == 7  # padding + shared bytes
    assert ack.material_body_bytes_by_phase(0, 4) == (0, 19)
    with pytest.raises(ProtocolError, match="phase accounting"):
        ack.material_body_bytes_by_phase(1, 4)


def test_phase_demand_terminal_pruning_client_placement_and_padding(monkeypatch):
    from pllm.runtime import model_binding
    monkeypatch.setattr(model_binding.Pipeline, "from_spec", lambda _: NS())
    steps = lambda ids: [{"executor": "remote_stage", "operation_ids": [name], "terminal_row_only": name == "head"} for name in ids]
    plan = NS(runtime_schedule=lambda _: NS(to_dict=lambda: {
        "prefill": {"steps": steps(["hidden", "head", "client"])}, "decode": {"steps": steps(["hidden", "head", "client"])}}))
    bound = NS(_canonical_composition="{}", _plan=plan,
        _stages=[NS(stage_id=name, role="lm_head" if name == "head" else "mlp", semantic_operations=(f"prefill:{name}", f"decode:{name}")) for name in ("hidden", "head", "client")],
        _bundle=NS(stages={name: NS(client_weight=object() if name == "client" else None) for name in ("hidden", "head", "client")}))
    bound.prepared_stage_phase_rows = lambda *args, **kwargs: model_binding.CompiledRuntimeModel.prepared_stage_phase_rows(bound, *args, **kwargs)
    assert bound.prepared_stage_phase_rows(7, 4) == {"prefill": {"hidden": 7, "head": 1}, "decode": {"hidden": 4, "head": 4}}
    assert model_binding.CompiledRuntimeModel.prepared_stage_rows(bound, 0, 0) == {"hidden": 1, "head": 1}
    assert bound.prepared_stage_phase_rows(2, 0, continuation=True)["prefill"]["head"] == 2


@pytest.mark.integration
def test_ordinary_tiny_sdk_prepaid_response_attribution(tmp_path):
    """Parent runs with fresh wheel; not part of lightweight agent validation."""
    import pllm
    from pllm.preparation import ModelAwareCorrections, PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import ClientBundleTransport, MaskedLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.servers import build_roles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)), model_id="traffic")
    pipeline = MaskedLinearCpu(source,
        linear=MaskedLinear(prefill_pruning="terminal", request_encoding="stage_packed", output_encoding="row_residues"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8), preparation=ModelAwareCorrections(storage="paged"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged"))
    experiment = pllm.Experiment("traffic", pipeline, pllm.Deployment.local(root=str(tmp_path / "roles")),
        pllm.ExecutionBudget(max_input_tokens=64, max_new_tokens=3, requests=1))
    with build_roles(experiment, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        client.prepare_response("A", 3)
        before = client.privacy_audit.to_dict()
        response = client.responses.create(input="A", max_output_tokens=3, temperature=0)
        after = client.privacy_audit.to_dict()
        delta = {key: value - before.get(key, 0) for key, value in after.items()}
        run = {"status": "completed", "privacy": delta, "tokens": {"authoritative": True, "output_tokens": response.usage.output_tokens}}
        data = {"schema_version": "pllm.loopback_benchmark.v1", "runs": [run], "warmup_runs": [], "topology_accounting": {
            "startup": prepared_body_accounting({"privacy": before}, initial_preparation=True), "warmups": [], "runs": [prepared_body_accounting(run)]}}
        result = benchmark_traffic_decomposition(data)
        assert result["startup"]["attribution"]["components_body_bytes"]["decode_material"] > 0
        assert result["runs"][0]["attribution"]["components_body_bytes"]["decode_material"] == 0
        cold = result["cold_first_request"]
        assert cold["total_all_link_body_bytes"] == cold["known_base_all_link_body_bytes"] + cold["known_output_dependent_all_link_body_bytes"] + cold["unattributed_all_link_body_bytes"]
        assert cold["base_all_link_mb_per_request"] is None and cold["incremental_all_link_mb_per_generated_output"] is None
