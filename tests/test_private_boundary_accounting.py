"""Unit contracts only; these tests do not claim SDK response execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from pllm.runtime.benchmark_memory import BenchmarkMemoryError, private_boundary_memory
from pllm.runtime.topology_accounting import (
    ACK_COVERAGE, BOUNDARY_COVERAGE, BOUNDARY_ROLES, body_coverage_identity,
    online_body_counter_total, prepared_body_accounting,
    private_boundary_body_accounting, source_delivery_accounting,
)


def audit():
    counters = dict(bundle_network_bytes=10,
        private_boundary_body_coverage_failures=0,
        session_authorization_upload_bytes=2, session_authorization_download_bytes=3,
        preparation_upload_bytes=20, preparation_download_bytes=30, correction_push_bytes=100,
        correction_ack_bytes=5, correction_ack_accounted_push_bytes=100,
        inference_upload_bytes=40, inference_download_bytes=60,
        inference_control_upload_bytes=7, inference_control_download_bytes=8,
        preparation_control_upload_bytes=9, preparation_control_download_bytes=10,
        preparation_attributed_body_bytes=150, preparation_prefill_material_body_bytes=60,
        preparation_decode_material_body_bytes=50, preparation_prefill_rows_issued=6,
        preparation_decode_rows_issued=5, preparation_padding_rows_issued=1,
        prepared_stage_rows_issued=12, prepared_stage_rows_reserved=11,
        prepared_stage_rows_claimed=11, prepared_stage_rows_burned=0,
        generation_completed_responses=1, generation_attribution_supported_responses=1,
        generation_output_tokens=3, generation_decode_output_tokens=2,
        generation_requested_output_tokens=6, generation_executed_decode_steps=3,
        generation_prefill_online_body_bytes=90, generation_decode_online_body_bytes=50)
    for role in ("boundary_a", "boundary_b"):
        for phase in ("setup", "online", "teardown"):
            for direction in ("upload", "download"):
                counters[f"role_link.{role}.{phase}_{direction}_bytes"] = 10
    return counters


def test_combined_sender_body_ledger_conserves_without_native_double_count():
    counters = audit()
    counters["private_boundary_native_head_payload_bytes"] = 99999999
    record = {"privacy": counters, "processes": {role: {"cpu_seconds": 1} for role in BOUNDARY_ROLES}}
    value = private_boundary_body_accounting(record)
    assert body_coverage_identity(value) == BOUNDARY_COVERAGE
    assert value["all_link_serialized_body_bytes"] == 304 + 120
    assert value["online_all_link_serialized_body_bytes"] == 140
    assert online_body_counter_total(counters, BOUNDARY_ROLES) == 140
    assert value["aggregate_run_window_cpu_seconds"] == 5
    assert value["total_wire_bytes"] is None
    part = value["traffic_attribution"]
    assert part["known_base_all_link_body_bytes"] == 15 + 90 + 60 + 80
    assert part["known_output_dependent_all_link_body_bytes"] == 50 + 50
    assert part["unattributed_all_link_body_bytes"] == 79
    assert part["total_all_link_body_bytes"] == (part["known_base_all_link_body_bytes"]
        + part["known_output_dependent_all_link_body_bytes"] + part["unattributed_all_link_body_bytes"])
    assert part["base_all_link_body_bytes"] is None  # Residual is explicit.
    assert body_coverage_identity(prepared_body_accounting(record)) == ACK_COVERAGE


@pytest.mark.parametrize("bad", [None, True, -1, 1.5])
def test_partial_boundary_counter_coverage_never_ranks(bad):
    counters = audit()
    counters["role_link.boundary_b.teardown_download_bytes"] = bad
    value = private_boundary_body_accounting({"privacy": counters})
    assert value["all_link_serialized_body_bytes"] is None
    assert body_coverage_identity(value) is None
    assert value["traffic_attribution"]["total_all_link_body_bytes"] is None


def test_cpu_missing_boundary_or_bool_not_accepted():
    record = {"privacy": audit(), "processes": {role: {"cpu_seconds": 1} for role in BOUNDARY_ROLES}}
    record["processes"]["boundary_b"]["cpu_seconds"] = True
    assert private_boundary_body_accounting(record)["aggregate_run_window_cpu_seconds"] is None


@pytest.mark.parametrize("value", [None, True, 1, -1])
def test_failed_or_unknown_transport_attempts_invalidate_directed_totals(value):
    counters = audit()
    counters["private_boundary_body_coverage_failures"] = value
    ledger = private_boundary_body_accounting({"privacy": counters})
    assert ledger["all_link_serialized_body_bytes"] is None
    assert ledger["unknown_reasons"]


def test_old_prepared_phase_counters_leave_boundary_online_residual():
    counters = audit()
    counters.update(generation_prefill_online_body_bytes=70, generation_decode_online_body_bytes=30)
    value = private_boundary_body_accounting({"privacy": counters})["traffic_attribution"]
    assert "decode_online" not in value["components_body_bytes"]
    assert any("unreconciled" in reason for reason in value["unknown_reasons"])
    assert value["output_dependent_all_link_body_bytes"] is None


def test_source_missing_measures_both_artifact_and_full_source_per_worker():
    assert source_delivery_accounting("source-present")["application_bytes"] is None
    assert source_delivery_accounting("missing-workers")["tcp_stream_bytes"] is None
    edges = [{"destination": role, "kind": kind, "application_bytes": 11, "tcp_stream_bytes": 13}
             for role in ("boundary_a", "boundary_b") for kind in ("artifact", "full-source")]
    measured = source_delivery_accounting("missing-workers", edges)
    assert measured["application_bytes"] == 44
    assert measured["tcp_stream_bytes"] == 52
    assert not source_delivery_accounting("missing-workers", edges[:-1])["measured"]
    bodies_only = [{key: value for key, value in edge.items() if key != "tcp_stream_bytes"} for edge in edges]
    assert source_delivery_accounting("missing-workers", bodies_only)["application_bytes"] == 44
    assert source_delivery_accounting("missing-workers", bodies_only)["tcp_stream_bytes"] is None
    edges[0]["application_bytes"] = True
    assert not source_delivery_accounting("missing-workers", edges)["measured"]


def test_memory_table_shared_but_scratch_ledgers_and_imports_not_erased():
    one = private_boundary_memory(1000, 64, lookup_rows=157, head_calls=8)
    two = private_boundary_memory(1000, 64, lookup_rows=157, head_calls=8, concurrent_requests=2)
    assert one["worker_native_table_bytes"] == two["worker_native_table_bytes"] == 64000 + 4000
    assert two["worker_lookup_scratch_bytes"] == 2 * one["worker_lookup_scratch_bytes"] == 128000
    assert one["worker_root_context_ledger_bytes"] == two["worker_root_context_ledger_bytes"]
    assert two["worker_session_ledger_bytes"] == 2 * one["worker_session_ledger_bytes"]
    assert one["worker_import_work_bytes"] > one["table_bytes"]
    assert one["worker_retained_metadata_and_source_scales_bytes"] >= one["scales_bytes"]
    assert one["client_tied_weight_bytes"] == 0
    assert one["publisher_table_export_bytes"] == 0
    assert one["publisher_descriptor_work_bytes"] >= 4 * one["scales_bytes"] + 64
    copied = private_boundary_memory(1000, 64, lookup_rows=157, head_calls=8, shared_matrix=False)
    assert copied["worker_native_table_bytes"] == 2 * one["worker_native_table_bytes"]
    assert copied["publisher_table_export_bytes"] >= 6 * one["table_bytes"]
    assert one["client_work_bytes"] >= 24 * 1000 + 4000


@pytest.mark.parametrize("kwargs", [{"lookup_rows": 4097}, {"head_calls": 4097},
    {"lookup_rows": True}, {"concurrent_requests": 0}, {"shared_matrix": 1}])
def test_memory_protocol_bounds(kwargs):
    arguments = dict(lookup_rows=157, head_calls=8)
    arguments.update(kwargs)
    with pytest.raises(BenchmarkMemoryError):
        private_boundary_memory(1000, 64, **arguments)


def driver():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("boundary_driver", scripts / "benchmark_private_boundary.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_driver_alternates_and_does_not_claim_unit_fixtures_are_sdk_execution():
    module = driver()
    assert module.replicate_order(3) == [(0, "baseline"), (0, "coupled"),
        (1, "coupled"), (1, "baseline"), (2, "baseline"), (2, "coupled")]
    with pytest.raises(ValueError):
        module.replicate_order(2)
    with pytest.raises(RuntimeError):
        module.verify_results([{"client": {"failure": {"stage": "sdk"}}, "input_tokens": 150}])


def test_case_rechecks_memory_before_launch_and_keeps_public_blocker(monkeypatch):
    module = driver()
    rejected = {"admitted": False, "required_bytes": 17,
                "topology": {"host": {"admission_budget_bytes": 16}}}
    monkeypatch.setattr(module, "admission", lambda *args: rejected)
    with pytest.raises(module.BenchmarkBlocked) as caught:
        module.run_case(None, {}, "coupled", 0, b"s" * 32,
            lambda *args: pytest.fail("rejected memory must not start SDK execution"),
            driver_implementation=module.DRIVER_IMPLEMENTATION)
    assert caught.value.code == "complete_case_memory_admission_rejected"
    assert caught.value.memory_admission == rejected


def test_driver_array_commitments_cover_full_exact_bytes_and_shape():
    import numpy as np
    module = driver()
    left = np.arange(12, dtype=np.float32).reshape(3, 4)
    right = left.copy()
    salt = b"x" * 32
    original = module.array_commitment(salt, [left])
    assert module.array_commitment(salt, [right]) == original
    right[-1, -1] += 1
    assert module.array_commitment(salt, [right]) != original
    assert module.array_commitment(salt, [left.reshape(4, 3)]) != original


def test_body_identity_survives_boundary_placement_but_rejects_mismatch():
    from types import SimpleNamespace
    module = driver()
    bundle = SimpleNamespace(privacy={"body_fingerprint": "a" * 64},
        manifest={"fingerprint": "local-boundary", "metadata": {"body_fingerprint": "a" * 64}})
    original = module.bound_body_fingerprint(bundle)
    bundle.manifest["fingerprint"] = "coupled-boundary"
    assert module.bound_body_fingerprint(bundle) == original
    bundle.privacy["body_fingerprint"] = "b" * 64
    with pytest.raises(RuntimeError, match="inconsistent"):
        module.bound_body_fingerprint(bundle)
    bundle.manifest["metadata"]["body_fingerprint"] = "b" * 64
    assert module.bound_body_fingerprint(bundle) != original


def test_matched_specs_keep_original_body_and_reject_unsupported_cache():
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.configuration import Pipeline
    from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
    from pllm.state import ClientPrefixReuse
    module = driver()
    base = Experiment("matched", MaskedLinearCpu(Model.tiny()), Deployment.local(root=".pllm/matched"),
                      ExecutionBudget(requests=1, max_input_tokens=150, max_new_tokens=8))
    pair = module.matched_experiments(base, 150)
    assert resolve_runtime_composition(pair["coupled"].pipeline).private_boundary
    assert not resolve_runtime_composition(pair["baseline"].pipeline).private_boundary
    assert pair["baseline"].pipeline.model == pair["coupled"].pipeline.model
    assert pair["baseline"].pipeline.components["linear"] == pair["coupled"].pipeline.components["linear"]
    with pytest.raises(ValueError, match="do not silently change"):
        cached = Pipeline(base.pipeline.model, {**base.pipeline.components,
            "cache": ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256)})
        module.matched_experiments(base.with_params(pipeline=cached), 150)


def test_target_equivalent_cold_recipe_adds_absent_slots_via_public_pipeline():
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.configuration import Pipeline
    from pllm.kernels import Cpu
    from pllm.preparation import ModelAwareCorrections, PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import ClientBundleTransport, MaskedLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead
    pipeline = MaskedLinearCpu(Model.tiny(), kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues",
                            prefill_pruning="terminal", prefill_chunk_rows=16),
        preparation=ModelAwareCorrections(storage="paged"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand", stage_window=4,
                                    stage_window_bytes=64 << 20),
        delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged", batch_objects=64),
        boundary=None, topology=None)
    baseline = Experiment("cold-only", pipeline, Deployment.local(root=".pllm/unit-cold"),
                          ExecutionBudget(requests=1, max_input_tokens=256, max_new_tokens=8))
    before = baseline.to_spec()
    assert {"boundary", "topology", "cache"}.isdisjoint(pipeline.components)
    pair = driver().matched_experiments(baseline, 150)
    assert baseline.to_spec() == before
    control, coupled = pair["baseline"], pair["coupled"]
    assert coupled.pipeline.components["boundary"] == PrivateLookupAndFullHead()
    assert coupled.pipeline.components["topology"] == PreparedBoundaryRoles()
    assert coupled.pipeline.model == control.pipeline.model == baseline.pipeline.model
    assert coupled.budget == control.budget
    assert coupled.deployment == control.deployment == baseline.deployment
    assert control.budget.max_input_tokens == 150
    stripped = Pipeline(coupled.pipeline.model, {key: value for key, value in coupled.pipeline.components.items()
                         if key not in {"boundary", "topology"}})
    assert stripped.to_spec() == control.pipeline.to_spec()


def test_isolated_cli_report_branch_uses_combined_v4_not_prepared_only():
    from pllm.runtime.benchmark_cli import build_loopback_report
    counters = audit()
    counters.update(plaintext_prompt_bytes_sent=0, plaintext_token_ids_sent=0,
                    preparation_requests_during_online=0)
    record = {"schema_version": "pllm.benchmark_run.v3", "run_id": "unit",
        "status": "completed", "model_id": "unit", "model_fingerprint": "unit",
        "privacy": counters, "durations": {"full_seconds": 1, "online_seconds": 1,
            "ttft_seconds": 0.1, "tokens_per_second": 3},
        "tokens": {"authoritative": True, "input_tokens": 6, "output_tokens": 3}}
    report = build_loopback_report(model_id="unit", tiny=True, max_output_tokens=6,
        warmup_runs=[], runs=[record], roles=BOUNDARY_ROLES, initial_preparation_audit=counters)
    assert report["checks"]["prepared_and_boundary_bodies_tracked"]
    assert report["summary"]["median_covered_all_link_serialized_body_bytes"] == 424
    assert report["topology_accounting"]["startup"]["body_counter_coverage"] == BOUNDARY_COVERAGE
    assert report["topology_accounting"]["runs"][0]["body_counter_coverage"] == BOUNDARY_COVERAGE


def test_connection_json_lists_restore_strict_sdk_tuple_contract():
    import json
    module = driver()
    original = {"base_url": "http://localhost:1", "api_key": "body-key",
        "boundary_urls": ("http://localhost:2", "http://localhost:3"),
        "boundary_api_keys": ("a-key", "b-key")}
    encoded = json.loads(json.dumps(original))
    assert module.normalize_sdk_connection(encoded) == original
    assert type(module.normalize_sdk_connection(encoded)["boundary_urls"]) is tuple
    assert type(encoded["boundary_urls"]) is list  # Do not mutate stdin payload.
    for bad in ({"boundary_urls": "xy"}, {"boundary_urls": ["x", "y"]},
                {"boundary_urls": ["x"], "boundary_api_keys": ["a", "b"]}):
        with pytest.raises(ValueError):
            module.normalize_sdk_connection(bad)


def test_sampler_freeze_is_public_and_does_not_touch_crypto_rng():
    from types import SimpleNamespace
    import numpy as np
    module = driver()
    crypto = object()
    left = SimpleNamespace(rng=np.random.default_rng(), cryptographic_rng=crypto)
    right = SimpleNamespace(rng=np.random.default_rng(), cryptographic_rng=crypto)
    module.freeze_sampler_rng(left)
    module.freeze_sampler_rng(right)
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    assert left.rng.bit_generator.state == np.random.default_rng(module.SAMPLER_SEED).bit_generator.state
    assert left.cryptographic_rng is right.cryptographic_rng is crypto


def test_response_state_commitment_excludes_composition_tags_not_numerical_state():
    from types import SimpleNamespace
    import numpy as np
    module = driver()
    cache = SimpleNamespace(length=2, key=np.zeros((2, 2), np.float32), value=np.ones((2, 2), np.float32))
    saved = SimpleNamespace(next_logits=np.arange(5, dtype=np.float32), token_ids=[1, 2],
        pending_token_ids=[2], snapshot=SimpleNamespace(position=2, caches=[cache], basis="baseline"))
    response = SimpleNamespace(id="fresh-private-id", output_text="private output")
    client = SimpleNamespace(_core=SimpleNamespace(_transformer_conversations={response.id: saved}))
    witness = module.response_state_commitment(client, response, b"x" * 32)
    saved.snapshot.basis = "coupled-composition"
    assert module.response_state_commitment(client, response, b"x" * 32) == witness
    assert witness["pending_count"] == 1
    cache.value[-1, -1] += 1
    assert module.response_state_commitment(client, response, b"x" * 32) != witness
    assert "private output" not in repr(witness) and response.id not in repr(witness)


def unit_result(variant="baseline", replicate=0, count=16):
    """Synthetic verifier evidence, not SDK execution or benchmark measurement."""
    counters = audit()
    counters.update(plaintext_prompt_bytes_sent=0, plaintext_token_ids_sent=0,
        preparation_requests_during_online=0, generation_output_tokens=8,
        generation_decode_output_tokens=7, generation_executed_decode_steps=7,
        generation_requested_output_tokens=8, prepared_stage_rows_discarded=1,
        private_boundary_lookup_rows=count + 7 if variant == "coupled" else 0,
        private_boundary_head_calls=8 if variant == "coupled" else 0)
    diagnostics = []
    if variant == "coupled":
        diagnostics = [{"lookup": {"calls": 2, "rows": count + 7, "upload_bytes": 10, "download_bytes": 10},
                        "head": {"calls": 8, "rows": 8, "upload_bytes": 10, "download_bytes": 10}}]
    else:
        counters = {key: value for key, value in counters.items() if not key.startswith("role_link.boundary_")}
        counters.update(generation_prefill_online_body_bytes=70, generation_decode_online_body_bytes=30)
    for operation in ("lookup", "head"):
        for direction in ("upload", "download"):
            counters[f"private_boundary_{operation}_{direction}_bytes"] = 10 if diagnostics else 0
    ledger = (private_boundary_body_accounting if variant == "coupled" else prepared_body_accounting)({"privacy": counters})
    roles = {"inference", "preparation"} | ({"boundary_a", "boundary_b"} if diagnostics else set())
    row = {"input_tokens": count, "output_tokens": 8, "status": "incomplete", "body_fingerprint": "original-body",
        "client_boundary_storage": {"private_boundary_selected": variant == "coupled", "tied_embeddings": True,
            "local_quantized_stage_ids": [] if diagnostics else ["lm_head", "token_lookup"],
            "boundary_client_weight_stage_ids": [] if diagnostics else ["lm_head", "token_lookup"],
            "retained_local_matrix_ids": [] if diagnostics else ["lm_head", "token_lookup"]},
        "state": {"position": count + 7, "pending_count": 1, "words": 123,
                  **{key: "a" * 64 for key in ("bits", "tokens", "pending", "output", "array_dtype_and_shape")}},
        "trajectory": [{"position": count + index, **{key: "b" * 64 for key in (
            "logits_and_kv", "sampled_token", "rng_before", "rng_after")}} for index in range(8)],
        "request_seconds": 1.0, "first_text_seconds": 0.1, "decode_tps": 10.0,
        "client_request_cpu_seconds": 0.1, "worker_entry_to_first_text_seconds": 0.5}
    client = {"failure": None, "responses": [row], "audit": counters, "ledger": ledger,
        "driver_implementation": driver().DRIVER_IMPLEMENTATION,
        "sampler_diagnostic": {"seed": 42, "temperature": 0, "frozen_runtime_count": 1,
                               "cryptographic_randomness_frozen": False},
        "boundary_operation_diagnostics": diagnostics, "client_peak_rss_bytes": 123456,
        "client_lifetime_cpu_seconds": 0.2, "client_worker_entry_cpu_seconds": 0.15, "client_window_seconds": 2.0,
        "cleanup": {"known_private_containers_empty": True,
            "container_counts": {name: 0 for name in ("cache", "histories", "message_histories",
                "_pending_response_store", "_transformer_states", "_transformer_conversations", "_decode_epoch_responses")},
            "owner_counters_after_close": [], "boundary_transports_observed": int(bool(diagnostics)),
            "boundary_transports_retired_and_owned_http_closed": True}}
    part = ledger["traffic_attribution"]
    return {"variant": variant, "replicate": replicate, "input_tokens": count,
        "source_lock_digest": "c" * 64, "source_case": "source-present", "wan_requested": False,
        "client": client, "provider_birth_cpu_seconds": {role: 1.0 for role in roles},
        "provider_sampled_peak_rss_bytes": {role: 12345 for role in roles}, "wan": None,
        "aggregate_cleanup_cpu_seconds": 7.0, "complete_case_seconds": 3.0,
        "cpu_accounting": driver().complete_case_cpu(10.0, 12.0, 100.0, 105.0),
        "provider_processes_reaped": True,
        "source_delivery": source_delivery_accounting("source-present"),
        "source_inclusive_all_link_application_bytes": None, "source_inclusive_all_link_tcp_stream_bytes": None,
        "priority_metrics": {"client_base_traffic_bytes": None, "client_output_traffic_bytes": None,
            "all_link_base_traffic_bytes": part.get("base_all_link_body_bytes"),
            "all_link_output_traffic_bytes": part.get("output_dependent_all_link_body_bytes"),
            "client_peak_rss_bytes": 123456, "ttft_seconds": 0.1, "decode_tps": 10.0}}


def unit_cohort(module, *, count=16):
    return [unit_result(variant, replicate, count) for replicate, variant in module.replicate_order(3)]


def test_verifier_requires_complete_unique_matched_replicates_not_partial_or_duplicate():
    module = driver()
    cohort = unit_cohort(module)
    verified = module.verify_results(cohort, input_tokens=(16,))
    assert verified["complete"] and verified["exact_full_state_parity"]
    assert verified["checked_responses"] == 6
    assert verified["maximum_client_rss_across_replicates_bytes"] == 123456
    assert module.verify_results(cohort[:-1], input_tokens=(16,), partial=True)["exact_full_state_parity"] is False
    for bad in ([], cohort[:-1], cohort + [deepcopy(cohort[0])], [cohort[0]] * 6):
        with pytest.raises(RuntimeError):
            module.verify_results(bad, input_tokens=(16,))
    with pytest.raises(RuntimeError):
        module.verify_results(cohort, input_tokens=(150,))
    with pytest.raises(ValueError):
        module.verify_results(cohort, input_tokens=(16,), replicates=2)


@pytest.mark.parametrize("path,value", [
    (("client", "failure"), {"stage": "SDK"}),
    (("client", "driver_implementation"), {}),
    (("client", "responses", 0, "state", "bits"), "d" * 64),
    (("client", "responses", 0, "state", "pending_count"), 0),
    (("client", "responses", 0, "output_tokens"), 7),
    (("client", "responses", 0, "trajectory", 3, "rng_after"), "d" * 64),
    (("client", "sampler_diagnostic", "cryptographic_randomness_frozen"), True),
    (("client", "sampler_diagnostic", "frozen_runtime_count"), 0),
    (("client", "audit", "plaintext_token_ids_sent"), 1),
    (("client", "audit", "private_boundary_lookup_rows"), 22),
    (("client", "audit", "private_boundary_head_calls"), 7),
    (("client", "audit", "prepared_stage_rows_discarded"), 0),
    (("client", "ledger", "all_link_serialized_body_bytes"), 99999),
    (("client", "boundary_operation_diagnostics", 0, "head", "rows"), 7),
    (("client", "cleanup", "container_counts", "histories"), 1),
    (("client", "cleanup", "boundary_transports_retired_and_owned_http_closed"), False),
    (("client", "client_peak_rss_bytes"), None),
    (("provider_processes_reaped",), False),
    (("source_lock_digest",), "d" * 64),
    (("source_inclusive_all_link_application_bytes",), 0),
    (("wan_requested",), True),
    (("wan",), {"enforced": True}),
    (("priority_metrics", "ttft_seconds"), 100.0),
    (("cpu_accounting", "parent_all_threads_cpu_seconds"), 0.0),
    (("cpu_accounting", "sampled_client_and_provider_cpu_added_again"), True),
    (("cpu_accounting", "synchronized_all_role_request_cpu_seconds"), 1.0),
    (("client", "responses", 0, "client_request_cpu_seconds"), 100.0),
    (("client", "responses", 0, "worker_entry_to_first_text_seconds"), 0.05),
    (("client", "responses", 0, "client_boundary_storage", "local_quantized_stage_ids"), ["lm_head"]),
    (("client", "responses", 0, "client_boundary_storage", "retained_local_matrix_ids"), ["lm_head"]),
])
def test_verifier_rejects_forged_or_stale_raw_evidence(path, value):
    module = driver()
    cohort = unit_cohort(module)
    parent = cohort[1]
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    with pytest.raises(RuntimeError):
        module.verify_results(cohort, input_tokens=(16,))


def test_verifier_all_workloads_share_one_source_and_checks_every_arm_resource():
    module = driver()
    cohort = unit_cohort(module) + unit_cohort(module, count=150)
    assert module.verify_results(cohort, input_tokens=(16, 150))["checked_responses"] == 12
    del cohort[1]["provider_birth_cpu_seconds"]["boundary_b"]
    with pytest.raises(RuntimeError, match="resource"):
        module.verify_results(cohort, input_tokens=(16, 150))


@pytest.mark.parametrize("input_tokens", ["0", "4090"])
def test_tiny_cli_input_bounds_fail_before_any_source_or_role_work(tmp_path, input_tokens):
    module = driver()
    with pytest.raises(SystemExit) as error:
        module.main(["--tiny", "--prompt-file", str(tmp_path / "not-read.txt"),
                     "--output", str(tmp_path / "not-written.json"), "--input-tokens", input_tokens])
    assert error.value.code == 2
    assert not (tmp_path / "not-written.json").exists()


def test_tiny_cli_rejects_ambiguous_source_and_duplicate_short_workload(tmp_path):
    module = driver()
    common = ["--tiny", "--prompt-file", str(tmp_path / "not-read.txt"),
              "--output", str(tmp_path / "not-written.json")]
    for tail in (["--baseline-spec", str(tmp_path / "not-read.json")],
                 ["--input-tokens", "16", "--short-prompt-file", str(tmp_path / "short.txt")]):
        with pytest.raises(SystemExit):
            module.main(common + tail)
    assert not (tmp_path / "not-written.json").exists()


def test_complete_case_cpu_charges_parent_wan_and_reaped_children_once_not_samples():
    module = driver()
    result = module.complete_case_cpu(10, 14, 100, 107)
    assert result["parent_all_threads_cpu_seconds"] == 4
    assert result["reaped_children_cpu_seconds"] == 7
    assert result["aggregate_cpu_seconds"] == 11
    assert result["sampled_client_and_provider_cpu_added_again"] is False
    assert result["synchronized_all_role_request_cpu_seconds"] is None
    for clocks in ((10, 9, 100, 101), (10, 11, 100, 99), (True, 11, 100, 101),
                   (10, float("nan"), 100, 101)):
        with pytest.raises(RuntimeError):
            module.complete_case_cpu(*clocks)


def test_reaped_cpu_cannot_omit_already_sampled_client_or_provider_cpu():
    module = driver()
    cohort = unit_cohort(module)
    cpu = cohort[1]["cpu_accounting"]
    cpu["reaped_children_cpu_seconds"] = 0.5
    cpu["aggregate_cpu_seconds"] = cpu["parent_all_threads_cpu_seconds"] + 0.5
    cohort[1]["aggregate_cleanup_cpu_seconds"] = cpu["aggregate_cpu_seconds"]
    with pytest.raises(RuntimeError, match="sampled provider/client CPU"):
        module.verify_results(cohort, input_tokens=(16,))


def test_known_storage_diagnostic_checks_all_stage_owners_not_only_head_payload():
    from types import SimpleNamespace
    module = driver()
    head = SimpleNamespace(id="lm_head", role="lm_head", client_weight=None, client_aux_weight=None)
    token = SimpleNamespace(id="token_lookup", role="token_lookup", client_weight=None, client_aux_weight=None)
    bundle = SimpleNamespace(stages={"lm_head": head, "token_lookup": token, "embed_tokens": token},
        _local_matrices={}, privacy={"private_boundary": {}}, manifest={"tied_embeddings": True})
    observed = module.boundary_storage_evidence(bundle)
    assert observed["local_quantized_stage_ids"] == observed["boundary_client_weight_stage_ids"] == []
    token.client_aux_weight = object()
    observed = module.boundary_storage_evidence(bundle)
    assert observed["boundary_client_weight_stage_ids"] == ["token_lookup"]  # Alias counted once.
    assert "object at" not in repr(observed)


def test_driver_source_lock_covers_helpers_and_rejects_mid_cohort_edits(monkeypatch):
    module = driver()
    expected = module.DRIVER_IMPLEMENTATION
    assert "frontier_probe_support.py" in expected and "benchmark_conversation_restore.py" in expected
    module.require_driver_implementation(expected)
    changed = {**expected, "benchmark_private_boundary.py": "f" * 64}
    monkeypatch.setattr(module, "driver_implementation_identity", lambda: changed)
    with pytest.raises(module.BenchmarkBlocked, match="measurement_driver_implementation_changed"):
        module.require_driver_implementation(expected)


def test_tiny_smoke_hi_prompt_declares_22_rendered_tokens_without_model_execution():
    from pllm.runtime.transformer_client import ClientBundle
    # Same portable template as generated Model.tiny(): literal backslash-n,
    # plus one BOS token. Render/tokenize only; no checkpoint or SDK response.
    descriptor = {"type": "byte", "vocab_size": 258, "bos_token_id": 0, "eos_token_id": 1,
        "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}assistant: "}
    bundle = ClientBundle("unit-render-only", {}, {"vocab_size": 258, "bos_token_id": 0, "eos_token_id": 1},
                          descriptor, {}, {}, {})
    rendered = bundle.render_prompt([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert len(bundle.tokenizer().encode(rendered, add_bos=True)) == 22
