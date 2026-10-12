from copy import deepcopy

import pytest

from pllm.metrics import communication_per_token
from pllm.runtime.benchmark_communication import (
    SCHEMA, benchmark_traffic_decomposition, body_decomposition,
)
from pllm.runtime.topology_accounting import prepared_body_accounting


def _audit():
    return dict(
        bundle_network_bytes=10,
        session_authorization_upload_bytes=2, session_authorization_download_bytes=3,
        preparation_upload_bytes=20, preparation_download_bytes=30, correction_push_bytes=100,
        correction_ack_bytes=5, correction_ack_accounted_push_bytes=100,
        inference_upload_bytes=40, inference_download_bytes=60,
        inference_control_upload_bytes=7, inference_control_download_bytes=8,
        preparation_control_upload_bytes=9, preparation_control_download_bytes=10,
        preparation_attributed_body_bytes=150,
        preparation_prefill_material_body_bytes=60, preparation_decode_material_body_bytes=50,
        preparation_prefill_rows_issued=6, preparation_decode_rows_issued=5,
        preparation_padding_rows_issued=1, prepared_stage_rows_issued=12,
        generation_completed_responses=1, generation_attribution_supported_responses=1,
        generation_output_tokens=3, generation_decode_output_tokens=2,
        generation_requested_output_tokens=6, generation_executed_decode_steps=3,
        generation_prefill_online_body_bytes=70, generation_decode_online_body_bytes=30,
    )


def _record(audit=None, n=3):
    return {"status": "completed", "cold": True, "privacy": _audit() if audit is None else audit,
            "tokens": {"authoritative": True, "input_tokens": 6, "output_tokens": n}}


def _report():
    record = _record()
    return {"schema_version": "pllm.loopback_benchmark.v1",
            "configuration": {"max_output_tokens": 6, "prefill_cache_mode": "prefix"},
            "runs": [record], "warmup_runs": [],
            "topology_accounting": {"startup": None, "warmups": [],
                                    "runs": [prepared_body_accounting(record)]}}


def _conserved(row):
    assert row["total_all_link_body_bytes"] == (
        row["known_base_all_link_body_bytes"]
        + row["known_output_dependent_all_link_body_bytes"]
        + row["unattributed_all_link_body_bytes"])


def test_direct_material_floor_includes_prepaid_eos_and_controls_stay_residual():
    report = _report()
    row = communication_per_token(report)["traffic_decomposition"]["runs"][0]
    assert row["schema"] == SCHEMA
    assert row["status"] == "Partial"
    assert row["total_all_link_body_bytes"] == 304
    assert row["known_base_all_link_body_bytes"] == 145  # setup + online prefill + material
    assert row["known_output_dependent_all_link_body_bytes"] == 80  # all prepaid material + decode
    assert row["unattributed_all_link_body_bytes"] == 79  # mixed envelopes, padding, ack, controls
    assert row["incremental_all_link_mb_per_generated_output"] is None
    assert row["base_all_link_mb_per_request"] is None
    assert row["unused_output_material_body_bytes"] is None  # not row-ratio extrapolation
    assert row["output_allowance"] == 6
    assert row["attribution"]["executed_decode_steps"] == 3  # EOS step differs from N-1
    assert row["known_output_dependent_all_link_mb_per_generated_output"] == 80 / 3 / 1e6
    _conserved(row)


@pytest.mark.parametrize("bad", [None, -1, True, 1.0])
def test_missing_or_invalid_measurement_never_becomes_zero(bad):
    row = body_decomposition(bad, base_body_bytes=10, output_dependent_body_bytes=20)
    assert row["total_all_link_body_bytes"] is None
    assert row["base_all_link_body_bytes"] is None
    assert row["output_dependent_all_link_body_bytes"] is None
    assert row["unattributed_all_link_body_bytes"] is None


def test_exact_contract_unused_is_subset_not_extra_charge():
    row = body_decomposition(100, base_body_bytes=20, output_dependent_body_bytes=80,
                             unused_output_material_body_bytes=30)
    assert row["status"] == "Attributed"
    assert row["unused_output_material_body_bytes"] == 30
    _conserved(row)
    for kwargs in ({"base_body_bytes": 101}, {"output_dependent_body_bytes": -1},
                   {"output_dependent_body_bytes": 20, "unused_output_material_body_bytes": 21}):
        invalid = body_decomposition(100, **kwargs)
        assert invalid["status"] == "Unattributed"
        assert invalid["unattributed_all_link_body_bytes"] == 100
        _conserved(invalid)


def test_row_mismatch_rejects_material_without_losing_total():
    record = _record()
    record["privacy"]["prepared_stage_rows_issued"] += 1
    row = prepared_body_accounting(record)["traffic_attribution"]
    assert row["known_base_all_link_body_bytes"] == 85
    assert row["known_output_dependent_all_link_body_bytes"] == 30
    assert "row totals mismatch" in row["unknown_reasons"][0]
    _conserved(row)


def test_cached_prefill_zero_is_observed_not_subtracted_from_prompt_length():
    audit = _audit()
    audit.update(generation_prefill_online_body_bytes=0, generation_decode_online_body_bytes=100,
                 preparation_prefill_material_body_bytes=0, preparation_decode_material_body_bytes=110,
                 preparation_prefill_rows_issued=0, preparation_decode_rows_issued=11)
    row = prepared_body_accounting(_record(audit))["traffic_attribution"]
    assert row["known_base_all_link_body_bytes"] == 15
    assert row["known_output_dependent_all_link_body_bytes"] == 210
    _conserved(row)


def test_missing_cache_recomputes_raw_counters_and_misalignment_stays_unknown():
    historical = deepcopy(_report())
    historical["topology_accounting"]["runs"][0].pop("traffic_attribution")
    row = benchmark_traffic_decomposition(historical)["runs"][0]
    assert row["status"] == "Partial"
    assert row["known_base_all_link_body_bytes"] == 145
    assert row["known_output_dependent_all_link_body_bytes"] == 80
    assert row["unattributed_all_link_body_bytes"] == 79
    historical["topology_accounting"]["runs"] = []
    result = benchmark_traffic_decomposition(historical)
    assert result["windows_aligned"] is False
    assert result["runs"][0]["total_all_link_body_bytes"] is None


def test_cold_startup_charged_once_warmups_separate_no_output_rate_undefined():
    report = _report()
    audit = {key: 0 for key in _audit()}
    audit.update(bundle_network_bytes=10, session_authorization_upload_bytes=2,
                 session_authorization_download_bytes=3, prepared_stage_rows_reserved=0,
                 prepared_stage_rows_claimed=0, prepared_stage_rows_burned=0)
    startup = prepared_body_accounting({"privacy": audit}, initial_preparation=True)
    report["topology_accounting"]["startup"] = startup
    warm_audit = _audit()
    warm_audit.update(generation_output_tokens=0, generation_decode_output_tokens=0,
                      generation_completed_responses=0, generation_attribution_supported_responses=0,
                      generation_executed_decode_steps=0, generation_prefill_online_body_bytes=0,
                      generation_decode_online_body_bytes=0)
    record = _record(warm_audit, n=0)
    report["warmup_runs"] = [record]
    warm = prepared_body_accounting(record)
    report["topology_accounting"]["warmups"] = [warm]
    result = benchmark_traffic_decomposition(report)
    cold = result["cold_first_request"]
    assert cold["total_all_link_body_bytes"] == 319
    assert cold["known_base_all_link_body_bytes"] == 90
    assert cold["known_output_dependent_all_link_body_bytes"] == 50
    assert cold["base_all_link_body_bytes"] is None
    assert cold["incremental_all_link_mb_per_generated_output"] is None
    assert result["warmups"][0]["generated_output_tokens"] == 0
    assert result["runs"][0]["total_all_link_body_bytes"] == 304
    _conserved(cold)


@pytest.mark.parametrize("source", [None, "forged-external-source", "sdk_audit_phase_and_prepared_row_payload_counters_v1"])
def test_cached_source_cannot_bypass_raw_counter_reconciliation(source):
    report = _report()
    ledger = report["topology_accounting"]["runs"][0]
    ledger["traffic_attribution"] = body_decomposition(304, base_body_bytes=304)
    if source is not None:
        ledger["traffic_attribution"]["source"] = source
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["status"] == "Partial"
    assert row["known_base_all_link_body_bytes"] == 145
    assert row["known_output_dependent_all_link_body_bytes"] == 80
    assert row["unattributed_all_link_body_bytes"] == 79
    assert row["base_all_link_mb_per_request"] is None
    assert row["incremental_all_link_mb_per_generated_output"] is None
    _conserved(row)
    report["runs"][0]["privacy"]["prepared_stage_rows_issued"] += 1
    stale = benchmark_traffic_decomposition(report)["runs"][0]
    assert stale["known_base_all_link_body_bytes"] == 85
    assert stale["known_output_dependent_all_link_body_bytes"] == 30
    assert stale["unattributed_all_link_body_bytes"] == 189
    _conserved(stale)


@pytest.mark.parametrize("privacy", [None, {}])
def test_external_cached_partition_without_raw_evidence_cannot_mint_headlines(privacy):
    report = _report()
    report["runs"][0]["privacy"] = privacy
    ledger = report["topology_accounting"]["runs"][0]
    ledger["traffic_attribution"] = body_decomposition(304, base_body_bytes=304)
    ledger["traffic_attribution"]["source"] = "external"
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["status"] == "Unattributed"
    assert row["unattributed_all_link_body_bytes"] == 304
    assert row["base_all_link_mb_per_request"] is None
    assert row["incremental_all_link_mb_per_generated_output"] is None
    _conserved(row)


def test_startup_cached_partition_requires_reconciled_bounded_counter_evidence():
    report = _report()
    startup = deepcopy(report["topology_accounting"]["runs"][0])
    cached = startup["traffic_attribution"]
    cached.pop("source")
    cached["known_base_all_link_body_bytes"] = 304
    cached["known_output_dependent_all_link_body_bytes"] = 0
    report["topology_accounting"]["startup"] = startup
    row = benchmark_traffic_decomposition(report)["startup"]
    assert row["known_base_all_link_body_bytes"] == 145
    assert row["known_output_dependent_all_link_body_bytes"] == 80
    assert row["base_all_link_body_bytes"] is None
    _conserved(row)
    cached.pop("audit_counters")
    row = benchmark_traffic_decomposition(report)["startup"]
    assert row["status"] == "Unattributed"
    assert row["unattributed_all_link_body_bytes"] == 304
    _conserved(row)


def test_missing_observed_output_count_preserves_known_setup_without_headline():
    audit = {key: 0 for key in _audit()}
    audit.update(bundle_network_bytes=10, session_authorization_upload_bytes=2,
                 session_authorization_download_bytes=3, prepared_stage_rows_reserved=0,
                 prepared_stage_rows_claimed=0, prepared_stage_rows_burned=0)
    audit.pop("generation_output_tokens")
    record = _record(audit)
    report = _report()
    report["runs"] = [record]
    report["topology_accounting"]["runs"] = [prepared_body_accounting(record)]
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["total_all_link_body_bytes"] == 15
    assert row["known_base_all_link_body_bytes"] == 15
    assert row["unattributed_all_link_body_bytes"] == 0
    assert row["base_all_link_mb_per_request"] is None
    assert row["incremental_all_link_mb_per_generated_output"] is None
    _conserved(row)


def test_omitted_body_counter_set_does_not_emit_headline_or_fake_rate():
    report = _report()
    report["topology_accounting"]["runs"][0]["tracked_body_counter_set_present"] = False
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["total_all_link_body_bytes"] is None
    assert row["base_all_link_mb_per_request"] is None
    assert row["incremental_all_link_mb_per_generated_output"] is None


def test_completed_no_output_preserves_bytes_without_inventing_phase_boundary():
    audit = _audit()
    audit.update(generation_completed_responses=0, generation_attribution_supported_responses=0,
                 generation_output_tokens=0, generation_decode_output_tokens=0,
                 generation_executed_decode_steps=0, generation_prefill_online_body_bytes=0,
                 generation_decode_online_body_bytes=0)
    report = _report()
    record = _record(audit, n=0)
    report["runs"] = [record]
    report["topology_accounting"]["runs"] = [prepared_body_accounting(record)]
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["total_all_link_body_bytes"] == 304
    assert row["known_output_dependent_all_link_body_bytes"] == 50
    assert row["known_output_dependent_all_link_mb_per_generated_output"] is None
    assert row["incremental_all_link_mb_per_generated_output"] is None
    _conserved(row)


def test_mixed_coverage_rejects_lifecycle_sum():
    report = _report()
    startup = deepcopy(report["topology_accounting"]["runs"][0])
    startup["body_counter_coverage"] = "other_coverage"
    report["topology_accounting"]["startup"] = startup
    result = benchmark_traffic_decomposition(report)
    assert result["body_counter_coverage"] is None
    assert result["covered_lifecycle"]["total_all_link_body_bytes"] is None
    assert result["summary"]["base_all_link_mb_per_request"] is None


def test_audit_output_count_mismatch_invalidates_attribution_not_measured_total():
    report = _report()
    report["runs"][0]["tokens"]["output_tokens"] = 4
    row = benchmark_traffic_decomposition(report)["runs"][0]
    assert row["status"] == "Unattributed"
    assert row["total_all_link_body_bytes"] == 304
    assert row["unattributed_all_link_body_bytes"] == 304
    assert row["known_output_dependent_all_link_mb_per_generated_output"] is None
