"""Synthetic evidence validates rejection/ranking, never claims model execution."""
from copy import deepcopy

import pytest

from pllm.metrics import BuildRef
from pllm.deployment import WanConditions
from pllm.runtime.topology_accounting import prepared_body_accounting, private_boundary_body_accounting
from benchmarks.research.private_boundary_decode import (
    BOUNDARY_COUNTERS, CHOICES, CONTROL, METRICS, NAMES, PREPARED_COUNTERS,
    Rejected, audit, matched_suite, recipes,
)


def snapshot(boundary, links=None):
    roles = {"client", "preparation", "inference"}
    edges = {(a, b) for a in roles for b in roles if a != b}
    if boundary:
        roles |= {"boundary_a", "boundary_b"}
        edges |= {(a, b) for role in ("boundary_a", "boundary_b")
                  for a, b in (("client", role), (role, "client"))}
    links = {f"{a}->{b}": 100 for a, b in edges} if links is None else links
    return {"schema": "pllm.native_wan.v1", "backend": "native-shared-tcp-pacer", "enforced": True,
        "conditions_digest": WanConditions().digest, "role_parties": {role: role for role in roles},
        "elapsed_seconds": 10, "burst_bytes": 16384, "directed_stream_bytes": links,
        "total_stream_bytes": sum(links.values()),
        "client_stream_bytes": sum(value for edge, value in links.items() if "client" in edge.split("->")),
        "parties": {role: {direction: {"bytes_per_second": rate,
            "admitted_bytes": sum(value for edge, value in links.items()
                if edge.split("->")[index] == role)}
            for direction, rate, index in (("upload", 5e6, 0), ("download", 12.5e6, 1))} for role in roles}}


def ledger(boundary, privacy, *, startup=False):
    account = private_boundary_body_accounting if boundary else prepared_body_accounting
    return account({"privacy": privacy}, initial_preparation=startup)


def body_links(ledger):
    result = {}
    for edge in ledger["body_bytes_by_edge"]:
        link = f'{edge["source"]}->{edge["destination"]}'
        result[link] = result.get(link, 0) + edge["serialized_body_bytes"]
    return result


@pytest.fixture
def cohort():
    suite = matched_suite(build=BuildRef(expected_digest="a" * 64))
    document = {"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(),
        "cohort": "b" * 64, "complete": True, "results": []}
    for index, case in enumerate(suite.cases):
        for replicate in range(3):
            boundary = case.name != "maintained"
            packed = boundary and CHOICES[case.name][1] == "uniform_residues"
            privacy = {key: 100 for key in (*PREPARED_COUNTERS, *BOUNDARY_COUNTERS)}
            privacy.update(inference_download_bytes=100, preparation_rows=100, inference_stage_calls=16,
                preparation_attributed_body_bytes=300, preparation_prefill_material_body_bytes=200,
                preparation_decode_material_body_bytes=20, preparation_prefill_rows_issued=92,
                preparation_decode_rows_issued=8, preparation_padding_rows_issued=0,
                prepared_stage_rows_reserved=100, prepared_stage_rows_claimed=100,
                prepared_stage_rows_burned=0, prepared_stage_rows_discarded=0,
                generation_executed_decode_steps=7, generation_output_tokens=8)
            privacy.update({key: 100 for key in ("bundle_network_bytes", "session_authorization_upload_bytes",
                "session_authorization_download_bytes", "correction_ack_accounted_push_bytes",
                *(f"{role}_control_{direction}_bytes" for role in ("inference", "preparation")
                  for direction in ("upload", "download")))})
            privacy.update(private_boundary_head_download_bytes=4000 - 1000 * packed,
                private_boundary_body_coverage_failures=0, private_boundary_lookup_rows=157,
                private_boundary_head_calls=8)
            for role in ("boundary_a", "boundary_b"):
                privacy[f"role_link.{role}.online_upload_bytes"] = 100
                privacy[f"role_link.{role}.online_download_bytes"] = 2050 - 500 * packed
                for phase in ("setup", "teardown"):
                    for direction in ("upload", "download"):
                        privacy[f"role_link.{role}.{phase}_{direction}_bytes"] = 25
            startup_privacy = {key: 0 for key in privacy}
            startup_privacy["bundle_network_bytes"] = 1000 + index * 10
            startup_ledger = ledger(boundary, startup_privacy, startup=True)
            run_ledger = ledger(boundary, privacy)
            total = startup_ledger["all_link_serialized_body_bytes"] + run_ledger["all_link_serialized_body_bytes"]
            startup_links, run_links = body_links(startup_ledger), body_links(run_ledger)
            before = snapshot(boundary, {link: 100 + startup_links.get(link, 0)
                for link in snapshot(boundary)["directed_stream_bytes"]})
            wire = snapshot(boundary, {link: count + 100 + run_links.get(link, 0)
                for link, count in before["directed_stream_bytes"].items()})
            metrics = {metric: 10 for metric in METRICS}
            metrics.update({"client.peak_rss": 180 << 20,
                "process.peak_rss": (300 if index == 4 and replicate == 2 else 220) << 20,
                "network.covered_body_bytes": total,
                "network.all_link_stream_bytes": wire["total_stream_bytes"]})
            report = {"checks": {"passed": True}, "runs": [{"cold": True,
                "tokens": {"authoritative": True, "input_tokens": 150, "output_tokens": 8},
                "privacy": privacy, "generation": {"output_text_digest": "same-output"}}],
                "configuration": {"source_lock_digest": "same-source", "prompt_digest": "same-salted-prompt",
                    "wan_emulation_digest": WanConditions().digest, "sampling": {"temperature": 0},
                    "provider_backend": "native", "wan_emulation": True},
                "client_process_memory": {"scope": "client and in-process dashboard",
                    "after": {"lifetime_peak_rss_bytes": 180 << 20}},
                "memory_guard": {"tripped": False, "maximum_host_swap_growth_bytes": 0},
                "wan_readiness": {"emulation": {"native_stream_rate_snapshots_checked": True}},
                "native_network_accounting": {"schema": "pllm.native_network_accounting.v1",
                    "samples": {"startup": {"wan_emulation": deepcopy(before)},
                        "runs": [{"before": {"wan_emulation": before},
                        "after": {"wan_emulation": wire}}]}},
                "topology_accounting": {"startup": startup_ledger, "warmups": [], "runs": [run_ledger]}}
            stage_edges = {"client->inference": 100, "inference->client": 100,
                "client->preparation": 100, "preparation->client": 100, "preparation->inference": 100}
            report["topology_accounting"]["stages"] = {"runs": [{
                "body_bytes_by_stage_and_edge": {"layer.0": dict(stage_edges)},
                "expected_stage_body_bytes_by_edge": dict(stage_edges),
                "measured_body_bytes_by_edge": dict(stage_edges), "reconciled_with_protocol_bodies": True}]}
            bindings = [{"sampler_seed": 42, "temperature": 0, "cryptographic_randomness_frozen": False,
                "failed": False, "numeric_body_fingerprint": "same-body", "selections": [
                    {"position": 150 + step, "cache_lengths": [150 + step], "checked_values": 100,
                     "logits_and_active_kv": "same-state", "selected_token": "same-token",
                     "rng_before": "same-rng", "rng_after": "same-rng"} for step in range(8)]}]
            document["results"].append({"status": "completed",
                "metrics": [{"id": key, "value": value} for key, value in metrics.items()],
                "execution": {"case": case.to_spec(), "replicate": replicate,
                    "suite_digest": suite.digest, "sampling": suite.sampling.to_spec(), "cohort": "b" * 64,
                    "build": {"digest": "a" * 64},
                    "fixture": {"digest": "same-source", "metadata": {
                        "cached_checkpoint": True, "owned_client_cache": "fresh"}},
                    "isolation": {"fresh_process": True, "controller_included": False},
                    "details": {"report": report, "response_state_audit": {"bindings": bindings,
                        "client_memory_owners": ["sdk", "dashboard", "native_wan_relay", "audit", "role_supervision"]}}}})
    return document


def test_recipes_only_change_head_options_and_keep_defaults():
    values = recipes()
    assert tuple(values) == NAMES
    assert values[CONTROL].pipeline.components["boundary"].to_spec()["params"] == {}
    stripped = []
    for name in CHOICES:
        spec = values[name].to_spec()
        del spec["pipeline"]["components"]["boundary"]
        stripped.append(spec)
    assert all(spec == stripped[0] for spec in stripped)
    assert all(exp.budget.max_input_tokens == 150 and exp.budget.max_new_tokens == 8 for exp in values.values())


def test_all_variants_compare_with_both_controls_without_historical_pooling(cohort):
    result = audit(cohort)
    assert result["clients"] == 15
    assert "tiers" not in result
    assert set(result["comparisons"]) == set(CHOICES)
    assert all(set(comparisons) == {CONTROL, "maintained"} for comparisons in result["comparisons"].values())
    assert not result["groups"][NAMES[-1]]["observed_tiers"]["low"]
    assert result["groups"][NAMES[-1]]["maximum_client_peak_bytes"] == 300 << 20
    reconciliation = result["body_reconciliation"][NAMES[-1]][0]
    assert reconciliation["head_reply_delta_bytes"] == -1000
    assert reconciliation["bootstrap_and_metadata_residual_bytes"] == 30


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "salt", "build", "body", "state", "wire",
    "pressure", "swap", "owners", "audit", "recipe", "metric", "scope", "head", "sampler", "bounds"])
def test_independently_reject_bad_evidence(cohort, mutation):
    row = cohort["results"][-1]
    execution = row["execution"]
    report = execution["details"]["report"]
    if mutation == "missing":
        cohort["results"].pop()
    elif mutation == "duplicate":
        execution["replicate"] = 0
    elif mutation == "salt":
        execution["cohort"] = "c" * 64
    elif mutation == "build":
        execution["build"]["digest"] = "c" * 64
    elif mutation == "body":
        report["runs"][0]["privacy"]["inference_upload_bytes"] += 1
    elif mutation == "state":
        execution["details"]["response_state_audit"]["bindings"][0]["selections"][-1]["rng_after"] = "changed"
    elif mutation == "wire":
        report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]["backend"] = "unpaced"
    elif mutation == "pressure":
        report["memory_guard"]["tripped"] = True
    elif mutation == "swap":
        report["memory_guard"]["maximum_host_swap_growth_bytes"] = 1
    elif mutation == "owners":
        execution["details"]["response_state_audit"]["client_memory_owners"].remove("native_wan_relay")
    elif mutation == "audit":
        del execution["details"]["response_state_audit"]["bindings"][0]["selections"][0]["rng_before"]
    elif mutation == "recipe":
        cohort["suite"]["cases"][-1]["target"]["parameters"]["state_audit"] = False
    elif mutation == "metric":
        next(m for m in row["metrics"] if m["id"] == "target.latency")["value"] = float("nan")
    elif mutation == "scope":
        report["topology_accounting"]["startup"]["boundary_body_counter_set_present"] = False
    elif mutation == "head":
        report["runs"][0]["privacy"]["private_boundary_head_download_bytes"] = 4000
    elif mutation == "sampler":
        execution["details"]["response_state_audit"]["bindings"][0]["cryptographic_randomness_frozen"] = True
    else:
        cohort["suite"]["sampling"]["processes"] = 1
    with pytest.raises(Rejected):
        audit(cohort)


def test_exactly_three_replicates_required():
    with pytest.raises(ValueError, match="three"):
        matched_suite(build=BuildRef(), processes=1)


def test_tiny_recipe_has_single_request_and_same_audit_envelope():
    suite = matched_suite(build=BuildRef(expected_digest="a" * 64), tiny=True)
    assert len(suite.cases) == 5
    for case in suite.cases:
        parameters = case.target.to_spec()["parameters"]
        assert parameters["state_audit"] is True
        assert parameters["model"]["experiment"]["budget"] == {
            "max_input_tokens": 128, "max_new_tokens": 2, "requests": 1}


def test_preflight_prices_every_choice_and_equal_audit_scratch(monkeypatch):
    from benchmarks.research.private_boundary_decode import preflight
    import pllm.metrics
    import pllm.runtime.benchmark_response_audit as state_audit

    seen, audits = [], []

    def memory(experiment, **kwargs):
        seen.append((experiment.pipeline.components.get("boundary"), kwargs))
        return {"admitted": True}

    def scratch(report, memory_bytes, *, enforced_wan):
        audits.append((memory_bytes, enforced_wan))
        return report

    monkeypatch.setattr(pllm.metrics, "benchmark_memory", memory)
    monkeypatch.setattr(state_audit, "state_audit_admission", scratch)
    result = preflight(matched_suite(build=BuildRef(expected_digest="a" * 64)))
    assert [row["case"] for row in result] == list(NAMES)
    assert len(seen) == 5 and seen[0][0] is None
    assert all(kwargs == {"backend": "native", "enforced_wan": True,
        "memory_budget_bytes": 8 << 30} for _, kwargs in seen)
    assert audits == [(8 << 30, True)] * 5


@pytest.mark.parametrize("window", ["startup", "request"])
@pytest.mark.parametrize("name", ["maintained", NAMES[-1]])
def test_cached_total_and_metric_collusion_cannot_hide_raw_bodies(cohort, window, name):
    row = next(row for row in cohort["results"] if row["execution"]["case"]["name"] == name)
    accounting = row["execution"]["details"]["report"]["topology_accounting"]
    target = accounting["startup"] if window == "startup" else accounting["runs"][0]
    target["all_link_serialized_body_bytes"] -= 1
    next(m for m in row["metrics"] if m["id"] == "network.covered_body_bytes")["value"] -= 1
    with pytest.raises(Rejected, match="decode_body_totals_differ_from_raw"):
        audit(cohort)


@pytest.mark.parametrize("mutation", ["owner", "phase", "duplicate", "control_total", "online_total",
    "prepared_raw", "ack_raw", "setup_raw", "teardown_raw", "startup_raw", "startup_missing"])
def test_directed_owner_phase_and_raw_counters_reconstructed(cohort, mutation):
    from benchmarks.research.private_boundary_decode import _body

    row = cohort["results"][-1]
    report = row["execution"]["details"]["report"]
    metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
    run = report["topology_accounting"]["runs"][0]
    raw = report["runs"][0]["privacy"]
    if mutation == "owner":
        run["body_bytes_by_edge"][0]["destination"] = "boundary_a"
    elif mutation == "phase":
        run["body_bytes_by_edge"][-1]["phase"] = "online"
    elif mutation == "duplicate":
        run["body_bytes_by_edge"].append(deepcopy(run["body_bytes_by_edge"][0]))
    elif mutation == "control_total":
        run["lifecycle_control_serialized_body_bytes"] -= 1
    elif mutation == "online_total":
        run["online_all_link_serialized_body_bytes"] -= 1
    elif mutation == "prepared_raw":
        raw["preparation_download_bytes"] += 1
    elif mutation == "ack_raw":
        raw["correction_ack_accounted_push_bytes"] -= 1
    elif mutation == "setup_raw":
        raw["role_link.boundary_a.setup_upload_bytes"] += 1
    elif mutation == "teardown_raw":
        raw["role_link.boundary_b.teardown_download_bytes"] += 1
    elif mutation == "startup_raw":
        report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"][
            "bundle_network_bytes"] += 1
    else:
        del report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"][
            "preparation_upload_bytes"]
    with pytest.raises(Rejected, match="decode_(body_|raw_body_)"):
        _body(report, metrics, True)


@pytest.mark.parametrize("window", ["startup", "request"])
def test_valid_pacer_totals_still_bound_each_body_link_in_its_own_window(cohort, window):
    from benchmarks.research.private_boundary_decode import _body, _wire

    row = cohort["results"][-1]
    report = row["execution"]["details"]["report"]
    metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
    samples = report["native_network_accounting"]["samples"]
    if window == "startup":
        point = samples["startup"]
        links = dict(point["wan_emulation"]["directed_stream_bytes"])
        links["inference->client"] = 0
    else:
        point = samples["runs"][0]["after"]
        links = dict(point["wan_emulation"]["directed_stream_bytes"])
        before = samples["runs"][0]["before"]["wan_emulation"]["directed_stream_bytes"]
        links["boundary_a->client"] = before["boundary_a->client"]
    point["wan_emulation"] = snapshot(True, links)
    metrics["network.all_link_stream_bytes"] = samples["runs"][0]["after"]["wan_emulation"]["total_stream_bytes"]
    _wire(report, metrics, True)  # Coherent, enforced pacer evidence alone is insufficient.
    with pytest.raises(Rejected, match="decode_body_exceeds_matching_tcp_window"):
        _body(report, metrics, True)


def test_startup_partitions_stay_unknown_and_tcp_overhead_need_not_equal_bodies(cohort):
    row = cohort["results"][-1]
    report = row["execution"]["details"]["report"]
    startup = report["topology_accounting"]["startup"]
    raw = dict(startup["traffic_attribution"]["audit_counters"])
    del raw["preparation_prefill_material_body_bytes"]
    startup = ledger(True, raw, startup=True)
    report["topology_accounting"]["startup"] = startup
    original = deepcopy(startup)
    assert startup["traffic_attribution"]["base_all_link_body_bytes"] is None
    # Missing optional phase/material counters are not invented as zero.
    assert "preparation_prefill_material_body_bytes" not in startup["traffic_attribution"]["audit_counters"]
    summary = audit(cohort)
    assert summary["complete"]
    assert startup == original
    metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
    assert metrics["network.covered_body_bytes"] < metrics["network.all_link_stream_bytes"]


@pytest.mark.parametrize("name", list(CHOICES))
def test_real_shaped_coupled_startup_without_boundary_lifecycle_evidence_cannot_rank(cohort, name):
    # Shape observed in private-boundary-decode-tiny-v1: preparation startup
    # captures ordinary zero counters before any boundary transport is retired.
    # No role_link entries exist yet; canonical v4 correctly retains unknown
    # startup coverage, edges and totals. Completed request coverage is separate.
    row = next(row for row in cohort["results"] if row["execution"]["case"]["name"] == name)
    report = row["execution"]["details"]["report"]
    counters = report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"]
    startup_raw = {key: 0 for key in counters if not key.startswith("role_link.")}
    startup = ledger(True, startup_raw, startup=True)
    report["topology_accounting"]["startup"] = startup
    assert startup["boundary_body_counter_set_present"] is False
    assert startup["tracked_body_counter_set_present"] is False
    assert startup["body_bytes_by_edge"] is None
    assert startup["all_link_serialized_body_bytes"] is None
    assert startup["lifecycle_control_serialized_body_bytes"] == 0
    assert report["topology_accounting"]["runs"][0]["boundary_body_counter_set_present"] is True
    next(m for m in row["metrics"] if m["id"] == "network.covered_body_bytes")["value"] = None
    with pytest.raises(Rejected, match="decode_body_scope_incomplete"):
        audit(cohort)


@pytest.mark.parametrize("field", [f"role_link.{role}.{phase}_{direction}_bytes"
    for role in ("boundary_a", "boundary_b") for phase in ("setup", "online", "teardown")
    for direction in ("upload", "download")])
def test_even_one_missing_startup_owner_phase_counter_stays_unknown(cohort, field):
    row = cohort["results"][-1]
    report = row["execution"]["details"]["report"]
    startup = report["topology_accounting"]["startup"]
    del startup["traffic_attribution"]["audit_counters"][field]
    # Cached totals/flags still claim complete coverage; raw reconstruction
    # must reject rather than interpret a missing zero-valued field as zero.
    with pytest.raises(Rejected, match="decode_raw_body_coverage_incomplete"):
        audit(cohort)


def test_real_shaped_zero_preparation_startup_with_explicit_boundary_counters_is_supported(cohort):
    for row in cohort["results"]:
        report = row["execution"]["details"]["report"]
        accounting = report["topology_accounting"]
        boundary = row["execution"]["case"]["name"] != "maintained"
        raw = accounting["startup"]["traffic_attribution"]["audit_counters"]
        accounting["startup"] = ledger(boundary, {key: 0 for key in raw}, startup=True)
        next(m for m in row["metrics"] if m["id"] == "network.covered_body_bytes")["value"] = (
            accounting["runs"][0]["all_link_serialized_body_bytes"])
    assert audit(cohort)["complete"] is True


def change_prepared_reply(row, delta, *, update_stage_evidence=True):
    """Update raw counters, canonical ledger and enclosing TCP evidence together."""
    report = row["execution"]["details"]["report"]
    privacy = report["runs"][0]["privacy"]
    privacy["inference_download_bytes"] += delta
    report["topology_accounting"]["runs"][0] = ledger(True, privacy)
    next(m for m in row["metrics"] if m["id"] == "network.covered_body_bytes")["value"] += delta
    samples = report["native_network_accounting"]["samples"]
    after = samples["runs"][0]["after"]
    links = dict(after["wan_emulation"]["directed_stream_bytes"])
    links["inference->client"] += delta
    after["wan_emulation"] = snapshot(True, links)
    next(m for m in row["metrics"] if m["id"] == "network.all_link_stream_bytes")["value"] += delta
    if update_stage_evidence:
        stage = report["topology_accounting"]["stages"]["runs"][0]
        stage["body_bytes_by_stage_and_edge"]["layer.0"]["inference->client"] += delta
        stage["expected_stage_body_bytes_by_edge"]["inference->client"] += delta
        stage["measured_body_bytes_by_edge"]["inference->client"] += delta


def test_real_shaped_msgpack_server_ns_width_variation_is_charged_with_independent_stage_evidence(cohort):
    import msgpack

    # residue_codec.pack_row_response uses this seven-field record. Fixed
    # ticket/stage/shape/layout/payload; only server_ns crosses uint16 -> uint32.
    envelope = ["a" * 32, "layers.0.self_attn.qkv_proj", 16, 32, b"x" * 32, 65535, b"y" * 100]
    original = len(msgpack.packb(envelope, use_bin_type=True))
    envelope[5] = 65536
    delta = len(msgpack.packb(envelope, use_bin_type=True)) - original
    assert delta == 2
    row = next(row for row in cohort["results"]
        if row["execution"]["case"]["name"] == "coupled-parallel-u32")
    change_prepared_reply(row, delta)
    summary = audit(cohort)
    paired = summary["body_reconciliation"]["coupled-parallel-u32"][0]
    assert paired["prepared_stage_reply_delta_bytes"] == 2
    assert paired["covered_body_delta_bytes"] == 12  # ten pre-existing bootstrap bytes remain charged
    assert paired["bootstrap_and_metadata_residual_bytes"] == 10
    assert paired["prepared_stage_reply_metadata_compatibility_bound_bytes"] >= 2


def test_reply_delta_without_independent_stage_evidence_is_rejected(cohort):
    change_prepared_reply(cohort["results"][-1], 2, update_stage_evidence=False)
    with pytest.raises(Rejected, match="decode_stage_body_evidence_unreconciled"):
        audit(cohort)


def test_reply_delta_beyond_possible_timing_metadata_is_rejected_even_with_matching_ledgers(cohort):
    row = cohort["results"][-1]
    change_prepared_reply(row, 10000)
    with pytest.raises(Rejected, match="decode_prepared_reply_metadata_bound_exceeded"):
        audit(cohort)


@pytest.mark.parametrize("counter", ["preparation_prefill_material_body_bytes", "preparation_decode_material_body_bytes",
    "preparation_rows", "prepared_stage_rows_claimed", "inference_stage_calls", "preparation_prefill_rows_issued"])
def test_metadata_variation_cannot_hide_changed_arithmetic_or_row_work(cohort, counter):
    row = cohort["results"][-1]
    change_prepared_reply(row, 2)
    row["execution"]["details"]["report"]["runs"][0]["privacy"][counter] += 1
    with pytest.raises(Rejected, match="decode_(prepared_row_or_material_work_invalid|unexpected_arithmetic_or_lookup_body_change)"):
        audit(cohort)


def test_matching_aggregate_totals_cannot_hide_changed_directed_stage_work(cohort):
    row = cohort["results"][-1]
    change_prepared_reply(row, 2)
    stages = row["execution"]["details"]["report"]["topology_accounting"]["stages"]["runs"][0]
    stages["body_bytes_by_stage_and_edge"]["layer.extra"] = {
        edge: 0 for edge in stages["body_bytes_by_stage_and_edge"]["layer.0"]}
    stages["body_bytes_by_stage_and_edge"]["layer.extra"]["client->inference"] = 1
    stages["body_bytes_by_stage_and_edge"]["layer.0"]["client->inference"] -= 1
    with pytest.raises(Rejected, match="decode_fixed_stage_work_changed"):
        audit(cohort)


def test_archived_sdk_screen_replays_without_checkpoint_or_benchmark_execution(monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    import pllm.metrics
    import pllm.model_loader

    def forbidden(*args, **kwargs):
        raise AssertionError("evidence replay must not execute a model or benchmark")

    monkeypatch.setattr(pllm.model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(pllm.metrics, "benchmark_suite", forbidden)
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/data/research/benchmark-cohorts.json").read_bytes())
    spec = next(item for item in registry["adapted_cohorts"] if item["adapter"] == "private-boundary-decode-v1")
    for path, digest in ((spec["report"], spec["sha256"]), (spec["plan"], spec["plan_sha256"]),
            (spec["summary"], spec["summary_sha256"]), (spec["replay_file"], spec["replay_sha256"]),
            (spec["frozen_factory"]["file"], spec["frozen_factory"]["sha256"])):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    raw = json.loads((root / spec["report"]).read_bytes())
    summary = audit(raw)
    assert summary == json.loads((root / spec["summary"]).read_bytes())
    assert summary["clients"] == 15 and summary["checked_values"] == 131404800
    assert summary["maximum_observed_swap_growth_bytes"] == 0
    assert summary["groups"][NAMES[-1]]["maximum_client_peak_bytes"] == 239779840
    assert all(group["observed_tiers"]["low"] for group in summary["groups"].values())
    for name in CHOICES:
        assert len(summary["body_reconciliation"][name]) == 3
        assert {item["head_reply_delta_bytes"] for item in summary["body_reconciliation"][name]} == (
            {0} if CHOICES[name][1] == "u32" else {-2125920})
    assert spec["evidenceTier"] == "screen"
