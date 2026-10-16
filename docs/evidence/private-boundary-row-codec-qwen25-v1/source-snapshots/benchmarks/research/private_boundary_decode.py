"""Matched ordinary SDK decode alternatives; configurations are not evidence.

Compose the tier audit's full-state/whole-owner invariants, not its hard-coded
candidate ranking. No historical measurements enter this cohort.
"""
from __future__ import annotations

from pathlib import Path

from pllm import Experiment, Pipeline
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario
from pllm.roles import PrivateLookupAndFullHead

from . import private_boundary_tiers as tiers
from .sota_combinations import Rejected

METRICS = tiers.METRICS
CHOICES = {
    "coupled-serial-u32": ("serial", "u32"),
    "coupled-parallel-u32": ("parallel", "u32"),
    "coupled-serial-uniform_residues": ("serial", "uniform_residues"),
    "coupled-parallel-uniform_residues": ("parallel", "uniform_residues"),
}
CONTROL = "coupled-serial-u32"
NAMES = ("maintained", *CHOICES)
PREPARED_COUNTERS = (
    "inference_upload_bytes", "correction_push_bytes",
    "correction_ack_bytes", "preparation_upload_bytes", "preparation_download_bytes",
    "prepared_stage_rows_issued",
)
FIXED_WORK_COUNTERS = (
    "preparation_rows", "inference_stage_calls", "preparation_attributed_body_bytes",
    "preparation_prefill_material_body_bytes", "preparation_decode_material_body_bytes",
    "preparation_prefill_rows_issued", "preparation_decode_rows_issued", "preparation_padding_rows_issued",
    "prepared_stage_rows_reserved", "prepared_stage_rows_claimed", "prepared_stage_rows_burned",
    "prepared_stage_rows_discarded", "generation_executed_decode_steps", "generation_output_tokens",
)
BOUNDARY_COUNTERS = (
    "private_boundary_lookup_upload_bytes", "private_boundary_lookup_download_bytes",
    "private_boundary_head_upload_bytes", "private_boundary_lookup_rows", "private_boundary_head_calls",
)


def recipes(*, tiny=False, artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    values = tiers.recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)
    result = {"maintained": values["maintained"].with_params(budget__requests=1)}
    base = values["coupled-boundary"].with_params(budget__requests=1)
    for name, (dispatch, encoding) in CHOICES.items():
        components = dict(base.pipeline.components)
        components["boundary"] = PrivateLookupAndFullHead(
            head_dispatch=dispatch, head_output_encoding=encoding)
        result[name] = base.with_params(pipeline=Pipeline.from_profile(base.pipeline.profile,
            model=base.pipeline.model, components=components))
    return result


def matched_suite(*, build, processes=3, tiny=False,
                  artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    if processes != 3:
        raise ValueError("exactly three balanced fresh replicates required")
    values = recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)
    cases = [BenchmarkCase(name, BenchmarkTarget.response(exp, backend="native",
        scenario=Scenario.cold(max_output_tokens=2 if tiny else 8), wan=WanConditions(), state_audit=True), build)
        for name, exp in values.items()]
    return BenchmarkSuite(cases, Sampling(processes=3, order="balanced", threads=1,
        repetitions=1, warmups=0, timeout_seconds=1200, memory_bytes=8 << 30), "private-boundary-decode-v1")


def preflight(suite):
    """Conservative topology and identical full-state audit admission for all arms."""
    return tiers.preflight(suite)


def _require(condition, code):
    if not condition:
        raise Rejected(code)


def _count(value):
    return type(value) is int and value >= 0


def _counter_values(privacy, fields):
    _require(all(_count(privacy.get(key)) for key in fields), "decode_body_counter_missing")
    return {key: privacy[key] for key in fields}


def _validate_suite(document, tiny):
    """Reject relabelled cells, changed algorithms, audit cost and public bounds."""
    suite = BenchmarkSuite.from_spec(document["suite"])
    sampling = suite.sampling
    _require(tuple(case.name for case in suite.cases) == NAMES
        and sampling.processes == 3 and sampling.order == "balanced"
        and sampling.threads == sampling.repetitions == 1 and sampling.warmups == 0
        and sampling.memory_bytes == 8 << 30, "decode_suite_contract_changed")
    experiments = {}
    for case in suite.cases:
        target = case.target.to_spec()["parameters"]
        exp = Experiment.from_spec(target["model"]["experiment"]).with_params(
            pipeline__profile=target["model"]["profile"])
        experiments[case.name] = exp
        _require(target == BenchmarkTarget.response(exp, backend="native",
            scenario=Scenario.cold(max_output_tokens=2 if tiny else 8), wan=WanConditions(),
            state_audit=True).to_spec()["parameters"], "decode_target_scope_changed")
        _require(exp.budget.max_input_tokens == (128 if tiny else 150)
            and exp.budget.max_new_tokens == (2 if tiny else 8)
            and exp.budget.requests == 1, "decode_public_bounds_changed")
    tokenizer = experiments["maintained"].pipeline.components.get("tokenizer")
    artifacts = Path(tokenizer.get_params()["path"]).parent if tokenizer else Path("target/prepared-frontier-artifacts")
    expected = recipes(tiny=tiny, artifacts=artifacts,
        generated_source=experiments["maintained"].pipeline.model if tiny else None)
    _require(all(experiments[name].to_spec() == expected[name].to_spec() for name in NAMES),
             "decode_recipe_changed")
    builds = [case.build.to_spec() for case in suite.cases]
    _require(all(build == builds[0] for build in builds) and builds[0]["expected_digest"],
             "decode_unlocked_or_mixed_build")
    return suite


def _wire(report, metrics, boundary):
    """Revalidate raw shared-pacer counters, not just cached readiness booleans."""
    from pllm.metrics.wan import _check_native_sample

    roles = {"client", "inference", "preparation"}
    edges = {(a, b) for a in roles for b in roles if a != b}
    if boundary:
        roles |= {"boundary_a", "boundary_b"}
        edges |= {(a, b) for role in ("boundary_a", "boundary_b")
                  for a, b in (("client", role), (role, "client"))}
    config = report["configuration"]
    _require(config.get("provider_backend") == "native" and config.get("wan_emulation") is True
        and config.get("wan_emulation_digest") == WanConditions().digest,
        "decode_source_or_wire_scope_changed")
    native = report["native_network_accounting"]
    windows = native["samples"]["runs"]
    _require(native["schema"] == "pllm.native_network_accounting.v1" and len(windows) == 1,
             "decode_wire_scope_incomplete")
    samples = [native["samples"]["startup"]["wan_emulation"],
        *(windows[0][key]["wan_emulation"] for key in ("before", "after"))]
    for index, snapshot in enumerate(samples):
        _check_native_sample(snapshot, WanConditions(), {role: role for role in roles})
        # Before demand preparation some directed links need not have sent data.
        observed = set(snapshot["directed_stream_bytes"])
        expected = {f"{a}->{b}" for a, b in edges}
        _require(observed == expected if index == 2 else observed <= expected,
                 "decode_wire_links_missing")
    for before, after in zip(samples, samples[1:]):
        _require(all(value <= after["directed_stream_bytes"].get(edge, 0)
            for edge, value in before["directed_stream_bytes"].items()), "decode_wire_counters_regressed")
    _require(windows[0]["after"]["wan_emulation"]["total_stream_bytes"]
        == metrics["network.all_link_stream_bytes"], "decode_wire_metric_differs")


def _body(report, metrics, boundary, *, chunk_rows=16):
    from pllm.runtime.topology_accounting import (
        ACK_COVERAGE, BOUNDARY_COVERAGE, body_coverage_identity,
        prepared_body_accounting, private_boundary_body_accounting,
    )

    accounting = report["topology_accounting"]
    _require(not accounting["warmups"] and len(accounting["runs"]) == 1, "decode_body_windows_changed")
    windows = [accounting["startup"], *accounting["runs"]]
    rebuilt = []
    account = private_boundary_body_accounting if boundary else prepared_body_accounting
    for index, window in enumerate(windows):
        _require(body_coverage_identity(window) == (BOUNDARY_COVERAGE if boundary else ACK_COVERAGE)
            and _count(window.get("all_link_serialized_body_bytes"))
            and _count(window.get("lifecycle_control_serialized_body_bytes"))
            and window["lifecycle_control_serialized_body_bytes"] <= window["all_link_serialized_body_bytes"]
             and window.get("total_wire_bytes") is None, "decode_body_scope_incomplete")
        # Startup has no request record. The runtime retains bounded raw counters
        # inside attribution, even when its generation/material partitions are
        # unknown. Rebuild only measured bodies, never infer those partitions.
        privacy = (window["traffic_attribution"]["audit_counters"] if index == 0
            else report["runs"][0]["privacy"])
        _require(isinstance(privacy, dict), "decode_raw_body_evidence_missing")
        prepared = prepared_body_accounting({"privacy": privacy}, initial_preparation=index == 0)
        _require(body_coverage_identity(prepared) == ACK_COVERAGE
            and prepared["tracked_body_counter_set_present"] is True,
            "decode_raw_body_coverage_incomplete")
        fresh = account({"privacy": privacy}, initial_preparation=index == 0)
        _require(body_coverage_identity(fresh) == (BOUNDARY_COVERAGE if boundary else ACK_COVERAGE)
            and fresh["tracked_body_counter_set_present"] is True, "decode_raw_body_coverage_incomplete")

        def edges(ledger):
            rows = ledger["body_bytes_by_edge"]
            _require(isinstance(rows, list) and len(rows) <= 32
                and all(set(row) == {"source", "destination", "phase", "serialized_body_bytes"}
                    and _count(row["serialized_body_bytes"]) for row in rows), "decode_body_edges_invalid")
            result = {(row["source"], row["destination"], row["phase"]): row["serialized_body_bytes"]
                for row in rows}
            _require(len(result) == len(rows), "decode_body_edges_duplicate")
            return result

        _require(edges(window) == edges(fresh), "decode_body_edges_differ_from_raw")
        for field in ("all_link_serialized_body_bytes", "client_serialized_body_bytes",
                "online_all_link_serialized_body_bytes", "online_client_serialized_body_bytes",
                "lifecycle_control_serialized_body_bytes"):
            _require(_count(window.get(field)) and window[field] == fresh[field],
                     "decode_body_totals_differ_from_raw")
        rebuilt.append(fresh)

    samples = report["native_network_accounting"]["samples"]
    startup = samples["startup"]["wan_emulation"]["directed_stream_bytes"]
    before = samples["runs"][0]["before"]["wan_emulation"]["directed_stream_bytes"]
    after = samples["runs"][0]["after"]["wan_emulation"]["directed_stream_bytes"]
    for ledger, bounds in ((rebuilt[0], startup), (rebuilt[1], {
            edge: count - before.get(edge, 0) for edge, count in after.items()})):
        directed = {}
        for edge in ledger["body_bytes_by_edge"]:
            link = f'{edge["source"]}->{edge["destination"]}'
            directed[link] = directed.get(link, 0) + edge["serialized_body_bytes"]
        # Bodies omit framing and some controls. Startup counters cover only
        # initial preparation, not all traffic since relay birth. Use a floor
        # within each enclosing TCP window, not equality or cross-window sums.
        _require(all(count <= bounds.get(link, 0) for link, count in directed.items()),
                 "decode_body_exceeds_matching_tcp_window")
    total = sum(window["all_link_serialized_body_bytes"] for window in windows)
    _require(total == metrics["network.covered_body_bytes"], "decode_body_metric_differs")
    privacy = report["runs"][0]["privacy"]
    result = {"total": total,
        "controls": sum(window["lifecycle_control_serialized_body_bytes"] for window in windows),
        "prepared": _counter_values(privacy, PREPARED_COUNTERS),
        "fixed_work": _counter_values(privacy, FIXED_WORK_COUNTERS),
        "prepared_reply": _counter_values(privacy, ("inference_download_bytes",))["inference_download_bytes"]}
    work = result["fixed_work"]
    rows = result["prepared"]["prepared_stage_rows_issued"]
    calls = work["inference_stage_calls"]
    _require(type(chunk_rows) is int and chunk_rows > 0 and 0 < calls <= rows
        and rows == work["preparation_rows"] == work["prepared_stage_rows_reserved"]
        == work["prepared_stage_rows_claimed"]
        and work["prepared_stage_rows_burned"] == work["prepared_stage_rows_discarded"] == 0
        and rows == sum(work[key] for key in ("preparation_prefill_rows_issued",
            "preparation_decode_rows_issued", "preparation_padding_rows_issued"))
        and work["preparation_prefill_material_body_bytes"] + work["preparation_decode_material_body_bytes"]
            <= work["preparation_attributed_body_bytes"]
        == sum(privacy[key] for key in ("preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes")),
        "decode_prepared_row_or_material_work_invalid")
    # Exact byte telemetry independently records each stage's directed bodies.
    # It is not an arithmetic-payload meter: response totals include server_ns.
    stages = accounting["stages"]["runs"]
    _require(len(stages) == 1, "decode_stage_body_evidence_missing")
    stage = stages[0]
    measured = stage["body_bytes_by_stage_and_edge"]
    fields = {"client->inference": "inference_upload_bytes", "inference->client": "inference_download_bytes",
        "client->preparation": "preparation_upload_bytes", "preparation->client": "preparation_download_bytes",
        "preparation->inference": "correction_push_bytes"}
    _require(bool(measured) and len(measured) <= 4096
        and all(set(edges) == set(fields) and all(_count(value) for value in edges.values())
            for edges in measured.values()), "decode_stage_body_evidence_missing")
    sums = {edge: sum(edges[edge] for edges in measured.values()) for edge in fields}
    _require(sums == {edge: privacy[field] for edge, field in fields.items()}
        == stage["measured_body_bytes_by_edge"] == stage["expected_stage_body_bytes_by_edge"]
        and stage["reconciled_with_protocol_bodies"] is True, "decode_stage_body_evidence_unreconciled")
    result["stage_bodies"] = measured
    # Sum ceil(rows_i/chunk) <= ceil(total_rows/chunk) + calls - 1.
    # Each reply's only variable-size envelope field is nonnegative server_ns,
    # whose MsgPack representation is 1,2,3,5 or 9 bytes. This is a bound,
    # NOT a fabricated response count or an exact timing-byte attribution.
    result["reply_metadata_delta_bound_bytes"] = 8 * ((rows + chunk_rows - 1) // chunk_rows + calls - 1)
    if boundary:
        _require(type(privacy.get("private_boundary_body_coverage_failures")) is int
            and privacy["private_boundary_body_coverage_failures"] == 0,
                 "decode_boundary_body_coverage_failed")
        result["boundary_fixed"] = _counter_values(privacy, BOUNDARY_COUNTERS)
        result["head_reply"] = _counter_values(privacy, ("private_boundary_head_download_bytes",))[
            "private_boundary_head_download_bytes"]
        _require(result["head_reply"] > 0, "decode_head_reply_missing")
        online = _counter_values(privacy, tuple(f"role_link.{role}.online_{direction}_bytes"
            for role in ("boundary_a", "boundary_b") for direction in ("upload", "download")))
        _require(sum(online.values()) == result["head_reply"] + sum(result["boundary_fixed"][key]
            for key in BOUNDARY_COUNTERS if key.endswith("_bytes")),
            "decode_boundary_online_body_does_not_reconcile")
    return result


def _compare(control, candidate):
    result = {}
    for metric, direction in METRICS.items():
        left, right = control["medians"][metric], candidate["medians"][metric]
        result[metric] = {"control": left, "candidate": right, "delta": right - left,
            "improvement_fraction": ((1 - right / left) if direction == "min" else right / left - 1)
                if left else None,
            "candidate_wins": right < left if direction == "min" else right > left,
            "paired_deltas": [b[metric] - a[metric] for a, b in zip(control["samples"], candidate["samples"])]}
    return result


def audit(document, *, tiny=False):
    """Fail closed on incomplete, mixed, weakened or differently scoped evidence."""
    try:
        suite = _validate_suite(document, tiny)
        _require(document["complete"] is True and len(document["results"]) == 15,
                 "decode_incomplete_cohort")
        cohort = document["cohort"]
        _require(type(cohort) is str and len(cohort) == 64
            and all(c in "0123456789abcdef" for c in cohort), "decode_private_cohort_missing")
        bodies = {}
        for row in document["results"]:
            execution = row["execution"]
            name, rep = execution["case"]["name"], execution["replicate"]
            _require(name in NAMES and execution["cohort"] == cohort
                and execution["suite_digest"] == suite.digest
                and execution["sampling"] == suite.sampling.to_spec()
                and execution["build"]["digest"] == suite.cases[0].build.expected_digest
                and execution["isolation"]["fresh_process"] is True
                and execution["isolation"]["controller_included"] is False,
                "decode_mixed_build_salt_or_isolation")
            report = execution["details"]["report"]
            fixture = execution["fixture"]
            _require(fixture["metadata"]["cached_checkpoint"] is True
                and fixture["metadata"]["owned_client_cache"] == "fresh"
                and fixture["digest"] == report["configuration"]["source_lock_digest"],
                "decode_source_present_scope_missing")
            _require(report["memory_guard"]["maximum_host_swap_growth_bytes"] == 0,
                     "decode_swap_growth")
            audit_record = execution["details"]["response_state_audit"]
            _require(set(audit_record["client_memory_owners"]) == {
                "sdk", "dashboard", "native_wan_relay", "audit", "role_supervision"},
                "decode_client_owners_missing")
            binding, = audit_record["bindings"]
            _require(binding["sampler_seed"] == 42 and binding["temperature"] == 0
                and binding["cryptographic_randomness_frozen"] is False,
                "decode_sampler_fixture_changed")
            for selection in binding["selections"]:
                _require(all(selection.get(key) is not None for key in (
                    "position", "cache_lengths", "logits_and_active_kv", "selected_token", "rng_before", "rng_after"))
                    and type(selection["checked_values"]) is int and selection["checked_values"] > 0
                    and type(selection["position"]) is int and selection["position"] >= 0
                    and bool(selection["cache_lengths"])
                    and all(type(length) is int and length > 0 for length in selection["cache_lengths"]),
                    "decode_full_state_missing")
            metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
            _wire(report, metrics, name != "maintained")
            experiment = Experiment.from_spec(execution["case"]["target"]["parameters"]["model"]["experiment"])
            chunk_rows = experiment.pipeline.components["linear"].get_params()["prefill_chunk_rows"]
            bodies[name, rep] = _body(report, metrics, name != "maintained", chunk_rows=chunk_rows)
            if name != "maintained":
                fixed = bodies[name, rep]["boundary_fixed"]
                _require(fixed["private_boundary_head_calls"] == (2 if tiny else 8)
                    and fixed["private_boundary_lookup_rows"] == report["runs"][0]["tokens"]["input_tokens"]
                        + (1 if tiny else 7), "decode_boundary_work_changed")
        # Tier audit checks exact case/replicate membership, state, body/source,
        # output, memory ownership, numeric metrics and complete-owner maxima.
        summary = tiers.audit(document, tiny=tiny)
        summary["scope"].update(
            incumbents="maintained prepared comparator and coupled serial/u32 control, measured in this cohort only",
            request=("single cold client, 128+2 generated-source smoke; source already present" if tiny else
                "single cold client, 150+8; source and indexed tokenizer already present"),
            public_bound="one request per fresh client; common input/output envelopes, original numeric body retained")
        groups = summary["groups"]
        for name in groups:
            groups[name]["samples"] = [next({m["id"]: m["value"] for m in row["metrics"] if m["id"] in METRICS}
                for row in document["results"] if row["execution"]["case"]["name"] == name
                and row["execution"]["replicate"] == rep) for rep in range(3)]
        reconciliation = {}
        for name in CHOICES:
            paired = []
            for rep in range(3):
                control, candidate = bodies[CONTROL, rep], bodies[name, rep]
                _require(candidate["prepared"] == control["prepared"]
                    and candidate["fixed_work"] == control["fixed_work"]
                    and candidate["boundary_fixed"] == control["boundary_fixed"],
                    "decode_unexpected_arithmetic_or_lookup_body_change")
                left, right = control["stage_bodies"], candidate["stage_bodies"]
                _require(set(left) == set(right) and all(
                    {edge: value for edge, value in left[stage].items() if edge != "inference->client"}
                    == {edge: value for edge, value in right[stage].items() if edge != "inference->client"}
                    for stage in left), "decode_fixed_stage_work_changed")
                metadata = candidate["prepared_reply"] - control["prepared_reply"]
                bound = control["reply_metadata_delta_bound_bytes"]
                _require(candidate["reply_metadata_delta_bound_bytes"] == bound
                    and sum(abs(right[stage]["inference->client"] - left[stage]["inference->client"])
                        for stage in left) <= bound, "decode_prepared_reply_metadata_bound_exceeded")
                delta = candidate["total"] - control["total"]
                reply = candidate["head_reply"] - control["head_reply"]
                controls = candidate["controls"] - control["controls"]
                same_encoding = CHOICES[name][1] == "u32"
                _require(reply == 0 if same_encoding else reply < 0, "decode_unexpected_head_reply_change")
                paired.append({"replicate": rep, "covered_body_delta_bytes": delta,
                    "head_reply_delta_bytes": reply, "lifecycle_control_delta_bytes": controls,
                    "prepared_stage_reply_delta_bytes": metadata,
                    "prepared_stage_reply_metadata_compatibility_bound_bytes": bound,
                    "bootstrap_and_metadata_residual_bytes": delta - reply - controls - metadata})
            reconciliation[name] = paired
        return {key: value for key, value in summary.items() if key not in ("tiers", "schema")} | {
            "schema": "pllm.private_boundary_decode.v1", "tiny": tiny,
            "comparisons": {name: {control: _compare(groups[control], groups[name])
                for control in (CONTROL, "maintained")} for name in CHOICES},
            "body_reconciliation": reconciliation,
            "body_reconciliation_scope": "Fixed prepared arithmetic/lookup/head requests; head reply deltas measured. "
                "Prepared stage replies include variable-width MsgPack server_ns; matching row/material work and "
                "independent directed stage-byte evidence bound metadata-compatible variation, not exact timing-byte attribution. "
                "Bootstrap, metadata and control variation remain charged; residual is not exact codec savings.",
            "evidence_scope": "One fresh same-build source-present cohort; no historical pooling. "
                "150+8 real-checkpoint cohort; 128+2 tiny smoke. TCP stream payload is not physical wire.",
        }
    except Rejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise Rejected("decode_missing_or_invalid_evidence") from exc
