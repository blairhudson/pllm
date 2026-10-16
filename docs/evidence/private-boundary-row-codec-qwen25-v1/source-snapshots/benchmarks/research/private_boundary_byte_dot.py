"""Matched ordinary-SDK head-kernel screen with independent raw evidence checks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from pllm import Experiment, Pipeline
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario
from pllm.roles import PrivateLookupAndFullHead

from . import private_boundary_decode as screen
from . import private_boundary_decode_confirmation as confirmation
from . import private_boundary_tiers as tiers
from .sota_combinations import Rejected

CONTROL = "parallel-residues-native"
CANDIDATE = "parallel-residues-byte-dot"
NAMES = (CONTROL, CANDIDATE)
ROLES = {"client", "inference", "preparation", "boundary_a", "boundary_b"}
METRICS = dict(screen.METRICS, **{"topology.cold_cpu": "min"})
preflight = tiers.preflight
_require = screen._require


def recipes(*, tiny=False, artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    base = screen.recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)[confirmation.WINNER]
    components = dict(base.pipeline.components)
    components["boundary"] = PrivateLookupAndFullHead(
        head_dispatch="parallel", head_output_encoding="uniform_residues", head_kernel="byte_dot")
    return {CONTROL: base, CANDIDATE: base.with_params(pipeline=Pipeline.from_profile(
        base.pipeline.profile, model=base.pipeline.model, components=components))}


def matched_suite(*, build, tiny=False, artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    values = recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)
    cases = [BenchmarkCase(name, BenchmarkTarget.response(exp, backend="native",
        scenario=Scenario.cold(max_output_tokens=2 if tiny else 8), wan=WanConditions(), state_audit=True), build)
        for name, exp in values.items()]
    return BenchmarkSuite(cases, Sampling(processes=3, order="balanced", threads=1,
        repetitions=1, warmups=0, timeout_seconds=3600, memory_bytes=8 << 30), "private-boundary-byte-dot-v1")


def _validate_suite(document, tiny):
    suite = BenchmarkSuite.from_spec(document["suite"])
    _require(tuple(case.name for case in suite.cases) == NAMES, "byte_dot_cells_changed")
    target = suite.cases[0].target.to_spec()["parameters"]
    exp = Experiment.from_spec(target["model"]["experiment"])
    tokenizer = exp.pipeline.components.get("tokenizer")
    artifacts = Path(tokenizer.get_params()["path"]).parent if tokenizer else Path("target/prepared-frontier-artifacts")
    expected = matched_suite(build=suite.cases[0].build, tiny=tiny, artifacts=artifacts,
        generated_source=exp.pipeline.model if tiny else None)
    _require(suite.to_spec() == expected.to_spec() and suite.cases[0].build.expected_digest,
             "byte_dot_suite_contract_changed")
    return suite


def _cold_cpu(report):
    record = report["process_cpu_accounting"]
    values = record["cold_first_response_cpu_seconds_by_role"]
    aggregate = record["aggregate_cold_first_response_cpu_seconds"]
    _require(record["schema"] == "pllm.topology_process_cpu.v1" and set(values) == ROLES
        and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in (*values.values(), aggregate))
        and math.isclose(aggregate, sum(values.values()), rel_tol=1e-12, abs_tol=1e-12),
        "byte_dot_cold_cpu_does_not_reconcile")
    return aggregate, dict(values)


def _reconcile(bodies):
    result = []
    for replicate in range(3):
        left, right = bodies[CONTROL, replicate], bodies[CANDIDATE, replicate]
        _require(all(left[key] == right[key] for key in ("prepared", "fixed_work", "boundary_fixed", "head_reply")),
                 "byte_dot_arithmetic_or_boundary_work_changed")
        a, b = left["stage_bodies"], right["stage_bodies"]
        _require(set(a) == set(b) and all(
            {edge: value for edge, value in a[stage].items() if edge != "inference->client"}
            == {edge: value for edge, value in b[stage].items() if edge != "inference->client"}
            for stage in a), "byte_dot_fixed_stage_work_changed")
        bound = left["reply_metadata_delta_bound_bytes"]
        _require(right["reply_metadata_delta_bound_bytes"] == bound
            and sum(abs(b[stage]["inference->client"] - a[stage]["inference->client"]) for stage in a) <= bound,
            "byte_dot_reply_metadata_bound_exceeded")
        total = right["total"] - left["total"]
        controls = right["controls"] - left["controls"]
        replies = right["prepared_reply"] - left["prepared_reply"]
        result.append({"replicate": replicate, "covered_body_delta_bytes": total,
            "head_reply_delta_bytes": 0, "lifecycle_control_delta_bytes": controls,
            "prepared_stage_reply_delta_bytes": replies,
            "prepared_stage_reply_metadata_compatibility_bound_bytes": bound,
            "bootstrap_and_metadata_residual_bytes": total - controls - replies})
    return result


def _compare(control, candidate):
    result = {}
    for metric, direction in METRICS.items():
        left, right = control["medians"][metric], candidate["medians"][metric]
        known = left is not None and right is not None
        result[metric] = {"control": left, "candidate": right,
            "delta": right - left if known else None,
            "improvement_fraction": ((1 - right / left) if direction == "min" else right / left - 1)
                if known and left else None,
            "candidate_wins": (right < left if direction == "min" else right > left) if known else None,
            "paired_deltas": [b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None
                for a, b in zip(control["samples"], candidate["samples"])]}
    return result


def audit(document, *, tiny=False):
    """Audit one complete six-client cohort without rewriting original evidence."""
    try:
        from pllm.metrics import BenchmarkRun
        try:
            BenchmarkRun(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise Rejected("byte_dot_invalid_v2_cohort") from exc
        suite = _validate_suite(document, tiny)
        _require(document["schema"] == "pllm.benchmark_run.v1" and document["complete"] is True
            and len(document["results"]) == 6, "byte_dot_incomplete_cohort")
        cohort = document["cohort"]
        _require(type(cohort) is str and len(cohort) == 64 and all(c in "0123456789abcdef" for c in cohort),
                 "byte_dot_private_cohort_missing")
        cases = {case.name: case for case in suite.cases}
        samples, bodies, work_rows, cpu_rows, seen = {}, {}, {}, {}, set()
        states = identity = common_contract = common_metrics = None
        arm_contracts = {}
        checked = 0
        cap = 2 if tiny else 8
        for row in document["results"]:
            execution = row["execution"]
            name, replicate = execution["case"]["name"], execution["replicate"]
            _require(row["status"] == "completed" and name in NAMES and type(replicate) is int
                and 0 <= replicate < 3 and (name, replicate) not in seen
                and execution["case"] == cases[name].to_spec() and execution["cohort"] == cohort
                and execution["suite_digest"] == suite.digest and execution["sampling"] == suite.sampling.to_spec()
                and execution["build"]["digest"] == suite.cases[0].build.expected_digest
                and execution["isolation"]["fresh_process"] is True
                and execution["isolation"]["controller_included"] is False, "byte_dot_changed_duplicate_or_mixed_cell")
            seen.add((name, replicate))
            report, metrics, bindings, current, work = confirmation._state_and_memory(row, cap, tiny)
            confirmation._checked_measurements(row)
            _require((identity is None or current == identity) and (states is None or bindings == states),
                     "byte_dot_source_history_logits_kv_selections_rng_or_output_mismatch")
            identity, states = current, bindings
            checked += sum(selection["checked_values"] for selection in bindings[0]["selections"])
            screen._wire(report, metrics, True)
            model = execution["case"]["target"]["parameters"]["model"]
            exp = Experiment.from_spec(model["experiment"]).with_params(pipeline__profile=model["profile"])
            _require(report["experiment"]["pipeline_digest"] == exp.pipeline.digest()
                and set(report["configuration"]["roles"]) == ROLES, "byte_dot_executed_pipeline_or_roles_changed")
            common, arm = confirmation._contracts(row, cases[name], exp, cap, work)
            _require(common_contract is None or common_contract == common, "byte_dot_cross_replicate_contract_changed")
            _require(name not in arm_contracts or arm_contracts[name] == arm, "byte_dot_arm_contract_changed")
            _require(common_metrics is None or common_metrics == arm["metrics"], "byte_dot_cross_arm_metric_contract_changed")
            common_contract, common_metrics, arm_contracts[name] = common, arm["metrics"], arm
            confirmation._material(report)
            body_audit = confirmation._eos_body if work["terminated_before_cap"] else screen._body
            body = body_audit(report, metrics, True,
                chunk_rows=exp.pipeline.components["linear"].get_params()["prefill_chunk_rows"])
            _require(body["boundary_fixed"]["private_boundary_head_calls"] == work["sampled_selections"]
                and body["boundary_fixed"]["private_boundary_lookup_rows"]
                == current["tokens"]["input_tokens"] + work["executed_decode_steps"], "byte_dot_executed_work_changed")
            metrics["topology.cold_cpu"], cpu_rows[name, replicate] = _cold_cpu(report)
            bodies[name, replicate], samples[name, replicate] = body, metrics
            work_rows[name, replicate] = work | {key: body["fixed_work"][key] for key in (
                "preparation_rows", "preparation_prefill_rows_issued", "preparation_decode_rows_issued",
                "preparation_padding_rows_issued", "inference_stage_calls", "prepared_stage_rows_reserved",
                "prepared_stage_rows_claimed", "prepared_stage_rows_burned", "prepared_stage_rows_discarded")}
        groups = {}
        for name in NAMES:
            rows = [samples[name, replicate] for replicate in range(3)]
            peak = max(row["process.peak_rss"] for row in rows)
            groups[name] = {"samples": rows, "maximum_client_peak_bytes": peak,
                "observed_tiers": {tier: peak <= ceiling for tier, ceiling in tiers.TIERS.items()},
                "medians": {metric: statistics.median(row[metric] for row in rows)
                    if all(row[metric] is not None for row in rows) else None for metric in METRICS},
                "executed_work": [work_rows[name, replicate] for replicate in range(3)],
                "cold_first_cpu_seconds_by_role": [cpu_rows[name, replicate] for replicate in range(3)]}
        return {"schema": "pllm.private_boundary_byte_dot.v1", "tiny": tiny, "complete": True,
            "cohort": cohort, "clients": len(seen), "replicates": 3, "output_cap": cap,
            "identity": identity, "checked_values": checked,
            "all_logits_active_kv_positions_selections_rng_and_outputs_match": True,
            "state_commitment": hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
            "maximum_observed_swap_growth_bytes": 0, "groups": groups,
            "comparisons": {CANDIDATE: {CONTROL: _compare(groups[CONTROL], groups[CANDIDATE])}},
            "body_reconciliation": {CANDIDATE: _reconcile(bodies)},
            "scope": {"request": f"single cold source-present client, {'128+2' if tiny else '150+8'} cap; natural EOS",
                "client": "whole fresh worker: SDK, supervisor, dashboard, native WAN relay and full-state audit; idle launcher excluded",
                "memory": "largest finalized whole-worker lifetime RSS across all three replicates; observed tiers do not relax admission",
                "cpu": "topology.cold_cpu recomputed from all five process CPU owners through the first response; cleanup excluded",
                "wire": "native shared 100/40 Mbps caps; directed TCP stream payload, not physical wire",
                "kernel": "opt-in native byte-dot API versus existing native wrap32; not an SDOT-instruction-only comparison",
                "evidence": "one new matched cohort and installed build; historical kernel/profiling timings remain separate",
                "limits": "checkpoint/tokenizer already present; distribution, Internet RTT, independent operators and representative quality unmeasured"}}
    except Rejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        raise Rejected("byte_dot_missing_or_invalid_evidence") from exc
