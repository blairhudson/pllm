"""Cold-request incumbent replay and exact-state, whole-client tier comparison.

Saved experiments supply configurations, never measurements. New source-present
clients own SDK, supervisor, dashboard, native WAN relay and bounded state audit
in one process. The idle benchmark-suite launcher is outside that owner.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from pllm import Experiment, Model, Pipeline
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario
from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead

from .historical_evidence import source_document
from .sota_combinations import MODEL, REVISION, Rejected

METRICS = {
    "client.peak_rss": "min", "client.cold_cpu": "min",
    "preparation.cold_cpu": "min", "preparation.peak_rss": "min",
    "response.ttft": "min", "response.decode_tps": "max",
    "network.all_link_stream_bytes": "min", "network.covered_body_bytes": "min",
    "request.0.full": "min", "target.latency": "min",
    "process.peak_rss": "min",
}
TIERS = {"low": 256 << 20, "medium": 2 << 30, "high": 16 << 30}


def _saved(source, name):
    document = source_document(source)
    case, = (c for c in document["suite"]["cases"] if c["name"] == name)
    model = case["target"]["parameters"]["model"]
    return Experiment.from_spec(model["experiment"]).with_params(pipeline__profile=model["profile"])


def recipes(*, tiny=False, artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    """Replay component choices under one common cold-request public bound."""
    if tiny:
        from examples.prepared_sdk import build_experiment
        from pllm.schedulers import MlpPrefillTiling
        base = build_experiment(generated_source or Model.tiny()).with_params(budget__max_input_tokens=128,
            budget__max_new_tokens=2, pipeline__cache__fixed_input_tokens=128,
            pipeline__linear__prefill_pruning="none")
        values = {"maintained": base, "tiling": base.with_params(pipeline__scheduler=MlpPrefillTiling(tile_rows=64))}
        components = dict(base.pipeline.components)
        del components["cache"]
        private = base.with_params(pipeline=Pipeline.from_profile(base.pipeline.profile,
            model=base.pipeline.model, components=components))
    else:
        values = {name: _saved("research-sota-combinations-qwen25", case) for name, case in (
            ("maintained", "150"), ("tiling", "150/tiling"), ("digit8", "150/digit8"),
            ("tiling-digit8", "150/both"), ("tiling-digit8-placement", "150/both-placement"))}
        values["digit8-placement"] = _saved("research-all-link-placement-sdk-confirmation", "selected")
        values["serial-digit8"] = _saved("research-prepared-study-qwen25-confirm", "prepared/serial-issuance+digit8")
        private = Experiment.from_spec(json.loads(Path("docs/evidence/private-boundary-qwen-control.json").read_bytes()))
    components = dict(private.pipeline.components)
    components.update(boundary=PrivateLookupAndFullHead(), topology=PreparedBoundaryRoles())
    values["coupled-boundary"] = private.with_params(pipeline=Pipeline.from_profile(private.pipeline.profile,
        model=private.pipeline.model, components=components))
    for name, experiment in values.items():
        if not tiny and experiment.pipeline.model.to_spec() != {"source": MODEL, "revision": REVISION}:
            raise Rejected("incumbent_source_changed")
        tokenizer = experiment.pipeline.components.get("tokenizer")
        if tokenizer is not None:
            path = "pipeline__tokenizer__path"
            if path not in experiment.get_params(deep=True):
                path = "pipeline__components__tokenizer__path"
            experiment = experiment.with_params(**{path: str((artifacts / "tokenizer").resolve())})
        if not tiny:
            changes = {"budget__max_input_tokens": 150, "budget__max_new_tokens": 8, "budget__requests": 1}
            if "cache" in experiment.pipeline.components:
                path = "pipeline__cache__fixed_input_tokens"
                if path not in experiment.get_params(deep=True):
                    path = "pipeline__components__cache__fixed_input_tokens"
                changes[path] = 150
            experiment = experiment.with_params(**changes)
        values[name] = experiment
    return values


def matched_suite(*, build, processes=3, tiny=False, artifacts=Path("target/prepared-frontier-artifacts"),
                  generated_source=None):
    values = recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)
    outputs = 2 if tiny else 8
    cases = [BenchmarkCase(name, BenchmarkTarget.response(exp, backend="native",
        scenario=Scenario.cold(max_output_tokens=outputs), wan=WanConditions(), state_audit=True), build)
        for name, exp in values.items()]
    return BenchmarkSuite(cases, Sampling(processes=processes, order="balanced", threads=1,
        repetitions=1, warmups=0, timeout_seconds=1200, memory_bytes=8 << 30), "private-boundary-tiers-v1")


def preflight(suite):
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_response_audit import state_audit_admission
    results = []
    for case in suite.cases:
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        report = benchmark_memory(exp, backend="native", enforced_wan=True,
                                  memory_budget_bytes=suite.sampling.memory_bytes)
        report = state_audit_admission(report, suite.sampling.memory_bytes, enforced_wan=True)
        results.append({"case": case.name, "admission": report})
    return results


def audit(document, *, tiny=False):
    """Cross-topology costs are explicit; ordinary privacy comparison is retained."""
    cases = {case["name"]: case for case in document["suite"]["cases"]}
    replicates = document["suite"]["sampling"]["processes"]
    if (not document["complete"] or len(document["results"]) != len(cases) * replicates
            or any(row["status"] != "completed" for row in document["results"])):
        raise Rejected("incomplete_or_failed_tier_cohort")
    seen, groups, states, identity = set(), {}, None, None
    checked = 0
    max_swap = 0
    for row in document["results"]:
        execution = row["execution"]
        name, replicate = execution["case"]["name"], execution["replicate"]
        if (name not in cases or execution["case"] != cases[name] or type(replicate) is not int
                or not 0 <= replicate < replicates or (name, replicate) in seen):
            raise Rejected("duplicate_or_changed_tier_cell")
        seen.add((name, replicate))
        detail = execution["details"]
        report = detail["report"]
        if report["checks"]["passed"] is not True or len(report["runs"]) != 1:
            raise Rejected("response_checks_failed")
        request = report["runs"][0]
        bindings = detail["response_state_audit"]["bindings"]
        if (len(bindings) != 1 or bindings[0]["failed"] or not bindings[0]["numeric_body_fingerprint"]
                or len(bindings[0]["selections"]) != (2 if tiny else 8)
                or request["tokens"]["authoritative"] is not True or not request["cold"]
                or request["tokens"]["output_tokens"] != (2 if tiny else 8)
                or not tiny and request["tokens"]["input_tokens"] != 150):
            raise Rejected("full_state_audit_incomplete")
        current = {"source": report["configuration"]["source_lock_digest"],
            "prompt": report["configuration"]["prompt_digest"],
            "body": bindings[0]["numeric_body_fingerprint"], "tokens": request["tokens"],
            "output": request["generation"]["output_text_digest"],
            "wan": report["configuration"]["wan_emulation_digest"],
            "sampling": report["configuration"]["sampling"], "build": execution["build"]}
        if identity is not None and current != identity or states is not None and bindings != states:
            raise Rejected("source_workload_output_logits_kv_or_sampler_mismatch")
        states, identity = bindings, current
        metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
        memory = report["client_process_memory"]
        peak = memory["after"]["lifetime_peak_rss_bytes"]
        guard = report["memory_guard"]
        swap = guard["maximum_host_swap_growth_bytes"]
        complete_peak = metrics.get("process.peak_rss")
        if (type(peak) is not int or peak <= 0 or peak != metrics["client.peak_rss"]
                or type(complete_peak) not in (int, float) or not math.isfinite(complete_peak)
                or complete_peak < peak or complete_peak != int(complete_peak)
                or "client and in-process dashboard" not in memory["scope"]
                or guard["tripped"] is not False or type(swap) is not int or swap < 0
                or report["wan_readiness"]["emulation"]["native_stream_rate_snapshots_checked"] is not True):
            raise Rejected("whole_client_memory_or_wan_scope_missing")
        if any(metrics.get(metric) is None for metric in METRICS if metric != "network.covered_body_bytes"):
            raise Rejected("priority_metric_missing")
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
               for metric in METRICS for value in (metrics.get(metric),)):
            raise Rejected("invalid_priority_metric")
        metrics["client.peak_rss"] = peak
        metrics["process.peak_rss"] = int(complete_peak)
        groups.setdefault(name, []).append({key: metrics[key] for key in METRICS})
        checked += sum(sample["checked_values"] for sample in bindings[0]["selections"])
        max_swap = max(max_swap, swap)
    summaries = {}
    for name, samples in groups.items():
        if len(samples) != replicates:
            raise Rejected("missing_tier_replicate")
        peak = max(row["process.peak_rss"] for row in samples)
        summaries[name] = {"samples": samples, "maximum_client_peak_bytes": peak,
            "observed_tiers": {tier: peak <= ceiling for tier, ceiling in TIERS.items()},
            "medians": {metric: statistics.median(row[metric] for row in samples)
                if all(row[metric] is not None for row in samples) else None for metric in METRICS},
            "experiment": cases[name]["target"]["parameters"]["model"]["experiment"]}
    tiers = {}
    for tier in TIERS:
        qualified = {name: data for name, data in summaries.items() if data["observed_tiers"][tier]}
        incumbents = {name: data for name, data in qualified.items() if name != "coupled-boundary"}
        candidate = qualified.get("coupled-boundary")
        comparisons = {}
        for metric, direction in METRICS.items():
            choose = min if direction == "min" else max
            measured = {name: data for name, data in incumbents.items() if data["medians"][metric] is not None}
            if not measured:
                comparisons[metric] = None
                continue
            best = choose(data["medians"][metric] for data in measured.values())
            value = candidate["medians"][metric] if candidate else None
            comparisons[metric] = {"incumbents": [name for name, data in incumbents.items() if data["medians"][metric] == best],
                "incumbent": best, "candidate": value,
                "improvement_fraction": (1 - value / best if direction == "min" else value / best - 1)
                    if value is not None and best else None,
                "candidate_wins": value is not None and (value < best if direction == "min" else value > best)}
        tiers[tier] = {"qualified": list(qualified), "comparisons": comparisons}
    return {"schema": "pllm.private_boundary_tiers.v1", "identity": identity,
        "complete": True, "replicates": replicates, "clients": len(seen), "checked_values": checked,
        "all_logits_active_kv_positions_selections_rng_and_outputs_match": True,
        "maximum_observed_swap_growth_bytes": max_swap, "groups": summaries, "tiers": tiers,
        "state_commitment": hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
        "scope": {"client": "whole fresh worker: SDK, supervisor, dashboard, WAN relay and audit; idle launcher excluded",
            "memory": "largest lifetime worker RSS through target finalization across every replicate; no sum of peaks",
            "comparison": "explicit cross-topology costs; two additional non-colluding boundary workers change role contract",
            "incumbents": "maintained, installed tiling/digit8/placement frontier and confirmed serial/placement recipes",
            "request": "single cold client, 150+8, no public-prefix reuse; source and indexed tokenizer already present",
            "public_bound": "all recipes rebind request and fixed-prefix-cache envelopes to 150+8; original algorithms retained",
            "limits": ["Not a global optimum or representative generation-quality result.",
                "Source/tokenizer distribution, physical wire, Internet RTT and independent operators unmeasured.",
                "Low/Medium/High are observed RSS ceilings, not conservative SDK memory admission.",
                "Full-state audit charged equally; timings remain separate from unaudited historical cohorts.",
                "Public-prefix and mostly-client-MLP configurations have different reuse/offload contracts."]}}
