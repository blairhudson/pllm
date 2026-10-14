"""Matched initial-attention ablations over frozen prepared-serving recipes."""
from __future__ import annotations

import json

from pllm import Experiment, _native, lower_model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget

from .sota_combinations import DIAGNOSTICS, OBJECTIVES, Rejected, preflight, study

RECIPES = ("tiling", "digit8", "both-placement")
METRICS = OBJECTIVES + DIAGNOSTICS + tuple(
    f"request.{index}.{metric}" for index in (0, 1)
    for metric in ("full", "preparation", "ttft", "decode_tps"))


def matched_suite(*, recipes=RECIPES, **kwargs):
    if not recipes or len(set(recipes)) != len(recipes) or set(recipes) - set(RECIPES):
        raise Rejected("attention_recipe_selection")
    base = study(**kwargs).plan(max_trials=256).suite
    cases = []
    for case in base.cases:
        if case.name.split("/")[-1] not in recipes:
            continue
        for enabled, name in ((False, "full"), (True, "tiled")):
            target = case.target.to_spec()
            target["parameters"]["client_attention"] = {
                "enabled": enabled, "query_rows": 16, "kv_heads": 2}
            cases.append(BenchmarkCase(f"{case.name}/{name}", BenchmarkTarget.from_spec(target), case.build))
    return BenchmarkSuite(tuple(cases), base.sampling, "attention-tiling-v1")


def admitted_suite(suite, *, tiny, placement_record, input_counts, checkpoint):
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_client_attention import attention_memory_admission
    from pllm.sources import _tiny_model_config

    records = preflight(suite, tiny=tiny, placement_record=placement_record,
                        input_counts=input_counts, checkpoint=checkpoint)
    for case, record in zip(suite.cases, records, strict=True):
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        config = (_tiny_model_config() if tiny else
                  json.loads((resolve_model(exp.pipeline.model).path / "config.json").read_bytes()))
        plan = lower_model(config, batch=1, max_input_tokens=exp.budget.max_input_tokens,
                           max_new_tokens=exp.budget.max_new_tokens)
        contract = json.loads(_native.decoder_attention_tiling(
            plan.canonical_bytes(), exp.pipeline.canonical_bytes(), 16, 2))
        record["attention"] = {"regions": len(contract["regions"]),
            "maximum_query_sequence": contract["maximum_query_sequence"],
            "maximum_key_sequence": contract["maximum_key_sequence"],
            "query_rows": 16, "kv_heads": 2}
        record["admission"] = attention_memory_admission(record["admission"],
            suite.sampling.memory_bytes, enforced_wan=True)
        checkpoint(records)
        if not record["admission"]["admitted"]:
            raise Rejected("attention_overlap_memory_limit")
    return records


def audit(run, *, tiny=False, placement_record=None):
    document = run.to_dict()
    suite = document["suite"]
    names = [case["name"] for case in suite["cases"]]
    processes = suite["sampling"]["processes"]
    rows = document["results"]
    if (not document["complete"] or len(rows) != len(names) * processes
            or any(row["status"] != "completed" for row in rows)):
        raise Rejected("incomplete_or_failed_cohort")
    seen, states, outputs, bodies, peaks, covered = set(), {}, {}, {}, {}, {}
    checked = 0
    maximum_swap = 0
    for row in rows:
        execution = row["execution"]
        name, replicate = execution["case"]["name"], execution["replicate"]
        if name not in names or type(replicate) is not int or not 0 <= replicate < processes or (name, replicate) in seen:
            raise Rejected("duplicate_or_unknown_cell")
        seen.add((name, replicate))
        workload, recipe, arm = name.split("/")
        group = f"{workload}/{recipe}"
        report = execution["details"]["report"]
        requests = report["runs"]
        research = execution["details"]["client_attention_research"]
        bindings = research["bindings"]
        if (report["checks"]["passed"] is not True or len(requests) != 2
                or research["selection"] != {"enabled": arm == "tiled", "query_rows": 16, "kv_heads": 2}
                or not bindings or any(b["failed"] or not b["phases"] or b["eligible_regions"] <= 0 for b in bindings)
                or (sum(b["tiles"] for b in bindings) > 0) != (arm == "tiled")):
            raise Rejected("response_or_attention_checks_failed")
        phases = [binding["phases"] for binding in bindings]
        if group in states and states[group] != phases:
            raise Rejected("full_logits_or_active_kv_differ")
        states[group] = phases
        for warm, request in enumerate(requests):
            if (request["tokens"]["authoritative"] is not True or request.get("cold") is not (warm == 0)
                    or request["privacy"].get("prefill_cache_hits") != warm
                    or request["privacy"].get("prefill_cache_misses") != 1 - warm):
                raise Rejected("cold_repeat_cache_contract_mismatch")
        if not tiny and (report["configuration"]["source_lock_digest"] != placement_record["source_lock_digest"]
                or any(r["model_fingerprint"] != placement_record["prior_body_fingerprint"]
                       or r["tokens"]["input_tokens"] != int(workload) for r in requests)):
            raise Rejected("measured_workload_or_body_mismatch")
        output = [(r["tokens"]["output_tokens"], r["generation"]["output_text_digest"]) for r in requests]
        if (any(not 1 <= count <= (2 if tiny else 8) or not commitment for count, commitment in output)
                or output[0] != output[1] or workload in outputs and outputs[workload] != output):
            raise Rejected("matched_outputs_differ")
        outputs[workload] = output
        metrics = {value["id"]: value["value"] for value in row["metrics"]}
        peak, body = metrics["client.peak_rss"], metrics["network.covered_body_bytes"]
        ledger = report["topology_accounting"]
        windows = [ledger["startup"], *ledger["warmups"], *ledger["runs"]]
        noncontrol = []
        for window in windows:
            total, controls = window.get("all_link_serialized_body_bytes"), window.get("lifecycle_control_serialized_body_bytes")
            if (type(total) is not int or type(controls) is not int or not 0 <= controls <= total
                    or window.get("control_body_counter_set_present") is not True):
                raise Rejected("body_control_coverage_missing")
            noncontrol.append(total - controls)
        if (peak is None or body is None or body != sum(w["all_link_serialized_body_bytes"] for w in windows)
                or group in bodies and bodies[group] != noncontrol):
            raise Rejected("peak_missing_or_application_bodies_differ")
        bodies[group] = noncontrol
        covered.setdefault(group, []).append(body)
        peaks.setdefault(name, []).append(peak)
        checked += sum(phase["checked_values"] for binding in phases for phase in binding)
        # The runtime monitor is authoritative; missing samples cannot mean zero.
        guard = report.get("memory_guard", {})
        swap = guard.get("maximum_host_swap_growth_bytes")
        if type(swap) is not int or swap < 0 or guard.get("tripped") is not False:
            raise Rejected("swap_observation_missing")
        maximum_swap = max(maximum_swap, swap)
    if any(len(values) != processes for values in peaks.values()):
        raise Rejected("missing_replicate")
    comparisons = [{"recipe": group, "metrics": {metric: run.compare(
        f"{group}/full", f"{group}/tiled", metric) for metric in METRICS}}
        for group in states]
    if any(row["metrics"]["client.peak_rss"]["status"] != "comparable" for row in comparisons):
        raise Rejected("matched_cohort_not_comparable")
    return {"schema": "pllm.attention_tiling_screen.v1", "completed_clients": len(rows),
        "completed_responses": 2 * len(rows), "checked_values": checked,
        "all_full_logits_active_kv_and_outputs_match": True,
        "noncontrol_application_bodies_match": True,
        "all_covered_application_bodies_match": all(len(set(values)) == 1 for values in covered.values()),
        "covered_body_ranges": {name: {"minimum": min(values), "maximum": max(values)} for name, values in covered.items()},
        "maximum_observed_swap_growth_bytes": maximum_swap,
        "observed_tiers": {name: {"largest_client_peak_bytes": max(values),
            "low": max(values) <= 256 << 20, "medium": max(values) <= 2 << 30}
            for name, values in peaks.items()},
        "comparisons": comparisons,
        "limitations": ["Matched screening cohort; independent finalist confirmation remains required.",
            "Full-logit/active-KV audit CPU and memory are charged identically to both arms.",
            "Observed tier eligibility is not conservative runtime admission or device memory.",
            "Cold/repeat timings and process-lifetime RSS have different windows.",
            "Variable lifecycle-control metadata stays charged; non-control bodies must match exactly.",
            "TCP streams exclude physical wire, checkpoint and pre-positioned tokenizer distribution."]}
