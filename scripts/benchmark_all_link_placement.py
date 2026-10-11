"""Measure explicit byte-aware layer ownership through ordinary SDK responses.

Selection uses a frozen prior byte ledger and the native current schedule. The
new cohort charges cold delivery, both prepared roles and exact-repeat execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

from filelock import FileLock

from pllm import Experiment, lower_model
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, benchmark_suite
from pllm.model_loader import resolve_model
from pllm.roles import ClientLinearRoles
from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role

from probe_all_link_frontier import choose_placement


def select(control, plan, pipeline, *, input_rows, output_tokens):
    """Bound both actual terminal-pruned prefill and ordinary decode body MACs."""
    schedule = plan.runtime_schedule(pipeline).to_dict()
    stages = {stage.id: stage for stage in scheduled_stage_specs(plan, pipeline)
              if stage.layer_index is not None}
    if set(stages) != set(control["stages"]):
        raise ValueError("prior ledger and compiled stage groups differ")
    operations = {op["id"]: op for op in plan.prefill["operations"]}
    steps = {(step["layer"], semantic_stage_role(step, operations)): step
             for step in schedule["prefill"]["steps"] if step["weight_ids"] and step["layer"] is not None}
    items = []
    for stage in stages.values():
        step = steps[stage.layer_index, stage.role]
        rows = 1 if step.get("terminal_row_only", False) else input_rows
        body = control["stages"][stage.id]
        items.append({"id": stage.id, "macs": rows * stage.in_features * stage.out_features,
            "weight_and_scale_bytes": (stage.in_features + 4) * stage.out_features,
            "pair_stage_bytes": sum(body.get(link, 0) for link in (
                "preparation->inference", "client->inference", "inference->client"))})
    total_prefill = sum(item["macs"] for item in items)
    selection = choose_placement(items, budget=total_prefill // 10, horizon=1)
    chosen = [stages[name] for name in selection["selected"]]
    total_decode = sum(s.in_features * s.out_features for s in stages.values())
    local_decode = sum(s.in_features * s.out_features for s in chosen)
    local_prefill = selection.pop("client_dense_macs_per_row")
    if not chosen or local_decode * 10 > total_decode:
        raise ValueError("selected stages exceed the separate decode MAC limit")
    pairs = tuple(sorted((stage.layer_index, stage.role) for stage in chosen))
    selected = pipeline.with_params(placement=ClientLinearRoles(layer_roles=pairs))
    checked = plan.runtime_schedule(selected).to_dict()
    for phase in ("prefill", "decode"):
        operations = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        actual = {(step["layer"], semantic_stage_role(step, operations))
                  for step in checked[phase]["steps"] if step["executor"] == "client_linear"}
        if actual != set(pairs):
            raise ValueError("native ownership differs from selected groups")
    selection.update(layer_roles=pairs, prefill_client_macs=local_prefill,
        prefill_body_macs=total_prefill, decode_client_macs=local_decode, decode_body_macs=total_decode,
        pair_client_macs=local_prefill + 2 * (output_tokens - 1) * local_decode,
        pair_body_macs=total_prefill + 2 * (output_tokens - 1) * total_decode,
        scope="prior-byte estimate; actual pruned prefill and decode each keep >=90% body MACs remote")
    return selected, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--prior", type=Path, default=Path("docs/evidence/research-all-link-frontier.json"))
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 6:
        parser.error("processes must be in [1,6]")
    if args.output.exists():
        raise FileExistsError(args.output)
    build_experiment = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/prepared_sdk.py"))["build_experiment"]
    base = Experiment.from_spec(json.loads((args.artifacts / "indexed.json").read_bytes()))
    control = build_experiment(base.pipeline.model, tokenizer=base.pipeline.components["tokenizer"])
    control = control.with_params(pipeline__preparation__kernel="digit8", pipeline__preparation__min_rows=64)
    source = resolve_model(control.pipeline.model)
    prior = json.loads(args.prior.read_bytes())["control"]
    plan = lower_model(json.loads((source.path / "config.json").read_bytes()), batch=1,
        max_input_tokens=control.budget.max_input_tokens, max_new_tokens=control.budget.max_new_tokens)
    start = time.process_time()
    pipeline, selection = select(prior, plan, control.pipeline, input_rows=150, output_tokens=8)
    selection["planning_cpu_seconds"] = time.process_time() - start
    selected = control.with_params(name="byte-aware-placement", pipeline=pipeline)
    build = BuildRef.capture()
    suite = BenchmarkSuite(tuple(BenchmarkCase(name, BenchmarkTarget.response(exp,
        scenario=Scenario.repeat(max_output_tokens=8), wan=WanConditions()), build)
        for name, exp in (("control", control), ("selected", selected))),
        Sampling(processes=args.processes, timeout_seconds=900, memory_bytes=8 << 30),
        "all-link-stage-placement-v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name("probe_all_link_frontier.py"),
               Path("examples/prepared_sdk.py"), args.prior]
    hashes = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    frozen = hashes()
    args.output.with_suffix(".plan.json").write_text(json.dumps({"suite": suite.to_spec(),
        "selection": selection, "frozen_sources": frozen, "source_lock_digest": source.source_lock_digest,
        "prior_body_fingerprint": prior["body_fingerprint"]}, indent=2) + "\n")
    if args.plan_only:
        return
    prompt = (args.artifacts / "prompt.txt").read_text()
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        run = benchmark_suite(suite, inputs={name: {"prompt": prompt} for name in ("control", "selected")},
                              output=args.output)
    if hashes() != frozen:
        raise RuntimeError("probe sources changed during the cohort")
    for result in run.results:
        row = result.to_dict()
        if row["status"] != "completed":
            raise RuntimeError("cohort contains a failed/blocked cell; retained evidence is not ranked")
        report = row["execution"]["details"]["report"]
        if (any(r["model_fingerprint"] != prior["body_fingerprint"] or r["tokens"]["input_tokens"] != 150
                or r["tokens"]["output_tokens"] != 8 for r in report["runs"])):
            raise RuntimeError("measured numeric/workload identity differs from selection")
    print(json.dumps({metric: run.compare("control", "selected", metric)
        for metric in ("network.covered_body_bytes", "network.all_link_stream_bytes", "client.peak_rss",
                       "client.cold_cpu", "preparation.cold_cpu", "preparation.peak_rss", "response.ttft",
                       "response.decode_tps")}, indent=2))


if __name__ == "__main__":
    main()
