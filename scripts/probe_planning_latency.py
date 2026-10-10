"""Bounded Qwen-shaped planning overhead; no weights, providers or inference timings.

Run: uv run --no-sync python scripts/probe_planning_latency.py --output target/planning-latency.json
Prior values are synthetic ranking fixtures, never claimed serving measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.compiler import plan
from pllm.deployment import NetworkSnapshot, PartyOffer
from pllm.deployment.network import canonical
from pllm.kernels import Cpu
from pllm.modeling import lower_model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.schedulers import MlpPrefillTiling
from pllm.search import PerformanceSample, PlanningPolicy, PlanningPrior, PlanningRequest
from pllm.search._planning_cache import _cache
from pllm.search.performance import assignment_context, runtime_identity


def fixture():
    config = {"model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
        "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
        "vocab_size": 151936, "max_position_embeddings": 32768, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True}
    bound = ExecutionBudget(1, 64, 8)
    model_plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
    candidates = []
    for paged, packed, tiled, threads, demand in itertools.product((False, True), repeat=5):
        pipeline = MaskedLinearCpu(Model.tiny(), kernels=Cpu(threads=2 if threads else 1),
            delivery=ClientBundleTransport("artifacts", storage="paged" if paged else "memory"),
            linear=MaskedLinear(request_encoding="stage_packed" if packed else "raw",
                                output_encoding="row_residues"),
            inventory=PreparedInventory("request-sized", rows=1, refill="on-demand",
                                        allocation="demand" if demand else "uniform"),
            **({"scheduler": MlpPrefillTiling(tile_rows=16)} if tiled else {}))
        candidates.append(Experiment(f"choice-{len(candidates)}", pipeline,
            Deployment.local(root="local://planning-overhead"), bound))
    stamp = time.time_ns() // 1_000_000
    offers = tuple(PartyOffer(party, party, str(index + 1) * 64, stamp + 300_000,
        (capability,), 32 << 30, 32 << 30, 4, 0, ("*",), ("*",))
        for index, (party, capability) in enumerate((("client", "trusted_client"),
            ("inference", "masked_linear_provider"), ("preparation", "trusted_preparation"))))
    snapshot = NetworkSnapshot("overhead", offers, (), "synthetic-geometry", stamp, stamp + 300_000)
    request = PlanningRequest(model_plan, tuple(candidates), PlanningPolicy(evaluated_at_ms=stamp,
        allocation_model="native_shared_v1", objectives=("prefill_dependent_exchanges",
        "online_all_link_body_bytes", "preprocessing_all_link_body_bytes", "client_weight_bytes")),
        source_lock_digest=hashlib.sha256(canonical(config)).hexdigest())
    context = assignment_context(snapshot, {name: name for name in
        ("client", "inference", "preparation")}, None)
    prior = PlanningPrior(model_plan.digest, request.source_lock_digest, request.workload_cohort_digest,
        runtime_identity(), "b" * 64, "c" * 64, stamp, stamp + 300_000, 64, 8, 8, False,
        tuple(PerformanceSample(candidate.pipeline.digest(), context, 1000.0 + index, 100.0, 3)
              for index, candidate in enumerate(request.candidates)))
    return replace(request, performance_prior=prior), snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--warm-decisions", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 12 or not 1 <= args.warm_decisions <= 32:
        parser.error("pairs must be 1..12 and warm-decisions 1..32")
    if args.output.exists():
        parser.error("output already exists")
    request, snapshot = fixture()
    rows, identities = [], set()
    for pair in range(args.pairs):
        _cache.clear()
        for index in range(args.warm_decisions + 1):
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            fresh = replace(request, policy=replace(request.policy,
                evaluated_at_ms=time.time_ns() // 1_000_000))
            result = plan(fresh, snapshot=snapshot)
            elapsed, consumed = time.perf_counter_ns() - wall, time.process_time_ns() - cpu
            assert result.status == "feasible" and result.exhaustive
            assert result.costs["selection_basis"] == "matched_performance_prior"
            identities.add(result.experiment.pipeline.digest())
            rows.append({"pair": pair, "cache": "cold" if index == 0 else "warm",
                "wall_ms": elapsed / 1e6, "cpu_ms": consumed / 1e6,
                "coverage": result.to_spec()["coverage"]})
    assert len(identities) == 1
    summary = {name: {metric: statistics.median(row[metric] for row in rows if row["cache"] == name)
        for metric in ("wall_ms", "cpu_ms")} for name in ("cold", "warm")}
    report = {"schema": "pllm.planning_latency_probe.v1", "candidates": len(request.candidates),
        "scope": "public Qwen2.5-shaped 64+8 geometry; request refresh, native planning and serialization",
        "excludes": ["source loading", "initial lowering", "discovery", "lease operations", "model execution"],
        "prior_scope": "synthetic ranking oracle; no serving-performance claim",
        "cache_scope": "cold cache within one initialized process; not interpreter cold start",
        "runtime_digest": runtime_identity(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "implementation_sha256": {name: hashlib.sha256(
            (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest() for name in (
                "scripts/probe_planning_latency.py", "python/pllm/search/placement.py",
                "python/pllm/search/_planning_cache.py", "python/pllm/search/performance.py",
                "crates/pllm-compiler/src/network_placement.rs", "crates/pllm-python/src/network.rs")},
        "model_plan_digest": request.model_plan.digest, "summary": summary, "samples": rows,
        "cache": {"entries": len(_cache._entries), "charged_bytes": _cache._bytes,
                  "maximum_bytes": _cache.max_bytes}, "same_selected_pipeline": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
