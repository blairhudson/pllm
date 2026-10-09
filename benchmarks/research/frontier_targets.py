"""Five tenfold challenge screens through installed, compiler-bound SDK targets.

Only public configuration is resolved. Stage weights and KV are synthetic, never
checkpoint-fidelity evidence. Run from the repository with ``python -m``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.evidence import EvidenceRegistry
from pllm.metrics import (BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef,
                          Sampling, TargetBudget, benchmark_suite)
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant

from benchmarks.research.common import MODEL, REVISION


def study(*, processes=3):
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REVISION,
                                            local_files_only=True)).read_bytes())
    model = Model.hf(MODEL, revision=REVISION, local_files_only=True)
    experiment = Experiment("tenfold-targets", MaskedLinearCpu(model),
        Deployment.local(root="local://frontier-targets"), ExecutionBudget(2, 150, 8))
    build = BuildRef.capture()
    def case(name, target, budget=()):
        return BenchmarkCase(name, target, build, budget)
    cases = [
        case("A-client-floor", BenchmarkTarget.runtime("client"), (TargetBudget("process.peak_rss", maximum=22_585_000),)),
        case("B-preparation-floor", BenchmarkTarget.runtime("preparation"), (TargetBudget("process.peak_rss", maximum=19_826_000),)),
    ]
    stages = []
    for role in ("qkv_projection", "mlp_gate_up", "mlp_down"):
        for rows in (1, 16, 150):
            name = f"C-{role}-{rows}"
            stages.append(name)
            cases.append(case(name, BenchmarkTarget.preparation(experiment, role=role, rows=rows, config=config)))
    # B × C prices the loss of matrix batching instead of multiplying gains.
    chunk_case = "C-mlp_gate_up-150"
    state = experiment.with_params(budget__max_input_tokens=2048)
    cases += [case("D-state", BenchmarkTarget.state(state, tokens=2048, operation="attention", config=config)),
              case("E-restart", BenchmarkTarget.state(experiment, tokens=128, operation="restart",
                                                       storage="sealed_local", config=config))]
    long = experiment.with_params(budget__max_input_tokens=32760)
    for floor in (62_000_000, 8_000_000):
        for live in (1, 2):
            cases.append(case(f"D-geometry-{floor}-{live}", BenchmarkTarget.projection(long, construction="layer_kv",
                parameters={"fixed_bytes": floor, "live_layers": live}, config=config)))
    for resumed in (8_000_000, 20_000_000):
        cases.append(case(f"E-horizon-{resumed}", BenchmarkTarget.projection(experiment, construction="restart",
            parameters={"bootstrap_bytes": 116_840_000, "publication_bytes": 6_000_000,
                        "control_bytes": 116_840_000, "resumed_bytes": resumed, "resumptions": 32}, config=config)))
    return FrontierStudy(BenchmarkSuite(tuple(cases), Sampling(processes=processes, timeout_seconds=300,
                                                               memory_bytes=768 << 20), "tenfold-challenges"),
        (Variant("digit4", {"kernel": "digit4"}, targets=stages),
         Variant("digit8", {"kernel": "digit8"}, targets=stages),
         Variant("rows4", {"chunk_rows": 4}, targets=(chunk_case,)),
         Variant("sealed", {"storage": "sealed_local"}, targets=("D-state",))),
        (("digit4", "rows4"), ("digit8", "rows4")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    current = study(processes=args.processes)
    work = current.plan(max_trials=256)
    result = benchmark_suite(work.suite, output=args.output)
    snapshot = current.snapshot(result, metrics={"target.cpu": "min", "process.peak_rss": "min"})
    registry = EvidenceRegistry().register_all(result.results)
    args.output.with_suffix(".frontier.json").write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
    args.output.with_suffix(".coverage.json").write_text(json.dumps(current.plan(registry).coverage.to_dict(), indent=2) + "\n")
    statuses = {}
    for row in result.results:
        status = row.to_dict()["status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(json.dumps({"processes": len(result.results), "status": statuses, "output": str(args.output)}))


if __name__ == "__main__":
    main()
