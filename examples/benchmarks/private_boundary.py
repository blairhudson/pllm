"""Benchmark ordinary local versus coupled vocabulary boundaries through the SDK.

Default: generated checkpoint, one fresh process per choice. Co-located workers
are a functionality fixture. Use --model-dir for an explicit local checkpoint;
whole-topology memory admission still applies. Neither choice sets a RAM tier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, benchmark_suite
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead


def suite(*, model=None, processes=1, memory_bytes=3 << 30):
    cases = []
    build = BuildRef.capture()
    for coupled in (False, True):
        name = "coupled-boundary" if coupled else "local-boundary"
        experiment = Experiment(name, MaskedLinearCpu(model or Model.tiny(),
            inventory=PreparedInventory(rows=1, allocation="demand", refill="on-demand"),
            boundary=PrivateLookupAndFullHead() if coupled else None,
            topology=PreparedBoundaryRoles() if coupled else None),
            Deployment.local(root=f".pllm/private-boundary-bench/{name}"),
            ExecutionBudget(requests=1, max_input_tokens=128, max_new_tokens=2))
        cases.append(BenchmarkCase(name, BenchmarkTarget.response(experiment,
            scenario=Scenario.cold(max_output_tokens=2), backend="native", wan=WanConditions()), build=build))
    return BenchmarkSuite(cases, Sampling(processes=processes, memory_bytes=memory_bytes,
                                         timeout_seconds=240))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--memory-mib", type=int, default=3072, help="whole-topology admission budget, not a client RAM tier")
    args = parser.parse_args(argv)
    model = Model.path(str(args.model_dir.resolve())) if args.model_dir else None
    prompt = args.prompt_file.read_text() if args.prompt_file else "Explain addition briefly."
    selected = suite(model=model, processes=args.processes, memory_bytes=args.memory_mib << 20)
    report = benchmark_suite(selected, inputs={case.name: {"prompt": prompt} for case in selected.cases},
                             output=args.output)
    statuses = [result.to_dict()["status"] for result in report.results]
    print({"output": args.output, "statuses": statuses})
    if any(status != "completed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
