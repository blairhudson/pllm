"""Screen or independently confirm the maintained prepared-serving SDK study.

The report, coverage plan, frontier snapshot and exact Experiment recipes share
one study identity. Historical evidence is retained; current response inputs and
checkpoint files are always revalidated. Outputs contain no prompt text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.evidence import EvidenceRegistry
from pllm.metrics import BenchmarkRun, BuildRef, Sampling, benchmark_suite

from benchmarks.research.prepared_frontier_study import study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true", help="Generated checkpoint transport control")
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--python", help="Installed interpreter for measured workers")
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--confirm", type=Path, help="Completed exploratory BenchmarkRun")
    parser.add_argument("--nominate", action="append", help="Measured case names; default: Pareto neighbors")
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if not 1 <= args.max_trials <= 1024:
        parser.error("--max-trials must be in [1,1024]")
    if args.output.exists():
        raise FileExistsError("output already exists")
    model = tokenizer = None
    if args.tiny:
        prompt, output_tokens = "Explain blue sky.", 2
    else:
        from pllm import Experiment
        # Explicit offline inputs; their source/object identities are revalidated
        # by ordinary SDK admission in every measured client.
        base = Experiment.from_spec(json.loads((args.artifacts / "indexed.json").read_text()))
        model, tokenizer = base.pipeline.model, base.pipeline.components["tokenizer"]
        prompt, output_tokens = (args.artifacts / "prompt.txt").read_text(), 8
    build = BuildRef.capture(args.python) if args.python else BuildRef.capture()
    current = study(model=model, tokenizer=tokenizer, build=build,
        sampling=Sampling(processes=1, timeout_seconds=args.timeout, memory_bytes=8 << 30),
        output_tokens=output_tokens)
    registry = EvidenceRegistry()
    for path in args.history:
        for result in BenchmarkRun.load(path).results:
            registry = registry.register(result)
    screen = BenchmarkRun.load(args.confirm) if args.confirm else None
    work = (current.confirm(screen, cases=args.nominate) if screen else
             current.plan(registry, max_trials=args.max_trials, changed_paths=args.changed_path))
    if work.coverage.to_dict()["selected_trials"] > args.max_trials:
        parser.error("confirmation exceeds --max-trials; narrow nominees or raise the explicit budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan_path = args.output.with_suffix(".plan.json")
    plan_path.write_text(json.dumps({"study": current.to_spec(), "coverage": work.coverage.to_dict()}, indent=2) + "\n")
    if args.plan_only or work.suite is None:
        print(json.dumps(work.coverage.to_dict(), indent=2))
        return
    run = benchmark_suite(work.suite, inputs=work.inputs({"prepared": {"prompt": prompt}}), output=args.output)
    snapshot = current.snapshot(screen or run, confirmation=run if screen else None)
    args.output.with_suffix(".frontier.json").write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
    print(json.dumps({"run": str(args.output), "study": current.digest,
        "statuses": [{"case": r.to_dict()["execution"]["case"]["name"], "status": r.to_dict()["status"]}
                     for r in run.results], "frontier": snapshot.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
