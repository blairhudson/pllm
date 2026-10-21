"""Compare scalar and source-stage-batched GKM using ordinary response targets.

Each sharing mode retains its own privacy contract. Compare batching against its
matching scalar control, not across sharing modes. Accepted-state audits, shared
native 100/40 Mbps caps, source delivery scope and fixed-count output semantics
come from the maintained sharing suite. Batching is explicit and HTTP-only;
admission failures stay visible rather than falling back to scalar execution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pllm import Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, benchmark_suite

from examples.benchmarks.gkm_sharing import suite as sharing_suite
from examples.papers.common import MODEL


def suite(*, model=MODEL, processes=3, input_tokens=8, output_tokens=2, memory_bytes=16 << 30):
    controls = sharing_suite(model=model, processes=processes, input_tokens=input_tokens,
                             output_tokens=output_tokens, memory_bytes=memory_bytes)
    cases = []
    for control in controls.cases:
        cases.append(control)
        cases.append(BenchmarkCase(
            f"{control.name}-batch",
            control.target.with_params(pipeline__linear__execution="source_stage_batch_v1"),
            build=control.build, budgets=control.budgets))
    return BenchmarkSuite(cases, controls.sampling, name="gkm-prefill-batch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--input-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--memory-mib", type=int, default=16384,
                        help="whole-topology admission budget; not a client memory tier")
    args = parser.parse_args(argv)
    model = Model.path(str(args.model_dir.resolve())) if args.model_dir else MODEL
    prompt = args.prompt_file.read_text() if args.prompt_file else "Hello world"
    selected = suite(model=model, processes=args.processes, input_tokens=args.input_tokens,
                     output_tokens=args.output_tokens, memory_bytes=args.memory_mib << 20)
    report = benchmark_suite(selected, inputs={case.name: {"prompt": prompt}
                                              for case in selected.cases}, output=args.output)
    statuses = [result.to_dict()["status"] for result in report.results]
    print({"output": args.output, "statuses": statuses})
    if any(status != "completed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
