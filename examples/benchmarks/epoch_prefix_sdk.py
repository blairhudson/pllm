"""Ordinary SDK epoch and same-client prefix-sharing benchmark suites.

Run installed wheel, with private prompt on stdin (never command-line metadata):
  python examples/benchmarks/epoch_prefix_sdk.py --output epoch-prefix.json < prompt.txt

No speed claims. Parent must serialize wheel builds/cohorts with frontier lock.
Use matched generated early-EOS/full-cap sources via ``suite(source=...)``; this
example never overrides natural EOS or patches runtime generation.
"""
from __future__ import annotations

import argparse
import sys

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, benchmark_suite
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


def experiment(*, source=None, epoch=None, inflight=False, requests=2, output_tokens=8):
    """Matched exact-numeric full-KV control/candidate; no runtime adapter."""
    return Experiment("epoch-prefix-sdk", MaskedLinearCpu(source or Model.tiny(),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand",
                                    allocation="demand", decode_epoch_tokens=epoch),
        cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=128,
                               inflight=inflight, inflight_max_requests=8, inflight_max_bytes=64 << 20)),
        Deployment.local(root="local://epoch-prefix-sdk"),
        ExecutionBudget(requests=requests, max_input_tokens=128, max_new_tokens=output_tokens))


def suite(*, source=None, epoch=2, concurrent_requests=2, output_tokens=8, memory_bytes=6 << 30):
    scenario = Scenario.cold(max_output_tokens=output_tokens)
    cases = []
    for name, span, sharing, concurrency in (
            ("serial-control", None, False, 1), ("serial-epochs", epoch, False, 1),
            ("group-control", None, False, concurrent_requests),
            ("group-inflight", None, True, concurrent_requests)):
        candidate = experiment(source=source, epoch=span, inflight=sharing,
                               requests=concurrent_requests, output_tokens=output_tokens)
        cases.append(BenchmarkCase(name, BenchmarkTarget.response(candidate, scenario=scenario,
                                                                  concurrent_requests=concurrency)))
    return BenchmarkSuite(tuple(cases), Sampling(processes=1, memory_bytes=memory_bytes), name="epoch-prefix-sdk")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--concurrent-requests", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--output-tokens", type=int, default=8)
    args = parser.parse_args()
    prompt = sys.stdin.read(16385)
    if not prompt or len(prompt) > 16384:
        parser.error("stdin requires a private prompt of 1..16384 characters")
    selected = suite(epoch=args.epoch, concurrent_requests=args.concurrent_requests,
                     output_tokens=args.output_tokens)
    result = benchmark_suite(selected, inputs={case.name: {"prompt": prompt} for case in selected.cases},
                             output=args.output)
    statuses = [row.to_dict()["status"] for row in result.results]
    print({"output": args.output, "statuses": statuses})
    if any(status != "completed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
