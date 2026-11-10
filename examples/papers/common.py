"""Shared source and workload bounds for the ten-paper SDK study."""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

from pllm import ExecutionBudget, Model


MODEL = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_files_only=True,
)
BUDGET = ExecutionBudget(requests=32, max_input_tokens=64, max_new_tokens=8)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def benchmark_arguments(
    paper, argv, *, task="response", max_input_tokens=32, max_new_tokens=2
) -> argparse.Namespace:
    """Parse a bounded workload without running or provisioning a benchmark.

    Token defaults are ceilings, not forced token counts. Explicit CLI values
    replace them without automatic capping. Forward JSON is passed through as a
    mapping; the SDK remains responsible for its authoritative input schema.
    """
    if task not in {"response", "forward"}:
        raise ValueError("task must be 'response' or 'forward'")
    parser = argparse.ArgumentParser(description=f"Benchmark {paper}", allow_abbrev=False)
    parser.add_argument(
        "--model-path", type=Path, required=task == "forward",
        help="offline model bundle (response defaults to the pinned MODEL)",
    )
    parser.add_argument(
        "--max-input-tokens", type=_positive_int, default=max_input_tokens,
        help="input token ceiling (default: %(default)s)",
    )
    if task == "response":
        parser.add_argument(
            "--max-new-tokens", type=_positive_int, default=max_new_tokens,
            help="output token ceiling (default: %(default)s)",
        )
        parser.add_argument("--prompt", default="hi")
    else:
        parser.add_argument("--forward-inputs", type=Path, required=True,
                            help="path to a JSON input mapping")
    parser.add_argument("--processes", type=_positive_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--warmups", type=_nonnegative_int, default=0)
    parser.add_argument("--memory-bytes", type=_positive_int,
                        help="explicit sampling memory limit; default is 512 MiB")
    parser.add_argument(
        "--output", type=Path,
        default=Path("target/paper-benchmarks") / f"{paper}.json",
    )
    args = parser.parse_args(argv)

    # Help and parse errors exit before constructing workload SDK declarations.
    from pllm import ForwardBudget
    from pllm.metrics import Sampling, Scenario

    if task == "forward":
        try:
            inputs = json.loads(args.forward_inputs.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            parser.error(f"cannot read --forward-inputs: {exc}")
        if not isinstance(inputs, Mapping):
            parser.error("--forward-inputs must contain a JSON mapping, not a list or scalar")
        args.inputs = inputs
    else:
        args.inputs = {"prompt": args.prompt}

    sampling_options = dict(
        processes=args.processes, repetitions=args.repetitions, warmups=args.warmups
    )
    if args.memory_bytes is not None:
        sampling_options["memory_bytes"] = args.memory_bytes
    try:
        args.sampling = Sampling(**sampling_options)
        args.model = MODEL if args.model_path is None else Model.path(str(args.model_path))
        if task == "forward":
            args.budget = ForwardBudget(requests=1, max_input_tokens=args.max_input_tokens)
            args.scenario = None
        else:
            args.budget = ExecutionBudget(
                requests=1, max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            args.scenario = Scenario.cold("request", max_output_tokens=args.max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    return args
