"""Fresh whole-SDK response screen for one compiler-bound native client stage.

Uses the maintained prepared recipe, exact build/cohort locks, ordinary memory
admission and shared 100/40 native pacing. Private prompts enter workers on stdin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Experiment, Model
from pllm.deployment import WanConditions
from pllm.metrics import (BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, BuildRef,
                          Sampling, Scenario, benchmark_suite)
from pllm.runtime.benchmark_response_target import SERVING_METRICS
from pllm.search import FrontierStudy, Variant

from examples.prepared_sdk import build_experiment


def study(*, model, tokenizer, build, workloads, processes=3, serial_issuance=False, input_bound=None):
    cases = []
    for name in workloads:
        maximum, outputs = {"short": (128, 8), "medium": (256, 8),
                             "long": (512, 8), "decode": (128, 32), "tiny": (128, 3),
                              "fresh": (256, 8), "growing": (512, 8)}[name]
        if input_bound is not None:
            maximum = input_bound
        count = 3 if name == "growing" else 2
        experiment = build_experiment(model, tokenizer=tokenizer).with_params(
            budget__max_input_tokens=maximum, budget__max_new_tokens=outputs,
            budget__requests=count, pipeline__cache__fixed_input_tokens=maximum)
        if serial_issuance:
            experiment = experiment.with_params(pipeline__inventory__stage_window=1,
                pipeline__inventory__stage_window_bytes=16 << 20)
        scenario = (Scenario.sequence(name, count=count, max_output_tokens=outputs)
                    if name in {"fresh", "growing"} else Scenario.repeat(name, max_output_tokens=outputs))
        cases.append(BenchmarkCase(name, BenchmarkTarget.response(experiment,
            scenario=scenario, wan=WanConditions(),
            client_stage={"owner": "python", "role": "mlp_gate_up", "layer": 0}), build))
    return FrontierStudy(BenchmarkSuite(tuple(cases), Sampling(processes=processes,
        timeout_seconds=1200, memory_bytes=8 << 30), "client-stage-ownership-v1"),
        (Variant("native", {"client_stage": {"owner": "native", "role": "mlp_gate_up", "layer": 0}}),))


def workload_inputs(short, medium, *, tiny=False):
    """Public fixed message histories; model outputs never select later inputs."""
    other = "Black holes?" if tiny else "Compare a black hole with a neutron star in two sentences."
    turns = (["Blue sky?", "Blue light scatters.", "Sunset?", "Red light remains.", "Clouds?"]
             if tiny else [short,
                 "Air scatters shorter wavelengths more strongly; longer paths at sunset favour red light.",
                 "How do clouds change the colours we see?",
                 "Water droplets scatter many wavelengths, so thick clouds often look white or grey.",
                 "Now explain how the same principles apply when viewing Earth from orbit."])
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
               for i, text in enumerate(turns)]
    return {"tiny": {"prompt": short}, "short": {"prompt": short}, "medium": {"prompt": medium},
            "long": {"prompt": medium + "\n" + medium + "\n" + short}, "decode": {"prompt": short},
            "fresh": {"prompts": [short, other]},
            "growing": {"prompts": [history[:end] for end in (1, 3, 5)]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python")
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--prompt-file", type=Path, help="Public medium-workload text, independent of model artifacts")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--serial-issuance", action="store_true",
                        help="Use explicitly serial Preparation in both controls and candidates")
    parser.add_argument("--input-bound", type=int,
                        help="Explicit public compiled input-token bound for each selected workload")
    parser.add_argument("--workload", choices=("short", "medium", "long", "decode", "fresh", "growing"), action="append")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--confirm", type=Path)
    parser.add_argument("--nominate", action="append")
    args = parser.parse_args(argv)
    if args.nominate and args.confirm is None:
        parser.error("--nominate requires --confirm")
    if args.output.exists():
        raise FileExistsError("output already exists")
    if args.tiny:
        model, tokenizer = Model.tiny(), None
        short = medium = "Explain blue sky."
        workloads = args.workload or ["tiny"]
    else:
        if (args.artifacts / "indexed.json").exists():
            base = Experiment.from_spec(json.loads((args.artifacts / "indexed.json").read_text()))
            model, tokenizer = base.pipeline.model, base.pipeline.components["tokenizer"]
        else:
            from pllm.tokenization import IndexedTokenizer
            publisher = json.loads((args.artifacts / "publisher.json").read_text())
            if publisher["schema"] != "pllm.public_tokenizer_build.v1":
                raise ValueError("unsupported public tokenizer publisher record")
            model = Model.hf(publisher["model"], revision=publisher["revision"], local_files_only=True)
            tokenizer = IndexedTokenizer(str((args.artifacts / "tokenizer").resolve()),
                                         digest=publisher["tokenizer_contract_sha256"])
        medium = (args.prompt_file or args.artifacts / "prompt.txt").read_text()
        # Public research workload text; only salted commitments enter evidence.
        short = "Explain why the sky is blue, then describe how its colour changes at sunset."
        workloads = args.workload or ["short", "medium", "long", "decode", "fresh", "growing"]
    inputs = workload_inputs(short, medium, tiny=args.tiny)
    build = BuildRef.capture(args.python) if args.python else BuildRef.capture()
    current = study(model=model, tokenizer=tokenizer, build=build, workloads=workloads,
                    processes=args.processes, serial_issuance=args.serial_issuance,
                    input_bound=args.input_bound)
    screen = BenchmarkRun.load(args.confirm) if args.confirm else None
    work = (current.confirm(screen, cases=args.nominate, processes=max(3, args.processes))
            if screen is not None else current.plan(max_trials=256))
    if work.suite is None:
        raise ValueError("no measured confirmation candidates selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".plan.json").write_text(json.dumps(
        {"study": current.to_spec(), "coverage": work.coverage.to_dict()}, indent=2) + "\n")
    run = benchmark_suite(work.suite,
        inputs=work.inputs({name: inputs[name] for name in workloads}), output=args.output)
    snapshot = current.snapshot(run) if screen is None else current.snapshot(screen, confirmation=run)
    args.output.with_suffix(".frontier.json").write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
    summaries = []
    measured = {case.name for case in work.suite.cases}
    for name in workloads:
        if name not in measured:
            continue
        summaries.append({"workload": name, "metrics": {
            metric: run.compare(name, name + "/native", metric) for metric in SERVING_METRICS}})
    print(json.dumps({"completed": sum(r.to_dict()["status"] == "completed" for r in run.results),
                      "trials": len(run.results), "comparisons": summaries}, indent=2))


if __name__ == "__main__":
    main()
