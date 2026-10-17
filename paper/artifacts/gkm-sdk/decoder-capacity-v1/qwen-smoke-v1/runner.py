"""Pinned-source GKM capacity smoke through the installed public benchmark API.

Preparation hashes source files and inspects native geometry without tensor-value
import. Execution rechecks ordinary host admission and uses its normal watchdog.
This is one functionality sample, not quality or comparative performance evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import (
    BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario,
    benchmark_memory, benchmark_suite,
)
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime.delegated_source import admit_source


ROOT = Path(__file__).resolve().parents[3]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
PROMPT = "Hello."


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == "prepare":
        directory.mkdir(parents=True, exist_ok=False)
        experiment = Experiment(
            "gkm-pinned-qwen-capacity-smoke",
            OpenWeightVerifiableCpu(Model.path(str(SOURCE)),
                linear=OpenWeightVerifiableOutsourcing(capacity="decoder")),
            Deployment.local(root=str(directory / "roles")),
            ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2),
        )
        with (directory / "experiment.json").open("xb") as stream:
            stream.write(experiment.canonical_bytes())
        source = admit_source(experiment)
        write(directory / "source-lock.json", source.source_lock)
        write(directory / "native-contract.json", source.contract)
        write(directory / "runtime-contract.json", source.runtime_contract)
        report = benchmark_memory(experiment, backend="native")
        write(directory / "preflight.json", report)
        write(directory / "inputs.json", {
            "source_repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "source_revision": REVISION,
            "public_prompt": PROMPT,
            "max_output_tokens": 2,
            "max_input_tokens": 64,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.executable,
            "scope": "one source-present local full-share functionality smoke",
        })
        (directory / "runner.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({"admitted": report["admitted"], "native": report["candidates"]["native"]}, indent=2))
        return int(not report["admitted"])

    experiment = Experiment.from_spec(json.loads((directory / "experiment.json").read_text()))
    if not json.loads((directory / "preflight.json").read_text())["admitted"]:
        raise ValueError("frozen preflight did not admit execution")
    inputs = json.loads((directory / "inputs.json").read_text())
    if inputs["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("smoke runner changed after preparation")
    case = BenchmarkCase("gkm-pinned-qwen-capacity-smoke", BenchmarkTarget.response(
        experiment, scenario=Scenario.cold(max_output_tokens=2), backend="native"),
        build=BuildRef.capture())
    result = benchmark_suite(BenchmarkSuite([case], Sampling(processes=1,
        timeout_seconds=1200, memory_bytes=16 << 30)),
        inputs={case.name: {"prompt": PROMPT}}, output=directory / "benchmark.json")
    status = result.results[0].to_dict()["status"]
    print(json.dumps({"status": status, "output": str(directory / "benchmark.json")}))
    return int(status != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
