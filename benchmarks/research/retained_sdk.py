"""Frozen prepared SDK retention factories and fresh-client public cohorts.

Each context has one ordinary CLI invocation/private cohort salt. Policies have
separate numeric contracts; the CLI deliberately refuses a cross-policy winner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.passes import KvCacheEviction
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientLocalKv

from benchmarks.research.common import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("dense", "full", "keep75", "keep50")
PROMPTS = {
    "short": "A public inventory lists three crates containing twelve, seven and nine books. "
             "Explain why keeping an accurate inventory is useful.",
    "long": "The following inventory journal is fictional and public. "
            + " ".join(f"Crate {i} contains {(i * 7) % 31 + 4} books checked on day {i + 1}."
                       for i in range(1, 25))
            + " Explain why preserving this journal helps detect missing books."
}


def dense():
    return Experiment("retained-sdk-dense", MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION, local_files_only=True),
        kernels=Cpu(threads=4),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        preparation=ModelAwareCorrections(storage="paged"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed",
                            prefill_pruning="terminal"),
        delivery=ClientBundleTransport("artifacts", storage="paged", compression="zlib", batch_objects=64)),
        Deployment.local(root="local://retained-sdk"),
        ExecutionBudget(requests=1, max_input_tokens=512, max_new_tokens=8))


def _retained(name, keep):
    return dense().with_params(name=f"retained-sdk-{name}", pipeline__state=ClientLocalKv(
        retention=KvCacheEviction(static_keep=keep, observation_window=(1, 4),
                                 dynamic_keep=(1, 1), share_adjacent_layers=False)))


def full():
    return _retained("full", (1, 1))


def keep75():
    return _retained("keep75", (3, 4))


def keep50():
    return _retained("keep50", (1, 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation():
    import pllm
    package = Path(pllm.__file__).parent
    return {str(path.relative_to(package)): sha(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so", ".json"}}


def run(directory, *, contexts=None, journal_rows=24):
    if type(journal_rows) is not int or not 1 <= journal_rows <= 24:
        raise ValueError("journal rows must be between 1 and 24")
    workloads = dict(PROMPTS)
    if journal_rows != 24:
        workloads["long"] = ("The following inventory journal is fictional and public. "
            + " ".join(f"Crate {i} contains {(i * 7) % 31 + 4} books checked on day {i + 1}."
                       for i in range(1, journal_rows + 1))
            + " Explain why preserving this journal helps detect missing books.")
    if contexts:
        if len(contexts) != len(set(contexts)) or any(name not in workloads for name in contexts):
            raise ValueError("contexts must select distinct public workloads")
        workloads = {name: workloads[name] for name in contexts}
    directory.mkdir(parents=True, exist_ok=False)
    frozen = implementation()
    targets = [f"benchmarks/research/retained_sdk.py:{name}" for name in POLICIES]
    base = [sys.executable, "-m", "pllm", "benchmark", "run", "--factory", "--trust-python",
            "--backend", "native", "--wan", "--isolate-candidates", "--temperature", "0",
            "--capture-output-digest", "--max-output-tokens", "8", "--warmups", "0",
            "--repetitions", "1", "--timeout", "1800"]
    for target in targets:
        base.extend(["--experiment", target])
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    metadata = {"schema": "pllm.retained_sdk_reproduction.v1", "targets": targets,
                "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "factory_sha256": sha(__file__), "installed_package_sha256": frozen,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}},
                "workloads": {name: {"public_prompt": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                      "output_cap": 8} for name, text in workloads.items()},
                "journal_rows": journal_rows,
                "execution": {"backend": "native", "wan_mbps": [100, 40], "fresh_clients": True,
                              "blas_threads": 1, "native_threads_per_executor": 4,
                              "cohort_salt": "private; separate salt per context; never retained",
                              "repetitions": 1, "temperature": 0, "warmups": 0,
                              "checkpoint_cache": "pre-positioned shared Hub/compiled cache",
                              "bundle_cache": "fresh private dashboard cache per candidate"}}
    (directory / "reproduction.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryDirectory(prefix="retained-sdk-", dir=directory) as temporary:
        for name, prompt in workloads.items():
            if implementation() != frozen:
                raise ValueError("installed implementation changed within cohort")
            prompt_path = Path(temporary) / "prompt.txt"
            prompt_path.write_text(prompt)
            arguments = base + ["--prompt-file", str(prompt_path)]
            subprocess.run(arguments + ["--preflight-only", "--output", str(directory / f"{name}-preflight.json")],
                           cwd=ROOT, env=environment, check=True)
            preflight = json.loads((directory / f"{name}-preflight.json").read_text())
            if not preflight["admitted"]:
                raise ValueError("retention SDK cohort failed memory admission")
            result = subprocess.run(arguments + ["--output", str(directory / f"{name}.json")],
                                    cwd=ROOT, env=environment, check=False)
            if result.returncode not in (0, 6):
                raise RuntimeError("retention SDK benchmark failed")
            report = json.loads((directory / f"{name}.json").read_text())
            if (report.get("schema_version") != "pllm.loopback_benchmark_comparison.v1"
                    or len(report["candidates"]) != 4 or not report["checks"]["all_candidates_passed"]
                    or report["checks"]["matched_numeric_contract"]
                    or any(report["rankings"].values())):
                raise ValueError("retention SDK cohort lacks completed, separately identified policies")
            if implementation() != frozen:
                raise ValueError("installed implementation changed within cohort")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--context", choices=tuple(PROMPTS), action="append")
    parser.add_argument("--journal-rows", type=int, default=24)
    args = parser.parse_args()
    run(args.output_directory.resolve(), contexts=args.context, journal_rows=args.journal_rows)
