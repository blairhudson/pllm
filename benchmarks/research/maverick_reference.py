"""Replay source-bound RAA stage probes; not an executable Experiment factory."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

from benchmarks.research.paper_baseline import experiment
from pllm import lower_model
from pllm.model_loader import resolve_model
from pllm.runtime.coded_resources import coded_delegation_storage_lower_bound
from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = {
    "paper": "maverick",
    "source_url": "https://arxiv.org/abs/2609.10264v1",
    "paper_sha256": "c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0",
    "source_sections": ["5.2", "6 / Protocol 3", "7.1", "Appendices A and B"],
    "field_modulus": 2_013_265_921,
    "code_rate_denominator": 8,
    "public_code_seed_byte": 13,
    "privacy_parameter_heuristic_bits": 128,
    "verification_sparsity": 41,
    "verification_repetitions": 2,
    "max_issued_queries": 65_536,
    "max_live_queries": 8,
    "reference_payload_bound_bytes": 256 << 20,
    "samples_per_admitted_shape": 8,
    "scope": "Qwen2.5 compiler-derived stage shapes, synthetic public weights and i8 inputs; single-thread native in-process reference",
}


def run():
    control = experiment()
    resolved = resolve_model(control.pipeline.model)
    config_path = resolved.path / "config.json"
    config_bytes = config_path.read_bytes()
    plan = lower_model(json.loads(config_bytes), batch=1, max_input_tokens=64, max_new_tokens=8)
    storage = coded_delegation_storage_lower_bound(
        plan, control.pipeline, resident_budget_bytes=2 << 30, cached_budget_bytes=6 << 30,
    )
    shapes = Counter((stage["output_width"], stage["input_width"]) for stage in storage["stage_dimensions"])
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-core", "--example", "coded_delegation_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"]) / "release/examples/coded_delegation_probe"
    host = host_memory()
    if host.available - host.reserve < GiB:
        raise RuntimeError("RAA reference requires 1 GiB safe RAM beyond the host reserve")
    active = []
    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    cases = []
    guard.start()
    try:
        for (rows, columns), count in sorted(shapes.items()):
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            child = subprocess.Popen([str(binary), str(rows), str(columns), str(CONFIGURATION["samples_per_admitted_shape"])],
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            active.append(child)
            try:
                stdout, stderr = child.communicate(timeout=120)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                active.remove(child)
            guard.check()
            if child.returncode or guard.error:
                raise RuntimeError(guard.error or stderr)
            case = json.loads(stdout)
            if (case["rows"], case["columns"]) != (rows, columns) or case["executable_sdk"] is not False:
                raise ValueError("native probe geometry or scope changed")
            case["compiled_stage_count"] = count
            if case["status"] == "reference_checks_passed":
                case["median_seconds"] = {key: statistics.median(row[key] for row in case["observations"])
                                          for key in case["observations"][0]}
            cases.append(case)
    finally:
        guard.close()
    passed_stages = sum(case["compiled_stage_count"] for case in cases if case["status"] == "reference_checks_passed")
    return {
        "schema": "pllm.raa_delegation_reference.v1",
        "configuration": CONFIGURATION,
        "model": control.pipeline.model.to_spec(),
        "source_lock_digest": resolved.source_lock_digest,
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "storage_lower_bound": storage,
        "cases": cases,
        "stages_with_admitted_reference_shape": passed_stages,
        "stages_rejected_before_preprocessing": storage["remote_stages"] - passed_stages,
        "executable_sdk": False,
        "privacy_certified": False,
        "verification_soundness_certified": False,
        "memory_guard": {"host": asdict(host), "required_headroom_bytes": GiB,
                         "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "source_gate": {
            "sampling": "uniform fixed-weight support without replacement; independent uniform nonzero coefficients from OS randomness",
            "code_construction": "public SHA256-seeded permutations and nonzero scales; separate role-domain seeds",
            "query_lifecycle": "non-cloneable Rust ownership; consume on finish including malformed/wrong-binding claims; erase correction on cancellation",
            "conditional_distance_target": 0.5,
            "distance_failure_probability": None,
            "appendix_b_cited_field_lower_bound_exclusive": 2**31,
            "babybear_meets_cited_field_range": False,
            "privacy_parameter_rule": "paper Eq. (1), a conditional linear-test heuristic, not a general security estimate",
        },
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "source_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in [
            "benchmarks/research/maverick_reference.py", "benchmarks/research/paper_baseline.py",
            "benchmarks/research/common.py", "python/pllm/runtime/coded_resources.py",
            "crates/pllm-core/src/coded_linear.rs", "crates/pllm-core/src/coded_delegation_reference.rs",
            "crates/pllm-core/examples/coded_delegation_probe.rs", "Cargo.lock",
        ]},
        "limitations": [
            "no RAA minimum-distance/failure certificate or independent dual-LPN review for BabyBear",
            "the paper uses verification sparsity 31; this conservative reference keeps 41 and two repetitions",
            "public seeded code generation is an engineering instantiation, not uniform-generator distance evidence",
            "synthetic weights: no checkpoint-value, logit, KV or generation-quality comparison",
            "timings are single-thread elapsed times; build, import, distribution, transport and full responses are excluded",
            "payload accounting is not measured peak RSS; allocator overhead, scratch and OS caches are extra",
            "one-use ownership is in-process only; no authenticated role sessions, wire codecs or batch protocol",
            "conditional coded rejection tests do not establish adversarial soundness",
            "privacy masks must not be reused or sent to the evaluator; fresh samples alone do not prove privacy",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new reproduction path")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"admitted_reference_stages": report["stages_with_admitted_reference_shape"],
                      "rejected_stages": report["stages_rejected_before_preprocessing"],
                      "preprocessing_bytes": report["storage_lower_bound"]["preprocessing_bytes_lower_bound"]}, indent=2))


if __name__ == "__main__":
    main()
