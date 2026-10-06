"""Reanalyse declared paper-study groups without altering original measurements."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from pllm.runtime.benchmark_cli import build_comparison_report


def extract(report_path: Path, targets: list[str]) -> dict:
    raw = report_path.read_bytes()
    original = json.loads(raw)
    if original.get("schema_version") != "pllm.loopback_benchmark_comparison.v1":
        raise ValueError("require one canonical multi-Experiment report with its original cohort salt")
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError("select at least two distinct Python factories")
    selected = {}
    for target in targets:
        module, function = target.split(":", 1)
        experiment = getattr(importlib.import_module(module), function)()
        if experiment.name in selected:
            raise ValueError("duplicate Experiment name")
        selected[experiment.name] = experiment
    pairs = []
    records = []
    for candidate in original["candidates"]:
        experiment = selected.get(candidate["name"])
        if experiment is None:
            continue
        if (candidate["configuration_digest"] != experiment.configuration_digest()
                or candidate["pipeline"] != experiment.pipeline.to_spec()
                or not candidate["report"]["checks"]["passed"]):
            raise ValueError("factory or runtime checks differ from the recorded observation")
        pairs.append((experiment, candidate["report"]))
        records.append(candidate)
    if len(pairs) != len(selected):
        raise ValueError("selected observation is missing or duplicated")
    result = build_comparison_report(pairs, compare_kernels=True)
    if not result["checks"]["passed"]:
        raise ValueError("selected cohort fails ordinary SDK comparison policy")
    if result["candidates"] != records:
        raise ValueError("reanalysis changed original measurements")
    result["analysis_of"] = {
        "file": report_path.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
        "candidates": [candidate["name"] for candidate in records],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--experiment", action="append", required=True,
                        help="trusted repository module:factory; repeat within one verifier/numeric contract")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new reproduction report path")
    result = extract(args.report, args.experiment)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Preserved {len(result['candidates'])} observations; matched cohort written to {args.output}")


if __name__ == "__main__":
    main()
