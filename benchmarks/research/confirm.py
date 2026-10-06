"""Remeasure SDK-selected finalists together through the ordinary benchmark CLI."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from pllm import BenchmarkResult, Experiment
from pllm.search import ParetoFrontier, SearchCandidate, evaluate_search
from benchmarks.research.artifacts import relocate_public_artifacts


ROOT = Path(__file__).resolve().parents[2]


def factory_source(candidate):
    source = candidate.python_source().replace("\ndef experiment():\n", "\ndef _recorded_experiment():\n", 1)
    return source + (
        "\n\n# An explicit artifact directory changes local paths and configuration identity.\n"
        "def experiment():\n"
        "    from benchmarks.research.artifacts import relocate_public_artifacts\n"
        "    return relocate_public_artifacts(_recorded_experiment())\n"
    )


def selected_candidates(document):
    """Use SDK identity/cohort validation and Pareto selection within each search."""
    if document.get("schema_version") != "pllm.search_benchmark.v1":
        raise ValueError("expected an executed SDK search report")
    candidates, results = [], {}
    for index, row in enumerate(document["search"]["evaluations"]):
        result = BenchmarkResult.from_dict(row["result"])
        # Profile labels are intentionally outside canonical Experiment specs.
        experiment = Experiment.from_spec(row["experiment"]).with_params(
            pipeline__profile=result.to_dict()["profile"])
        candidate = SearchCandidate(row["trial_id"], index, experiment,
                                    row["parameters"], experiment.configuration_digest())
        candidates.append(candidate)
        results[candidate.configuration_digest] = result
    evaluations = evaluate_search(candidates, lambda candidate: results[candidate.configuration_digest])
    scopes = [("all", evaluations)]
    remote_body = tuple(value for value in evaluations
                        if "placement" not in value.candidate.experiment.pipeline.components
                        and len(value.candidate.experiment.resolve().role_graph.roles) > 1)
    if remote_body and len(remote_body) != len(evaluations):
        scopes.append(("remote_body", remote_body))
    selected = {}
    for scope, values in scopes:
        for objective, direction in (("request_tps", "max"), ("covered_bytes", "min"), ("online_bytes", "min")):
            measured = tuple(value for value in values if any(
                metric["id"] == objective and metric["origin"] == "external_measured"
                for metric in value.result.to_dict()["metrics"]))
            for winner in ParetoFrontier(measured, objectives={objective: direction}).results():
                digest = winner.candidate.configuration_digest
                if digest not in selected:
                    selected[digest] = {"candidate": winner.candidate, "selected_for": []}
                selected[digest]["selected_for"].append({"scope": scope, "objective": objective})
                # Preserve the first observed configuration on exact metric ties.
                break
    if not selected:
        raise ValueError("search has no measured finalist")
    return tuple(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--search", action="append", required=True, metavar="LABEL=REPORT")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true")
    mode.add_argument("--recompare", action="store_true",
                      help="validate saved measurements and write a separate cross-kernel comparison")
    args = parser.parse_args()
    directory = ROOT / "benchmarks/research/finalists"
    directory.mkdir(parents=True, exist_ok=True)
    controls = ["benchmarks.research.baseline:experiment", "benchmarks.research.sota:experiment",
                "benchmarks.research.public_prefix:experiment"]
    selected, targets, labels = [], list(controls), set()
    for selection in args.search:
        label, separator, filename = selection.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", label) or label in labels:
            raise ValueError("each search needs a unique Python-safe label and report path")
        labels.add(label)
        path = Path(filename)
        raw = path.read_bytes()
        for item in selected_candidates(json.loads(raw)):
            original = item["candidate"]
            name = f"searched-{label}-{original.configuration_digest[:12]}"
            # Keep saved source immutable; its factory applies explicit path relocation at call time.
            experiment = original.experiment.with_params(name=name)
            candidate = SearchCandidate(name, 0, experiment, {}, experiment.configuration_digest())
            source = factory_source(candidate)
            module = name.replace("-", "_")
            factory = directory / f"{module}.py"
            if factory.exists() and factory.read_text() != source:
                raise ValueError("saved finalist factory has changed")
            factory.write_text(source)
            targets.append(f"benchmarks.research.finalists.{module}:experiment")
            selected.append({
                "name": name, "file": str(factory.relative_to(ROOT)),
                "file_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "configuration_digest": relocate_public_artifacts(experiment).configuration_digest(),
                "search_configuration_digest": original.configuration_digest,
                "search_report": filename, "search_report_sha256": hashlib.sha256(raw).hexdigest(),
                "selected_for": item["selected_for"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".selection.json").write_text(json.dumps({
        "schema_version": "pllm.research_finalists.v1",
        "baseline": controls[0], "controls": controls, "finalists": selected,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selected, indent=2), flush=True)
    if args.export_only:
        return
    if args.recompare:
        from pllm.runtime.benchmark_cli import build_comparison_report

        original = args.output.read_bytes()
        measured = json.loads(original)
        by_name = {row["name"]: row for row in measured["candidates"]}
        pairs = []
        for target in targets:
            module, attribute = target.split(":")
            experiment = getattr(importlib.import_module(module), attribute)()
            row = by_name.pop(experiment.name)
            if (row["configuration_digest"] != experiment.configuration_digest()
                    or row["pipeline"] != experiment.pipeline.to_spec()):
                raise ValueError("saved finalist no longer matches its Python configuration")
            pairs.append((experiment, row["report"]))
        if by_name:
            raise ValueError("saved report contains unselected candidates")
        comparison = build_comparison_report(pairs, compare_kernels=True)
        if not comparison["checks"]["passed"]:
            raise ValueError(f"saved comparison rejected: {comparison['checks']}")
        comparison["analysis_of"] = {"file": str(args.output),
                                     "sha256": hashlib.sha256(original).hexdigest()}
        destination = args.output.with_suffix(".comparison.json")
        destination.write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n")
        print(f"Validated saved measurements: {destination}")
        return
    command = [sys.executable, "-m", "pllm", "benchmark", "run", "--factory", "--trust-python"]
    for target in targets:
        command.extend(["--experiment", target])
    command.extend([
        "--prompt-file", str(Path(os.environ["PLLM_RESEARCH_ARTIFACTS"]) / "prompt.txt"),
        "--max-output-tokens", "8", "--warmups", "0", "--repetitions", "1",
        "--temperature", "0", "--capture-output-digest", "--compare-kernels", "--backend", "native",
        "--timeout", "900", "--output", str(args.output),
    ])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
