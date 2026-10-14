"""Read-only benchmark coverage command."""
from __future__ import annotations

import json

from .errors import ResolutionError
from .output import emit_machine


def run(args, output_format, dry_run):
    from pllm.evidence import EvidenceRegistry
    from pllm.metrics import (
        BenchmarkRequirement, BenchmarkRun, BenchmarkSuite, BuildRef,
        benchmark_capabilities, benchmark_coverage,
    )

    def load(path):
        with path.open("rb") as stream:
            payload = stream.read((256 << 20) + 1)
        if len(payload) > 256 << 20:
            raise ValueError("coverage input exceeds 256 MiB")
        return json.loads(payload)

    try:
        if len(args.evidence) + len(args.suite) > 256:
            raise ValueError("coverage accepts at most 256 input documents")
        suites = [BenchmarkSuite.from_spec(load(path)) for path in args.suite]
        registry = EvidenceRegistry()
        for path in args.evidence:
            evidence = BenchmarkRun(load(path))
            registry = registry.register_all(evidence.results)
            suites.append(BenchmarkSuite.from_spec(evidence.to_dict()["suite"]))
        requirements = benchmark_capabilities()
        if args.requirements:
            requirements += tuple(BenchmarkRequirement.from_spec(r) for r in load(args.requirements))
        build = BuildRef.capture().expected_digest if args.current_build else args.build_digest
        report = benchmark_coverage(suites=suites, registry=registry, requirements=requirements,
                                    build_digest=build).to_dict()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ResolutionError("BENCHMARK_COVERAGE", str(exc)) from exc
    required = [row for row in report["requirements"] if row["requirement"]["disposition"] == "required"]
    if args.require == "mapped" and any(r["status"] == "unmapped" for r in required):
        raise ResolutionError("BENCHMARK_COVERAGE_GAP", "required capabilities have no benchmark case; omit --require to inspect the report")
    if args.require == "measured" and any(r["status"] != "measured" or r["unmeasured_cases"] for r in required):
        raise ResolutionError("BENCHMARK_COVERAGE_GAP", "required capabilities have unmeasured cases; omit --require to inspect the report")
    if output_format == "human":
        print("Capability coverage (configuration evidence; no timing ranking)")
        for status, count in report["counts"].items():
            print(f"{status}: {count}")
        for row in report["requirements"]:
            print(f'{row["status"]}\t{row["requirement"]["name"]}')
    else:
        emit_machine("benchmark.coverage", {"report": report, "dry_run": dry_run}, output_format)
