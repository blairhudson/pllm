#!/usr/bin/env python3
"""Inventory public SDK contracts, benchmark source references, and scoped evidence.

Static source references are navigational links, never execution evidence.
Unknown historical report formats remain linked-only. No model values, private
fixtures, provider factories or benchmark entry points are executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pllm.components import list_component_classes, list_components
from pllm.evidence import EvidenceRegistry
from pllm.metrics import (
    BenchmarkRequirement, BenchmarkRun, BenchmarkSuite, BuildRef,
    benchmark_capabilities, benchmark_coverage,
)
from pllm.metrics.coverage import _components
from scripts.generate_developer_reference import public_exports


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def read(path):
    with path.open("rb") as stream:
        payload = stream.read((256 << 20) + 1)
    if len(payload) > 256 << 20:
        raise ValueError(f"coverage input exceeds 256 MiB: {path.name}")
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def signature(value):
    try:
        # Strip unstable repr addresses while preserving public arguments and
        # annotation/return contracts. The docs use these same public objects.
        text = str(inspect.signature(value)).replace(sys.executable, "<python>")
        return re.sub(r"0x[0-9a-fA-F]+", "0x…", text)
    except (ValueError, TypeError):
        return None


def surface():
    grouped = {}
    for export in public_exports():
        value = export["value"]
        alias = f'{export["module"]}.{export["name"]}'
        canonical = f"{value.__module__}.{value.__qualname__}" if callable(value) else alias
        group = grouped.setdefault(canonical, {"aliases": [], "value": value})
        group["aliases"].append(alias)
    for name, group in grouped.items():
        value = group["value"]
        members = {}
        attributes = {}
        if inspect.isclass(value):
            for member in dir(value):
                if not member.startswith("_"):
                    raw = inspect.getattr_static(value, member, None)
                    if isinstance(raw, property):
                        attributes[member] = {"kind": "property", "signature": signature(raw.fget)}
                    elif callable(resolved := getattr(value, member, None)):
                        members[member] = signature(resolved)
                    else:
                        attributes[member] = {"kind": f"{type(raw).__module__}.{type(raw).__qualname__}"}
        group["contract"] = {"aliases": sorted(group["aliases"]), "signature": signature(value),
                             "members": members, "attributes": attributes}
    return grouped


def source_references(root, aliases):
    """Resolve explicit Python import aliases and attribute references statically."""
    references = defaultdict(set)
    files = sorted(set(root.glob("benchmarks/**/*.py")) | set(root.glob("scripts/benchmark*.py"))
                   | set(root.glob("examples/benchmarks/*.py")))
    for path in files:
        if path.name == "benchmark_coverage.py":
            continue
        tree = ast.parse(path.read_text())
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for item in node.names:
                    names[item.asname or item.name] = f"{node.module}.{item.name}"
            elif isinstance(node, ast.Import):
                for item in node.names:
                    names[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        def resolve(node):
            if isinstance(node, ast.Name):
                return names.get(node.id)
            if isinstance(node, ast.Attribute):
                parent = resolve(node.value)
                return f"{parent}.{node.attr}" if parent else None
            return None
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = resolve(node)
                if name in aliases:
                    references[aliases[name]].add(f"{path.relative_to(root)}:{node.lineno}")
    return {key: sorted(value) for key, value in references.items()}


def external_decode(report, file_hash, build_digest):
    from scripts.summarize_decode_frontier_sdk import summarize

    summary = summarize(report, file_hash)
    return {"status": "validated_historical" if build_digest is None else
            "validated_current" if summary["build"]["expected_digest"] == build_digest else "stale_build",
        "kind": "response", "build_digest": summary["build"]["expected_digest"],
        "cohort": summary["cohort_commitment"], "source_lock_digest": summary["source_lock_digest"],
        "configurations": {key: {"target": {"kind": "response", "parameters": {"model": {"experiment": value}}},
            "client_samples": summary["groups"][key]["client_samples"],
            "request_samples": summary["groups"][key]["request_samples"]}
            for key, value in summary["experiments"].items()},
        "scope": summary["evidence_class"], "limits": summary["limits"]}


def external_private_boundary(report, file_hash, build_digest, control, control_hash):
    from scripts.summarize_private_boundary_sdk import summarize

    summary = summarize(report, file_hash, control, control_hash)
    observed = summary["build"]["expected_digest"]
    return {"status": "validated_historical" if build_digest is None else
            "validated_current" if observed == build_digest else "stale_build",
        "kind": "response", "build_digest": observed,
        "cohort": summary["cohort_commitment"], "source_lock_digest": summary["source_lock_digest"],
        "configurations": {key: {"target": {"kind": "response", "parameters": {
            "model": {"experiment": value["experiment"]}}},
            "client_samples": value["sample_count"], "request_samples": value["sample_count"]}
            for key, value in summary["groups"].items()},
        "scope": summary["evidence_class"], "limits": summary["limits"]}


def inventory(root=ROOT, *, extra_evidence=(), build_digest=None):
    api = surface()
    aliases = {alias: name for name, item in api.items() for alias in item["aliases"]}
    references = source_references(root, aliases)
    catalog, _ = read(root / "benchmarks/coverage.json")
    if catalog.get("schema") != "pllm.sdk_benchmark_catalog.v1":
        raise ValueError("invalid SDK benchmark catalog")
    requirements = benchmark_capabilities() + tuple(BenchmarkRequirement.from_spec(r) for r in catalog["requirements"])
    registry, suites, sources = EvidenceRegistry(), [], []
    paths = sorted(set(root.glob("docs/evidence/*.json")) | {Path(p).resolve() for p in extra_evidence})
    for path in paths:
        document, file_hash = read(path)
        if not isinstance(document, dict):
            continue
        schema = document.get("schema", document.get("schema_version"))
        configs = list(_components(document))
        if schema not in {"pllm.benchmark_run.v1", "pllm.private_boundary_sdk_benchmark.v1"} and not configs:
            continue
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        source = {"path": label, "sha256": file_hash, "schema": schema, "status": "linked_only",
                  "component_ids": sorted({c["component"] for c in configs}),
                  "configurations": list({sha(c): c for c in configs}.values())}
        try:
            if schema == "pllm.benchmark_run.v1":
                run = BenchmarkRun(document)
                registry = registry.register_all(run.results)
                suites.append(BenchmarkSuite.from_spec(document["suite"]))
                source.update(status="typed_evidence", complete=document["complete"],
                              result_ids=[r.id for r in run.results])
            elif schema in {"pllm.growing_epochs_sdk_benchmark.v1", "pllm.decode_cache_sdk_benchmark.v1"}:
                source.update(external_decode(document, file_hash, build_digest))
            elif schema == "pllm.private_boundary_sdk_benchmark.v1":
                control, control_hash = read(root / "docs/evidence/private-boundary-qwen-control.json")
                source.update(external_private_boundary(document, file_hash, build_digest, control, control_hash))
                source["component_ids"] = sorted({component["component"] for item in source["configurations"].values()
                    for component in _components(item["target"])})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            # Preserve the failed import as a visible evidence-adapter gap.
            source.update(status="validation_failed", error_type=type(exc).__name__)
        sources.append(source)
    # Repeated paths/suites are aliases, not additional evidence.
    suites = tuple({s.digest: s for s in suites}.values())
    report = benchmark_coverage(suites=suites, registry=registry, requirements=requirements,
                                build_digest=build_digest).to_dict()
    rows = {r["requirement"]["name"]: r for r in report["requirements"]}
    classes = {f"{c.__module__}.{c.__qualname__}": c.describe().component for c in list_component_classes()}
    for requirement in requirements:
        row = rows[requirement.name]
        row["external_evidence"] = []
        row["linked_sources"] = [{"path": source["path"], "sha256": source["sha256"],
            "schema": source["schema"], "status": source["status"]} for source in sources
            if source["status"] in {"linked_only", "validation_failed"}
            and row["requirement"]["component"] in source["component_ids"]]
        for source in sources:
            if source["status"] in {"validated_historical", "validated_current", "stale_build"}:
                for name, item in source["configurations"].items():
                    if requirement.matches(item["target"]):
                        row["external_evidence"].append({"path": source["path"], "configuration": name,
                            "status": source["status"], "kind": source["kind"], "build_digest": source["build_digest"]})
        row["typed_status"] = row["status"]
        external_states = {e["status"] for e in row["external_evidence"]}
        if external_states & {"validated_historical", "validated_current"}:
            row["has_measurement"] = True
            if row["status"] not in {"measured", "partial"}:
                row["status"] = "partial" if row["unmeasured_cases"] else "measured_external"
        elif row["status"] not in {"measured", "partial"} and "stale_build" in external_states:
            row["status"] = "stale_build"
    report["counts"] = dict(sorted(Counter(row["status"] for row in rows.values()).items()))
    apis = []
    for name, item in sorted(api.items()):
        value = item["value"]
        requirement_ids = catalog["api_bindings"].get(name, [])
        if name in classes:
            requirement_ids = [classes[name], *requirement_ids]
        unknown = set(requirement_ids) - set(rows)
        if unknown:
            raise ValueError(f"unknown benchmark requirements for {name}: {sorted(unknown)}")
        planned = getattr(value, "planned_identity", None)
        declaration = not callable(value) or inspect.isclass(value) and issubclass(value, Exception)
        status = ("unsupported" if planned in rows else "declaration" if declaration else
                  "resolved" if requirement_ids else "source_reference_only" if references.get(name) else "unmapped")
        apis.append({"name": name, "aliases": sorted(item["aliases"]), "status": status,
            "requirement_ids": requirement_ids, "source_references": references.get(name, []),
            "planned_requirement": planned if planned in rows else None,
            "reason": "public constant or exception declaration" if declaration else None})
    unknown_apis = set(catalog["api_bindings"]) - set(api)
    if unknown_apis:
        raise ValueError(f"catalog binds absent public APIs: {sorted(unknown_apis)}")
    review = {"schema": "pllm.sdk_benchmark_surface.v1",
        "api": {name: sha(item["contract"]) for name, item in sorted(api.items())},
        "components": {d.component: sha(d.to_dict()) for d in list_components()},
        "requirements": {r.name: sha(r.to_spec()) for r in requirements},
        "measured_requirements": sorted(r["requirement"]["name"] for r in rows.values() if r["has_measurement"]),
        "coverage_links": {a["name"]: sha({"status": a["status"], "requirements": a["requirement_ids"],
            "source_files": sorted({r.rsplit(":", 1)[0] for r in a["source_references"]})}) for a in apis}}
    return {"schema": "pllm.sdk_benchmark_inventory.v1", "coverage": report,
        "public_api": apis, "api_counts": dict(sorted(Counter(a["status"] for a in apis).items())),
        "evidence_sources": sources, "surface": review,
        "limitations": ["Source references link recipes; they do not prove executed API/method coverage.",
            "Unadapted historical reports are linked-only, never current-build measurements.",
            "Named interactions are explicit; individual option coverage does not cover their product."]}


def review_changes(previous, current):
    changes = {kind: sorted(k for k in set(previous.get(kind, {})) | set(current[kind])
        if previous.get(kind, {}).get(k) != current[kind].get(k))
        for kind in ("api", "components", "requirements", "coverage_links")}
    changes["measurement_losses"] = sorted(set(previous.get("measured_requirements", ()))
                                          - set(current["measured_requirements"]))
    if previous.get("schema") != current["schema"]:
        changes["schema"] = [current["schema"]]
    return {kind: keys for kind, keys in changes.items() if keys}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--current-build", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail on unreviewed public API/parameter contract changes")
    parser.add_argument("--update-surface", action="store_true", help="explicitly acknowledge the current surface, including gaps")
    args = parser.parse_args(argv)
    if args.check and args.update_surface:
        parser.error("--check and --update-surface are mutually exclusive")
    report = inventory(extra_evidence=args.evidence,
                       build_digest=BuildRef.capture().expected_digest if args.current_build else None)
    lock = ROOT / "benchmarks/coverage-surface.json"
    if args.update_surface:
        lock.write_text(json.dumps(report["surface"], indent=2, sort_keys=True) + "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"api": report["api_counts"], "capabilities": report["coverage"]["counts"],
        "evidence_sources": dict(Counter(s["status"] for s in report["evidence_sources"])),
        "unmeasured_cases": sum(not c["measured"] for c in report["coverage"]["case_coverage"])}, sort_keys=True))
    if args.check:
        previous, _ = read(lock)
        changed = review_changes(previous, report["surface"])
        if changed:
            print(json.dumps({"unreviewed_contracts": changed}, sort_keys=True), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
