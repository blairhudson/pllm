"""Serial exact-context branch/restart probe; full attention, not SDK serving.

Controls: immutable-block resident continuation and existing tiered full manifests
sharing the same prefix pages. Candidate changes only manifest publication.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import sys
import tempfile
import time

# Use assigned source Python runtime with read-only installed native extension.
# No wheel rebuild or modification of shared environment is needed.
import pllm.runtime
pllm.runtime.__path__.insert(0, str(Path(__file__).resolve().parents[1] / "python/pllm/runtime"))


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def freeze_sources():
    """Freeze without importing model modules or loading checkpoint values."""
    root = Path(__file__).resolve().parents[1]
    installed = Path(sys.modules["pllm"].__file__).resolve().parent
    paths = {Path(__file__).resolve(), root / "scripts/decoder_probe_support.py",
             root / "scripts/frontier_probe_support.py"}
    paths.update((root / "python/pllm/runtime").rglob("*.py"))
    # Include installed Python and native files: deferred imports must also come
    # from bytes frozen BEFORE execution, not hashes discovered afterward.
    paths.update(p for p in installed.rglob("*") if p.is_file() and p.suffix in {".py", ".so", ".dylib"})
    return {str(p.resolve()): file_hash(p) for p in sorted(paths)}


def verify_sources(frozen):
    for path, expected in frozen.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"probe source changed during cohort: {path}")


def imported_sources(frozen):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "pllm" or name.startswith("pllm.")
                or name in {"decoder_probe_support", "frontier_probe_support"}):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = str(Path(path).resolve())
        if path not in frozen or file_hash(path) != frozen[path]:
            raise RuntimeError(f"imported module absent from frozen source set: {name}")
        result[name] = {"path": path, "sha256": frozen[path]}
    return result


def storage_order(token_index, repeat):
    return ("tiered", "graph") if (token_index + repeat) % 2 == 0 else ("graph", "tiered")


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def disk(root):
    paths = [p for p in Path(root).iterdir() if p.name != ".writer"]
    return {"logical_file_bytes": sum(p.stat().st_size for p in paths),
        "physical_allocated_bytes": sum(p.stat().st_blocks * 512 for p in paths),
        "files": len(paths)}


def storage_counters(owner):
    return {"read_bytes": owner.read_bytes, "written_bytes": owner.written_bytes,
        "read_and_authentication_cpu_seconds": owner.read_cpu_seconds,
        "write_and_authentication_cpu_seconds": owner.write_cpu_seconds}


def counter_delta(before, after):
    return {name: value - before[name] for name, value in after.items()}


def erase(snapshot):
    for cache in snapshot.caches:
        cache.key.fill(0)
        cache.value.fill(0)


def kv_hash(runtime):
    import numpy as np
    owner = runtime._tiered
    result = hashlib.sha256()
    if owner is not None:
        owner.begin(runtime, np.array([0], np.int64), False)
    for layer in range(runtime.layers):
        if owner is not None:
            owner.load(runtime, layer)
        cache = runtime.caches[layer]
        result.update(cache.key[:runtime.position].tobytes())
        result.update(cache.value[:runtime.position].tobytes())
        if owner is not None:
            owner._retire(runtime)
    return result.hexdigest()


def worker(spec):
    import numpy as np
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.context_graph import context_graph_runtime
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.prefill_cache import ExactPrefillCache
    from pllm.runtime.tiered_execution import TieredExecution
    from decoder_probe_support import DecoderFixture

    class MeteredTiered(TieredExecution):
        def __init__(self, *args, **kwargs):
            self.read_cpu_seconds = self.write_cpu_seconds = 0.0
            self.manifest_read_bytes = self.manifest_written_bytes = 0
            self.logical_manifest_written_bytes = 0
            super().__init__(*args, **kwargs)

        def _read(self, record, metadata, maximum):
            start = time.process_time()
            try:
                raw = super()._read(record, metadata, maximum)
                if "manifest" in metadata:
                    self.manifest_read_bytes += record["bytes"]
                return raw
            finally:
                self.read_cpu_seconds += time.process_time() - start

        def _write(self, raw, metadata):
            start = time.process_time()
            try:
                record = super()._write(raw, metadata)
                if "manifest" in metadata:
                    self.manifest_written_bytes += record["bytes"]
                    self.logical_manifest_written_bytes += len(raw)
                return record
            finally:
                self.write_cpu_seconds += time.process_time() - start

    phases = {}
    def phase(name, call):
        wall, cpu = time.perf_counter(), time.process_time()
        value = call()
        phases[name] = {"wall_seconds": time.perf_counter() - wall,
                        "cpu_seconds": time.process_time() - cpu, "memory": process_memory()}
        return value

    frozen = spec["frozen_sources"]
    def audit_before():
        verify_sources(frozen)
        return imported_sources(frozen)
    initial_imports = phase("source_integrity_audit_before", audit_before)
    parity_cpu = 0.0
    def inspect(name, runtime):
        nonlocal parity_cpu
        # Cumulative cold CPU at this boundary includes startup/loading and all
        # preceding lifecycle work, but excludes earlier parity/audit scans.
        audits = sum(p["cpu_seconds"] for n, p in phases.items() if n.startswith("source_integrity_audit"))
        result.setdefault("pre_parity_totals", []).append({"inspection": name,
            "cold_process_cpu_seconds": time.process_time(),
            "cold_nonparity_nonaudit_cpu_seconds": time.process_time() - parity_cpu - audits,
            "memory": process_memory()})
        value = phase(name, lambda: kv_hash(runtime))
        parity_cpu += phases[name]["cpu_seconds"]
        return value

    def load():
        fixture = DecoderFixture(inputs=spec["tokens"], outputs=spec["steps"] + 4)
        fixture.composition = fixture.composition.with_params(
            quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
        fixture.compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=fixture.composition)
        return fixture
    fixture = phase("source_and_initial_construction", load)
    mode, kind = spec["mode"], spec["kind"]
    ids = [(i * 31 + 11) % 10000 for i in range(spec["tokens"])]
    result = {"kind": kind, "mode": mode, "source": fixture.lock(), "phases": phases,
              "imported_sources_before_model_load": initial_imports}
    if kind == "resident":
        runtime = phase("owner_creation", lambda: fixture.compiled.runtime(fixture.remote))
        _, logits, _ = phase("initial_prefill", lambda: runtime.prepare_ids(ids))
        cache = ExactPrefillCache(256 << 20, causal_reduction="prefix_f32")
        def publish():
            snapshot = runtime.snapshot()
            try:
                if not cache.put("prefix", snapshot, logits):
                    raise RuntimeError("resident prefix exceeds selected cache bound")
            finally:
                erase(snapshot)
        phase("immutable_block_publication", publish)
        result["resident_cache_bytes"] = cache.size_bytes
        result["branches"] = []
        for branch in range(spec["branches"]):
            runtime.reset()
            def restore():
                saved, scores = cache.get("prefix", position=len(ids), layers=runtime.layers)
                runtime.restore(saved)
                erase(saved)
                return scores
            scores = phase(f"branch_{branch}_restore", restore)
            hashes = [digest(scores)]
            def execute():
                nonlocal scores
                for step in range(spec["steps"] + 1):
                    token = (int(np.argmax(scores)) + (branch if step == 0 else 0)) % len(scores)
                    scores = runtime.forward_ids([token])[-1]
                    hashes.append(digest(scores))
            phase(f"branch_{branch}_full_attention", execute)
            result["branches"].append({"logit_hashes": hashes,
                                       "kv_hash": inspect(f"branch_{branch}_parity_inspection", runtime),
                                       "position": runtime.position})
        cache.clear()
        runtime.reset()
    else:
        def construct():
            if kind == "graph":
                return context_graph_runtime(fixture.compiled, fixture.remote, spec["root"],
                    bytes.fromhex(spec["key"]), head=spec.get("head"), page_rows=64)
            runtime = fixture.compiled.runtime(fixture.remote)
            runtime._tiered = MeteredTiered(fixture.compiled, runtime, spec["root"],
                bytes.fromhex(spec["key"]), spec.get("head"), page_rows=64)
            return runtime
        runtime = phase("authenticated_restore_or_owner_creation", construct)
        owner = runtime._tiered
        result["restore_or_creation_storage"] = storage_counters(owner)
        result["disk_before"] = disk(owner.root)
        before_prefill = storage_counters(owner)
        if mode == "issue":
            _, scores, _ = phase("initial_prefill_and_publication", lambda: runtime.prepare_ids(ids))
        else:
            scores = owner.logits.copy()
        result["initial_prefill_publication_storage"] = counter_delta(before_prefill, storage_counters(owner))
        hashes = [digest(scores)]
        prefix_names = {p[k]["name"] for rows in owner.records for p in rows for k in ("key", "value")}
        def execute():
            nonlocal scores
            for step in range(spec.get("execute_steps", 0)):
                token = (int(np.argmax(scores)) + (spec.get("branch", 0) if step == 0 else 0)) % len(scores)
                scores = runtime.forward_ids([token])[-1]
                hashes.append(digest(scores))
        before_attention = storage_counters(owner)
        phase("restored_full_attention_and_publication", execute)
        result["restored_full_attention_publication_storage"] = counter_delta(before_attention, storage_counters(owner))
        names = {p[k]["name"] for rows in owner.records for p in rows for k in ("key", "value")}
        result.update(logit_hashes=hashes, position=runtime.position, head=owner.head,
            read_bytes=owner.read_bytes, written_bytes=owner.written_bytes,
            manifest_read_bytes=owner.manifest_read_bytes, manifest_written_bytes=owner.manifest_written_bytes,
            logical_full_manifest_written_bytes=owner.logical_manifest_written_bytes,
            read_and_authentication_cpu_seconds=owner.read_cpu_seconds,
            write_and_authentication_cpu_seconds=owner.write_cpu_seconds,
            disk_after=disk(owner.root), shared_prefix_pages=len(prefix_names & names),
            final_page_count=len(names), peak_live_kv_bytes=owner.peak_live_kv_bytes,
            branch_logical_kv_bytes=runtime.position * runtime.layers * np.prod(owner.shape).item() * 8)
        # Parity inspection is outside measurement; its extra authenticated reads
        # are reported separately, never confused with serving attention reads.
        reads = owner.read_bytes
        result["kv_hash"] = inspect("parity_inspection", runtime)
        result["parity_inspection_read_bytes"] = owner.read_bytes - reads
        runtime.reset()
    result["parity_inspection_cpu_seconds"] = parity_cpu
    audits = sum(p["cpu_seconds"] for n, p in phases.items() if n.startswith("source_integrity_audit"))
    result["cold_nonparity_nonaudit_lifecycle_cpu_seconds"] = time.process_time() - parity_cpu - audits
    def audit_after():
        verify_sources(frozen)
        return imported_sources(frozen)
    result["imported_sources"] = phase("source_integrity_audit_after", audit_after)
    result["cold_process_cpu_including_parity_and_audits_seconds"] = time.process_time()
    result["memory_after_parity"] = process_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512], choices=[128, 512])
    parser.add_argument("--processes", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=18, choices=range(2, 25))
    parser.add_argument("--branches", type=int, default=2, choices=[2, 3, 4])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("new --output required")
    from frontier_probe_support import fresh_workers
    frozen = freeze_sources()
    verify_sources(frozen)
    parent_imports = imported_sources(frozen)
    samples = []
    with fresh_workers(__file__, memory_bytes=4 << 30, timeout=600) as (raw_run, guard, host):
        def run(payload):
            verify_sources(frozen)
            value = raw_run(payload | {"frozen_sources": frozen})
            verify_sources(frozen)
            return value
        for token_index, tokens in enumerate(args.tokens):
            for repeat in range(args.processes):
                base = {"tokens": tokens, "steps": args.steps, "branches": args.branches}
                resident = run(base | {"kind": "resident", "mode": "resident"})
                order = storage_order(token_index, repeat)
                cohort = {"resident": resident, "tokens": tokens, "repeat": repeat, "storage_order": order}
                with tempfile.TemporaryDirectory(prefix="pllm-context-graph-") as directory:
                    for kind in order:
                        common = base | {"kind": kind, "root": directory + "/" + kind,
                                         "key": secrets.token_bytes(32).hex()}
                        issuer = run(common | {"mode": "issue"})
                        head = issuer.pop("head")
                        branches = []
                        for branch in range(args.branches):
                            continued = run(common | {"mode": "branch", "head": head,
                                "branch": branch, "execute_steps": args.steps})
                            branch_head = continued.pop("head")
                            restarted = run(common | {"mode": "restart", "head": branch_head,
                                                      "execute_steps": 1})
                            restarted.pop("head")
                            control = resident["branches"][branch]
                            if (continued["logit_hashes"] + restarted["logit_hashes"][1:] != control["logit_hashes"]
                                    or restarted["kv_hash"] != control["kv_hash"]
                                    or restarted["position"] != control["position"]
                                    or any(x["source"] != resident["source"] for x in (issuer, continued, restarted))):
                                raise AssertionError("context graph/control decoder parity differs")
                            branches.append({"continued": continued, "restarted": restarted})
                        cohort[kind] = {"issuer": issuer, "branches": branches}
                samples.append(cohort)
    verify_sources(frozen)
    report = {"schema": "pllm.context_graph_probe.v1", "scope": __doc__, "samples": samples,
        "host": asdict(host), "build": raw_run.build, "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        "source_hashes_frozen_before_workers": frozen, "parent_imported_sources": parent_imports,
        "source_hashes_verified_before_and_after_every_worker": True,
        "exact_parity": True, "serving_metrics": None,
        "limitations": ["Local-clear decoder, not SDK/provider traffic or whole-response metrics.",
            "Read/write CPU combines file I/O, hashes and AEAD; auth-only CPU not separately isolated.",
            "Cold inclusive CPU contains parity/integrity audits; use explicitly separated lifecycle/phase CPU.",
            "Storage/authentication CPU is a nested subset of prefill/attention phase CPU, not an additive cost.",
            "Initial publication charged. All accepted heads/orphans remain owned; no unsafe GC.",
            "Resident cache fits this cohort. Histories exceeding resident cache not measured."]}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps({"samples": len(samples), "exact_parity": True}))


if __name__ == "__main__":
    main()
