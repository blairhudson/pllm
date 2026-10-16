"""Offline, bounded stage launcher. Real checkpoint stages are parent-operated."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import importlib
import json
import os as os  # Public test hook for filesystem fault injection.
from pathlib import Path
import platform
import re
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.research import checkpoint_compression as c

DEPENDENCIES = (
    "scripts/benchmark_checkpoint_compression.py", "benchmarks/research/checkpoint_compression.py",
    "scripts/train_checkpoint_tiny_model.py", "benchmarks/research/checkpoint_tiny_model.py",
    "scripts/decoder_probe_support.py", "scripts/frontier_probe_support.py",
    c.PRIOR_MANIFEST,
)


def atomic(path, doc, exclusive=False):
    c.publish_json(path, doc, exclusive=exclusive)


class ProvenanceError(ValueError):
    """Evidence from this attempt cannot be reused, even if some items finished."""


def snapshot_inventory(root, choice):
    """Hash whole snapshot, all index references and SDK lock before any engine load."""
    from pllm.model_loader import _source_lock
    root = Path(root)
    paths = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    for name, path in list(paths.items()):
        if name.endswith(".safetensors.index.json"):
            index = json.loads(path.read_text())
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ProvenanceError("invalid shard index")
            for shard in mapping.values():
                if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ProvenanceError("shard index escapes frozen snapshot")
                if shard not in paths or not shard.endswith(".safetensors"):
                    raise ProvenanceError("missing/unfrozen index-referenced shard")
    files = {name: {"sha256": c.file_hash(path), "bytes": path.stat().st_size}
             for name, path in sorted(paths.items())}
    checkpoint, source_lock_digest, source_lock = _source_lock(root, choice)
    parameters = 0
    import math
    for name, path in paths.items():
        if name.endswith(".safetensors"):
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 8 << 20:
                    raise ProvenanceError("unbounded checkpoint metadata")
                header = json.loads(stream.read(size))
            parameters += sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    return {"files": files, "teacher_parameters": parameters, "checkpoint_digest": checkpoint,
            "source_lock_digest": source_lock_digest, "resolved_source_lock": source_lock}


def make_teacher(expected):
    """Guard shared Teacher's resolver return BEFORE its constructor loads engine values.

    Scoped interception is child-local and serial; shared sources are not edited.
    """
    from unittest.mock import patch
    from pllm import model_loader
    from scripts.train_checkpoint_tiny_model import Teacher
    resolve = model_loader.resolve_model
    def checked(choice):
        try:
            resolved = resolve(choice)
            actual = snapshot_inventory(resolved.path, choice)
            if any(actual[key] != expected[key] for key in actual):
                raise ProvenanceError("actual resolved execution inventory/source lock differs before engine load")
            if (resolved.source_lock_digest != expected["source_lock_digest"]
                    or resolved.checkpoint_digest != expected["checkpoint_digest"]
                    or resolved.manifest.metadata.get("source_lock") != expected["resolved_source_lock"]):
                raise ProvenanceError("actual resolved source/checkpoint lock differs before engine load")
            return resolved
        except Exception as error:
            raise ProvenanceError(str(error)) from error
    with patch.object(model_loader, "resolve_model", checked):
        return Teacher(expected)


def environment():
    import pllm
    import importlib.util
    # Bind actual imported SDK, not merely checkout paths; includes installed native binary.
    sdk = Path(pllm.__file__).resolve().parent
    paths = [ROOT / p for p in DEPENDENCIES]
    paths += sorted(p for p in sdk.rglob("*") if p.is_file() and p.suffix in {".py", ".so", ".dylib", ".pyd"})
    # Versions alone do not bind a rebuilt NumPy/BLAS or tokenizer native wheel.
    for package in ("numpy", "tokenizers", "safetensors", "filelock", "psutil", "jinja2", "huggingface_hub"):
        base = Path(importlib.import_module(package).__file__).resolve().parent
        bases = [base, base.parent / (base.name + ".libs")]
        paths += sorted(p for root in bases for p in root.rglob("*") if p.is_file()
                        and p.suffix in {".py", ".so", ".dylib", ".pyd"})
    paths.append(Path(sys.executable).resolve())
    native = importlib.util.find_spec("pllm._native")
    if native is None or native.origin is None:
        raise RuntimeError("installed native dependency unavailable; no build attempted")
    paths.append(Path(native.origin).resolve())
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": sorted([d.metadata["Name"], d.version] for d in importlib.metadata.distributions()),
            "code_native": {str(p): c.file_hash(p) for p in paths},
            "threads": 4, "numeric": "W8A8-prefix_f32-local-clear"}


def metadata(source):
    from huggingface_hub import try_to_load_from_cache
    if not source.get("files") or "config.json" not in source["files"] or not any(n.endswith((".safetensors", ".safetensors.index.json")) for n in source["files"]):
        raise ValueError("complete source spec required")
    if not re.fullmatch("[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("immutable 40-character checkpoint revision required")
    if any(Path(n).name != n for n in source["files"]):
        raise ValueError("source names must be flat cache artifacts")
    path = try_to_load_from_cache(source["model"], "config.json", revision=source["revision"])
    if not isinstance(path, str) or not Path(path).is_file():
        raise RuntimeError("preposition pinned snapshot; downloads forbidden")
    from pllm import Model
    choice = Model.hf(source["model"], revision=source["revision"], local_files_only=True)
    execution = snapshot_inventory(Path(path).parent, choice)
    files = execution["files"]
    for name in set(source["files"]) | {"tokenizer.json", "tokenizer_config.json"}:
        if name not in files:
            raise ProvenanceError(f"missing snapshot artifact: {name}")
        if name in source["files"] and files[name]["sha256"] != source["files"][name]:
            raise ValueError("source hash mismatch")
    source_weight_bytes = sum(v["bytes"] for n, v in files.items() if n.endswith(".safetensors"))
    if source_weight_bytes * 3 + (256 << 20) > c.POLICY["post_reserve_ram_bytes"]:
        raise RuntimeError("target outside this bounded pilot's conservative 4GiB resource estimate")
    inventory = c.Encoder("inventory").inventory
    return {"pilot": source, **execution,
            "helper_inventory": inventory,
            "entire_helper_parameters": c.runtime_parameters(inventory),
            "research_teacher_plus_helper_parameters": execution["teacher_parameters"] + sum(inventory.values()),
            "research_evaluation_teacher_plus_two_helpers_parameters": execution["teacher_parameters"] + 2 * sum(inventory.values()),
            "placement": "local-clear research teacher; target provider-owned in future private serving"}


def prepare(data, manifest, source=None):
    from benchmarks.research.checkpoint_tiny_model import PILOT
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    doc = {"schema": c.SCHEMA, "status": "prepared_unadmitted", "policy": c.POLICY,
           "data_path": str(Path(data).resolve()), "data": c.prepare_data(data),
           "source": metadata(source or PILOT), "environment": environment()}
    doc["preparation_cost"] = {"cpu_seconds": time.process_time() - start_cpu,
                               "wall_seconds": time.perf_counter() - start_wall,
                               "source_prepositioned_bytes": sum(f["bytes"] for f in doc["source"]["files"].values()),
                               "public_data_bytes": Path(data).stat().st_size,
                               "download_cost": "prepositioned; prior download/setup unknown, not zero"}
    doc["work_budget"] = work_budget(doc["data"]["rows"])
    doc["digest"] = c.digest(doc)
    atomic(manifest, doc, exclusive=True)
    return doc


def _validate(manifest):
    doc = json.loads(Path(manifest).read_text())
    identity = doc.pop("digest")
    if doc["schema"] != c.SCHEMA or c.digest(doc) != identity or doc["policy"] != c.POLICY:
        raise ValueError("manifest/policy binding")
    if doc["environment"] != environment():
        raise ValueError("code/native/environment changed; never combine cohorts")
    if doc["data"] != c.prepare_data(doc["data_path"]):
        raise ValueError("canonical public data re-derivation failed")
    if doc["source"] != metadata(doc["source"]["pilot"]):
        raise ValueError("checkpoint/tokenizer source changed")
    return dict(doc, digest=identity)


def validate(manifest):
    try:
        return _validate(manifest)
    except Exception as error:
        raise ProvenanceError(str(error)) from error


class Ledger:
    """Immutable attempts + fsynced per-item snapshots; failed work never erased."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "lineage.json"
        expected = {"schema": c.SCHEMA, "manifest_digest": manifest["digest"], "serving_qualified": False}
        try:
            atomic(path, expected, exclusive=True)
        except FileExistsError:
            if json.loads(path.read_text()) != expected:
                raise ValueError("incompatible cohort lineage") from None

    def start(self, stage, ids):
        previous = [json.loads(p.read_text()) for s in ("baseline", "labels", "fit", "evaluate")
                    for p in self.root.glob(f"{s}-*.json")]
        if len(previous) >= c.POLICY["max_child_attempts"]:
            raise RuntimeError("frozen total attempt budget exhausted; originals retained")
        if any(sum(key in p["ids"] for p in previous if p["stage"] == stage) >= c.POLICY["max_attempts_per_item"] for key in ids):
            raise RuntimeError("frozen per-item attempt budget exhausted; originals retained")
        path = self.root / f"{stage}-{uuid.uuid4().hex}.json"
        doc = {"stage": stage, "ids": ids, "manifest_digest": self.manifest["digest"],
               "status": "incomplete", "outcome": "running", "items": {}, "started": time.time(),
               "provenance_invalid": False,
               "cost_scope": "partial worker window after validation/import; NOT entire child lifetime; unknown interrupted work never zero"}
        atomic(path, doc, exclusive=True)
        return path

    def completed(self, stage):
        results = {}
        for path in sorted(self.root.glob(f"{stage}-*.json")):
            doc = json.loads(path.read_text())
            if doc["manifest_digest"] != self.manifest["digest"]:
                raise ValueError("incompatible attempt")
            if doc.get("provenance_invalid"):
                continue
            if doc.get("guard", {}).get("error") or doc.get("guard", {}).get("maximum_swap_growth_bytes", 0) > 0:
                continue  # Retained/charged, but pressure-invalidated items require fresh work.
            for key, item in doc["items"].items():
                if item.get("status") == "complete":
                    if key in results:
                        raise ValueError("duplicate completed item; refuse pooled cohorts")
                    results[key] = item
        return results


class StrictHostSampler:
    """Tighten shared watchdog: preserve full reserve and reject ANY new swap."""
    def __init__(self, sample, host):
        self.sample, self.minimum_swap, self.reserve = sample, host.swap_used, host.reserve
        self.maximum_swap_growth_bytes = 0
        self.error = None

    def __call__(self):
        current = self.sample()
        self.minimum_swap = min(self.minimum_swap, current.swap_used)
        growth = current.swap_used - self.minimum_swap
        self.maximum_swap_growth_bytes = max(self.maximum_swap_growth_bytes, growth)
        if growth > 0:
            self.error = "no-new-swap gate: observed swap growth"
            raise RuntimeError(self.error)
        if current.available < self.reserve:
            self.error = "full host reserve gate"
            raise RuntimeError(self.error)
        return current


def work_budget(rows):
    counts = {s: sum(r["split"] == s for r in rows) for s in c.POLICY["counts"]}
    labels = sum(len(c.masks(r)) for r in rows if r["split"] == "train")
    chunks = {s: (n + 7) // 8 for s, n in counts.items()}
    evaluation = 4 * counts["heldout"] + 2 * chunks["heldout"]
    return {"items": counts, "baseline_target_passes": counts["admission"],
            "assumption": "default eight-item chunks; smaller chunks change fallback/process caps, never measured savings",
            "teacher_label_target_passes": labels, "evaluation_target_passes_including_fallback_cap": evaluation,
            "maximum_target_passes_no_failures": counts["admission"] + labels + evaluation,
            "maximum_sequence_rows_no_failures": (counts["admission"] + labels + evaluation) * 535,
            "local_neural_updates": counts["train"] * c.POLICY["epochs"],
            "fresh_children_default_eight_item_chunks": sum(chunks.values()) + 1,
            "seconds_per_child_timeout": c.POLICY["timeout_seconds"],
            "max_attempts_per_item": c.POLICY["max_attempts_per_item"],
            "max_child_attempts_including_failures": c.POLICY["max_child_attempts"],
            "retry_cost": "at most two attempts/item and 138 child attempts/root; fixed fit only once; no automatic retry; all originals charged"}


def admission(manifest, ledger):
    rows = [r for r in manifest["data"]["rows"] if r["split"] == "admission"]
    records = ledger.completed("baseline")
    results = {k: v["full"] for k, v in records.items()}
    gate = c.baseline_gate(rows, results)
    optimistic = c.optimistic_baseline(rows, results)
    return dict(gate, optimistic=optimistic,
                status="baseline_rejected" if optimistic["baseline_rejected"] else "admitted" if gate["admitted"] else "pending_or_weak")


def worker(payload):
    path = Path(payload["attempt"])
    doc = json.loads(path.read_text())
    stage = doc["stage"]
    start_cpu = start_wall = None
    from pllm.runtime.benchmark_memory import process_memory
    def save():
        doc["measured_worker_cpu_seconds"] = time.process_time() - start_cpu if start_cpu is not None else None
        doc["measured_worker_wall_seconds"] = time.perf_counter() - start_wall if start_wall is not None else None
        doc["worker_measurement_scope"] = "partial: after validation/import through last snapshot, not entire child lifetime"
        doc["process_memory"] = process_memory()
        atomic(path, doc)
    try:
        manifest = validate(payload["manifest"])
        if doc["manifest_digest"] != manifest["digest"] or doc["status"] != "incomplete":
            raise ProvenanceError("attempt lineage/state")
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        rows = {r["id"]: r for r in manifest["data"]["rows"]}
        binding = c.digest(manifest["source"])
        ledger = Ledger(path.parent, manifest)
        if stage in {"labels", "fit", "evaluate"} and not admission(manifest, ledger)["admitted"]:
            raise RuntimeError("training/evaluation blocked: missing or weak useful baseline")
        if stage == "fit":
            doc["artifact"] = {"path": str(Path(payload["artifact"]).resolve()),
                               "exists": Path(payload["artifact"]).exists(), "serving_qualified": False,
                               "status": "planned_unqualified"}
            save()
            train = [r for r in rows.values() if r["split"] == "train"]
            labels = ledger.completed("labels")
            if set(labels) != {r["id"] for r in train}:
                raise RuntimeError("complete frozen training labels required")
            def epoch(number, loss):
                doc["epoch"], doc["loss"] = number, loss
                save()
            model = c.fit(train, [labels[r["id"]]["losses"] for r in train], binding, epoch)
            model.save(payload["artifact"])
            doc["artifact"] = {"path": str(Path(payload["artifact"]).resolve()),
                               "sha256": c.file_hash(payload["artifact"]),
                               "bytes": Path(payload["artifact"]).stat().st_size,
                               "inventory": model.inventory, "exists": True,
                               "status": "fit_unqualified", "serving_qualified": False}
        else:
            if stage == "baseline" and admission(manifest, ledger)["status"] == "baseline_rejected":
                doc.update(status="complete", outcome="baseline_rejected", baseline=admission(manifest, ledger))
                save()
                return {"attempt": str(path), "status": "baseline_rejected"}
            teacher = make_teacher(manifest["source"])
            doc["teacher_lock"] = teacher.lock
            save()
            model = None
            if stage == "evaluate":
                fits = [json.loads(p.read_text()) for p in path.parent.glob("fit-*.json")]
                fits = [f for f in fits if f["status"] == "complete" and not f.get("provenance_invalid") and not f.get("guard", {}).get("error")]
                if len(fits) != 1 or c.file_hash(payload["artifact"]) != fits[0]["artifact"]["sha256"]:
                    raise ValueError("exact completed fit artifact required")
                model = c.Encoder.load(payload["artifact"], binding)
                if model.train_digest != c.digest([r for r in rows.values() if r["split"] == "train"]):
                    raise ValueError("training split artifact mismatch")
                doc["artifact"] = fits[0]["artifact"]
            fallbacks = 0
            for key in doc["ids"]:
                if stage == "baseline" and admission(manifest, ledger)["status"] == "baseline_rejected":
                    doc["baseline"] = admission(manifest, ledger)
                    doc["outcome"] = "baseline_rejected"
                    break
                row = rows[key]
                expected_split = {"baseline": "admission", "labels": "train", "evaluate": "heldout"}[stage]
                if row["split"] != expected_split:
                    raise ValueError("stage split violation")
                item = doc["items"][key] = {"status": "incomplete", "passes": []}
                save()
                def execute(context, mode, answer=None):
                    # Record intent BEFORE execution; failure is never interpreted as zero work.
                    attempt = {"mode": mode, "status": "incomplete", "context_digest": c.digest(context)}
                    item["passes"].append(attempt)
                    save()
                    result = teacher.execute(c.prompt(row, context), answer=answer)
                    attempt.update(status="complete", result=result)
                    save()
                    return result
                if stage == "baseline":
                    item["full"] = execute(row["context"], "full")
                elif stage == "labels":
                    item["losses"] = []
                    for mask in c.masks(row):
                        item["losses"].append(execute(c.extract(row, mask), "teacher_candidate", row["answers"][0])["answer_nll"])
                        save()
                else:
                    item["full"] = execute(row["context"], "full")
                    choices = {"lexical": lambda: c.lexical_mask(row),
                               "untrained": lambda: c.Encoder(binding).choose(row),
                               "learned": lambda: model.choose(row)}
                    for mode, choose in choices.items():
                        item.setdefault("local", {})[mode] = {"status": "incomplete", "cpu_seconds": None}
                        save()
                        local_start = time.process_time()
                        mask = choose()
                        context = c.extract(row, mask)
                        item["local"][mode] = {"status": "complete", "mask": mask, "cpu_seconds": time.process_time() - local_start,
                            "answer_retained_diagnostic_only": any(a in context for a in row["answers"])}
                        result = execute(context, mode)
                        item[mode + "_raw"] = result
                        fallback = c.needs_fallback(result["text"], result["eos"], context)
                        if fallback and fallbacks < c.POLICY["max_fallbacks_per_chunk"]:
                            fallbacks += 1
                            result = execute(row["context"], mode + "_fresh_full_fallback")
                        elif fallback:
                            item.setdefault("fallback_budget_exhausted", []).append(mode)
                        item[mode] = result
                item["status"] = "complete"
                save()
        validate(payload["manifest"])
        doc.update(status="complete", outcome=doc.get("outcome") if doc.get("outcome") == "baseline_rejected" else "complete")
        save()
        return {"attempt": str(path), "status": "baseline_rejected" if doc["outcome"] == "baseline_rejected" else doc["status"]}
    except BaseException as error:
        doc["outcome"] = "failed"
        if isinstance(error, ProvenanceError):
            doc["provenance_invalid"] = True
        if stage == "fit" and "artifact" in doc:
            doc["artifact"]["exists"] = Path(doc["artifact"]["path"]).exists()
        doc["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        save()
        raise


def run_stage(manifest_path, root, stage, artifact=None, limit=8):
    # Metadata may be inspected first; all target values/work stay in guarded child.
    from scripts.frontier_probe_support import fresh_workers
    parent_cpu, parent_wall = time.process_time(), time.perf_counter()
    manifest = validate(manifest_path)
    ledger = Ledger(root, manifest)
    if not 1 <= limit <= c.POLICY["chunk_items"]:
        raise ValueError("chunk bound 1..8")
    if stage != "baseline" and not admission(manifest, ledger)["admitted"]:
        raise RuntimeError("training blocked: baseline missing/weak")
    split = {"baseline": "admission", "labels": "train", "evaluate": "heldout", "fit": "train"}[stage]
    completed = ledger.completed(stage)
    if stage == "baseline" and admission(manifest, ledger)["status"] == "baseline_rejected":
        rejected = {"status": "baseline_rejected", "manifest_digest": manifest["digest"], "gate": admission(manifest, ledger)}
        atomic(ledger.root / "baseline_rejected.json", rejected)
        return rejected
    ids = [r["id"] for r in manifest["data"]["rows"] if r["split"] == split and r["id"] not in completed][:limit]
    if stage == "fit":
        ids = []
        if any(ledger.root.glob("fit-*.json")):
            raise RuntimeError("fit attempted already; retain attempt, no heldout-driven refit")
    elif not ids:
        return {"status": "cohort_complete", "stage": stage}
    path = ledger.start(stage, ids)
    try:
        from pllm.runtime.benchmark_memory import host_memory
        if host_memory().disk_free < c.POLICY["disk_free_bytes"]:
            raise RuntimeError("requires 3GiB free scratch disk")
        with fresh_workers(Path(__file__).resolve(), memory_bytes=c.POLICY["post_reserve_ram_bytes"],
                           timeout=c.POLICY["timeout_seconds"]) as (run, guard, host):
            strict = StrictHostSampler(guard.sample, host)
            guard.sample = strict
            doc = json.loads(path.read_text())
            doc.update(host=asdict(host), build=run.build)
            atomic(path, doc)
            try:
                result = run({"manifest": str(Path(manifest_path).resolve()), "attempt": str(path.resolve()),
                              "artifact": str(Path(artifact).resolve()) if artifact else None})
                guard.check()
            finally:
                doc = json.loads(path.read_text())
                doc["guard"] = {"maximum_swap_growth_bytes": strict.maximum_swap_growth_bytes,
                                "minimum_available_bytes": guard.minimum_available_bytes, "error": strict.error or guard.error}
                atomic(path, doc)
        validate(manifest_path)
        if stage == "baseline" and admission(manifest, ledger)["status"] == "baseline_rejected":
            rejected = {"status": "baseline_rejected", "manifest_digest": manifest["digest"], "gate": admission(manifest, ledger)}
            atomic(ledger.root / "baseline_rejected.json", rejected)
            return rejected
        return result
    except BaseException as error:
        doc = json.loads(path.read_text())
        try:
            validate(manifest_path)
        except ProvenanceError as mismatch:
            doc.update(provenance_invalid=True, provenance_error=str(mismatch)[:2048])
        if isinstance(error, ProvenanceError) or "frontier build changed" in str(error):
            doc["provenance_invalid"] = True
        doc.update(status="incomplete", outcome="parent_aborted",
                   parent_error={"type": type(error).__name__, "message": str(error)[:2048]})
        atomic(path, doc)
        raise
    finally:
        doc = json.loads(path.read_text())
        doc["parent_cpu_seconds"] = time.process_time() - parent_cpu
        doc["parent_wall_seconds"] = time.perf_counter() - parent_wall
        atomic(path, doc)


def measured_sum(records, key):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    missing = len(records) - len(values)
    return {"total": sum(values) if not missing else None,
            "known_subtotal": sum(values), "missing_records": missing}


def pass_cost(passes):
    completed = [p["result"] for p in passes if p["status"] == "complete"]
    incomplete = sum(p["status"] != "complete" for p in passes)
    counts = {key: measured_sum(completed, key) for key in
              ("input_tokens", "selected_tokens", "executed_sequence_rows", "stage_rows", "stage_calls", "cpu_seconds", "wall_seconds")}
    directions = ("client_to_inference", "inference_to_client", "preparation_to_inference")
    wire = {d: measured_sum([r.get("projected_arithmetic_bytes", {}) for r in completed], d) for d in directions}
    if incomplete:
        for metric in [*counts.values(), *wire.values()]:
            metric["total"] = None
    return {"attempted_passes": len(passes), "completed_passes": len(completed),
            "incomplete_passes_unknown_not_zero": incomplete,
            "counts": counts, "projected_arithmetic_bytes": wire}


def report(manifest, root):
    ledger = Ledger(root, manifest)
    attempts = [json.loads(p.read_text()) for p in sorted(Path(root).glob("*-*.json"))]
    rows = [r for r in manifest["data"]["rows"] if r["split"] == "heldout"]
    evaluation = ledger.completed("evaluate")
    complete = (len(rows) == c.POLICY["counts"]["heldout"] + 4 * c.POLICY["controls_per_kind"]["heldout"]
                and set(evaluation) == {r["id"] for r in rows})
    result = {"schema": c.SCHEMA, "manifest_digest": manifest["digest"], "baseline": admission(manifest, ledger),
              "heldout_complete": complete, "useful_quality_pass": False, "learned_value_pass": False,
              "serving_qualified": False,
              "costs": {"attempts": len(attempts), "child_full_lifetime_cpu_seconds": None,
                        "child_full_lifetime_wall_seconds": None,
                        "worker_partial_window_cpu_seconds": measured_sum(attempts, "measured_worker_cpu_seconds"),
                        "worker_partial_window_wall_seconds": measured_sum(attempts, "measured_worker_wall_seconds"),
                        "worker_unknown_tail_attempts": sum(a["status"] != "complete" for a in attempts),
                        "parent_partial_window_cpu_seconds": measured_sum(attempts, "parent_cpu_seconds"),
                        "parent_partial_window_wall_seconds_including_child": measured_sum(attempts, "parent_wall_seconds"),
                        "preparation": manifest.get("preparation_cost", "unknown, not zero"),
                        "helper_artifacts": [a["artifact"] for a in attempts if a["stage"] == "fit" and "artifact" in a],
                        "failed_attempts": sum(a["status"] != "complete" for a in attempts),
                        "incomplete_passes_unknown_not_zero": sum(p["status"] != "complete" for a in attempts for i in a["items"].values() for p in i["passes"]),
                        "actual_passes": sum(len(i["passes"]) for a in attempts for i in a["items"].values()),
                        "provenance_invalid_attempts": sum(bool(a.get("provenance_invalid")) for a in attempts)},
              "unmeasured": ["private full cold traffic", "whole-client Low/Medium RSS", "prior provider source setup/delivery", "transport/one-use preparation", "uncatchable post-snapshot interrupted work"],
              "traffic_target_status": "future measured >=20% net H=1 reduction; arithmetic is not admission"}
    passes = [p for a in attempts for i in a["items"].values() for p in i["passes"]]
    result["costs"]["all_passes_including_invalid_and_failed_attempts"] = pass_cost(passes)
    eval_passes = [p for a in attempts if a["stage"] == "evaluate" for i in a["items"].values() for p in i["passes"]]
    result["costs"]["per_arm_all_attempts"] = {arm: {
        "scope": "target pass totals below; local selection CPU separately charged; not complete serving cost",
        "local_selection_cpu_seconds": measured_sum([i["local"][arm] for a in attempts if a["stage"] == "evaluate"
            for i in a["items"].values() if arm in i.get("local", {})], "cpu_seconds"),
        "raw": pass_cost([p for p in eval_passes if p["mode"] == arm]),
        "fallback_inclusive": pass_cost([p for p in eval_passes if p["mode"] in {arm, arm + "_fresh_full_fallback"}])}
        for arm in ("full", "lexical", "untrained", "learned")}
    result["costs"]["helper_local_cpu_seconds"] = measured_sum([v for a in attempts for i in a["items"].values() for v in i.get("local", {}).values()], "cpu_seconds")
    peaks = [a.get("process_memory", {}).get("lifetime_peak_rss_bytes") for a in attempts]
    result["costs"]["largest_observed_child_lifetime_peak_rss_bytes"] = max((p for p in peaks if p is not None), default=None)
    result["costs"]["peak_rss_missing_attempts"] = sum(p is None for p in peaks)
    if complete:
        metrics = {mode: c.quality(rows, {k: v[mode] for k, v in evaluation.items()}) for mode in ["full", "lexical", "untrained", "learned"]}
        result["heldout"] = metrics
        raw = {mode: c.quality(rows, {k: v.get(mode + "_raw", v[mode]) for k, v in evaluation.items()})
               for mode in ["full", "lexical", "untrained", "learned"]}
        result["heldout_before_fallback"] = raw
        baseline = c.baseline_gate(rows, {k: v["full"] for k, v in evaluation.items()}, "heldout")
        learned = metrics["learned"]
        fallback_counts = {mode: sum(any(p["mode"] == mode + "_fresh_full_fallback" for p in v["passes"]) for v in evaluation.values()) for mode in ["lexical", "untrained", "learned"]}
        useful = baseline["admitted"] and all(learned[k]["em"] >= metrics["full"][k]["em"] - c.POLICY["max_quality_drop"] for k in learned)
        useful &= learned["public_qa"]["em"] >= c.POLICY["qa_min_em"] and all(learned[k]["em"] >= c.POLICY["control_min_em"] for k in c.POLICY["control_kinds"])
        useful &= fallback_counts["learned"] / len(rows) <= c.POLICY["max_fallback_fraction"]
        useful &= not any("learned" in v.get("fallback_budget_exhausted", []) for v in evaluation.values())
        result.update(useful_quality_pass=bool(useful),
                      learned_value_pass=bool(useful
                          and learned["public_qa"]["em"] >= metrics["lexical"]["public_qa"]["em"] + c.POLICY["learned_min_em_gain_over_lexical"]
                          and raw["learned"]["public_qa"]["em"] >= raw["lexical"]["public_qa"]["em"] + c.POLICY["learned_min_em_gain_over_lexical"]),
                      fresh_fallback_counts=fallback_counts)
    return result


def mechanics():
    """Tiny analytic-label fixture, never a checkpoint/useful-quality result."""
    from pllm.runtime.benchmark_memory import process_memory
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    rows = c.controls("train", 1)
    losses = [[0.0 if row["answers"][0] in c.extract(row, mask) else 4.0
               for mask in c.masks(row)] for row in rows]
    initial = c.Encoder("SYNTHETIC-NOT-CHECKPOINT")
    fitted = c.fit(rows, losses, "SYNTHETIC-NOT-CHECKPOINT")
    before = sum(c.objective(initial, r, y)[0] for r, y in zip(rows, losses, strict=True)) / len(rows)
    after = sum(c.objective(fitted, r, y)[0] for r, y in zip(rows, losses, strict=True)) / len(rows)
    return {"schema": c.SCHEMA, "status": "synthetic_mechanics_only", "real_checkpoint_tensor_loads": 0,
            "useful_quality_established": False, "serving_qualified": False,
            "baseline_status": "not_run_training_on_public_checkpoint_blocked",
            "teacher": "analytic span-retention labels, NOT target logits or language evidence",
            "fixture_digest": c.digest({"rows": rows, "teacher_losses": losses, "policy": c.POLICY}),
            "inventory": fitted.inventory, "entire_helper_runtime_parameters": sum(fitted.inventory.values()),
            "frozen_policy": c.POLICY, "train_rows": len(rows),
            "objective_before": before, "objective_after": after,
            "all_neural_owners_changed": all(not np.array_equal(v, fitted.weights[k]) for k, v in initial.weights.items()),
            "fresh_procedural_masks": [{"id": r["id"], "mask": fitted.choose(r)} for r in c.controls("heldout", 1)],
            "cpu_seconds": time.process_time() - start_cpu, "wall_seconds": time.perf_counter() - start_wall,
            "process_memory": process_memory(), "environment": environment()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", nargs="?", choices=["prepare", "preflight", "baseline", "labels", "fit", "evaluate", "report", "mechanics"])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--data")
    parser.add_argument("--source-spec")
    parser.add_argument("--run-dir")
    parser.add_argument("--artifact")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.worker:
        result = worker(json.load(sys.stdin))
    elif args.stage == "mechanics":
        result = mechanics()
        atomic(args.output, result, exclusive=True)
        result = {"status": result["status"], "output": args.output,
                  "fixture_digest": result["fixture_digest"],
                  "entire_helper_runtime_parameters": result["entire_helper_runtime_parameters"]}
    elif args.stage == "prepare":
        result = prepare(args.data, args.manifest, json.loads(Path(args.source_spec).read_text()) if args.source_spec else None)
        result = {"status": result["status"], "manifest_digest": result["digest"]}
    elif args.stage == "preflight":
        from pllm.runtime.benchmark_memory import host_memory
        manifest = validate(args.manifest)
        host = host_memory()
        result = {"manifest_digest": manifest["digest"], "status": "metadata_only_no_values",
                  "host": asdict(host), "resource_admitted_now": host.available - host.reserve >= c.POLICY["post_reserve_ram_bytes"] and host.disk_free >= c.POLICY["disk_free_bytes"]}
        atomic(args.output, result, exclusive=True)
    elif args.stage == "report":
        result = report(validate(args.manifest), args.run_dir)
        atomic(args.output, result, exclusive=True)
    elif args.stage:
        result = run_stage(args.manifest, args.run_dir, args.stage, args.artifact, args.limit)
    else:
        parser.error("stage required")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
