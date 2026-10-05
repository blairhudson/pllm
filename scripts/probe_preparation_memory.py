"""Fresh-process resident/paged Preparation engine ablation on locked checkpoints.

The probe-local loader replaces each retained CPU matrix with an authenticated
private raw-file snapshot. It exercises the ordinary seeded correction method
and row-residue codec, with a shared native executor and bounded concurrent
stage requests. This is isolated loader/issuance evidence, not a selectable
Pipeline, HTTP deployment, whole-response benchmark or filesystem-cache bound.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


def source(*, tiny=False, four_b=False):
    from pllm import Model

    if tiny:
        return Model.tiny(model_id="preparation-memory-probe")
    return Model.hf(
        "Qwen/Qwen3-4B" if four_b else "Qwen/Qwen2.5-0.5B-Instruct",
        revision=("1cfa9a7208912126459214e8b04321603b3df60c" if four_b
                  else "7ae557604adf67be50417f59c2c2f167def9a775"),
        local_files_only=True,
    )


def paged_loader(engine, scratch: Path, snapshots: list):
    """Test-local substitution; the installed engine and its defaults are intact."""
    from pllm.native import PagedGEMM
    from pllm.runtime.transformer_engine import _weight_chunks

    original = engine._load_stage

    def load(*args, **kwargs):
        runtime = original(*args, **kwargs)
        if runtime.compiled_weight is None:
            return runtime
        # The normal compressed correction codec needs these committed row bounds
        # after the weight array has been retired.
        _ = runtime.output_residue_bits
        with tempfile.NamedTemporaryFile(dir=scratch, prefix="stage-", suffix=".i8") as artifact:
            for chunk in _weight_chunks(runtime.weight.values):
                artifact.write(memoryview(chunk))
            artifact.flush()
            matrix = PagedGEMM._from_raw_executor(
                artifact.name, runtime.weight_digest,
                (runtime.spec.out_features, runtime.spec.in_features), engine.kernel._executor,
            )
        snapshots.append(matrix)
        if matrix.weight_digest != runtime.weight_digest:
            raise ValueError("paged stage differs from the committed preparation weight")
        runtime.quantized_weight = None
        runtime.compiled_weight = matrix
        return runtime

    return load


async def measure(model, mode: str, rows: int, stage_window: int, scratch: Path, salt: bytes):
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.preparation_protocol import PreparationRequest
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    if mode not in {"resident", "paged"} or not 1 <= rows <= 128 or stage_window not in {1, 4}:
        raise ValueError("invalid bounded preparation probe")
    if len(salt) != 32:
        raise ValueError("probe cohort salt must contain 32 bytes")
    start, start_cpu = time.perf_counter(), time.process_time()
    resolved = resolve_model(model)
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
        weight_residency="provider", prepared_output_encoding="row_residues")
    snapshots = []
    try:
        substitute = (patch.object(engine, "_load_stage", paged_loader(engine, scratch, snapshots))
                      if mode == "paged" else nullcontext())
        with substitute:
            await engine.load(resolved.manifest)
        loaded = engine.models[resolved.manifest.id]
        stages = [s for s in loaded.stages.values() if engine._provider_owns_stage(s.spec)]
        if not stages:
            raise ValueError("probe requires provider-owned semantic stages")
        for stage in stages:
            if rows * max(stage.spec.in_features, stage.spec.out_features) > 4_000_000:
                raise ValueError("probe exceeds paged/row-codec capacity")
            _ = stage.output_residue_bits
        loaded_memory = process_memory()
        load_cpu, load_seconds = time.process_time() - start_cpu, time.perf_counter() - start
        metadata = hashlib.sha256()
        for stage in loaded.stages.values():
            metadata.update(stage.metadata.pack())
        corrections = hashlib.sha256()
        correction_bytes = 0
        begin, begin_cpu = time.perf_counter(), time.process_time()

        async def issue(stage):
            # Same fresh diagnostic cohort in both children. No live inventory or
            # user request consumes this test-only material.
            domain = salt + stage.spec.id.encode()
            request = PreparationRequest(
                attempt_id=hashlib.sha256(b"attempt" + domain).hexdigest()[:32],
                session_id="preparation-memory-probe",
                model=resolved.manifest.id,
                body_fingerprint=loaded.manifest.metadata["body_fingerprint"],
                stage_id=stage.spec.id, weight_digest=stage.weight_digest,
                rows=rows, in_features=stage.spec.in_features, out_features=stage.spec.out_features,
                weight_bits=8, activation_bits=8,
                signed_output_bound=stage.signed_output_bound,
                ring=stage.seeded_profile.ring, modulus=stage.seeded_profile.modulus,
                wire_bits=stage.seeded_profile.wire_bits,
                seed=hashlib.sha256(b"seed" + domain).digest(),
            )
            request._validate(max_rows=128, max_tensor_elements=4_000_000)
            result = await engine.prepare_seeded_stage(request)
            # Compare the complete serialized frame, excluding only its clock.
            frame = replace(result, server_ns=0).pack()
            return len(frame), hashlib.sha256(frame).digest()

        for start_index in range(0, len(stages), stage_window):
            group = stages[start_index:start_index + stage_window]
            tasks = [asyncio.create_task(issue(stage)) for stage in group]
            try:
                for size, digest in await asyncio.gather(*tasks):
                    correction_bytes += size
                    corrections.update(digest)
            finally:
                # A failing sibling must finish before its file handles retire.
                await asyncio.gather(*tasks, return_exceptions=True)
        issue_cpu, issue_seconds = time.process_time() - begin_cpu, time.perf_counter() - begin
        result = {
            "mode": mode, "checkpoint_digest": resolved.checkpoint_digest,
            "source_lock_digest": resolved.source_lock_digest,
            "body_fingerprint": loaded.manifest.metadata["body_fingerprint"],
            "stage_commitment": loaded.manifest.metadata["seeded_stage_commitment"],
            "metadata_digest": metadata.hexdigest(), "correction_digest": corrections.hexdigest(),
            "correction_body_bytes": correction_bytes, "stage_count": len(stages),
            "rows_per_stage": rows, "stage_window": stage_window, "threads": 1,
            "weight_bits": 8, "activation_bits": 8, "output_encoding": "row_residues",
            "logical_weight_bytes": sum(s.spec.in_features * s.spec.out_features for s in stages),
            "resident_weight_bytes": sum(s.weight.values.nbytes for s in stages
                                         if s.quantized_weight is not None),
            "private_snapshot_disk_bytes": sum(s.artifact_bytes for s in snapshots),
            "paged_metadata_bytes": sum(s.metadata_bytes for s in snapshots),
            "paged_weight_buffer_bound_bytes": sum(sorted(
                (s.transient_weight_bytes for s in snapshots), reverse=True)[:stage_window]),
            "after_load": loaded_memory, "after_issuance": process_memory(),
            "load_cpu_seconds": load_cpu, "load_seconds": load_seconds,
            "issuance_cpu_seconds": issue_cpu, "issuance_seconds": issue_seconds,
            "load_and_issuance_cpu_seconds": time.process_time() - start_cpu,
            "load_and_issuance_seconds": time.perf_counter() - start,
            "process_cpu_seconds": time.process_time(),
        }
    finally:
        if resolved.manifest.id in engine.models:
            await engine.unload(resolved.manifest.id)
        for snapshot in snapshots:
            snapshot.close()
    result["after_unload"] = process_memory()
    return result


def check_parity(samples):
    if {s["mode"] for s in samples} != {"resident", "paged"}:
        raise ValueError("both storage controls are required")
    for key in ("checkpoint_digest", "source_lock_digest", "body_fingerprint", "stage_commitment",
                "metadata_digest", "correction_digest", "correction_body_bytes", "stage_count",
                "rows_per_stage", "stage_window", "threads", "weight_bits", "activation_bits",
                "output_encoding", "logical_weight_bytes"):
        if samples[0][key] is None or any(s[key] != samples[0][key] for s in samples):
            raise ValueError(f"preparation cohort mismatch: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--four-b", action="store_true")
    parser.add_argument("--rows", type=int, default=71)
    parser.add_argument("--stage-window", type=int, choices=(1, 4), default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("resident", "paged"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tiny and args.four_b or not 1 <= args.rows <= 128 or not 1 <= args.repetitions <= 3:
        parser.error("choose one model, 1..128 rows and 1..3 repetitions")
    if args.output and args.output.exists():
        parser.error("output already exists")
    model = source(tiny=args.tiny, four_b=args.four_b)
    if args.worker:
        with tempfile.TemporaryDirectory(prefix="pllm-preparation-memory-", dir=args.scratch_dir) as directory:
            result = asyncio.run(measure(model, args.worker, args.rows, args.stage_window, Path(directory),
                bytes.fromhex(os.environ["PLLM_PREPARATION_MEMORY_SALT"])))
        print(json.dumps(result))
        return

    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.metrics import benchmark_memory
    from pllm.preparation import PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    experiment = Experiment("preparation-memory-probe", MaskedLinearCpu(model,
        inventory=PreparedInventory(rows=args.rows, stage_window=args.stage_window)),
        Deployment.local(root="local://preparation-memory-probe"),
        ExecutionBudget(requests=1, max_input_tokens=args.rows, max_new_tokens=1))
    # Reuse the ordinary geometry admission; only one Preparation engine and the
    # lightweight controller are live. Its resident estimate bounds both modes.
    preflight = benchmark_memory(experiment, backend="native")
    if preflight["estimate"]["limitations"]:
        raise RuntimeError(str(preflight["estimate"]["limitations"]))
    required = preflight["estimate"]["provider_peak_bytes"]["preparation"] + GiB // 2
    disk = 2 * preflight["estimate"]["components"]["per_engine_i8_and_native_bytes"] + GiB
    host = host_memory()
    if host.available - host.reserve < required or host.disk_free < disk:
        raise RuntimeError("isolated preparation probe exceeds admitted physical headroom or disk")
    active = []

    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PLLM_PREPARATION_MEMORY_SALT=os.urandom(32).hex())
    samples = []
    guard.start()
    try:
        for repetition in range(args.repetitions):
            modes = ("resident", "paged") if repetition % 2 == 0 else ("paged", "resident")
            for mode in modes:
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                command = [sys.executable, __file__, "--worker", mode, "--rows", str(args.rows),
                           "--stage-window", str(args.stage_window)]
                command += [flag for flag, enabled in (("--tiny", args.tiny), ("--four-b", args.four_b)) if enabled]
                if args.scratch_dir:
                    command.extend(("--scratch-dir", str(args.scratch_dir)))
                print(f"[{repetition + 1}/{args.repetitions}] {mode} preparation", flush=True)
                child = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                active.append(child)
                try:
                    stdout, stderr = child.communicate(timeout=600)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
                    active.remove(child)
                if child.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                samples.append(json.loads(stdout))
        check_parity(samples)
        guard.check()
        report = {
            "schema": "pllm.preparation_memory_probe.v1", "scope": __doc__, "parity": True,
            "measured_at": datetime.now(timezone.utc).isoformat(), "samples": samples,
            "host": asdict(host), "required_physical_headroom_bytes": required,
            "maximum_host_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "python": platform.python_version(), "platform": platform.platform(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preflight": preflight,
        }
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps([{key: s[key] for key in ("mode", "after_issuance", "issuance_cpu_seconds",
            "load_and_issuance_cpu_seconds", "private_snapshot_disk_bytes")} for s in samples], indent=2))
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
