"""Bounded cross-inventory correction factory; synthetic stages, not serving.

Existing immutable native/Metal matrices do arithmetic. Python owns admission,
native mask cursors and authenticated row attribution only. CUDA is unavailable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
import importlib
from importlib import metadata
import json
from pathlib import Path
import secrets
import tempfile
import time

import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ROLES = ("qkv_projection", "mlp_gate_up", "mlp_down")


def file_identity(path):
    path = Path(path).resolve()
    state = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            state.update(block)
    return {"path": str(path), "sha256": state.hexdigest()}


def runtime_identity():
    """Freeze installed implementations without creating a GPU owner/device."""
    from pllm.runtime._native_support import capabilities, extension

    modules = ("pllm.runtime.native", "pllm.runtime.metal",
               "pllm.runtime.preparation_protocol", "pllm.runtime.stage_protocol",
               "pllm.runtime.protocol", "pllm.runtime._native_support",
               "pllm.runtime.benchmark_memory", "pllm.runtime.shutdown",
               "msgpack", "msgpack._cmsgpack", "numpy", "numpy._core._multiarray_umath")
    files = {name: file_identity(importlib.import_module(name).__file__) for name in modules}
    files["probe"] = file_identity(__file__)
    files["frontier_probe_support"] = file_identity(Path(__file__).with_name("frontier_probe_support.py"))
    native = extension()
    if native is not None:
        files["pllm._native"] = file_identity(native.__file__)
    versions = {name: metadata.version(name) for name in ("numpy", "msgpack")}
    for package in ("mlx", "mlx-metal"):
        try:
            distribution = metadata.distribution(package)
            versions[package] = distribution.version
            # mlx-metal owns libmlx and shader library on current split wheels.
            # No mlx.core import or GPU availability query.
            for entry in distribution.files or ():
                if str(entry).endswith((".so", ".dylib", ".metallib")):
                    files[f"{package}:{entry}"] = file_identity(distribution.locate_file(entry))
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"files": files, "versions": versions, "native_capabilities": capabilities()}


def execute_cells(specs, run, guard, expected_identity, *, capture_identity=runtime_identity):
    """Stop globally on pressure or identity loss; never retry remaining cells."""
    samples, failures, stopped = [], [], None

    def identity_stop(index):
        try:
            changed = capture_identity() != expected_identity
        except Exception as error:
            return {"reason": "runtime_identity_unverifiable", "next_cell_index": index,
                    "error_class": type(error).__name__}
        return {"reason": "runtime_identity_changed", "next_cell_index": index} if changed else None

    for index, spec in enumerate(specs):
        if guard.error:
            stopped = {"reason": "host_pressure", "next_cell_index": index}
            break
        stopped = identity_stop(index)
        if stopped:
            break
        try:
            samples.append(run({**spec, "expected_identity": expected_identity}))
        except Exception as error:
            failures.append({"spec": spec, "error_class": type(error).__name__})
            if guard.error:
                stopped = {"reason": "host_pressure", "next_cell_index": index + 1}
                break
            stopped = identity_stop(index + 1)
            if stopped:
                break
    return samples, failures, stopped


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def admission_groups(arrivals, *, capacity=4, deadline_seconds=0.005):
    """Public virtual-arrival schedule, not measured live-provider waiting."""
    if not 1 <= capacity <= 8 or not 0 <= deadline_seconds <= 1:
        raise ValueError("invalid admission bound")
    if not arrivals or any(not np.isfinite(x) or x < 0 for x in arrivals):
        raise ValueError("invalid arrivals")
    if sorted(arrivals) != list(arrivals):
        raise ValueError("arrivals must be ordered")
    groups, start = [], 0
    while start < len(arrivals):
        limit = arrivals[start] + deadline_seconds
        end = start + 1
        while end < len(arrivals) and end - start < capacity and arrivals[end] <= limit:
            end += 1
        ready = arrivals[end - 1] if end - start == capacity else limit
        groups.append((list(range(start, end)), ready))
        start = end
    return groups


@dataclass
class Ticket:
    request: object
    cursor: object
    binding: str
    file: object
    state: str = "PENDING"
    issued: int = 0
    received: int = 0

    def burn(self):
        self.state = "BURNED"
        self.cursor.cancel()
        self.file.close()


class PreparationFactory:
    """Research-local owner. No reusable tickets, provider or production API."""

    def __init__(self, matrix, *, backend="integer", gpu=None, window_rows=128,
                 min_gpu_rows=64, max_tickets=8, workspace_bytes=64 << 20):
        if backend not in {"integer", "digit8", "metal", "adaptive"}:
            raise ValueError("unknown backend")
        if not 1 <= window_rows <= 128 or not 1 <= max_tickets <= 8:
            raise ValueError("factory capacity exceeded")
        if not 1 <= min_gpu_rows <= 128 or not 0 < workspace_bytes <= 64 << 20:
            raise ValueError("factory workspace exceeded")
        if backend in {"metal", "adaptive"} and gpu is None:
            raise ValueError("Metal snapshot required")
        self.matrix, self.gpu, self.backend = matrix, gpu, backend
        self.weight_digest = hashlib.sha256(matrix.weight_view().tobytes()).hexdigest()
        self.window_rows, self.min_gpu_rows = window_rows, min_gpu_rows
        self.max_tickets, self.workspace_bytes = max_tickets, workspace_bytes
        self.key = secrets.token_bytes(32)
        self.tickets = []
        self.closed = False
        self.calls = {"integer": 0, "digit8": 0, "metal": 0}

    def add(self, request):
        from pllm.runtime.preparation_protocol import prepared_mask_rows

        cursor = file = None
        try:
            if self.closed or len(self.tickets) >= self.max_tickets:
                raise ValueError("factory closed or ticket capacity exceeded")
            request._validate(max_rows=150, max_tensor_elements=4_000_000)
            if (request.out_features, request.in_features) != self.matrix.shape:
                raise ValueError("ticket geometry differs")
            if (request.weight_digest != self.weight_digest or request.weight_bits != 8
                    or request.activation_bits != 8
                    or request.signed_output_bound != self.matrix.shape[1] * 128 * 128):
                raise ValueError("ticket weight or numeric binding differs")
            # All materials bind exactly the same public stage and numeric contract.
            context = asdict(request.context)
            if self.tickets and context != asdict(self.tickets[0].request.context):
                raise ValueError("ticket public stage differs")
            if any(t.request.attempt_id == request.attempt_id for t in self.tickets):
                raise ValueError("ticket replay")
            out, width = self.matrix.shape
            if 32 * self.window_rows * (width + out) > self.workspace_bytes:
                raise ValueError("row workspace admission exceeded")
            cursor = prepared_mask_rows(request)
            if cursor is None:
                raise ValueError("native mask cursor required")
            file = tempfile.TemporaryFile()
            binding = digest({"context": context, "attempt": request.attempt_id,
                              "session": request.session_id, "rows": request.rows})
            ticket = Ticket(request, cursor, binding, file)
            self.tickets.append(ticket)
            return ticket
        except BaseException:
            if cursor is not None:
                cursor.cancel()
            if file is not None:
                file.close()
            self.close()
            raise

    def _accept(self, ticket, offset, count, payload, tag):
        from pllm.runtime.stage_protocol import unpack_residues

        header = json.dumps([ticket.binding, offset, count], separators=(",", ":")).encode()
        expected = hmac.digest(self.key, header + payload, "sha256")
        if (ticket.state != "PENDING" or offset != ticket.received or count <= 0
                or offset + count > ticket.request.rows or not hmac.compare_digest(tag, expected)):
            raise ValueError("row attribution or authentication differs")
        values = unpack_residues(payload, (count, ticket.request.out_features), ticket.request.wire_bits)
        ticket.file.write(values.astype("<u4", copy=False).tobytes())
        ticket.received += count

    def execute(self, tickets):
        from pllm.runtime.stage_protocol import pack_residues

        metrics = {f"{phase}_{unit}": 0.0 for phase in ("mask", "kernel", "encoding")
                   for unit in ("seconds", "cpu_seconds")}
        metrics.update(encoded_bytes=0, windows=0, rows=0)
        start, cpu = time.perf_counter(), time.process_time()
        try:
            if (self.closed or not tickets or len({id(t) for t in tickets}) != len(tickets)
                    or any(not any(t is owned for owned in self.tickets)
                           or t.state != "PENDING" or t.issued for t in tickets)):
                raise ValueError("ticket not fresh or factory closed")
            while any(t.issued < t.request.rows for t in tickets):
                before, before_cpu = time.perf_counter(), time.process_time()
                pieces, masks, ledger, capacity = [], [], [], self.window_rows
                # Round-robin shares one window; long inventories stream over windows.
                active = [t for t in tickets if t.issued < t.request.rows]
                for index, ticket in enumerate(active):
                    count = min(ticket.request.rows - ticket.issued,
                                max(1, capacity // (len(active) - index)))
                    if not capacity:
                        break
                    count = min(count, capacity)
                    offset = ticket.issued
                    raw_r, raw_s = ticket.cursor.take(offset, count)
                    pieces.append(np.frombuffer(raw_r, "<u4").reshape(count, ticket.request.in_features))
                    masks.append(np.frombuffer(raw_s, "<u4").reshape(count, ticket.request.out_features))
                    ledger.append((ticket, offset, count))
                    ticket.issued += count
                    capacity -= count
                inputs, output_masks = np.concatenate(pieces), np.concatenate(masks)
                metrics["mask_seconds"] += time.perf_counter() - before
                metrics["mask_cpu_seconds"] += time.process_time() - before_cpu
                before, before_cpu = time.perf_counter(), time.process_time()
                backend = self.backend
                if backend == "adaptive":
                    backend = "metal" if len(inputs) >= self.min_gpu_rows else "integer"
                bits = tickets[0].request.wire_bits
                if backend == "metal":
                    result = self.gpu.wrap32(inputs)
                elif backend == "digit8":
                    result = self.matrix.digit8(inputs, bits)
                else:
                    result = self.matrix.wrap32(inputs)
                correction = np.subtract(result, output_masks, dtype=np.uint32)
                correction &= np.uint32((1 << bits) - 1)
                self.calls[backend] += 1
                metrics["kernel_seconds"] += time.perf_counter() - before
                metrics["kernel_cpu_seconds"] += time.process_time() - before_cpu
                before, before_cpu = time.perf_counter(), time.process_time()
                row = 0
                for ticket, offset, count in ledger:
                    payload = pack_residues(correction[row:row + count], bits)
                    header = json.dumps([ticket.binding, offset, count], separators=(",", ":")).encode()
                    tag = hmac.digest(self.key, header + payload, "sha256")
                    self._accept(ticket, offset, count, payload, tag)
                    metrics["encoded_bytes"] += len(header) + len(payload) + len(tag)
                    row += count
                metrics["encoding_seconds"] += time.perf_counter() - before
                metrics["encoding_cpu_seconds"] += time.process_time() - before_cpu
                metrics["windows"] += 1
                metrics["rows"] += len(inputs)
            if any(t.received != t.request.rows for t in tickets):
                raise ValueError("incomplete inventory cannot seal")
            for ticket in tickets:
                ticket.file.flush()
                ticket.state = "SEALED"
                ticket.cursor.cancel()
            metrics["service_seconds"] = time.perf_counter() - start
            metrics["service_cpu_seconds"] = time.process_time() - cpu
            return metrics
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        for ticket in self.tickets:
            ticket.burn()
        self.key = b""


def request_for(weights, *, rows, stage="test-stage", plan_digest="test-plan"):
    from pllm.runtime.preparation_protocol import PreparationRequest, seeded_ring_profile

    out, width = weights.shape
    # Conservative signed W8A8 bound, independent of private materials.
    bound = width * 128 * 128
    profile = seeded_ring_profile(bound)
    weight_digest = hashlib.sha256(weights.tobytes()).hexdigest()
    return PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), MODEL,
        plan_digest, stage, weight_digest, rows, width, out, 8, 8, bound,
        profile.ring, profile.modulus, profile.wire_bits, secrets.token_bytes(32))


def check_ticket(ticket, weights, *, chunk_rows=8):
    """Independent bounded signed-i64 oracle; outside measured service/RSS window.

    Test-local request retained only to reconstruct oracle masks after cursor burn.
    Never use reconstructed material for another execution attempt.
    """
    from pllm.runtime.preparation_protocol import prepared_mask_rows

    if ticket.state != "SEALED":
        raise ValueError("oracle cannot read incomplete inventory")
    cursor = prepared_mask_rows(ticket.request)
    checked = 0
    ticket.file.seek(0)
    try:
        for offset in range(0, ticket.request.rows, chunk_rows):
            count = min(chunk_rows, ticket.request.rows - offset)
            r, s = cursor.take(offset, count)
            r = np.frombuffer(r, "<u4").reshape(count, weights.shape[1])
            s = np.frombuffer(s, "<u4").reshape(count, weights.shape[0])
            expected = ((r.astype(np.int64) @ weights.astype(np.int64).T - s.astype(np.int64))
                        & ((1 << ticket.request.wire_bits) - 1)).astype(np.uint32)
            actual = np.frombuffer(ticket.file.read(count * weights.shape[0] * 4), "<u4").reshape(expected.shape)
            if not np.array_equal(actual, expected):
                raise AssertionError("factory changed exact W*r-s or row attribution")
            checked += actual.size
        if ticket.file.read(1):
            raise AssertionError("extra correction rows")
        return checked
    finally:
        cursor.cancel()


def compiled_geometry(role):
    from huggingface_hub import hf_hub_download
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime.semantic_stages import scheduled_stage_specs

    raw = Path(hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True)).read_bytes()
    plan = lower_model(raw, batch=1, max_input_tokens=150, max_new_tokens=8)
    pipeline = MaskedLinearCpu(Model.hf(MODEL, revision=REVISION, local_files_only=True))
    stage = next(s for s in scheduled_stage_specs(plan, pipeline) if s.role == role and s.layer_index == 0)
    return stage, {"model": MODEL, "revision": REVISION, "config_sha256": hashlib.sha256(raw).hexdigest(),
                   "plan_digest": plan.digest, "schedule_digest": plan.runtime_schedule(pipeline).digest,
                   "weights": "public deterministic synthetic i8; not checkpoint tensor values"}


def worker(spec):
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.native import MaskedGEMM

    expected_identity = spec.pop("expected_identity")
    observed_identity = runtime_identity()
    if observed_identity != expected_identity:
        raise RuntimeError("worker runtime identity changed before execution")
    start, cpu = time.perf_counter(), time.process_time()
    stage, source = compiled_geometry(spec["role"])
    weights = np.random.default_rng(7143).integers(-128, 128,
        (stage.out_features, stage.in_features), dtype=np.int8)
    engine = MaskedGEMM(threads=4)
    if not engine.native:
        raise RuntimeError("installed native matrix required")
    matrix = engine.compile(weights)
    gpu = None
    if spec["backend"] in {"metal", "adaptive"}:
        from pllm.runtime.metal import MetalGEMM
        gpu = MetalGEMM().compile(matrix.weight_view())
        import mlx.core as mx
        if file_identity(mx.__file__) not in expected_identity["files"].values():
            raise RuntimeError("loaded MLX core differs from frozen implementation")
    factory = PreparationFactory(matrix, backend=spec["backend"], gpu=gpu)
    tickets = [factory.add(request_for(weights, rows=spec["rows"], stage=stage.id,
                                      plan_digest=source["plan_digest"]))
               for _ in range(spec["inventories"])]
    construction = {"seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
    arrivals = [i * spec["arrival_interval_seconds"] for i in range(len(tickets))]
    capacity = len(tickets) if spec["layout"] == "batch" else 1
    groups = admission_groups(arrivals, capacity=capacity,
        deadline_seconds=spec["deadline_seconds"] if spec["layout"] == "batch" else 0)
    services, delays, busy_until = [], [], 0.0
    try:
        for indices, ready in groups:
            service = factory.execute([tickets[i] for i in indices])
            admitted = max(ready, busy_until)
            busy_until = admitted + service["service_seconds"]
            for i in indices:
                delays.append({"inventory_index": i, "fill_wait_seconds": ready - arrivals[i],
                    "busy_queue_seconds": admitted - ready,
                    "modeled_arrival_to_completion_seconds": busy_until - arrivals[i]})
            services.append(service)
        memory = process_memory()
        gpu_stats = None
        if gpu is not None:
            import mlx.core as mx
            gpu_stats = {"active_bytes": mx.get_active_memory(), "peak_bytes": mx.get_peak_memory(),
                         "cache_bytes": mx.get_cache_memory()}
        oracle_start = time.perf_counter()
        checked = sum(check_ticket(ticket, weights) for ticket in tickets)
        oracle_seconds = time.perf_counter() - oracle_start
        if runtime_identity() != expected_identity:
            raise RuntimeError("worker runtime identity changed during execution")
        return {"spec": spec, "source": source, "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "shape": list(weights.shape), "construction": construction, "services": services,
            "modeled_delays": delays, "calls": factory.calls, "service_window_memory": memory,
            "cpu_weight_bytes": matrix.weight_bytes, "gpu_weight_bytes": 0 if gpu is None else gpu.weight_bytes,
            "gpu_workspace_admission_bytes": 0 if gpu is None else 32 * 128 * sum(weights.shape),
            "mlx_memory": gpu_stats, "checked_values": checked, "oracle_seconds": oracle_seconds,
            "exact_parity": True, "process_cpu_seconds_including_oracle": time.process_time(),
            "disk_correction_payload_bytes": spec["inventories"] * spec["rows"] * stage.out_features * 4,
            "cuda": "unavailable; no CUDA measurement", "gpu_compute_seconds": None,
            "physical_device_memory_bytes": None, "provider_traffic_bytes": None,
            "runtime_identity_digest": digest(expected_identity), "runtime_identity_verified": True}
    finally:
        factory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--rows", type=int, nargs="+", choices=(1, 16, 150), default=[1, 16, 150])
    parser.add_argument("--backends", nargs="+", choices=("integer", "digit8", "metal", "adaptive"),
                        default=["integer", "digit8", "metal", "adaptive"])
    parser.add_argument("--inventories", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--arrival-interval-ms", type=float, choices=(0, 1, 10), default=0)
    args = parser.parse_args()
    if args.worker:
        import sys
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("fresh --output path required")
    from frontier_probe_support import fresh_workers

    frozen_identity = runtime_identity()
    specs = []
    for repeat in range(args.processes):
        for role in ROLES:
            for rows in args.rows:
                choices = [(backend, layout) for backend in args.backends for layout in ("separate", "batch")]
                for backend, layout in choices if repeat % 2 == 0 else reversed(choices):
                    specs.append({"role": role, "rows": rows, "backend": backend, "layout": layout,
                        "inventories": args.inventories, "arrival_interval_seconds": args.arrival_interval_ms / 1000,
                        "deadline_seconds": 0.005, "repeat": repeat})
    with fresh_workers(__file__, memory_bytes=768 << 20, timeout=300) as (run, guard, host):
        samples, failures, stopped = execute_cells(specs, run, guard, frozen_identity)
        if guard.error and stopped is None:
            stopped = {"reason": "host_pressure", "next_cell_index": len(specs)}
        try:
            identity_after = runtime_identity()
        except Exception as error:
            identity_after = None
            stopped = {"reason": "runtime_identity_unverifiable",
                       "next_cell_index": len(samples) + len(failures),
                       "error_class": type(error).__name__}
        if identity_after is not None and identity_after != frozen_identity:
            stopped = {"reason": "runtime_identity_changed",
                       "next_cell_index": len(samples) + len(failures)}
    if guard.error and stopped is None:
        stopped = {"reason": "host_pressure", "next_cell_index": len(samples) + len(failures)}
    report = {"schema": "pllm.preparation_factory_probe.v1", "scope": __doc__, "samples": samples,
        "failures": failures, "host": asdict(host), "build": run.build,
        "probe_sha256": frozen_identity["files"]["probe"]["sha256"],
        "runtime_identity_before": frozen_identity, "runtime_identity_after": identity_after,
        "runtime_identity_unchanged": identity_after == frozen_identity,
        "status": "interrupted" if stopped else "finished", "stopped": stopped,
        "planned_cells": len(specs), "unattempted_cells": specs[len(samples) + len(failures):],
        "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        "whole_response_metrics": None, "production_activation": False,
        "waiting_scope": "virtual public arrival schedule; fill/queue delays modeled, service measured",
        "private_material_policy": "fresh independent requests; no seeds, masks, tickets or correction hashes persisted"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed": len(samples), "failed": len(failures),
                      "unattempted": len(report["unattempted_cells"]), "status": report["status"]}))


if __name__ == "__main__":
    main()
