"""Source-shaped installed CorrectionBatch target with independent integer checks."""
from __future__ import annotations

import secrets
import time

import numpy as np

from pllm._benchmark_contract import TargetUnavailable, digest
from pllm._benchmark_worker import peak_rss


def execute_batch(p, m):
    from pllm.native import MaskedGEMM
    from pllm.preparation import CorrectionBatch, PreparationRequest
    from .benchmark_targets import model_context, _select, _weight_file, _oracle
    from .preparation_protocol import prepared_mask_rows, seeded_ring_profile
    from .stage_protocol import unpack_residues

    context = model_context(p, m)
    stage = _select(context[3], p)
    if context[4].verification_component is not None:
        raise TargetUnavailable("verified_batch_target_not_implemented")
    rows, width, out, batch = p["rows"], stage.in_features, stage.out_features, p["batch"]
    count, window = batch["inventories"], batch["window_rows"]
    if (rows > context[0].budget.max_input_tokens or rows * max(width, out) > 4_000_000
            or 4 * count * rows * out > 64 << 20 or 32 * window * (width + out) > 64 << 20
            or width * out > 128 << 20):
        raise TargetUnavailable("preparation_batch_allocation_bound", status="blocked")
    if count * rows * width * out * (m.sampling.repetitions + m.sampling.warmups) > 8_000_000_000:
        raise TargetUnavailable("preparation_batch_work_bound", status="blocked")
    if 4 * count * rows * out * (m.sampling.repetitions + m.sampling.warmups) > 256 << 20:
        raise TargetUnavailable("preparation_batch_audit_disk_bound", status="blocked")
    gpu_bytes = width * out if p["kernel"] in {"metal", "adaptive"} else 0
    m.admit((224 << 20) + 4 * width * out + 32 * window * (width + out)
            + 2 * gpu_bytes + ((224 << 20) if p["kernel"] == "digit8_native" else 0))
    path, weight_hash, bound = _weight_file(stage, p, context, m)
    weights = np.memmap(path, dtype=np.int8, mode="r", shape=(out, width))
    cpu, wall = time.process_time(), time.perf_counter()
    executor = MaskedGEMM(threads=m.sampling.threads)
    if not executor.native:
        raise TargetUnavailable("native_extension_required")
    matrix = executor.compile(weights)
    m.details["kernel_import"] = {"cpu_seconds": time.process_time() - cpu,
                                   "wall_seconds": time.perf_counter() - wall}
    profile = seeded_ring_profile(bound)

    def execute(index):
        requests = [PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), "benchmark-stage",
            context[2].digest, stage.id, weight_hash, rows, width, out, stage.weight_bits, stage.activation_bits,
            bound, profile.ring, profile.modulus, profile.wire_bits, secrets.token_bytes(32)) for _ in range(count)]
        destination = m.directory / f"batch-{index}.bin"
        with CorrectionBatch(matrix, kernel="digit8" if p["kernel"] == "digit8_native" else p["kernel"],
                             window_rows=window, min_gpu_rows=batch["min_gpu_rows"]) as factory:
            tickets = [factory.add(request) for request in requests]
            groups = []
            for start in range(0, count, batch["batch_size"]):
                groups.append(factory.execute(tickets[start:start + batch["batch_size"]]))
            with destination.open("xb") as sink:
                for ticket in tickets:
                    for start in range(0, rows, 128):
                        sink.write(ticket.take_rows(min(128, rows - start)))
            return destination, requests, {"groups": groups,
                "allocation_bytes_estimate": factory.allocation_bytes_estimate}

    observed = m.run(execute)
    m.details.update(target_peak_rss_bytes=peak_rss(), target_cpu_from_birth_seconds=time.process_time())
    oracle_cpu = time.process_time()
    checked = 0
    for destination, requests, _ in observed:
        with destination.open("rb") as source:
            for request in requests:
                cursor = prepared_mask_rows(request)
                try:
                    for start in range(0, rows, 8):
                        take = min(8, rows - start)
                        r, s = cursor.take(start, take)
                        r = np.frombuffer(r, "<u4").reshape(take, width)
                        s = np.frombuffer(s, "<u4").reshape(take, out)
                        expected = (_oracle(weights, r, profile.modulus) - s) & np.uint32(profile.modulus - 1)
                        payload = source.read(take * out * (profile.wire_bits // 8))
                        actual = unpack_residues(payload, expected.shape, profile.wire_bits)
                        if not np.array_equal(actual, expected):
                            raise TargetUnavailable("preparation_batch_integer_parity_failed", status="failed")
                        checked += actual.size
                finally:
                    cursor.cancel()
            if source.read(1):
                raise ValueError("extra correction payload")
    m.oracle = {"kind": "independent_signed_i64_with_fresh_mask_cursors", "outcome": "passed", "checked_values": checked}
    m.details.update(batch=batch, kernel=p["kernel"], ring=profile.to_dict(),
        samples=[row[2] for row in observed], oracle_cpu_seconds=time.process_time() - oracle_cpu,
        cpu_weight_bytes=width * out, gpu_weight_bytes=gpu_bytes,
        exported_file_bytes=count * rows * out * (profile.wire_bits // 8),
        retained_audit_file_bytes=count * rows * out * (profile.wire_bits // 8) *
                                 (m.sampling.repetitions + m.sampling.warmups))
    m.comparison.update(workload_digest=digest({"shape": [count, rows, width, out], "seed": p["seed"]}),
        output_digest=m.commitment({"algebra": "W-r-minus-s", "weight": weight_hash,
                                   "ring": profile.to_dict(), "shape": [count, rows, out]}),
        measurement_digest=digest({"scope": "local_batch_import_expand_kernel_attribution_seal_export_retire_v1"}))
    m.metric("batch.correction_bytes", count * rows * out * (profile.wire_bits // 8), kind="bytes",
             origin="deterministic", scope="unframed_arithmetic_payload")
    m.metric("batch.gpu_weight_bytes", gpu_bytes, kind="allocated", scope="gpu_snapshot_payload")
    m.limitations += ["Single source-selected stage with independent one-use inventories; no provider queue, arrival waiting or HTTP transport.",
        "Factory construction, GPU snapshot, fresh masks, kernel, local attribution, sealing, export and retirement are timed; CPU matrix import is separately reported.",
        "Independent integer validation follows peak capture; retained test-only seed copies are never reissued to an execution.",
        "Logical local frames, correction files, CPU/GPU weights and process RSS are distinct owners; no full-response throughput claim."]
