"""One-use, party-local masked relay screen; includes the displaced peer traffic."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import statistics
import time


@dataclass(frozen=True, slots=True)
class MaskedAggregationProbe:
    """Measure a native output boundary, not matrix execution or HTTP transport.

    B adds an independent client-known output mask before relaying its share to
    A. Each role burns before malformed Python arguments can be converted.
    """

    repetitions: int = 3

    def __post_init__(self):
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 32:
            raise ValueError("repetitions must be in [1, 32]")

    def run(self, share_a, share_b, widths) -> dict:
        import numpy as np
        from pllm import _native

        a, b = np.asarray(share_a), np.asarray(share_b)
        widths = bytes(widths)
        if (a.dtype != np.uint32 or b.dtype != np.uint32 or a.ndim != 2
                or a.shape != b.shape or a.shape[1] != len(widths) or not 1 <= a.size <= 4_000_000):
            raise ValueError("expected equal bounded u32 worker matrices and public column widths")
        rows = int(a.shape[0])
        raw_a, raw_b = a.astype("<u4", copy=False).tobytes(), b.astype("<u4", copy=False).tobytes()
        pa = _native.offset_pack_rows(raw_a, widths, rows)
        pb = _native.offset_pack_rows(raw_b, widths, rows)
        context = hashlib.sha256(b"pllm.masked-aggregation.probe.v1\0" + widths + rows.to_bytes(8, "little")).digest()
        control_cpu, client_cpu, worker_cpu = [], [], []
        for _ in range(self.repetitions):
            started = time.process_time_ns()
            expected = _native.offset_reconstruct_rows(pa, pb, widths, rows)
            control_cpu.append((time.process_time_ns() - started) / 1e9)
            started = time.process_time_ns()
            worker, relay, client = _native.output_aggregation_issue(context, widths, rows)
            issuance_cpu = (time.process_time_ns() - started) / 1e9
            started = time.process_time_ns()
            peer = worker.apply(raw_b)
            response = relay.combine(raw_a, peer)
            worker_cpu.append((time.process_time_ns() - started) / 1e9)
            started = time.process_time_ns()
            actual = client.finish(response)
            client_cpu.append(issuance_cpu + (time.process_time_ns() - started) / 1e9)
            if actual != expected:
                raise AssertionError("masked relay changed reconstructed integers")
        rings = np.left_shift(np.uint64(1), np.frombuffer(widths, dtype=np.uint8).astype(np.uint64))
        values = (a.astype(np.uint64) + b.astype(np.uint64)) % rings
        signed = values.astype(np.int64) - np.where(values >= rings // 2, rings, 0).astype(np.int64)
        if not np.array_equal(np.frombuffer(actual, dtype="<i8").reshape(a.shape), signed):
            raise AssertionError("masked relay differs from independent ring oracle")
        direct = len(pa) + len(pb) + 2 * 41
        issuance = 64  # Fresh seed and public nonce; additional control is unmeasured.
        total = len(peer) + len(response) + issuance
        return {"schema": "pllm.masked_aggregation_probe.v1", "complete": False,
            "executable": False, "scope": "bounded in-process output boundary; no HTTP or input/matrix costs",
            "rows": rows, "columns": a.shape[1], "exact_integer_parity": True,
            "direct_client_download_bytes": direct, "relay_client_download_bytes": len(response),
            "worker_b_to_a_bytes": len(peer), "fresh_issuance_minimum_bytes": issuance,
            "direct_all_link_bytes": direct, "relay_all_link_minimum_bytes": total,
            "all_link_reduction": 1 - total / direct,
            "control_client_cpu_seconds_median": statistics.median(control_cpu),
            "relay_client_cpu_seconds_median": statistics.median(client_cpu),
            "extra_worker_cpu_seconds_median": statistics.median(worker_cpu),
            "client_peak_memory_bytes": None, "full_wire_bytes": None,
            "all_link_gate_passed": total < direct,
            "privacy_scope": "independent fresh AES mask; ideal small-ring view test; no independent cryptographic review"}
