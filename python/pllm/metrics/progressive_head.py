"""Certified greedy bitplanes with explicit private-refinement cost gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import statistics
import time


@dataclass(frozen=True, slots=True)
class ProgressiveHeadProbe:
    """Exact W8 greedy decisions; sampling distributions need a separate contract.

    The native oracle retains the full matrix. Sparse refinement counts are
    input-dependent and cannot be exposed to providers under the baseline
    privacy contract. A measured one-record PIR control prices the alternative.
    """

    prefix_bits: int = 5
    measure_private_lookup: bool = True

    def __post_init__(self):
        if type(self.prefix_bits) is not int or not 1 <= self.prefix_bits <= 7:
            raise ValueError("prefix_bits must be in [1,7]")
        if type(self.measure_private_lookup) is not bool:
            raise ValueError("measure_private_lookup must be boolean")

    def run(self, weights, scales, queries, activation_scales) -> dict:
        import numpy as np
        from pllm import _native
        from pllm.runtime.native import MaskedGEMM
        from pllm.runtime.quantization import dequantize_matmul
        from .private_pages import PrivatePageLookupProbe

        w, x = np.asarray(weights), np.asarray(queries)
        s, a = np.asarray(scales, dtype=np.float32), np.asarray(activation_scales, dtype=np.float32)
        if (w.dtype != np.int8 or x.dtype != np.int8 or w.ndim != 2 or x.ndim != 2
                or not 1 <= x.shape[0] <= 32 or x.shape[1] != w.shape[1]
                or s.shape != (w.shape[0],) or a.shape != (x.shape[0],)):
            raise ValueError("head probe requires bounded i8 matrices and matching per-row f32 scales")
        head = _native._ProgressiveHead(w.tobytes(), s.tolist(), w.shape[1])
        clear = MaskedGEMM(threads=1).compile(w)
        rows, times, control_times = [], [], []
        for query, scale in zip(x, a, strict=True):
            started = time.process_time_ns()
            scores = dequantize_matmul(clear.clear(query[None, :]), np.array([scale], np.float32),
                s, output_shape=(1, len(s)))[0]
            expected = int(np.argmax(scores))
            control_times.append((time.process_time_ns() - started) / 1e9)
            started = time.process_time_ns()
            winner, rounds = head.query(query.tobytes(), float(scale), self.prefix_bits)
            times.append((time.process_time_ns() - started) / 1e9)
            if winner != expected:
                raise AssertionError("progressive head violated exact greedy certificate")
            rows.append({"certificate_bits": rounds[-1][0], "initial_survivors": rounds[0][2],
                "rounds": [{"bits": b, "scanned_rows": n, "surviving_rows": k} for b, n, k in rounds]})
        record_bytes = (w.shape[1] * (8 - self.prefix_bits) + 7) // 8
        private = None
        if self.measure_private_lookup:
            private = PrivatePageLookupProbe(records=w.shape[0], record_bytes=record_bytes,
                queries=1).run(head.residual_table(self.prefix_bits))
        max_refinement = max(row["initial_survivors"] if row["certificate_bits"] > self.prefix_bits else 0
                             for row in rows)
        return {"schema": "pllm.progressive_head_probe.v1", "complete": False, "executable": False,
            "scope": "local exact greedy certificate and isolated native PIR control",
            "weight_digest": hashlib.sha256(w.tobytes()).hexdigest(), "vocabulary": w.shape[0],
            "hidden": w.shape[1], "queries": len(rows), "exact_greedy_agreements": len(rows),
            "prefix_bits": self.prefix_bits, "certificates": rows,
            "packed_prefix_weight_bytes": (w.size * self.prefix_bits + 7) // 8,
            "full_weight_bytes": w.nbytes, "oracle_retained_native_weight_bytes": w.nbytes,
            "weight_scale_bytes": s.nbytes, "full_residual_table_bytes": record_bytes * w.shape[0],
            "control_head_cpu_seconds_median": statistics.median(control_times),
            "progressive_head_cpu_seconds_median": statistics.median(times),
            "observed_maximum_refinement_rows": max_refinement,
            "public_capacity_certificate": False,
            "private_lookup_one_record_control": private,
            "projected_maximum_sparse_pir_bytes": None if private is None else max_refinement * private["body_bytes_per_query"],
            "projected_maximum_sparse_pir_worker_cpu_seconds": None if private is None else
                max_refinement * sum(private["worker_cpu_seconds_per_query_median"]),
            "privacy_safe_full_tail_delivery_bytes": record_bytes * w.shape[0],
            "client_peak_memory_bytes": None, "full_wire_bytes": None,
            "admission": "closed: input-dependent refinement count; no public capacity certificate or cheaper admitted private fetch"}
