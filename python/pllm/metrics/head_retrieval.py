"""Public compressed-index / private original-row retrieval quality gate."""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrivateHeadRetrievalProbe:
    """Public-weight fit and exact greedy certificates; not a serving component.

    Rust owns query projection, quantization, GEMM, bounds, ranking and private
    page execution. Python owns the offline public fit and independent oracle.
    """

    rank: int = 8
    candidate_counts: tuple[int, ...] = (1, 8, 32)
    fit_rows: int = 2048
    page_records: int = 8

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.rank, 1, 256),
            (self.fit_rows, 1, 8192),
            (self.page_records, 1, 64),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("head retrieval configuration exceeds bounded policy")
        if (
            type(self.candidate_counts) is not tuple
            or not 1 <= len(self.candidate_counts) <= 8
            or any(type(k) is not int or not 1 <= k <= 4096 for k in self.candidate_counts)
            or tuple(sorted(set(self.candidate_counts))) != self.candidate_counts
        ):
            raise ValueError("candidate counts must be increasing bounded integers")
        if self.fit_rows < self.rank:
            raise ValueError("public fit needs at least rank rows")

    def run(
        self, *, weights=None, scales=None, inputs=None, input_scales=None, public_calibration=None
    ) -> dict:
        import numpy as np

        from pllm import _native
        from pllm.native import MaskedGEMM
        from pllm.runtime.quantization import dequantize_matmul, quantize_weight_per_row

        if weights is None and scales is None and inputs is None and input_scales is None:
            rng = np.random.default_rng(319)
            weights = rng.integers(-127, 128, (64, 32), dtype=np.int8)
            scales = np.full(64, 0.01, np.float32)
            inputs = rng.integers(-127, 128, (4, 32), dtype=np.int8)
            input_scales = np.full(4, 0.01, np.float32)
        if (
            not isinstance(weights, np.ndarray)
            or weights.ndim != 2
            or weights.dtype != np.int8
            or not isinstance(inputs, np.ndarray)
            or inputs.ndim != 2
            or inputs.dtype != np.int8
        ):
            raise ValueError("head weights and quantized queries must be rank-two int8 arrays")
        rows, features = weights.shape
        if (
            not 1 <= rows <= 262144
            or not 1 <= features <= 4096
            or rows * features > 256 * 1024**2
            or not 1 <= len(inputs) <= 128
            or inputs.shape[1] != features
            or self.rank > min(rows, features)
            or self.candidate_counts[-1] > rows
            or self.page_records * (features + 4) > 65536
        ):
            raise ValueError("head data exceed bounded shape/resource policy")
        scales = np.asarray(scales, dtype=np.float32)
        input_scales = np.asarray(input_scales, dtype=np.float32)
        if (
            scales.shape != (rows,)
            or input_scales.shape != (len(inputs),)
            or not np.all(np.isfinite(scales))
            or not np.all(np.isfinite(input_scales))
            or np.any(scales <= 0)
            or np.any(input_scales <= 0)
            or max(float(scales.max()), float(input_scales.max())) > 1e6
        ):
            raise ValueError("head scales must be positive finite bounded row vectors")

        # No prompt, query or held-out winner participates in this public fit.
        start = time.process_time_ns()
        rng = np.random.default_rng(7309)
        calibration_digest = None
        if public_calibration is None:
            sample_ids = rng.choice(rows, min(rows, self.fit_rows), replace=False)
            sample = weights[sample_ids].astype(np.float64) * scales[sample_ids, None]
        else:
            if (
                not isinstance(public_calibration, np.ndarray)
                or public_calibration.ndim != 2
                or public_calibration.dtype.kind != "f"
                or public_calibration.shape[1] != features
                or not self.rank <= len(public_calibration) <= 8192
                or not np.all(np.isfinite(public_calibration))
                or np.max(np.abs(public_calibration)) > 1e6
            ):
                raise ValueError("public calibration exceeds bounded index contract")
            calibration_digest = hashlib.sha256(
                public_calibration.astype("<f8").tobytes()
            ).hexdigest()
            sample_ids = rng.choice(
                len(public_calibration), min(len(public_calibration), self.fit_rows), replace=False
            )
            sample = public_calibration[sample_ids].astype(np.float64)
        width = min(self.rank + 8, *sample.shape)
        omega = rng.normal(size=(features, width))
        q, _ = np.linalg.qr(sample @ omega, mode="reduced")
        for _ in range(2):
            q, _ = np.linalg.qr(sample @ (sample.T @ q), mode="reduced")
        _, _, vt = np.linalg.svd(q.T @ sample, full_matrices=False)
        basis = np.ascontiguousarray(vt[: self.rank].T)
        basis_norm = float(np.linalg.norm(basis))
        quantized = np.empty((rows, self.rank), np.int8)
        profiles = np.empty((rows, 3), dtype="<f8")
        for first in range(0, rows, 512):
            end = min(first + 512, rows)
            source = weights[first:end].astype(np.float64) * scales[first:end, None]
            exact_coefficients = source @ basis
            compressed = quantize_weight_per_row(exact_coefficients, bits=8)
            quantized[first:end] = compressed.values
            reduced = compressed.values.astype(np.float64) * compressed.scales[:, None]
            residual = exact_coefficients - reduced
            source_norm = np.linalg.norm(source, axis=1)
            padding = 1e-10 * (source_norm * basis_norm + np.linalg.norm(reduced, axis=1) + 1)
            profiles[first:end, 0] = compressed.scales
            profiles[first:end, 1] = np.nextafter(
                np.linalg.norm(residual, axis=1) + padding, np.inf
            )
            profiles[first:end, 2] = np.nextafter(source_norm + padding, np.inf)
            del source, residual, reduced, compressed, exact_coefficients
        projection_bytes = basis.T.astype("<f8").tobytes()
        profile_bytes, index_weights = profiles.tobytes(), quantized.tobytes()
        index = _native.HeadRetrievalIndex(
            rows, features, self.rank, index_weights, projection_bytes, profile_bytes
        )
        index_digest = hashlib.sha256(index_weights + projection_bytes + profile_bytes).hexdigest()
        fit_cpu = (time.process_time_ns() - start) / 1e9
        index_wire_bytes = len(index_weights) + len(projection_bytes) + len(profile_bytes)
        del sample, q, vt, omega, quantized, profiles, basis
        del profile_bytes, projection_bytes, index_weights

        kernel = MaskedGEMM(threads=1).compile(weights)
        controls, reference_cpu = [], []
        for query, scale in zip(inputs, input_scales, strict=True):
            start = time.process_time_ns()
            scores = dequantize_matmul(kernel.clear(query[None, :]), np.asarray([scale]), scales)[0]
            reference_cpu.append((time.process_time_ns() - start) / 1e9)
            if not np.all(np.isfinite(scores)):
                raise ValueError("reference head produced nonfinite scores")
            controls.append(scores)

        candidates = []
        selected = None
        for count in self.candidate_counts:
            agreement = certified = 0
            times = []
            for query, scale, scores in zip(inputs, input_scales, controls, strict=True):
                start = time.process_time_ns()
                raw, outside_upper = index.candidates(query.tobytes(), float(scale), count)
                ids = np.frombuffer(raw, "<u4")
                times.append((time.process_time_ns() - start) / 1e9)
                winner = int(ids[np.lexsort((ids, -scores[ids]))[0]])
                expected = int(np.argmax(scores))
                omitted = np.ones(rows, dtype=bool)
                omitted[ids] = False
                if np.any(omitted) and float(scores[omitted].max()) > outside_upper:
                    raise AssertionError("compressed head residual bound missed a source row")
                has_certificate = bool(float(scores[winner]) > outside_upper)
                if has_certificate and winner != expected:
                    raise AssertionError("head certificate selected the wrong source winner")
                agreement += winner == expected
                certified += has_certificate
                if selected is None:
                    selected = int(ids[0])
            candidates.append(
                {
                    "candidate_count": count,
                    "query_rows": len(inputs),
                    "winner_agreement": agreement,
                    "certified_winners": certified,
                    "native_index_cpu_seconds_median": statistics.median(times),
                }
            )

        # One original-page execution; full fixed-budget totals are projections.
        record_bytes = features + 4
        records = np.empty((rows, record_bytes), np.uint8)
        records[:, :features] = weights.view(np.uint8)
        records[:, features:] = scales.astype("<f4").view(np.uint8).reshape(rows, 4)
        table = records.tobytes()
        page_width = record_bytes * self.page_records
        padded = table + bytes((-len(table)) % page_width)
        a, b = (_native.PrivatePageServer(padded, page_width, party) for party in (0, 1))
        assert selected is not None
        qa, qb, decoder = _native.private_page_issue(*a.descriptor(), selected // self.page_records)
        start = time.process_time_ns()
        ra, rb = a.evaluate(qa), b.evaluate(qb)
        worker_cpu = (time.process_time_ns() - start) / 1e9
        page = decoder.decode(ra, rb)
        offset = selected % self.page_records * record_bytes
        if (
            page[offset : offset + record_bytes]
            != table[selected * record_bytes : (selected + 1) * record_bytes]
        ):
            raise AssertionError("private head page differs from original weights/scales")
        body_bytes = len(qa) + len(qb) + len(ra) + len(rb)
        for item in candidates:
            item["fixed_budget_head_body_bytes_per_token"] = item["candidate_count"] * body_bytes
        return {
            "schema": "pllm.private_head_retrieval_probe.v1",
            "scope": "same-hidden W8A8 greedy head screen; one native private original-page sample",
            "source_digest": hashlib.sha256(
                weights.tobytes() + scales.astype("<f4").tobytes()
            ).hexdigest(),
            "index_digest": index_digest,
            "rows": rows,
            "features": features,
            "rank": self.rank,
            "bound_contract": "pllm.head_index_query_residual.v1",
            "projection_source": "public_weight_rows"
            if public_calibration is None
            else "public_query_calibration",
            "public_calibration_digest": calibration_digest,
            "public_index_fit_cpu_seconds": fit_cpu,
            "client_index_payload_bytes": index.payload_bytes,
            "index_distribution_body_bytes": index_wire_bytes,
            "original_table_bytes": len(table),
            "reference_head_cpu_seconds_median": statistics.median(reference_cpu),
            "private_page_body_bytes": body_bytes,
            "private_page_worker_cpu_seconds": worker_cpu,
            "private_page_exact": True,
            "all_checked_bounds_valid": True,
            "candidates": candidates,
            "whole_decoder_executable": False,
            "limitations": [
                "Public-weight index fit, no model training; trusted offline residual bounds require source binding",
                "Candidate membership and inputs stay client-local; fresh DPF shares hide fixed-budget page indices",
                "Projected retrieval pads duplicate pages to fixed counts; data-dependent early stopping is not admitted",
                "Index CPU excludes private page transfer, candidate re-evaluation and sampling",
                "Only greedy winner certificates; no temperature/top-p or whole-generation quality claim",
                "Original source head is fully evaluated for this oracle; fixture memory is not client peak RSS",
                "Full wire, independent parties and deployment source distribution remain unmeasured",
            ],
        }
