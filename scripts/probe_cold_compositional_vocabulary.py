"""Bounded synthetic additive tied vocabulary probe; no SDK/checkpoint loading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


CONTRACT = "fp16-centroids-f32-left-sum-scale-last-sequential-dot-v1"
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = [
    {"url": "https://arxiv.org/html/1711.01068v2", "bytes": 172496,
     "sha256": "98e8f501a382e4476c268c936560705d38a62dafa943a12fe44c1b4f117e17db"},
    {"url": "https://arxiv.org/abs/1608.05859v3", "bytes": 40343,
     "sha256": "f6fc1ef5c292f1000c6b7825c74c874e8e7ba7074db730844ac943365db3ef04"},
    {"url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/"
     "7ae557604adf67be50417f59c2c2f167def9a775/config.json", "bytes": 659,
     "sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"},
]


class Vocabulary:
    """Small research model. All public weights arrive before private token lookup."""

    def __init__(self, books, indices, scales):
        b, i, s = map(np.asarray, (books, indices, scales))
        if (b.ndim != 3 or b.dtype != np.float16 or i.ndim != 2
                or i.dtype != np.uint8 or s.dtype != np.float32):
            raise ValueError("require fp16 books, uint8 indices, f32 scales")
        m, k, h = b.shape
        v = len(i)
        if (not 1 <= m <= 8 or not 1 <= k <= 256 or not 1 <= h <= 128
                or not 1 <= v <= 1024 or i.shape != (v, m) or s.shape != (v,)
                or np.any(i >= k) or not np.all(np.isfinite(b))
                or not np.all(np.isfinite(s)) or np.any(s <= 0)):
            raise ValueError("invalid or unadmitted public geometry/numerics")
        self.books, self.indices, self.scales = [x.copy() for x in (b, i, s)]
        for x in (self.books, self.indices, self.scales):
            x.flags.writeable = False

    def lookup(self, tokens):
        t = np.asarray(tokens)
        if t.ndim != 1 or len(t) > 1024 or t.dtype.kind not in "iu" or np.any(t >= len(self.indices)) or np.any(t < 0):
            raise ValueError("invalid private token ids")
        result = np.zeros((len(t), self.books.shape[2]), np.float32)
        for m in range(len(self.books)):
            result = np.add(result, self.books[m, self.indices[t, m]].astype(np.float32))
        return np.multiply(result, self.scales[t, None])

    def logits(self, hidden):
        """Factor first, then gather all V scores. Not dense-f32 bit-equivalent."""
        x = np.asarray(hidden)
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != self.books.shape[2] or len(x) > 32 or not np.all(np.isfinite(x)):
            raise ValueError("invalid or unadmitted hidden states")
        scores = [dense_logits(b.astype(np.float32), x) for b in self.books]
        result = np.zeros((len(x), len(self.indices)), np.float32)
        for m in range(len(self.books)):
            result = np.add(result, scores[m][:, self.indices[:, m]])
        return np.multiply(result, self.scales[None, :])

    def artifact(self):
        payload = (self.books.astype("<f2").tobytes() + self.indices.tobytes()
                   + self.scales.astype("<f4").tobytes())
        descriptor = canonical({"contract": CONTRACT, "shape": list(self.books.shape),
                                "vocabulary": len(self.indices),
                                "payload_sha256": hashlib.sha256(payload).hexdigest()})
        # Demonstration framing, not existing runtime wire protocol or authentication.
        encoded = zlib.compress(payload, 1)
        frame = struct.pack(">II", len(payload), len(encoded)) + encoded
        return {"books_bytes": self.books.nbytes, "indices_bytes": self.indices.nbytes,
                "scales_bytes": self.scales.nbytes, "raw_bytes": len(payload),
                "descriptor_bytes": len(descriptor), "frame_bytes": len(frame),
                "charged_artifact_bytes": len(frame) + len(descriptor),
                "payload_sha256": hashlib.sha256(payload).hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dense_logits(weights, hidden):
    """Feature-order float32 dot; no BLAS-dependent reductions/FMA."""
    out = np.zeros((len(hidden), len(weights)), np.float32)
    for h in range(weights.shape[1]):
        out = np.add(out, np.multiply(hidden[:, h, None], weights[None, :, h]))
    return out


def fit_additive(target, *, components=4, centroids=8, iterations=6):
    """Greedy residual k-means, deterministic seed rows; intentionally tiny."""
    w = np.asarray(target)
    if (w.dtype != np.float32 or w.ndim != 2 or not 1 <= len(w) <= 128
            or not 1 <= w.shape[1] <= 32 or not np.all(np.isfinite(w))
            or not 1 <= components <= 4 or not 1 <= centroids <= min(16, len(w))
            or not 1 <= iterations <= 8):
        raise ValueError("fit admission exceeded")
    residual = w.copy()
    books, codes = [], []
    for _ in range(components):
        centers = residual[np.linspace(0, len(w) - 1, centroids, dtype=int)].copy()
        for _ in range(iterations):
            distance = np.sum((residual[:, None, :] - centers[None, :, :]) ** 2,
                              axis=2, dtype=np.float32)
            assignment = np.argmin(distance, axis=1)
            for k in range(centroids):
                selected = residual[assignment == k]
                if len(selected):
                    centers[k] = np.mean(selected, axis=0, dtype=np.float32)
        # Freeze delivered fp16 values, then reassign against delivered centroids.
        centers = centers.astype(np.float16)
        assignment = np.argmin(np.sum((residual[:, None, :] - centers.astype(np.float32)[None, :, :]) ** 2, axis=2), axis=1)
        residual = np.subtract(residual, centers[assignment].astype(np.float32))
        books.append(centers)
        codes.append(assignment.astype(np.uint8))
    return Vocabulary(np.array(books), np.stack(codes, axis=1), np.ones(len(w), np.float32))


def independent_dense(model):
    """Scalar reconstruction oracle independent of vectorized lookup."""
    v, h = len(model.indices), model.books.shape[2]
    out = np.empty((v, h), np.float32)
    for t in range(v):
        for j in range(h):
            value = np.float32(0)
            for m in range(len(model.books)):
                value = np.float32(value + np.float32(model.books[m, int(model.indices[t, m]), j]))
            out[t, j] = np.float32(value * model.scales[t])
    return out


def compare(model, hidden, target=None):
    dense = independent_dense(model)
    lookup = model.lookup(np.arange(len(dense)))
    expected, actual = dense_logits(dense, hidden), model.logits(hidden)
    result = {"lookup_values": dense.size, "lookup_bit_mismatches": int(np.count_nonzero(dense.view(np.uint32) != lookup.view(np.uint32))),
              "logit_values": expected.size,
              "factor_bit_mismatches": int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))),
              "factor_max_absolute_error": float(np.max(np.abs(expected.astype(np.float64) - actual))),
              "artifact": model.artifact()}
    if target is not None:
        source = dense_logits(target, hidden)
        result.update(reconstruction_rmse=float(np.sqrt(np.mean((dense.astype(np.float64) - target) ** 2))),
                      source_logit_max_absolute_error=float(np.max(np.abs(source.astype(np.float64) - actual))),
                      source_argmax_mismatches=int(np.count_nonzero(np.argmax(source, axis=1) != np.argmax(actual, axis=1))))
    return result


def run():
    rng = np.random.default_rng(20261014)
    indices = rng.integers(0, 8, (96, 4), dtype=np.uint8)
    scales = rng.choice(np.array([0.25, 0.5, 1, 2], np.float32), 96)
    exact = Vocabulary(rng.integers(-4, 5, (4, 8, 12)).astype(np.float16), indices, scales)
    hidden = rng.integers(-4, 5, (7, 12)).astype(np.float32)
    target = rng.normal(size=(96, 12)).astype(np.float32)
    fitted = fit_additive(target)
    random_model = Vocabulary(rng.normal(size=(4, 8, 12)).astype(np.float16), indices,
                              rng.uniform(0.1, 2, 96).astype(np.float32))
    # Deliberate cancellation: dense row is [1, 1], but summing large centroid
    # products loses low bits. Algebraic distribution is not float32 equivalence.
    cancellation = Vocabulary(np.array([[[65504, 0]], [[1, 1]], [[-65504, 0]]], np.float16),
                              np.zeros((1, 3), np.uint8), np.ones(1, np.float32))
    ch = np.array([[1e8, 1]], np.float32)
    # Tiny embedding RMSE still flips winner: absent low-margin residual.
    adversarial = Vocabulary(np.zeros((1, 1, 2), np.float16), np.zeros((2, 1), np.uint8), np.ones(2, np.float32))
    adv_source = np.array([[0, 0], [0.0001, 0]], np.float32)
    # Rounding is nonlinear; summing rounded codebooks is not rounding row sums.
    rounding = {"centroid_values": [0.5, 0.5], "round_each_then_sum": int(np.rint(0.5) + np.rint(0.5)),
                "sum_then_round": int(np.rint(1.0)), "rule": "ties-to-even"}
    paths = ["scripts/probe_cold_compositional_vocabulary.py", "tests/test_cold_compositional_vocabulary.py",
             "docs/research/cold-compositional-vocabulary-reference.md", "ARCHITECTURE.md",
             "docs/research/packed-embedding-reference.md", "docs/research/predictive-weight-codec-reference.md"]
    return {"schema": "cold-compositional-vocabulary-synthetic-v1", "seed": 20261014,
            "contract": CONTRACT, "numpy_version": np.__version__, "scope": "synthetic correctness only",
            "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
            "upstream_sources": UPSTREAM,
            "structured": compare(exact, hidden), "random_float_order": compare(random_model, hidden / np.float32(7)),
            "fit": compare(fitted, hidden / np.float32(7), target),
            "cancellation": compare(cancellation, ch),
            "adversarial": compare(adversarial, np.array([[1, 0]], np.float32), adv_source),
            "requantization": rounding,
            "qwen_structural_not_wire": {"vocabulary": 151936, "hidden": 896, "components": 4, "centroids": 256,
                "fp16_books_bytes": 1835008, "uint8_indices_bytes": 607744, "f32_scales_bytes": 607744,
                "raw_total_bytes": 3050496, "dense_i8_entries_bytes": 136134656,
                "dense_i8_plus_f32_scale_bytes": 136742400, "full_f32_logits_per_row_bytes": 607744,
                "expanded_dense_f32_bytes": 544538624, "expanded_books_f32_bytes": 3670016,
                "codebook_dot_macs_per_hidden": 917504, "dense_dot_macs_per_hidden": 136134656,
                "gather_adds_per_hidden": 607744},
            "unmeasured": {k: None for k in ["qwen_quality", "wire_savings", "whole_cold_bytes", "cpu_seconds", "gpu_seconds", "peak_rss", "latency", "training_cost"]},
            "synthetic_extra_array_envelope_bytes": 8 * 1024 * 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    report = run()
    if args.verify and json.loads(args.verify.read_text()) != report:
        raise SystemExit("evidence replay mismatch")
    if args.output:
        with args.output.open("x") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not args.output and not args.verify:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
