"""Research-only local W8/A8 head box certificates; no SDK or runtime changes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


CONTRACT = "w8-a8-exact-integer-f32-cast-mul-mul-add-first-tie-v1"


def frozen(array, dtype):
    value = np.asarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape)


def dequant(dot, activation_scale, scale, bias):
    # Separate ufuncs prohibit fused multiply-add and float64 intermediates.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.multiply(np.multiply(np.asarray(dot, np.float32),
                        np.float32(activation_scale)), np.asarray(scale, np.float32))
        if bias is not None:
            result = np.add(result, np.asarray(bias, np.float32))
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite dequant result: reject certificate domain")
    return result


@dataclass(frozen=True)
class Node:
    start: int
    stop: int
    low: np.ndarray
    high: np.ndarray
    scale_low: np.float32
    scale_high: np.float32
    bias_high: np.float32
    children: tuple


class HeadIndex:
    """Immutable private-copy public source. Storage figures are wire payloads only."""

    def __init__(self, weights, scales, bias=None, *, leaf_rows=8):
        w = np.asarray(weights)
        if (w.dtype != np.int8 or w.ndim != 2 or not 1 <= w.shape[0] <= 4096
                or not 1 <= w.shape[1] <= 16384 or w.nbytes > 16 << 20
                or not 1 <= leaf_rows <= 4096):
            raise ValueError("bounded signed-i8 matrix required")
        self.weights = frozen(w, np.int8)
        self.scales = frozen(scales, np.float32)
        self.has_bias = bias is not None
        self.bias = frozen(np.zeros(w.shape[0], np.float32) if bias is None else bias,
                           np.float32)
        if (self.scales.shape != (w.shape[0],) or self.bias.shape != self.scales.shape
                or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0)
                or not np.all(np.isfinite(self.bias))):
            raise ValueError("finite positive row scales and finite biases required")
        self.source_bytes = self.weights.nbytes + self.scales.nbytes + self.bias.nbytes
        self.original_source_bytes = self.weights.nbytes + self.scales.nbytes + (
            self.bias.nbytes if self.has_bias else 0)
        self.source_digest = hashlib.sha256(CONTRACT.encode() + bytes([self.has_bias]) + self.weights.tobytes()
                                + self.scales.tobytes() + self.bias.tobytes()).hexdigest()
        self.nodes = []

        def build(start, stop):
            children = ()
            if stop - start > leaf_rows:
                middle = (start + stop) // 2
                children = (build(start, middle), build(middle, stop))
            rows = self.weights[start:stop]
            node = Node(start, stop, frozen(rows.min(axis=0), np.int8),
                        frozen(rows.max(axis=0), np.int8), self.scales[start:stop].min(),
                        self.scales[start:stop].max(), self.bias[start:stop].max(), children)
            self.nodes.append(node)
            return node

        self.root = build(0, w.shape[0])
        # Proposed canonical wire: endpoints i8, 3 f32, 4 u32 geometry/children.
        self.index_payload_bytes = 52 + len(self.nodes) * (2 * w.shape[1] + 28)
        # Deliberately conservative operation accounting for this simple builder.
        self.build_weight_coordinate_visits = sum((n.stop - n.start) * w.shape[1]
                                                  for n in self.nodes)

    def serialize(self):
        """Public geometry, source binding and conservative metadata; no query data."""
        positions = {id(node): i for i, node in enumerate(self.nodes)}
        out = bytearray(b"PLLMHC01" + bytes.fromhex(self.source_digest))
        out.extend(struct.pack("<III", *self.weights.shape, len(self.nodes)))
        for node in self.nodes:
            child_ids = [positions[id(child)] for child in node.children] or [0xffffffff] * 2
            out.extend(struct.pack("<IIIIfff", node.start, node.stop, *child_ids,
                                   node.scale_low, node.scale_high, node.bias_high))
            out.extend(node.low.tobytes())
            out.extend(node.high.tobytes())
        assert len(out) == self.index_payload_bytes
        return bytes(out)

    def upper(self, node, query, activation_scale):
        x = query.astype(np.int64)
        lo = sum(int(a) * int(b) for a, b in zip(np.where(x >= 0, node.low, node.high), x))
        hi = sum(int(a) * int(b) for a, b in zip(np.where(x >= 0, node.high, node.low), x))
        # Positive activation scale; sign of dot means scale extrema cannot simply
        # be assigned high-to-high. All four corners are conservative and monotone.
        values = dequant(np.array([lo, lo, hi, hi], np.int64), activation_scale,
                        np.array([node.scale_low, node.scale_high] * 2, np.float32),
                        node.bias_high)
        return np.max(values)

    def query(self, query, activation_scale=1.0, *, max_bound_nodes=None):
        x = np.asarray(query)
        a = np.float32(activation_scale)
        if (x.dtype != np.int8 or x.shape != (self.weights.shape[1],)
                or not np.isfinite(a) or a <= 0
                or (max_bound_nodes is not None and max_bound_nodes < 0)):
            raise ValueError("bounded signed-i8 query and positive finite f32 scale required")
        # Reject possible overflow for ANY row before pruning, not merely tested rows.
        magnitude = sum(abs(int(v)) * 128 for v in x)
        dequant(np.array([-magnitude, magnitude], np.int64), a,
                self.scales.max(), np.max(np.abs(self.bias)))
        tested, skipped, bound_nodes = [], [], 0
        best_row, best_value = 0, None

        def exact(row):
            return dequant(sum(int(w) * int(v) for w, v in zip(self.weights[row], x)),
                           a, self.scales[row], self.bias[row] if self.has_bias else None).item()

        best_value = exact(0)
        tested.append(0)
        stack = [self.root]
        fallback = False
        while stack:
            node = stack.pop()
            if max_bound_nodes is not None and bound_nodes >= max_bound_nodes:
                fallback = True
                break
            upper = self.upper(node, x, a)
            bound_nodes += 1
            if upper < best_value or (upper == best_value and node.start > best_row):
                skipped.append((node.start, node.stop, float(upper), best_row, best_value))
                continue
            if node.children:
                stack.extend(reversed(node.children))
            else:
                for row in range(node.start, node.stop):
                    if row == 0:
                        continue
                    value = exact(row)
                    tested.append(row)
                    if value > best_value or (value == best_value and row < best_row):
                        best_row, best_value = row, value
        extra_full_rows = 0
        if fallback:
            # Explicit full exact recomputation; prior index/test work remains charged.
            scores = [exact(row) for row in range(self.weights.shape[0])]
            extra_full_rows = len(scores)
            best_row = max(range(len(scores)), key=lambda row: (scores[row], -row))
            best_value = scores[best_row]
            skipped = []
        return dict(winner=best_row, score=best_value, tested_rows=tested,
                    skipped=skipped, bound_nodes=bound_nodes, fallback=fallback,
                    exact_row_evaluations=len(tested) + extra_full_rows,
                    bound_coordinate_visits=2 * bound_nodes * self.weights.shape[1],
                    domain_check_coordinate_visits=self.weights.shape[1],
                    domain_check_scale_bias_visits=2 * self.weights.shape[0],
                    bound_endpoint_dequants=4 * bound_nodes)


def verify_oracle(index, x, activation_scale, result):
    """Independent vectorized full scan verifies EACH skipped row and exact tie rule."""
    dots = index.weights.astype(np.int64) @ x.astype(np.int64)
    scores = dots.astype(np.float32)
    scores = (scores * np.float32(activation_scale)).astype(np.float32)
    scores = (scores * index.scales).astype(np.float32)
    if index.has_bias:
        scores = (scores + index.bias).astype(np.float32)
    assert np.all(np.isfinite(scores))
    assert result["winner"] == int(np.argmax(scores))
    assert np.float32(result["score"]).tobytes() == scores[result["winner"]].tobytes()
    verified = 0
    for start, stop, upper, incumbent, value in result["skipped"]:
        for row in range(start, stop):
            assert scores[row] <= upper
            assert scores[row] < value or (scores[row] == value and row > incumbent)
            verified += 1
    return verified


def source_locks():
    root = Path(__file__).resolve().parents[1]
    paths = ["crates/pllm-core/src/offset_transport.rs", "crates/pllm-core/src/progressive_head.rs",
             "crates/pllm-compiler/src/lib.rs", "crates/pllm-compiler/src/decoder_continuation.rs",
             "python/pllm/runtime/quantization.py", "python/pllm/runtime/conversation_checkpoint.py",
             "python/pllm/runtime/model_binding.py",
             "python/pllm/runtime/public_prefix.py", "python/pllm/runtime/tiered_execution.py"]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screen(payload):
    locks_before = source_locks()
    rng = np.random.default_rng(501)
    source = {"kind": "tiny synthetic, no timing or checkpoint scan"}
    if payload.get("weights"):
        rows, columns = payload["rows"], payload["columns"]
        if not 1 <= columns <= 16384 or not 1 <= rows <= 262144:
            raise ValueError("source geometry exceeds bounded domain")
        for name, size in (("weights", rows * columns), ("scales", rows * 4)):
            path = Path(payload[name])
            if path.stat().st_size != size or file_digest(path) != payload[name + "_sha256"]:
                raise ValueError("public raw artifact source lock differs")
        count = min(rows, payload["scan_rows"])
        if not 1 <= count <= 256 or count * columns > 4 << 20:
            raise ValueError("bounded prefix scan exceeds 4 MiB")
        w = np.memmap(payload["weights"], mode="r", dtype=np.int8,
                      shape=(rows, columns))[:count].copy()
        s = np.memmap(payload["scales"], mode="r", dtype="<f4", shape=(rows,))[:count].copy()
        datasets = [("public-row-prefix-not-global-vocabulary", w, s)]
        source = {"kind": "source-locked raw W8/f32 public prefix; synthetic queries only",
                  "weights_sha256": payload["weights_sha256"],
                  "scales_sha256": payload["scales_sha256"], "admitted_rows": count,
                  "full_artifact_delivery_bytes": rows * (columns + 4),
                  "source_verification_bytes": rows * (columns + 4)}
    else:
        random = rng.integers(-128, 128, (64, 64), dtype=np.int8)
        easy = np.zeros((64, 64), np.int8)
        easy[0] = 127
        easy[1:] = -64
        datasets = [("separated", easy, np.ones(64, np.float32)),
                    ("high-dimensional-random-box", random, np.ones(64, np.float32))]
    cases = []
    for name, w, s in datasets:
        index = HeadIndex(w, s, leaf_rows=8)
        public_index = index.serialize()
        queries = [np.ones(w.shape[1], np.int8),
                   rng.integers(-128, 128, w.shape[1], dtype=np.int8)]
        for x in queries:
            result = index.query(x, 0.03125)
            verified = verify_oracle(index, x, 0.03125, result)
            # No private queries, candidate IDs, page decisions, or logits in reports.
            cases.append(dict(fixture=name, rows=w.shape[0], columns=w.shape[1],
                tested_rows=len(result["tested_rows"]), skipped_rows_verified=verified,
                **{k: result[k] for k in ("bound_nodes", "fallback", "exact_row_evaluations",
                    "bound_coordinate_visits", "domain_check_coordinate_visits",
                    "domain_check_scale_bias_visits", "bound_endpoint_dequants")},
                exact_dot_macs=result["exact_row_evaluations"] * w.shape[1],
                full_head_macs=w.size, oracle_macs=w.size,
                oracle_weight_bytes=w.nbytes, oracle_logits_bytes=4 * w.shape[0],
                index_payload_bytes=index.index_payload_bytes,
                index_sha256=hashlib.sha256(public_index).hexdigest(),
                index_build_coordinate_visits=index.build_weight_coordinate_visits,
                immutable_source_bytes=index.source_bytes,
                original_head_delivery_bytes=index.original_source_bytes,
                source_admission_hash_bytes=index.source_bytes,
                exact_weight_read_bytes=result["exact_row_evaluations"] * w.shape[1],
                exact_scale_bias_read_bytes=result["exact_row_evaluations"] * (8 if index.has_bias else 4),
                domain_scan_read_bytes=12 * w.shape[0],
                oracle_int64_weight_scratch_bytes=8 * w.size,
                full_head_float32_materialization_bytes=4 * w.shape[0],
                index_publication_delivery_bytes=index.index_payload_bytes,
                native_or_full_logit_parity_claim=False))
    if source_locks() != locks_before:
        raise RuntimeError("native/cache source changed during screen")
    return dict(schema="pllm.local_head_certificate_screen.v1", contract=CONTRACT,
        source=source, source_locks=locks_before,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases,
        limitations=["Scalar greedy screen only; complete logits and cache contract unchanged.",
            "No timings, native promotion, tier qualification, or network savings claim.",
            "Python resident allocations and serialization overhead exceed payload figures.",
            "Full public artifact fetched; no data-dependent provider retrieval."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-lock", type=Path,
                        help="Require the source, driver and installed build from a prior report")
    parser.add_argument("--weights")
    parser.add_argument("--scales")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--scales-sha256")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--scan-rows", type=int, default=128)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(screen(json.load(sys.stdin)), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output required")
    if args.weights and not all((args.scales, args.weights_sha256, args.scales_sha256,
                                  args.rows, args.columns)):
        parser.error("raw scan requires both files, SHA256 locks, rows and columns")
    if not args.weights and any((args.scales, args.weights_sha256, args.scales_sha256,
                                args.rows, args.columns)):
        parser.error("partial raw scan arguments cannot select synthetic mode")
    from frontier_probe_support import fresh_workers
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    locked = None
    if args.source_lock:
        locked = json.loads(args.source_lock.read_text())
        if (locked["source_locks"] != source_locks()
                or locked["driver_sha256"] != file_digest(__file__)):
            raise RuntimeError("research screen source lock changed")
    with fresh_workers(Path(__file__).resolve(), memory_bytes=128 << 20, timeout=120) as (run, _, _):
        if locked is not None and locked["installed_build"] != run.build:
            raise RuntimeError("research screen installed build changed")
        report = run(vars(args) | {"output": str(args.output), "source_lock": None})
        report["installed_build"] = run.build
        report["source_revision"] = revision
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != revision:
        raise RuntimeError("source revision changed during screen")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
