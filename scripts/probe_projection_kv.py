"""Bounded, non-production quantization-origin KV bit-packing reference.

No checkpoint loading, model execution, timing cohort or SDK component registration.
Captured coefficients are mandatory: recomputing RoPE on smaller arrays is not
assumed bit-identical to the original phase's libm/vector path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

MAX_ROWS = 64
MAX_COLUMNS = 4096
MAX_ELEMENTS = 262144


def source_identity():
    modules = ["pllm", "pllm._native", "pllm.runtime.quantization",
               "pllm.runtime.semantic_executor", "pllm.runtime.rotary_coefficients",
               "pllm.runtime.semantic_attention", "numpy", "numpy._core._multiarray_umath"]
    files = {}
    for name in modules:
        path = Path(importlib.import_module(name).__file__).resolve()
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    lock = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "files": files}
    lock["digest"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    return lock


def require_identity(expected):
    if expected != source_identity()["digest"]:
        raise ValueError("source/numeric identity changed")


def checked_float(value, shape, *, positive=False):
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError("invalid finite float32 metadata")
    if positive and np.any(value <= 0):
        raise ValueError("scales must be positive")
    return value


@dataclass(frozen=True)
class Profile:
    """Public, immutable per-output bounds and original projection metadata."""
    bounds: tuple[int, ...]
    widths: bytes
    weight_scales: bytes
    bias: bytes | None

    def __post_init__(self):
        n = len(self.bounds)
        if (not 1 <= n <= MAX_COLUMNS or len(self.widths) != n
                or any(type(b) is not int or not 0 <= b < 1 << 31 for b in self.bounds)
                or any(w != max(1, b.bit_length() + 1) for b, w in zip(self.bounds, self.widths))
                or len(self.weight_scales) != 4 * n or (self.bias is not None and len(self.bias) != 4 * n)):
            raise ValueError("invalid immutable public profile")
        checked_float(np.frombuffer(self.weight_scales, "<f4"), (n,), positive=True)
        if self.bias is not None:
            checked_float(np.frombuffer(self.bias, "<f4"), (n,))

    @classmethod
    def from_weights(cls, weights, activation_bits, scales, bias=None):
        from pllm import _native
        w = np.asarray(weights)
        if (w.dtype != np.int8 or w.ndim != 2 or min(w.shape) < 1
                or w.shape[0] > MAX_COLUMNS or w.size > MAX_ELEMENTS
                or type(activation_bits) is not int or not 2 <= activation_bits <= 8):
            raise ValueError("invalid public weight profile")
        qmax = (1 << (activation_bits - 1)) - 1
        bounds = np.abs(w.astype(np.int64)).sum(axis=1) * qmax
        if np.any(bounds >= 1 << 31):
            raise ValueError("signed wrap32 bound exceeded")
        widths = _native.offset_row_bits(w.tobytes(), w.shape[1], qmax)
        s = checked_float(scales, (w.shape[0],), positive=True).astype("<f4", copy=False).tobytes()
        b = None if bias is None else checked_float(bias, (w.shape[0],)).astype("<f4", copy=False).tobytes()
        return cls(tuple(map(int, bounds)), widths, s, b)


@dataclass(frozen=True)
class Page:
    """One public-size page. Bytes are not authenticated durable storage."""
    rows: int
    payload: bytes
    scales: bytes
    positions: bytes
    commitment: str

    @classmethod
    def capture(cls, integer, scales, positions, profile, *, verified):
        from pllm import _native
        a = np.asarray(integer)
        columns = len(profile.widths)
        if verified is not True:
            raise ValueError("unverified integer output cannot become KV state")
        if (a.dtype not in (np.dtype("int32"), np.dtype("int64")) or a.ndim != 2
                or a.shape[1] != columns or not 1 <= a.shape[0] <= MAX_ROWS
                or a.size > MAX_ELEMENTS):
            raise ValueError("integer page exceeds public shape bounds")
        bound = np.asarray(profile.bounds, dtype=np.int64)
        if np.any(a < -bound) or np.any(a > bound):
            raise ValueError("integer output exceeds public weight-L1 bound")
        rows = a.shape[0]
        s = checked_float(scales, (rows,), positive=True).astype("<f4", copy=False).tobytes()
        p = np.asarray(positions)
        if (p.dtype != np.int64 or p.shape != (rows,) or np.any(p < 0)
                or np.any(p[1:] <= p[:-1])):
            raise ValueError("invalid absolute positions")
        p = p.astype("<i8", copy=False).tobytes()
        payload = _native.offset_pack_rows(a.astype("<u4").tobytes(), profile.widths, rows)
        return cls(rows, payload, s, p, hashlib.sha256(payload + s + p).hexdigest())

    def integers(self, profile):
        from pllm import _native
        columns = len(profile.widths)
        if (not 1 <= self.rows <= MAX_ROWS or not 1 <= columns <= MAX_COLUMNS
                or self.rows * columns > MAX_ELEMENTS
                or len(self.scales) != 4 * self.rows or len(self.positions) != 8 * self.rows
                or len(self.payload) != (self.rows * sum(profile.widths) + 7) // 8):
            raise ValueError("restored page exceeds public bounds")
        if hashlib.sha256(self.payload + self.scales + self.positions).hexdigest() != self.commitment:
            raise ValueError("page commitment differs")
        raw = _native.prepared_unpack_output(self.payload, bytes(4 * self.rows * columns),
                                             profile.widths, self.rows)
        result = np.frombuffer(raw, "<i8").reshape(self.rows, columns)
        bound = np.asarray(profile.bounds, dtype=np.int64)
        if np.any(result < -bound) or np.any(result > bound):
            raise ValueError("restored integer exceeds bound")
        checked_float(np.frombuffer(self.scales, "<f4"), (self.rows,), positive=True)
        positions = np.frombuffer(self.positions, "<i8")
        if np.any(positions < 0) or np.any(positions[1:] <= positions[:-1]):
            raise ValueError("invalid restored positions")
        return result

    def projection(self, profile):
        from pllm.runtime.quantization import dequantize_matmul
        result = dequantize_matmul(self.integers(profile), np.frombuffer(self.scales, "<f4"),
                                  np.frombuffer(profile.weight_scales, "<f4"))
        if profile.bias is not None:
            result = result + np.frombuffer(profile.bias, "<f4")
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite reconstructed projection")
        return result

    @property
    def stored_bytes(self):
        return len(self.payload) + len(self.scales) + len(self.positions) + 32


class CapturedCoefficients:
    """Exact slices of originally computed phase tables, never inverse RoPE."""
    def __init__(self, positions, cosine, sine, width, theta):
        rows = len(positions)
        if not 1 <= rows <= MAX_ROWS or not 0 < width <= MAX_COLUMNS or rows * width > MAX_ELEMENTS:
            raise ValueError("coefficient slab exceeds bound")
        self.positions = np.asarray(positions).copy()
        if (self.positions.dtype != np.int64 or self.positions.shape != (rows,)
                or np.any(self.positions < 0) or np.any(self.positions[1:] <= self.positions[:-1])
                or width % 2 or not np.isfinite(theta) or theta <= 0):
            raise ValueError("invalid coefficient domain")
        self.tables = tuple(checked_float(t, (1, 1, rows, width)).copy() for t in (cosine, sine))
        self.positions.flags.writeable = False
        for table in self.tables:
            table.flags.writeable = False
        self.width, self.theta = width, theta

    def get(self, positions, width, theta):
        if width != self.width or theta != self.theta or not np.array_equal(positions, self.positions):
            raise ValueError("original rotary domain differs")
        return self.tables


def replay_key(page, profile, *, heads, coefficients, norm_weight=None, epsilon=1e-6):
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    if type(heads) is not int or heads < 1 or len(profile.widths) % heads:
        raise ValueError("invalid head layout")
    value = page.projection(profile).reshape(page.rows, heads, -1).transpose(1, 0, 2)[None]
    runtime = object.__new__(SemanticDecoderRuntime)
    if norm_weight is not None:
        weight = checked_float(norm_weight, (value.shape[-1],))
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid RMS epsilon")
        runtime._weight = lambda _: weight
        value = runtime._local({"operator": "rms_norm", "inputs": ["x"],
                                "attributes": {"weight": "k_norm", "epsilon": epsilon, "weight_offset": 0}},
                               {"x": value}, {}, {})
    attrs = {"rotary_dimensions": coefficients.width, "theta": coefficients.theta}
    result = runtime._rotary(value, np.frombuffer(page.positions, "<i8"), attrs, _coefficients=coefficients)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite reconstructed key")
    return result


def original_attention(query, key, value):
    """Execute original semantic score/scale/softmax/value operations unchanged."""
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    r = object.__new__(SemanticDecoderRuntime)
    r._causal_reduction = None
    group = query.shape[1] // key.shape[1]
    def op(kind, x, y=None, attrs=None):
        return r._local({"operator": kind, "inputs": ["x", "y"], "attributes": attrs or {}},
                        {"x": x, "y": y}, {}, {})
    scores = op("attention_scores", query, key, {"group_size": group})
    scores = op("attention_scale", scores, attrs={"head_dim": query.shape[-1]})
    probabilities = op("softmax", scores)
    return op("attention_values", probabilities, value, {"group_size": group})


def synthetic_probe():
    from pllm.runtime.quantization import dequantize_matmul
    from pllm.runtime.rotary_coefficients import default_rotary_coefficients
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    rng = np.random.default_rng(20261014)
    rows, heads, dim, inner = 7, 2, 8, 19
    positions = np.arange(37, 37 + rows, dtype=np.int64)
    activation = rng.integers(-127, 128, (rows, inner), dtype=np.int32)
    scales = np.linspace(.0007, .004, rows, dtype=np.float32)
    results = []
    for family in ("qwen2", "qwen3", "llama"):
        ws = np.linspace(.001, .002, heads * dim, dtype=np.float32)
        bias = rng.normal(0, .01, heads * dim).astype(np.float32) if family == "qwen2" else None
        weights = [rng.integers(-127, 128, (heads * dim, inner), dtype=np.int8) for _ in range(2)]
        integers = [activation.astype(np.int64) @ w.astype(np.int64).T for w in weights]
        profiles = [Profile.from_weights(w, 8, ws, bias) for w in weights]
        pages = [Page.capture(i, scales, positions, p, verified=True) for i, p in zip(integers, profiles)]
        plain = [dequantize_matmul(i, scales, ws) for i in integers]
        if bias is not None:
            plain = [x + bias for x in plain]
        key = plain[0].reshape(rows, heads, dim).transpose(1, 0, 2)[None]
        norm = rng.uniform(.8, 1.2, dim).astype(np.float32) if family == "qwen3" else None
        if norm is not None:
            key = key / np.sqrt(np.mean(key * key, axis=-1, keepdims=True) + 1e-6) * norm
        attrs = {"rotary_dimensions": dim, "theta": 10000.0}
        key = SemanticDecoderRuntime._rotary(key, positions, attrs)
        cos, sin = default_rotary_coefficients(positions, dim, 10000.0)
        # Slice exact original phase tables to public codec pages, including an odd tail.
        decoded = []
        for start, end in ((0, 3), (3, 6), (6, 7)):
            page = Page.capture(integers[0][start:end], scales[start:end], positions[start:end], profiles[0], verified=True)
            coeff = CapturedCoefficients(positions[start:end], cos[:, :, start:end], sin[:, :, start:end], dim, 10000.0)
            decoded.append(replay_key(page, profiles[0], heads=heads, coefficients=coeff, norm_weight=norm))
        recovered_key = np.concatenate(decoded, axis=2)
        value = plain[1].reshape(rows, heads, dim).transpose(1, 0, 2)[None]
        recovered_value = pages[1].projection(profiles[1]).reshape(rows, heads, dim).transpose(1, 0, 2)[None]
        query = rng.normal(size=(1, heads * 2, 1, dim)).astype(np.float32)
        expected = original_attention(query, key, value)
        actual = original_attention(query, recovered_key, recovered_value)
        for x, y in ((key, recovered_key), (value, recovered_value), (expected, actual)):
            if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
                raise AssertionError("original KV/attention bits differ")
        results.append({"family_fixture": family, "kv_and_attention_bits_equal": True,
                        "public_widths": list(profiles[0].widths),
                        "dense_kv_bytes": sum(x.nbytes for x in plain),
                        "page_bytes_including_duplicate_positions_and_commitments": sum(p.stored_bytes for p in pages),
                        "original_coefficient_bytes": cos.nbytes + sin.nbytes,
                        "attention_sha256": hashlib.sha256(expected.tobytes()).hexdigest()})
    return {"schema": "pllm.projection-kv-probe.v1", "scope": "bounded synthetic; no checkpoint or timing",
            "identity": source_identity(), "fixtures": results, "promotion": "research only; no tier or SOTA qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", help="Reject changed installed source/numeric identity before replay")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if args.expected_source:
        require_identity(args.expected_source)
    report = synthetic_probe()
    from pllm.metrics import BuildRef
    report["build"] = BuildRef.capture().to_spec()
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
