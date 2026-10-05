"""Second-round numeric offload references; never imported by serving code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rope_contract(plan_digest: str, attributes: dict, capacity: int) -> dict:
    # Admit only the ordinary unscaled float32 contract used by this experiment.
    fixed = {"coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
             "input_layout": "batch_heads_sequence_feature",
             "output_layout": "batch_heads_sequence_feature", "pairing": "split_half",
             "position_policy": "sequential_absolute", "tail_policy": "unchanged"}
    if (not isinstance(plan_digest, str) or len(plan_digest) != 64
            or any(c not in "0123456789abcdef" for c in plan_digest)
            or type(capacity) is not int or not 1 <= capacity <= 40960
            or set(attributes) != set(fixed) | {"rotary_dimensions", "theta"}
            or any(attributes[k] != v for k, v in fixed.items())):
        raise ValueError("unsupported public rotary contract")
    width, theta = attributes["rotary_dimensions"], float(attributes["theta"])
    if type(width) is not int or not 2 <= width <= 256 or width % 2 or not 0 < theta < float("inf"):
        raise ValueError("unsupported rotary dimensions or theta")
    return {"schema": "pllm.research.public_rope.v1", "plan_digest": plan_digest,
            "attributes": attributes, "capacity": capacity, "dtype": "<f4",
            "numpy": np.__version__, "arithmetic": "semantic_executor._rotary/default-f32/v1"}


def export_rope(path: Path, contract: dict) -> str:
    checked = rope_contract(contract["plan_digest"], contract["attributes"], contract["capacity"])
    if checked != contract:
        raise ValueError("changed public rotary numeric contract")
    width, theta = contract["attributes"]["rotary_dimensions"], float(contract["attributes"]["theta"])
    frequency = 1.0 / (theta ** (np.arange(0, width, 2, dtype=np.float32) / width))
    angles = np.arange(contract["capacity"], dtype=np.float32)[:, None] * frequency[None, :]
    cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)
    sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)
    header = canonical(contract)
    with path.open("xb") as output:
        output.write(len(header).to_bytes(4, "little"))
        output.write(header)
        output.write(np.stack((cos, sin)).astype("<f4", copy=False).tobytes())
    return digest(path)


class PublicRope:
    """Trusted public compiler digest; private activation arithmetic stays local."""

    def __init__(self, path: Path, trusted_digest: str, contract: dict):
        checked = rope_contract(contract["plan_digest"], contract["attributes"], contract["capacity"])
        if checked != contract:
            raise ValueError("changed rotary contract")
        header = canonical(contract)
        width, capacity = contract["attributes"]["rotary_dimensions"], contract["capacity"]
        expected_size = 4 + len(header) + 2 * capacity * width * 4
        if path.stat().st_size != expected_size:
            raise ValueError("public rotary size differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != trusted_digest:
            raise ValueError("public rotary identity differs from trusted compiler")
        if raw[:4] != len(header).to_bytes(4, "little") or raw[4:4 + len(header)] != header:
            raise ValueError("public rotary binding differs")
        self.table = np.frombuffer(raw, "<f4", offset=4 + len(header)).reshape(2, capacity, width)
        if not np.isfinite(self.table).all():
            raise ValueError("public rotary table is not finite")
        self.width, self.capacity = width, capacity

    def apply(self, value: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if (value.dtype != np.float32 or value.ndim != 4 or value.shape[-1] < self.width
                or positions.ndim != 1 or positions.dtype.kind not in "iu"
                or len(positions) != value.shape[2] or not len(positions)
                or np.min(positions) < 0 or np.max(positions) >= self.capacity):
            raise ValueError("private rotary input exceeds public contract")
        cos, sin = (self.table[i, positions][None, None, :, :] for i in (0, 1))
        current = value[..., :self.width]
        half = self.width // 2
        rotated = np.concatenate((-current[..., half:], current[..., :half]), axis=-1)
        result = value.copy()
        result[..., :self.width] = current * cos + rotated * sin
        return result


def sqrt_index(value: np.float32) -> tuple[int, int]:
    """Normalize a positive normal f32 to [1,4); keep its power-of-two exponent local."""
    if type(value) is not np.float32 or not np.isfinite(value) or value < np.finfo(np.float32).tiny:
        raise ValueError("private sqrt lookup requires a positive normal float32")
    bits = int(value.view(np.uint32))
    exponent = (bits >> 23) - 127
    return ((exponent & 1) << 23) | (bits & 0x7fffff), exponent // 2


def sqrt_restore(encoded: bytes, exponent: int) -> np.float32:
    if len(encoded) != 4 or type(exponent) is not int or not -63 <= exponent <= 63:
        raise ValueError("private sqrt output has invalid representation")
    mantissa = np.frombuffer(encoded, "<f4")[0]
    if not np.isfinite(mantissa) or not 1 <= mantissa <= 2:
        raise ValueError("private sqrt normalized output is invalid")
    # Every square root of a positive normal f32 remains normal: exact exponent shift.
    return np.ldexp(mantissa, exponent, dtype=np.float32)


def export_sqrt(path: Path) -> str:
    """Public 64 MiB sqrt table, produced with bounded 256 KiB float chunks."""
    with path.open("xb") as output:
        for exponent in (127, 128):
            for start in range(0, 1 << 23, 1 << 16):
                values = (np.arange(start, start + (1 << 16), dtype=np.uint32)
                          | np.uint32(exponent << 23)).view(np.float32)
                output.write(np.sqrt(values).astype("<f4", copy=False).tobytes())
    return digest(path)


def folding_probe(weight: np.ndarray, gamma: np.ndarray, inputs: np.ndarray) -> dict:
    """Public norm-gamma folding, checked through actual W8A8 stage primitives."""
    from pllm.native import MaskedGEMM
    from pllm.runtime.quantization import (
        dequantize_matmul, quantize_activation_per_row, quantize_weight_per_row,
    )
    import time

    if (weight.dtype != np.float32 or gamma.dtype != np.float32 or inputs.dtype != np.float32
            or weight.ndim != 2 or inputs.ndim != 2 or gamma.shape != (weight.shape[1],)
            or inputs.shape[1] != weight.shape[1]):
        raise ValueError("norm folding geometry differs")
    normalized = inputs / np.sqrt(np.mean(inputs * inputs, axis=-1, keepdims=True) + 1e-6)
    started = time.process_time()
    folded = weight * gamma[None, :]
    compile_cpu = time.process_time() - started
    a = quantize_activation_per_row(normalized * gamma, bits=8)
    b = quantize_activation_per_row(normalized, bits=8)
    kernels = MaskedGEMM(threads=1)
    outputs = []
    weight_hashes = []
    for w, activation in ((weight, a), (folded, b)):
        q = quantize_weight_per_row(w, bits=8)
        kernel = kernels.compile(q.values)
        outputs.append(dequantize_matmul(kernel.clear(activation.values), activation.scales, q.scales))
        weight_hashes.append(hashlib.sha256(q.values.tobytes() + q.scales.tobytes()).hexdigest())
    float_control = (normalized * gamma) @ weight.T
    float_candidate = normalized @ folded.T
    start = time.process_time()
    for _ in range(100):
        normalized * gamma
    saved = (time.process_time() - start) / 100
    return {"shape": list(weight.shape), "synthetic_input_rows": len(inputs),
            "float32_changed_elements": int(np.count_nonzero(float_control.view("u4") != float_candidate.view("u4"))),
            "float32_max_absolute_error": float(np.max(np.abs(float_control - float_candidate))),
            "w8a8_changed_elements": int(np.count_nonzero(outputs[0].view("u4") != outputs[1].view("u4"))),
            "w8a8_max_absolute_error": float(np.max(np.abs(outputs[0] - outputs[1]))),
            "w8a8_changed_input_codes": int(np.count_nonzero(a.values != b.values)),
            "original_and_folded_stage_sha256": weight_hashes,
            "public_fold_cpu_seconds": compile_cpu, "client_gamma_multiply_cpu_seconds": saved,
            "privacy": "only public gamma and public weights enter offline folding",
            "decision": "numeric contract changes; cannot substitute under original W8A8 plan",
            "whole_decoder_or_quality_evidence": False}
