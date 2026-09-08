"""Measure complete Python-to-native client calls, not just isolated Rust loops."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--width', type=int, default=5120)
    parser.add_argument('--output', type=Path, default=Path('results/client-native.json'))
    parser.add_argument('--allow-missing-rust', action='store_true')
    args = parser.parse_args()
    if min(args.samples, args.rows, args.width) < 1:
        parser.error('samples and dimensions must be positive')
    present = importlib.util.find_spec('pllm._native') is not None
    if not present and not args.allow_missing_rust:
        parser.error('Build the native extension with uv sync before benchmarking')
    # This benchmark intentionally includes the explicit reference as a control.
    # A missing extension still aborts native comparisons; this is not test fallback.
    os.environ.pop('PLLM_REQUIRE_RUST', None)
    os.environ['PLLM_KERNEL_BACKEND'] = 'python'
    from pllm.runtime.compact import pack_unsigned, unpack_unsigned
    from pllm.runtime.quantization import quantize_activation_per_row
    from pllm.runtime.secure_random import uniform_residues
    from pllm.runtime.stage_protocol import StageCorrelation, encode_signed_with_mask, decode_unmasked
    from pllm.runtime._native_support import capabilities

    rng = np.random.default_rng(17)  # Public fixtures, not protocol secrets.
    shape = (args.rows, args.width)
    floats = rng.normal(size=shape).astype(np.float32)
    residues = rng.integers(0, 786433, size=shape, dtype=np.uint32)
    signed = rng.integers(-127, 128, size=shape, dtype=np.int8)
    correlation = StageCorrelation('benchmark', 'stage', residues, residues, 786433, 'prime')
    packed = pack_unsigned(residues, width=3)
    masked = encode_signed_with_mask(signed, correlation)
    quantized = quantize_activation_per_row(floats, bits=8)
    expected_quantized = (quantized.values.copy(), quantized.scales.copy())

    def quantize():
        result = quantize_activation_per_row(floats, bits=8)
        return result.values, result.scales

    def check_quantize(value):
        for a, b in zip(value, expected_quantized):
            np.testing.assert_array_equal(a, b)

    def check_random(value):
        assert value.shape == shape and value.dtype == np.uint32
        assert (value < 786433).all()

    cases = [
        ('pack_u24', lambda: pack_unsigned(residues, width=3), lambda x: np.testing.assert_equal(x, packed)),
        ('unpack_u24', lambda: unpack_unsigned(packed, width=3), lambda x: np.testing.assert_array_equal(x, residues.ravel())),
        ('quantize_a8', quantize, check_quantize),
        ('mask', lambda: encode_signed_with_mask(signed, correlation), lambda x: np.testing.assert_array_equal(x, masked)),
        ('unmask', lambda: decode_unmasked(masked, correlation), lambda x: np.testing.assert_array_equal(x, signed)),
        ('os_random_masks', lambda: uniform_residues(786433, shape), check_random),
    ]
    rows = []
    for name, function, validate in cases:
        row = {'operation': name, 'shape': shape}
        for backend in ('python', 'rust'):
            if backend == 'rust' and not present:
                row[backend] = {'status': 'not_run', 'reason': 'compiled extension missing'}
                continue
            os.environ['PLLM_KERNEL_BACKEND'] = backend
            assert capabilities()['compiled'] == (backend == 'rust')
            for _ in range(3):
                validate(function())
            samples = []
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                result = function()
                samples.append((time.perf_counter_ns() - start) / 1e6)
                validate(result)
            row[backend] = {'median_ms': statistics.median(samples), 'p95_ms': float(np.percentile(samples, 95)), 'samples_ms': samples, 'validated_every_sample': True}
        if present:
            row['speedup_vs_python'] = row['python']['median_ms'] / row['rust']['median_ms']
        rows.append(row)
    report = {'scope': 'Complete client helper calls including buffer conversion; not language model TPS', 'rust_executed': present, 'python': sys.version, 'platform': platform.platform(), 'operations': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'rust_executed': present, 'operations': len(rows)}))


if __name__ == '__main__':
    main()
