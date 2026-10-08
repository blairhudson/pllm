"""Fresh-process synthetic client-boundary screen, without matrix execution.

Public Qwen3-shaped geometry; private-value stand-ins use reproducible randomness.
Times cover NumPy/bytes boundaries, native quantization/masking/packing and reply
reconstruction. Allocation tracing is separate from timing. No whole-client RSS,
network, checkpoint quality or cryptographic claim follows from this screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc


def run(mode, rows, width, output):
    import numpy as np
    from pllm import _native
    from pllm.runtime.native import mask_prepared_input
    from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul

    rng = np.random.default_rng(481)
    value = rng.normal(size=(rows, width)).astype("<f4")
    mask = rng.integers(0, 1 << 24, (rows, width), dtype="<u4")
    output_mask = rng.integers(0, 1 << 24, (rows, output), dtype="<u4")
    widths = bytes([23]) * output
    packed = _native.offset_pack_rows(
        rng.integers(0, 1 << 23, (rows, output), dtype="<u4").tobytes(), widths, rows)
    weights = rng.uniform(.0001, .01, output).astype("<f4")
    bias = rng.normal(size=output).astype("<f4")

    def execute():
        if mode == "separate":
            quantized = quantize_activation_per_row(value, bits=8)
            complement = mask_prepared_input(quantized.values, mask, 24)
            request = _native.offset_pack_rows(complement.astype("<u4", copy=False).tobytes(), bytes([23]) * width, rows)
            integer = _native.prepared_unpack_output(packed, output_mask.tobytes(), widths, rows)
            result = dequantize_matmul(np.frombuffer(integer, "<i8").reshape(rows, output),
                                      quantized.scales, weights) + bias
            return request, quantized.scales.tobytes(), result.tobytes()
        request, scales = _native.prepared_quantize_pack(value.tobytes(), mask.tobytes(), rows, width, 8, 23)
        result = _native.prepared_dequantize_output(packed, output_mask.tobytes(), widths,
                                                  rows, scales, weights.tobytes(), bias.tobytes())
        return request, scales, result

    expected = execute()
    digest = hashlib.sha256(b"".join(expected)).hexdigest()
    del expected
    times = []
    for _ in range(9):
        start, cpu = time.perf_counter(), time.process_time()
        result = execute()
        times.append({"wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu})
        if hashlib.sha256(b"".join(result)).hexdigest() != digest:
            raise ValueError("boundary results changed across iterations")
        del result
    traced = []
    for _ in range(3):
        tracemalloc.start()
        result = execute()
        traced.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        del result
    return {"mode": mode, "shape": [rows, width, output], "output_sha256": digest,
            "checked_values": rows * (width + output), "samples": times,
            "median_cpu_seconds": statistics.median(t["cpu_seconds"] for t in times),
            "median_wall_seconds": statistics.median(t["wall_seconds"] for t in times),
            "python_traced_peak_bytes": traced,
            "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("separate", "fused"))
    parser.add_argument("--shape", nargs=3, type=int)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run(args.child, *args.shape)))
        return
    if args.output is None or args.output.exists():
        parser.error("a new output path is required")
    records = []
    for rows, width, output in ((1, 2560, 9728), (150, 2560, 9728), (150, 9728, 2560)):
        for repetition in range(3):
            pair = []
            for mode in (("separate", "fused") if repetition % 2 == 0 else ("fused", "separate")):
                value = subprocess.check_output([sys.executable, "-m", "scripts.probe_prepared_boundaries",
                    "--child", mode, "--shape", str(rows), str(width), str(output)], text=True, timeout=120)
                record = json.loads(value)
                record["repetition"] = repetition
                pair.append(record)
            if pair[0]["output_sha256"] != pair[1]["output_sha256"]:
                raise ValueError("fused boundaries differ from the separate control")
            records.extend(pair)
    args.output.write_text(json.dumps({"schema": "pllm.prepared_boundaries.v1", "scope": __doc__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records},
        indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "processes": len(records)}))


if __name__ == "__main__":
    main()
