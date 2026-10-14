"""Bounded exact exception-split screening; no production or timing claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1 << 24
WORKSPACE = 64 << 20
MAX_ELEMENTS = 4 << 20


def bounds(base_bits):
    if type(base_bits) is not int or base_bits not in (4, 5, 6, 7, 8):
        raise ValueError("base width must be 4..8")
    h = 1 << (base_bits - 1)
    products = [a * b for a in (-128, 127) for b in (-h, h - 1)]
    maximum = max(abs(v) for v in products)
    block = (LIMIT - 1) // maximum
    return {"base_bits": base_bits, "base_min": -h, "base_max": h - 1,
            "product_min": min(products), "product_max": max(products),
            "block": block, "partial_abs_max": block * maximum,
            "exception_min": -128 + h, "exception_max": 128 - h}


def validate_weights(weights):
    if not isinstance(weights, np.ndarray) or weights.dtype != np.int8 or weights.ndim != 2:
        raise ValueError("rank-two signed-i8 weights required")
    if min(weights.shape) < 1 or max(weights.shape) > 16384 or weights.size > MAX_ELEMENTS:
        raise ValueError("bounded reference geometry exceeded")


def split(weights, base_bits):
    validate_weights(weights)
    proof = bounds(base_bits)
    base = np.clip(weights, proof["base_min"], proof["base_max"]).astype(np.int8)
    difference = weights.astype(np.int16) - base.astype(np.int16)
    row, column = np.nonzero(difference)
    pointers = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=weights.shape[0])))).astype(np.uint32)
    return base, pointers, column.astype(np.uint32), difference[row, column].astype(np.int8)


def validate_csr(base, pointers, indices, coefficients):
    validate_weights(base)
    rows, columns = base.shape
    if (pointers.dtype != np.uint32 or pointers.shape != (rows + 1,)
            or indices.dtype != np.uint32 or indices.ndim != 1
            or coefficients.dtype != np.int8 or coefficients.shape != indices.shape):
        raise ValueError("canonical u32 CSR pointers/indices and i8 coefficients required")
    p = pointers.astype(np.uint64)
    if p[0] != 0 or p[-1] != len(indices) or np.any(p[1:] < p[:-1]) or np.any(indices >= columns):
        raise ValueError("invalid CSR bounds")
    if np.any(coefficients == 0):
        raise ValueError("zero exception coefficient")
    for row in range(rows):
        ix = indices[int(p[row]):int(p[row + 1])]
        if len(ix) > 1 and np.any(ix[1:] <= ix[:-1]):
            raise ValueError("CSR indices must be strictly increasing in each row")


def pack_base(base, base_bits):
    validate_weights(base)
    proof = bounds(base_bits)
    if np.any(base < proof["base_min"]) or np.any(base > proof["base_max"]):
        raise ValueError("base outside signed width")
    output = bytearray((base.size * base_bits + 7) // 8)
    for i, value in enumerate(base.flat):
        unsigned = int(value) & ((1 << base_bits) - 1)
        for bit in range(base_bits):
            offset = i * base_bits + bit
            output[offset // 8] |= ((unsigned >> bit) & 1) << (offset % 8)
    return bytes(output)


def unpack_base(packed, shape, base_bits):
    proof = bounds(base_bits)
    if len(shape) != 2 or any(type(v) is not int or not 1 <= v <= 16384 for v in shape):
        raise ValueError("invalid packed geometry")
    count = shape[0] * shape[1]
    if count > MAX_ELEMENTS or len(packed) != (count * base_bits + 7) // 8:
        raise ValueError("invalid packed length")
    if count * base_bits % 8 and packed[-1] >> (count * base_bits % 8):
        raise ValueError("noncanonical padding")
    output = np.empty(count, dtype=np.int8)
    for i in range(count):
        value = sum(((packed[(i * base_bits + bit) // 8] >> ((i * base_bits + bit) % 8)) & 1) << bit
                    for bit in range(base_bits))
        output[i] = value if value < -proof["base_min"] else value - (1 << base_bits)
    return output.reshape(shape)


def project(base, pointers, indices, coefficients, inputs, ring_bits, base_bits, *, block=None):
    """NumPy float32 base reference plus exact integer CSR, not Accelerate."""
    validate_csr(base, pointers, indices, coefficients)
    proof = bounds(base_bits)
    if np.any(base < proof["base_min"]) or np.any(base > proof["base_max"]):
        raise ValueError("base outside signed width")
    if np.any(coefficients < proof["exception_min"]) or np.any(coefficients > proof["exception_max"]):
        raise ValueError("exception coefficient outside exact split bound")
    for row in range(base.shape[0]):
        lo, hi = int(pointers[row]), int(pointers[row + 1])
        for e in range(lo, hi):
            endpoint = proof["base_max"] if coefficients[e] > 0 else proof["base_min"]
            if int(base[row, int(indices[e])]) != endpoint:
                raise ValueError("exception must correct a clipped base endpoint")
    if ring_bits not in (16, 24, 32):
        raise ValueError("ring must be 16, 24 or 32")
    if (inputs.dtype != np.uint32 or inputs.ndim != 2 or inputs.shape[1] != base.shape[1]
            or not 1 <= inputs.shape[0] <= 4096 or inputs.size > MAX_ELEMENTS
            or inputs.shape[0] * base.shape[0] > MAX_ELEMENTS
            or np.any(inputs.astype(np.uint64) >= 1 << ring_bits)):
        raise ValueError("invalid canonical ring input")
    block = proof["block"] if block is None else block
    if type(block) is not int or not 1 <= block <= proof["block"]:
        raise ValueError("block violates exact partial-sum bound")
    output = np.zeros((inputs.shape[0], base.shape[0]), dtype=np.uint64)
    mask = (1 << ring_bits) - 1
    for start in range(0, base.shape[1], block):
        x = inputs[:, start:start + block].astype(np.uint64)
        weight = base[:, start:start + block].astype(np.float32)
        carry = 0
        for shift in range(0, ring_bits, 8):
            remaining = (x + np.uint64(carry)) >> np.uint64(shift)
            digits = ((remaining + np.uint64(128)) & np.uint64(255)).astype(np.int16) - 128
            product = digits.astype(np.float32) @ weight.T
            if np.any(~np.isfinite(product)) or np.any(np.abs(product) >= LIMIT) or np.any(product != np.trunc(product)):
                raise ArithmeticError("float32 result violated exact bound")
            # First reduce signed block results, then shift as unsigned residues.
            residue = (product.astype(np.int64) & mask).astype(np.uint64)
            output = (output + (residue << np.uint64(shift))) & np.uint64(mask)
            carry += 128 << shift
    # Python integers avoid signed overflow for all ring widths and coefficients.
    for row in range(base.shape[0]):
        lo, hi = int(pointers[row]), int(pointers[row + 1])
        for sample in range(inputs.shape[0]):
            correction = sum(int(coefficients[e]) * int(inputs[sample, int(indices[e])]) for e in range(lo, hi))
            output[sample, row] = (int(output[sample, row]) + correction) & mask
    return output.astype(np.uint32)


def integer_oracle(weights, inputs, ring_bits):
    mask = (1 << ring_bits) - 1
    return np.asarray([[sum(int(w) * int(x) for w, x in zip(row, sample)) & mask
                        for row in weights] for sample in inputs], dtype=np.uint32)


def stage_cost(rows, columns, batch, ring_bits, base_bits, exceptions):
    if (any(type(v) is not int or v <= 0 for v in (rows, columns, batch))
            or max(rows, columns) > 16384 or batch > 4096
            or batch * max(rows, columns) > MAX_ELEMENTS or ring_bits not in (16, 24, 32)
            or type(exceptions) is not int or not 0 <= exceptions <= rows * columns):
        raise ValueError("invalid cost geometry")
    proof = bounds(base_bits)
    k = min(columns, proof["block"])
    fixed = 4 * (batch * rows + batch * k)
    tile = min(rows, 1024, max(0, (WORKSPACE - fixed) // (4 * (k + batch))))
    blocks = (columns + k - 1) // k
    baseline_blocks = (columns + 1022) // 1023
    baseline_tiles = (rows + 1023) // 1024
    tiles = (rows + tile - 1) // tile if tile else None
    digits = ring_bits // 8
    csr = 4 * (rows + 1) + 5 * exceptions
    dense = rows * columns
    return {"rows": rows, "columns": columns, "batch": batch, "ring_bits": ring_bits,
            "base_bits": base_bits, "exceptions": exceptions, "density": exceptions / dense,
            "proof": proof, "block_width": k, "output_tile": tile,
            "native_digit8_sgemm_calls": baseline_blocks * baseline_tiles * digits,
            "proposed_sgemm_calls": blocks * tiles * digits if tile else None,
            "native_digit8_dense_products": digits * batch * dense,
            "proposed_dense_products": digits * batch * dense,
            "extra_integer_exception_macs": batch * exceptions,
            "original_i8_owner_bytes": dense,
            "base_i8_owner_bytes": dense,
            "packed_base_owner_bytes": (dense * base_bits + 7) // 8,
            "csr_owner_bytes": csr,
            "retained_i8_split_bytes": dense + csr,
            "retained_packed_split_bytes": (dense * base_bits + 7) // 8 + csr,
            "packed_storage_better_than_original": (dense * base_bits + 7) // 8 + csr < dense,
            "workspace_bytes": fixed + 4 * tile * (k + batch),
            "native_digit8_workspace_bytes": 4 * (batch * rows + batch * min(columns, 1023)
                + min(rows, 1024) * min(columns, 1023) + batch * min(rows, 1024)),
            "float_weight_conversion_elements": dense,
            "native_digit8_float_weight_conversion_elements": dense,
            "mask_digit_conversion_elements": digits * batch * columns * tiles if tiles else None,
            "native_digit8_mask_digit_conversion_elements": digits * batch * columns * baseline_tiles,
            "partial_output_reconstruction_elements": digits * batch * rows * blocks,
            "native_digit8_partial_output_reconstruction_elements": digits * batch * rows * baseline_blocks,
            "mask_expansion_bytes_per_role": 4 * batch * (columns + rows),
            "correction_ring_storage_bytes": 4 * batch * rows,
            "correction_encoded_body_bytes": (ring_bits // 8) * batch * rows,
            "binding_input_copy_bytes": 4 * batch * columns,
            "binding_output_copy_bytes": 4 * batch * rows,
            "platform_blas_allowance_bytes": 64 << 20,
            "current_role_admission_extra_bytes": 224 << 20,
            "client_budgets_bytes": {"Low": 256 << 20, "Medium": 2 << 30},
            "whole_topology_admitted": None,
            "native_candidate_implemented": False, "setup_cpu_s": None,
            "issuance_cpu_s": None, "setup_amortization_queries": None}


def histogram_summary(histogram, rows, columns, base_bits, row_counts):
    proof = bounds(base_bits)
    values = np.arange(-128, 128)
    count = int(histogram[(values < proof["base_min"]) | (values > proof["base_max"])].sum())
    return {"base_bits": base_bits, "exceptions": count, "density": count / (rows * columns),
            "per_row_exception_min": int(min(row_counts)), "per_row_exception_max": int(max(row_counts)),
            "per_row_exception_p95": float(np.percentile(row_counts, 95)),
            "cost_150_ring32": stage_cost(rows, columns, 150, 32, base_bits, count)}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def scan(payload):
    """Explicit public tensors only, complete rows, installed current W8 quantizer."""
    from safetensors import safe_open
    import ml_dtypes  # noqa: F401 - registers BF16 for NumPy Safetensors
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime import quantization, _native_support
    root = Path(payload["checkpoint"]).resolve()
    lock_path = Path(payload["source_lock"]).resolve()
    lock_digest = digest_file(lock_path)
    lock = json.loads(lock_path.read_text())
    # Lock is supplied from independently retained public source provenance.
    # It is not manufactured by this scan from whatever files happen to be present.
    if not lock.get("revision") or not isinstance(lock.get("sha256"), dict) or not lock["sha256"]:
        raise ValueError("source lock requires public revision and sha256 file mapping")
    files = {}
    for name, expected in lock["sha256"].items():
        relative = Path(name)
        # Hugging Face snapshot files may be symlinks to sibling blob storage.
        # Their bytes are authenticated by the independently supplied hash.
        if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
            raise ValueError("invalid source lock path/hash")
        path = root / relative
        if digest_file(path) != expected:
            raise ValueError("checkpoint source hash mismatch")
        files[name] = expected
    if "config.json" not in files:
        raise ValueError("config.json must be source-locked")
    keys = payload["keys"]
    if not 1 <= len(keys) <= 8 or len(set(keys)) != len(keys):
        raise ValueError("select 1..8 distinct tensor keys")
    locations = {}
    for name in files:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(root / name, framework="np") as handle:
            for key in keys:
                if key in handle.keys():
                    if key in locations:
                        raise ValueError("duplicate tensor source")
                    view = handle.get_slice(key)
                    shape = tuple(view.get_shape())
                    if len(shape) != 2 or min(shape) < 1 or max(shape) > 16384 or view.get_dtype() not in ("F32", "F16", "BF16"):
                        raise ValueError("unsupported source geometry/dtype/orientation")
                    locations[key] = (name, shape)
    if set(locations) != set(keys):
        raise ValueError("all selected keys must exist in locked shards")
    if sum(r * c for _, (r, c) in locations.values()) > 32 << 20:
        raise MemoryError("scan element cap exceeded before value reads")
    # Source maps can fault all locked file pages; reject an oversized mapping.
    mapped = sum((root / n).stat().st_size for n in files if n.endswith(".safetensors"))
    scratch = 32 * 32 * max(shape[1] for _, shape in locations.values())
    if mapped + scratch + (128 << 20) > 2 << 30:
        raise MemoryError("scan conservative mapped-page budget exceeds 2 GiB")
    extension = _native_support.extension()
    if extension is None:
        raise RuntimeError("installed native quantizer required; no fallback/build")
    source_paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py",
                    ROOT / "tests/test_exception_corrections_probe.py",
                    Path(quantization.__file__), Path(_native_support.__file__), Path(extension.__file__)]
    frozen = {str(p.resolve()): digest_file(p) for p in source_paths}
    stages = []
    for key in keys:
        name, (rows, columns) = locations[key]
        hist = np.zeros(256, dtype=np.int64)
        tails = {b: [] for b in range(4, 8)}
        quant_hash, scales_hash = hashlib.sha256(), hashlib.sha256()
        with safe_open(root / name, framework="np") as handle:
            view = handle.get_slice(key)
            for start in range(0, rows, 32):
                raw = np.asarray(view[start:min(rows, start + 32), :]).astype(np.float32)
                quantized = quantize_weight_per_row(raw, bits=8)
                weights = quantized.values
                hist += np.bincount(weights.astype(np.int16).ravel() + 128, minlength=256)
                quant_hash.update(weights.tobytes())
                scales_hash.update(quantized.scales.astype("<f4").tobytes())
                for b in tails:
                    p = bounds(b)
                    tails[b].extend(np.count_nonzero((weights < p["base_min"]) | (weights > p["base_max"]), axis=1).tolist())
        stages.append({"key": key, "shape": [rows, columns], "orientation": "checkpoint output-by-input; no transpose",
                       "quantized_i8_sha256": quant_hash.hexdigest(), "scales_f32_sha256": scales_hash.hexdigest(),
                       "signed_i8_histogram": hist.tolist(),
                       "candidates": [histogram_summary(hist, rows, columns, b, tails[b]) for b in tails]})
    for name, expected in files.items():
        if digest_file(root / name) != expected:
            raise RuntimeError("checkpoint changed during scan")
    for path, expected in frozen.items():
        if digest_file(path) != expected:
            raise RuntimeError("scan implementation changed")
    if digest_file(lock_path) != lock_digest:
        raise RuntimeError("source lock changed during scan")
    return {"kind": "public_checkpoint_tail_screen", "revision": lock["revision"],
            "source_lock_sha256": lock_digest, "checkpoint_files": files,
            "implementation_sha256": frozen, "stages": stages,
            "estimated_scan_peak_bytes": mapped + scratch + (128 << 20),
            "selection": "explicit public tensor keys, not complete compiled topology",
            "native_timing": None, "whole_response": None}


def screening():
    rng = np.random.default_rng(20261014)
    checked = 0
    commitments = []
    for b in range(4, 8):
        weights = rng.integers(-128, 128, (5, 37), dtype=np.int16).astype(np.int8)
        arrays = split(weights, b)
        for ring in (16, 24, 32):
            x = rng.integers(0, 1 << ring, (3, 37), dtype=np.uint32)
            x[0, :6] = np.asarray([0, 127, 128, 255, 1 << (ring - 1), (1 << ring) - 1], dtype=np.uint32)
            actual = project(*arrays, x, ring, b, block=13)
            if not np.array_equal(actual, integer_oracle(weights, x, ring)):
                raise AssertionError("independent integer mismatch")
            commitments.append(hashlib.sha256(actual.tobytes()).hexdigest())
            checked += actual.size
    return {"kind": "synthetic_correctness_and_symbolic_cost_only", "seed": 20261014,
            "oracle_output_values": checked, "output_sha256": commitments,
            "proofs": [bounds(b) for b in range(4, 9)],
            "no_checkpoint_values_scanned": True, "native_timing": None,
            "real_tail_densities": None, "whole_response": None,
            "stage_cost_zero_tail_lower_bounds": [stage_cost(r, c, 150, 32, b, 0)
                for r, c in ((896, 896), (1152, 896), (9728, 896), (896, 4864)) for b in range(4, 8)],
            "uniform_i8_counterexample_tail_density": {str(b): 1 - (1 << b) / 256 for b in range(4, 8)},
            "tests_sha256": digest_file(ROOT / "tests/test_exception_corrections_probe.py"),
            "implementation_sha256": digest_file(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-checkpoint", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(scan(json.load(sys.stdin))))
        return
    if args.output is None:
        parser.error("--output required")
    if args.scan_checkpoint:
        if not args.source_lock or not args.tensor:
            parser.error("scan requires --source-lock and explicit --tensor")
        from frontier_probe_support import fresh_workers
        with fresh_workers(Path(__file__).resolve(), memory_bytes=3 << 30, timeout=240) as (run, guard, host):
            report = run({"checkpoint": str(args.scan_checkpoint.resolve()),
                          "source_lock": str(args.source_lock.resolve()), "keys": args.tensor})
            report["installed_build"] = run.build
            report["observed_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    else:
        if args.source_lock or args.tensor:
            parser.error("source options require --scan-checkpoint")
        report = screening()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
