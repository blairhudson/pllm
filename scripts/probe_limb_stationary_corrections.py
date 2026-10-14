"""Bounded digit8 schedule reference; no SDK mutation, model loads or timing claims."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

BLOCK, TILE = 1023, 1024
MAX_ELEMENTS, NATIVE_CAP, STAGE_CAP = 4 << 20, 64 << 20, 224 << 20
ROOT = Path(__file__).resolve().parents[1]


def preflight(batch, rows, cols, bits, *, stationary=True, page_rows=None,
              page_transient_bytes=0):
    """Native payload accounting, not Python interpreter/allocator RSS admission.

    Resident immutable signed weights are separate base ownership. Paged callers
    must supply their actual maximum decode/transient weight owner bound.
    """
    if any(type(v) is not int or v <= 0 for v in (batch, rows, cols)):
        raise ValueError("positive integer geometry required")
    if bits not in (16, 24, 32) or type(bits) is not int or batch > 4096:
        raise ValueError("ring/batch domain")
    if max(batch * rows, batch * cols) > MAX_ELEMENTS:
        raise ValueError("input/output element bound")
    if page_rows is not None and (type(page_rows) is not int or page_rows <= 0):
        raise ValueError("page rows")
    if type(page_transient_bytes) is not int or page_transient_bytes < 0:
        raise ValueError("page transient owner")
    local_rows = rows if page_rows is None else min(rows, page_rows)
    k, n, d = min(cols, BLOCK), min(local_rows, TILE), bits // 8
    slabs = d if stationary else 1
    local = 4 * (batch * local_rows + slabs * batch * k + n * k + batch * n)
    if local > NATIVE_CAP:
        raise ValueError("native workspace exceeds 64 MiB; fallback required")
    input_bytes, output_bytes = 4 * batch * cols, 4 * batch * rows
    # Conservative simultaneous input ndarray + wire bytes + decoded Rust Vec;
    # output core + serializer + PyBytes + ndarray-copy allowance. Count local
    # output once in local; subtract it from full output owners in resident mode.
    extra = (local + 3 * input_bytes + 4 * output_bytes
             - (output_bytes if page_rows is None else 0)
             + (64 << 20) + page_transient_bytes)
    if extra > STAGE_CAP:
        raise ValueError("stage extra exceeds 224 MiB; fallback required")
    pages = [(first, min(local_rows, rows - first))
             for first in range(0, rows, local_rows)]
    tiles = sum((r + TILE - 1) // TILE for _, r in pages)
    blocks = (cols + BLOCK - 1) // BLOCK
    return dict(native_bytes=local, stage_extra_bytes=extra,
                admitted_stage_extra_bytes=STAGE_CAP,
                resident_weight_bytes=rows * cols if page_rows is None else 0,
                page_transient_bytes=page_transient_bytes,
                mask_pack_elements=batch * cols * d * (len(pages) if stationary else tiles),
                weight_pack_elements=rows * cols,
                gemm_calls=tiles * blocks * d,
                integer_reconstruction_elements=batch * rows * blocks * d,
                float_mac_products=batch * rows * cols * d,
                useful_within_page_reuse=local_rows > TILE,
                pages=len(pages), tiles=tiles, blocks=blocks)


class OneUseMask:
    """Synthetic fresh-mask owner. Reuse allowed inside one call, never across calls."""
    def __init__(self, values):
        self.values, self.spent = values, False

    def claim(self):
        if self.spent:
            raise ValueError("mask already consumed")
        self.spent = True  # Remains spent on failure; no implicit retry/recycling.
        return self.values


def oracle(weights, x, bits):
    """Independent signed Python integers; no limb decomposition or NumPy reduction."""
    modulus = 1 << bits
    return [[sum(int(x[r, i]) * int(weights[j, i]) for i in range(x.shape[1]))
             % modulus for j in range(weights.shape[0])] for r in range(x.shape[0])]


def reference(weights, ticket, bits, *, stationary, page_rows=None):
    x = ticket.values
    if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
            or weights.ndim != 2 or not isinstance(x, np.ndarray)
            or x.dtype != np.uint32 or x.ndim != 2 or x.shape[1] != weights.shape[1]):
        raise ValueError("int8 weights / uint32 mask geometry")
    m, cols = x.shape
    rows = weights.shape[0]
    counts = preflight(m, rows, cols, bits, stationary=stationary, page_rows=page_rows,
                       page_transient_bytes=0 if page_rows is None else 2 * min(rows, page_rows) * cols)
    if bits < 32 and np.any(x >= (1 << bits)):
        raise ValueError("mask outside ring")
    x = ticket.claim()
    # No full-mask digit cache, no full-weight float copy. Views stand in for
    # decoded pages only; this reference does not authenticate/read disk pages.
    out = np.zeros((m, rows), dtype=np.uint32)
    local_rows = rows if page_rows is None else min(page_rows, rows)
    observed = dict(mask_pack_elements=0, weight_pack_elements=0, gemm_calls=0)
    for page in range(0, rows, local_rows):
        nr = min(local_rows, rows - page)
        partial = out[:, page:page + nr]
        for start in range(0, cols, BLOCK):
            k = min(BLOCK, cols - start)

            def pack_all():
                a = np.empty((bits // 8, m, k), dtype=np.float32)
                carry = 0
                for digit, shift in enumerate(range(0, bits, 8)):
                    for r in range(m):
                        for i in range(k):
                            remaining = (int(x[r, start + i]) + carry) >> shift
                            a[digit, r, i] = ((remaining + 128) & 255) - 128
                    carry += 128 << shift
                observed['mask_pack_elements'] += a.size
                return a

            a = pack_all() if stationary else None
            for first in range(0, nr, TILE):
                n = min(TILE, nr - first)
                b = weights[page + first:page + first + n, start:start + k].astype(np.float32)
                observed['weight_pack_elements'] += b.size
                # Baseline uses one slab, as native does (not an all-digit cache).
                carry = 0
                for digit, shift in enumerate(range(0, bits, 8)):
                    if stationary:
                        slab = a[digit]
                    else:
                        slab = np.empty((m, k), dtype=np.float32)
                        for r in range(m):
                            for i in range(k):
                                remaining = (int(x[r, start + i]) + carry) >> shift
                                slab[r, i] = ((remaining + 128) & 255) - 128
                        observed['mask_pack_elements'] += slab.size
                    c = slab @ b.T
                    observed['gemm_calls'] += 1
                    if not np.all(np.isfinite(c) & (np.abs(c) < (1 << 24)) & (c == np.floor(c))):
                        raise ArithmeticError("SGEMM exact integer bound")
                    target = partial[:, first:first + n]
                    for r in range(m):
                        for j in range(n):
                            target[r, j] = (int(target[r, j]) + (int(c[r, j]) << shift)) & 0xffffffff
                    carry += 128 << shift
                del b, slab, c
            if a is not None:
                del a
    if bits < 32:
        out &= np.uint32((1 << bits) - 1)
    for key, value in observed.items():
        assert counts[key] == value, (key, counts[key], value)
    return out, counts


def synthetic_case(rows, cols, batch, bits):
    weights = np.resize(np.array([-128, 127, 0, -1, 1], dtype=np.int8), rows * cols).reshape(rows, cols)
    extremes = [0, (1 << bits) - 1, 127, 128, 255, 256, 32767, 32768,
                0x7fffffff & ((1 << bits) - 1), 0x80808080 & ((1 << bits) - 1)]
    x = np.resize(np.array(extremes, dtype=np.uint32), batch * cols).reshape(batch, cols)
    return weights, x


def source_hashes():
    paths = ['scripts/probe_limb_stationary_corrections.py',
             'tests/test_limb_stationary_corrections_probe.py',
             'scripts/research_native/limb_stationary_corrections/fixture.c',
             'crates/pllm-core/src/digit_gemm.rs', 'crates/pllm-core/src/paged.rs',
             'crates/pllm-python/src/lib.rs', 'python/pllm/runtime/benchmark_memory.py']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def correctness():
    cases = []
    for bits in (16, 24, 32):
        for rows, cols, batch, page_rows in [(7, 11, 3, None), (1025, 3, 2, None),
                                             (3, 1024, 2, None), (9, 17, 2, 4),
                                             (1027, 2, 2, 1025)]:
            w, x = synthetic_case(rows, cols, batch, bits)
            expected = oracle(w, x, bits)
            results = []
            for stationary in (False, True):
                result, counts = reference(w, OneUseMask(x), bits, stationary=stationary, page_rows=page_rows)
                assert result.tolist() == expected
                results.append(counts)
            cases.append(dict(bits=bits, rows=rows, cols=cols, batch=batch,
                              page_rows=page_rows, baseline=results[0], stationary=results[1]))
    coherent_cases = 0
    for bits in (16, 24, 32):
        for value in (128, (1 << bits) - 1, 0x80808080 & ((1 << bits) - 1)):
            w = np.full((2, 1023), -128, dtype=np.int8)
            x = np.full((1, 1023), value, dtype=np.uint32)
            expected = oracle(w, x, bits)
            for stationary in (False, True):
                result, _ = reference(w, OneUseMask(x), bits, stationary=stationary)
                assert result.tolist() == expected
            coherent_cases += 1
    return dict(kind='synthetic correctness, not native performance', cases=cases,
                cases_passed=len(cases), source_sha256=source_hashes(),
                coherent_adversarial_cases_passed=coherent_cases,
                missing_gates=['native fixture timing/correctness', 'authenticated paged native',
                               'complete response CPU/RSS/TTFT/decode/all-link traffic',
                               'matched fresh-process low/medium tier replicates',
                               'prefill and decode remote body MAC >=90% independently'])


def benchmark(args):
    """Parent-only resident C/Accelerate fixture. Transfers included, not full response."""
    counts = preflight(args.batch, args.rows, args.cols, args.bits)
    if args.rows * args.cols > 16 << 20 or args.repeats not in range(1, 6):
        raise ValueError("fixture bounded to 16 MiB weights and 1..5 repetitions")
    lib = ctypes.CDLL(str(Path(args.native_library).resolve()))
    fn = lib.limb_stationary_corrections
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int]
    fn.restype = ctypes.c_int
    w, _ = synthetic_case(args.rows, args.cols, args.batch, args.bits)
    measurements = []
    # Each mode/rep gets separate deterministic fresh mask. No warmed persistent
    # packing state. Public fixture values are not production random masks.
    digest_by_rep = {}
    for rep in range(args.repeats):
        for stationary in ((False, True) if rep % 2 == 0 else (True, False)):
            _, x = synthetic_case(1, args.cols, args.batch, args.bits)
            x ^= np.uint32(rep)  # Distinct fixture masks, still within ring.
            cpu, wall = time.process_time(), time.perf_counter()
            wire = x.tobytes()
            decoded = (ctypes.c_uint32 * x.size).from_buffer_copy(wire)
            output = (ctypes.c_uint32 * (args.batch * args.rows))()
            status = fn(w.ctypes.data, decoded, output, args.batch, args.rows,
                        args.cols, args.bits, int(stationary))
            response = bytes(output)
            received = np.frombuffer(response, dtype=np.uint32).copy()
            wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
            if status:
                raise RuntimeError(f"native status {status}")
            digest = hashlib.sha256(received.tobytes()).hexdigest()
            if rep in digest_by_rep:
                assert digest_by_rep[rep] == digest
            digest_by_rep[rep] = digest
            # Independent Python oracle, outside timing. Fully check small native
            # fixtures; bounded samples for parent performance geometry.
            if args.batch * args.rows * args.cols <= 100000:
                assert received.reshape(args.batch, args.rows).tolist() == oracle(w, x, args.bits)
            else:
                for r in range(min(args.batch, 2)):
                    for j in (0, args.rows - 1):
                        expected = sum(int(x[r, i]) * int(w[j, i]) for i in range(args.cols)) % (1 << args.bits)
                        assert int(received[r * args.rows + j]) == expected
            measurements.append(dict(rep=rep, stationary=stationary, wall_seconds=wall,
                                     process_cpu_seconds=cpu, output_sha256=digest))
            del x, wire, decoded, output, response, received
    return dict(kind='resident standalone C Accelerate, transfers included; not SDK or full response',
                geometry=[args.batch, args.rows, args.cols, args.bits], preflight=counts,
                measurements=measurements, source_sha256=source_hashes(),
                native_library_sha256=hashlib.sha256(Path(args.native_library).read_bytes()).hexdigest(),
                oracle_scope='full' if args.batch * args.rows * args.cols <= 100000 else '2 rows x 2 output columns',
                peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--native-library')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--cols', type=int, default=1024)
    parser.add_argument('--bits', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args) if args.native_library else correctness()
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
