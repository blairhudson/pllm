"""Bounded exact digit8 certificates and explicit parent-only native research gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1 << 24
L1_LIMIT = (LIMIT - 1) // 128  # 131071; retain native strict postcondition.
WORKSPACE = 64 << 20
MAX_ELEMENTS = 4 << 20
MAX_WEIGHTS = 16 << 20
NATIVE_SOURCE = ROOT / "crates/pllm-core/src/digit_gemm.rs"
FIXTURE_SOURCE = ROOT / "scripts/research_native/certified_digit_blocks/fixture.c"


class NativeFixture:
    """Test-only ctypes owner. Copies source and each issuance across C ABI."""

    def __init__(self, library, weights, batch, bits, certified, certificate=None):
        import ctypes as ct

        geometry(weights, batch, bits)
        self.ct = ct
        self.shape, self.batch, self.bits = weights.shape, batch, bits
        self.lib = ct.CDLL(str(Path(library).resolve()))
        pointer, size = ct.c_void_p, ct.c_size_t
        self.lib.cdb_create.argtypes = [pointer, size, size, size, size, ct.c_uint, ct.c_int, ct.POINTER(pointer)]
        self.lib.cdb_create.restype = ct.c_int
        self.lib.cdb_destroy.argtypes = [pointer]
        self.lib.cdb_destroy.restype = None
        self.lib.cdb_stats.argtypes = [pointer, pointer, size]
        self.lib.cdb_stats.restype = ct.c_int
        self.lib.cdb_records.argtypes = [pointer, pointer, size]
        self.lib.cdb_records.restype = ct.c_int
        self.lib.cdb_execute.argtypes = [pointer, pointer, size, pointer, size, pointer, size]
        self.lib.cdb_execute.restype = ct.c_int
        create_wall, create_cpu = time.perf_counter(), time.process_time()
        source = (ct.c_int8 * weights.size).from_buffer_copy(weights.tobytes())
        self.handle = pointer()
        status = self.lib.cdb_create(source, weights.size, *weights.shape, batch, bits,
                                     int(certified), ct.byref(self.handle))
        if status:
            raise ValueError(f"cdb_create status {status}")
        self.create_timing = {"wall_s": time.perf_counter() - create_wall,
                              "cpu_s": time.process_time() - create_cpu,
                              "scope": "source serialization/ABI copy, C snapshot, source planning and independent C validation"}
        try:
            values = (ct.c_uint64 * 8)()
            if self.lib.cdb_stats(self.handle, values, 8):
                raise ValueError("cdb_stats failed")
            names = ("sgemm_calls", "max_block", "workspace_bytes", "records",
                     "certificate_reserved_bytes", "dense_products", "source_bytes", "handle_owner_bytes")
            self.stats = dict(zip(names, map(int, values)))
            record_data = (ct.c_uint32 * (6 * self.stats["records"]))()
            if self.lib.cdb_records(self.handle, record_data, len(record_data)):
                raise ValueError("cdb_records failed")
            self.records = np.ctypeslib.as_array(record_data).reshape(-1, 6).copy().tolist()
            if certified:
                expected = certificate if certificate is not None else plan(weights, batch, bits)
                if [r[:5] for r in self.records] != expected["records"]:
                    raise ValueError("independent Python/C source certificate mismatch")
                if self.stats["workspace_bytes"] != expected["cost"]["workspace_bytes"]:
                    raise ValueError("Python/C workspace mismatch")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.lib.cdb_destroy(self.handle)
            self.handle = self.ct.c_void_p()

    def __call__(self, inputs, output_mask):
        if not self.handle:
            raise ValueError("native fixture closed")
        ring_input(inputs, (self.batch, self.shape[1]), self.bits)
        ring_input(output_mask, (self.batch, self.shape[0]), self.bits)
        ct = self.ct
        x = (ct.c_uint32 * inputs.size).from_buffer_copy(inputs.tobytes())
        mask = (ct.c_uint32 * output_mask.size).from_buffer_copy(output_mask.tobytes())
        output = (ct.c_uint32 * output_mask.size)()
        status = self.lib.cdb_execute(self.handle, x, len(x), mask, len(mask), output, len(output))
        if status:
            raise ValueError(f"cdb_execute status {status}; output invalid")
        return np.ctypeslib.as_array(output).reshape(output_mask.shape).copy()


def native_identity(payload):
    """Require parent compilation record; never compile implicitly."""
    import platform

    if payload.get("admit_native_run") is not True:
        raise ValueError("explicit native admission required")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("fixture requires Apple Silicon Accelerate")
    record_path = Path(payload["native_build_record"]).resolve()
    record = json.loads(record_path.read_text())
    library = Path(payload["native_library"]).resolve()
    if (record.get("fixture_source_sha256") != file_hash(FIXTURE_SOURCE)
            or record.get("library_sha256") != file_hash(library)
            or record.get("library") != str(library)):
        raise ValueError("native source/build record mismatch")
    return {"library": str(library), "library_sha256": file_hash(library),
            "build_record_sha256": file_hash(record_path),
            "fixture_source_sha256": file_hash(FIXTURE_SOURCE), "compiler": record["compiler"],
            "compile_command": record["compile_command"],
            "probe_sha256": file_hash(__file__),
            "tests_sha256": file_hash(ROOT / "tests/test_certified_digit_blocks_probe.py")}


def compile_worker(payload):
    """Parent-invoked, locked, bounded standalone build; no workspace registration."""
    import platform
    import subprocess

    if payload.get("admit_native_run") is not True:
        raise ValueError("explicit native admission required")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("fixture requires Apple Silicon Accelerate")
    library = Path(payload["native_library"]).resolve()
    if library.exists():
        raise ValueError("refuse overwriting existing native fixture; choose new build path")
    library.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(FIXTURE_SOURCE)
    command = ["/usr/bin/clang", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
               "-Wno-deprecated-declarations",  # Deliberate LP64 Accelerate ABI, as in installed digit8.
               "-dynamiclib", str(FIXTURE_SOURCE), "-framework", "Accelerate", "-o", str(library)]
    compiler = subprocess.check_output([command[0], "--version"], text=True, timeout=10)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    if file_hash(FIXTURE_SOURCE) != source_hash:
        raise RuntimeError("fixture source changed during compile")
    return {"kind": "standalone_certified_digit_blocks_compile", "compiler": compiler,
            "compile_command": command, "compile_stdout": result.stdout, "compile_stderr": result.stderr,
            "fixture_source_sha256": source_hash, "library": str(library),
            "library_sha256": file_hash(library), "platform": platform.platform()}


def native_check_worker(payload):
    """Parent-only adversarial C execution, all outputs versus Python integers."""
    identity = native_identity(payload)
    rng = np.random.default_rng(20261015)
    fixtures = [np.full((3, 2053), -128, dtype=np.int8),
                np.full((3, 4097), 32, dtype=np.int8),
                np.tile(np.array([-128, 127], dtype=np.int8), (3, 2049)),
                rng.integers(-128, 128, (3, 2053), dtype=np.int16).astype(np.int8),
                np.zeros((1025, 17), dtype=np.int8)]
    fixtures[-1][1024] = -128  # Independent tail tile with negative extreme.
    results = []
    for weights in fixtures:
        for bits in (16, 24, 32):
            mask = (1 << bits) - 1
            inputs = rng.integers(0, 1 << bits, (2, weights.shape[1]), dtype=np.uint32)
            inputs[0] = 128
            inputs[1, :6] = np.array([0, 127, 128, 255, 0xffffffff, 0x80808080], dtype=np.uint32) & mask
            outmask = rng.integers(0, 1 << bits, (2, weights.shape[0]), dtype=np.uint32)
            expected = integer_oracle(weights, inputs, bits, outmask)
            for certified in (False, True):
                fixture = NativeFixture(identity["library"], weights, 2, bits, certified)
                try:
                    original = weights.copy()
                    weights[:] = 0
                    try:
                        actual = fixture(inputs, outmask)
                    finally:
                        weights[:] = original
                    np.testing.assert_array_equal(actual, expected)
                    # Bypass Python validation to exercise C's fail-closed domain gate.
                    if bits < 32:
                        ct = fixture.ct
                        bad = (ct.c_uint32 * inputs.size)(*map(int, inputs.ravel()))
                        bad[0] = 1 << bits
                        masks = (ct.c_uint32 * outmask.size)(*map(int, outmask.ravel()))
                        output = (ct.c_uint32 * outmask.size)()
                        if fixture.lib.cdb_execute(fixture.handle, bad, len(bad), masks, len(masks), output, len(output)) != 5:
                            raise AssertionError("C canonical domain gate failed")
                        bad[0] = 0
                        masks[0] = 1 << bits
                        if fixture.lib.cdb_execute(fixture.handle, bad, len(bad), masks, len(masks), output, len(output)) != 5:
                            raise AssertionError("C output-mask domain gate failed")
                    ct = fixture.ct
                    tiny = (ct.c_int8 * 1)(1)
                    rejected = ct.c_void_p()
                    status = fixture.lib.cdb_create(tiny, 1, 16385, 1, 1, bits, 1, ct.byref(rejected))
                    if status != 1 or rejected:
                        raise AssertionError("C geometry rejection failed")
                    results.append({"shape": list(weights.shape), "bits": bits, "certified": certified,
                                    "all_outputs_checked": int(actual.size), "stats": fixture.stats,
                                    "source_sha256": weight_hash(weights),
                                    "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
                finally:
                    fixture.close()
    return {"kind": "adversarial_native_fixture_all_output_integer_oracle", "native_identity": identity,
            "cases": results, "timing": None, "whole_response": None}


def native_measure(weights, payload, certificate):
    """One fresh arm, bounded repeats, full installed exact-integer controls."""
    import resource
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime._native_support import capabilities

    identity = native_identity(payload)
    if weights.shape != (896, 4864) or payload["batch"] not in (1, 150) or payload["bits"] != 32:
        raise ValueError("representative timing restricted to down [896,4864], batch1/150, ring32")
    arm = payload["arm"]
    if arm not in ("fixed", "certified", "installed_integer", "installed_digit8"):
        raise ValueError("unknown native timing arm")
    repeats = payload["repeats"]
    if type(repeats) is not int or not 1 <= repeats <= 4:
        raise ValueError("timing repeats must be 1..4")
    executor = MaskedGEMM(threads=4, build=False)
    if not executor.native or not capabilities().get("digit8_cpu"):
        raise RuntimeError("installed exact integer/digit8 controls unavailable; no fallback")
    if not Path(executor.library_path).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("installed control shadowed by checkout")
    installed_hash = file_hash(executor.library_path)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    integer_matrix = executor.compile(weights)
    installed_setup = {"wall_s": time.perf_counter() - start_wall, "cpu_s": time.process_time() - start_cpu}
    fixture = None
    try:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if arm in ("fixed", "certified"):
            fixture = NativeFixture(identity["library"], weights, payload["batch"], 32,
                                    arm == "certified", certificate=certificate)
        setup = {"wall_s": time.perf_counter() - start_wall, "cpu_s": time.process_time() - start_cpu}
        if fixture is None:
            setup = installed_setup

        def issue(inputs, outmask):
            if fixture is not None:
                return fixture(inputs, outmask)
            transformed = (integer_matrix.wrap32(inputs) if arm == "installed_integer"
                           else integer_matrix.digit8(inputs, 32))
            return np.subtract(transformed, outmask, dtype=np.uint32)

        rng = np.random.default_rng(20261015 + payload["batch"])
        samples = []
        # One equal, untimed warm issuance per fresh process; same fresh-mask
        # sequence across arms. Installed oracle runs AFTER each timed issuance.
        for iteration in range(repeats + 1):
            inputs = rng.integers(0, 1 << 32, (payload["batch"], weights.shape[1]), dtype=np.uint32)
            outmask = rng.integers(0, 1 << 32, (payload["batch"], weights.shape[0]), dtype=np.uint32)
            wall, cpu = time.perf_counter(), time.process_time()
            actual = issue(inputs, outmask)
            elapsed = {"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu}
            expected = np.subtract(integer_matrix.wrap32(inputs), outmask, dtype=np.uint32)
            np.testing.assert_array_equal(actual, expected)  # ALL outputs, never sampled.
            samples.append({"iteration": iteration, "warmup": iteration == 0,
                            "transfer_inclusive_issuance": elapsed, "all_outputs_checked": int(actual.size),
                            "input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
                            "output_mask_sha256": hashlib.sha256(outmask.tobytes()).hexdigest(),
                            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
        if file_hash(executor.library_path) != installed_hash or native_identity(payload) != identity:
            raise RuntimeError("native build changed during timing")
        return {"arm": arm, "batch": payload["batch"], "bits": 32, "shape": list(weights.shape),
                "native_identity": identity, "installed_extension_sha256": installed_hash,
                "installed_oracle_setup": installed_setup, "arm_setup_including_source_copy_and_certificate": setup,
                "c_source_certificate_setup": fixture.create_timing if fixture else None,
                "c_stats": fixture.stats if fixture else None, "samples": samples,
                "process_peak_before_post_source_validation_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "threads": 4, "mask_expansion_timed": False, "oracle_timed": False,
                "network_transfer_timed": False, "whole_response": None,
                "comparison_scope": "C-versus-C isolated; installed arms controls, not C-to-Rust optimization proof"}
    finally:
        if fixture is not None:
            fixture.close()


def validate_matched_native_runs(runs):
    """No pooling if source, mask sequence, exact outputs or builds differ."""
    groups = {}
    for run in runs:
        if len(run["stages"]) != 1:
            raise ValueError("matched cohort needs one stage per worker")
        stage = run["stages"][0]
        measurement = stage["native_measurement"]
        key = (run["replicate"], measurement["batch"])
        arm = measurement["arm"]
        groups.setdefault(key, {})
        if arm in groups[key]:
            raise ValueError("duplicate native cohort arm")
        groups[key][arm] = {
            "revision": run["revision"], "source_lock": run["source_lock_sha256"],
            "implementation_sources": run.get("source_sha256"),
            "source_i8": stage["certificate"]["weight_sha256"], "source_scales": stage["scales_f32_sha256"],
            "key": stage["key"], "native_identity": measurement["native_identity"],
            "installed_extension": measurement["installed_extension_sha256"],
            "samples": [{k: sample[k] for k in ("iteration", "warmup", "input_sha256",
                                                 "output_mask_sha256", "output_sha256", "all_outputs_checked")}
                        for sample in measurement["samples"]],
        }
    if not groups:
        raise ValueError("empty native cohort")
    for arms in groups.values():
        if set(arms) != {"fixed", "certified", "installed_integer", "installed_digit8"}:
            raise ValueError("incomplete native cohort controls")
        reference = arms["fixed"]
        if any(value != reference for value in arms.values()):
            raise ValueError("native cohort source/input/output/build mismatch")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def weight_hash(weights):
    digest = hashlib.sha256()
    for row in weights:
        digest.update(row.tobytes())
    return digest.hexdigest()


def geometry(weights, batch, bits):
    if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
            or weights.ndim != 2 or not weights.flags.c_contiguous
            or min(weights.shape) < 1 or max(weights.shape) > 16384
            or weights.size > MAX_WEIGHTS):
        raise ValueError("bounded contiguous output-by-input i8 source required")
    if (type(batch) is not int or not 1 <= batch <= 4096
            or type(bits) is not int or bits not in (16, 24, 32)
            or batch * max(weights.shape) > MAX_ELEMENTS):
        raise ValueError("native batch/ring geometry exceeded")


def plan(weights, batch, bits, *, tile_rows=1024, max_block=None):
    """Greedy longest contiguous blocks per output tile, using only public weights.

    Costs describe a prospective native implementation, not this Python object's
    heap. Metadata is conservatively reserved for one record per source column
    per tile (24-byte native records), plus 32 bytes per row for planner scratch.
    """
    geometry(weights, batch, bits)
    rows, columns = weights.shape
    if type(tile_rows) is not int or not 1 <= tile_rows <= 1024:
        raise ValueError("tile_rows must be 1..1024")
    tile = min(rows, tile_rows)
    tiles = (rows + tile - 1) // tile
    metadata = 24 * columns * tiles + 32 * tile
    fixed = 4 * (batch * rows + batch * tile) + metadata
    width_cap = min(columns, (WORKSPACE - fixed) // (4 * (batch + tile)))
    if max_block is not None:
        if type(max_block) is not int or max_block < 1:
            raise ValueError("max_block must be positive")
        width_cap = min(width_cap, max_block)
    if width_cap < 1:
        raise MemoryError("certified native workspace exceeds 64 MiB")
    records = []
    for first in range(0, rows, tile):
        last = min(rows, first + tile)
        start = 0
        sums = np.zeros(last - first, dtype=np.uint64)
        for column in range(columns):
            magnitude = np.abs(weights[first:last, column].astype(np.int16)).astype(np.uint64)
            candidate = sums + magnitude
            if column - start == width_cap or np.any(candidate > L1_LIMIT):
                records.append([first, last, start, column, int(sums.max())])
                start = column
                sums = magnitude
            else:
                sums = candidate
        records.append([first, last, start, columns, int(sums.max())])
    peak_k = max(end - start for _, _, start, end, _ in records)
    digits = bits // 8
    base_blocks = (columns + 1022) // 1023
    base_tiles = (rows + 1023) // 1024
    reconstruction = digits * batch * sum((last - first) for first, last, _, _, _ in records)
    cost = {
        "sgemm_calls": digits * len(records),
        "baseline_sgemm_calls": digits * base_blocks * base_tiles,
        "dense_products": digits * batch * rows * columns,
        "baseline_dense_products": digits * batch * rows * columns,
        "weight_conversion_elements": rows * columns,
        "baseline_weight_conversion_elements": rows * columns,
        "mask_digit_conversion_elements": digits * batch * columns * tiles,
        "baseline_mask_digit_conversion_elements": digits * batch * columns * base_tiles,
        "partial_output_reconstruction_elements": reconstruction,
        "baseline_partial_output_reconstruction_elements": digits * batch * rows * base_blocks,
        "workspace_bytes": 4 * (batch * rows + batch * peak_k + tile * peak_k + batch * tile) + metadata,
        "baseline_workspace_bytes": 4 * (batch * rows + batch * min(columns, 1023)
                                         + min(rows, 1024) * min(columns, 1023) + batch * min(rows, 1024)),
        "certificate_reserved_bytes": metadata,
        "certificate_used_bytes": 24 * len(records),
        "public_i8_owner_bytes": weights.size,
        "binding_input_copy_bytes": 4 * batch * columns,
        "binding_output_copy_bytes": 4 * batch * rows,
        "mask_owners_bytes": 4 * batch * (columns + rows),
        "correction_owner_bytes": 4 * batch * rows,
        "wire_correction_bytes": (bits // 8) * batch * rows,
        "blas_internal_workspace_bytes": None,
        "paged_decode_owners_bytes": None,
        "native_timing": None, "whole_client_peak_bytes": None,
    }
    return {"shape": [rows, columns], "batch": batch, "bits": bits,
            "scope": "resident matrix or one decoded page; not whole paged schedule",
            "weight_sha256": weight_hash(weights), "l1_limit": L1_LIMIT,
            "tile_rows": tile, "width_cap": width_cap, "records": records, "cost": cost}


def validate_plan(weights, certificate, batch, bits):
    geometry(weights, batch, bits)
    if (certificate["shape"] != list(weights.shape) or certificate["batch"] != batch
            or certificate["bits"] != bits or certificate["l1_limit"] != L1_LIMIT
            or certificate["weight_sha256"] != weight_hash(weights)):
        raise ValueError("certificate source/domain mismatch")
    # Recompute rather than trusting hashes or caller-supplied sums/coverage.
    expected = plan(weights, batch, bits, tile_rows=certificate["tile_rows"],
                    max_block=certificate["width_cap"])
    if expected != certificate:
        raise ValueError("certificate coverage, bounds or costs modified")


def ring_input(values, shape, bits):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint32
            or values.shape != shape or np.any(values.astype(np.uint64) >= 1 << bits)):
        raise ValueError("canonical u32 ring array required")


def execute(weights, inputs, bits, certificate, output_mask=None):
    """Float32 digit GEMM reference; independent of installed Accelerate kernel."""
    batch = inputs.shape[0]
    validate_plan(weights, certificate, batch, bits)
    ring_input(inputs, (batch, weights.shape[1]), bits)
    modulus_mask = (1 << bits) - 1
    output = np.zeros((batch, weights.shape[0]), dtype=np.uint64)
    for first, last, start, end, _ in certificate["records"]:
        x = inputs[:, start:end].astype(np.uint64)
        w = weights[first:last, start:end].astype(np.float32)
        carry = 0
        for shift in range(0, bits, 8):
            remaining = (x + np.uint64(carry)) >> np.uint64(shift)
            digit = ((remaining + np.uint64(128)) & np.uint64(255)).astype(np.int16) - 128
            product = digit.astype(np.float32) @ w.T
            if (np.any(~np.isfinite(product)) or np.any(np.abs(product) >= LIMIT)
                    or np.any(product != np.trunc(product))):
                raise ArithmeticError("float32 result violates certified bound")
            residue = (product.astype(np.int64) & modulus_mask).astype(np.uint64)
            output[:, first:last] = (output[:, first:last] + (residue << np.uint64(shift))) & modulus_mask
            carry += 128 << shift
    result = output.astype(np.uint32)
    if output_mask is not None:
        ring_input(output_mask, result.shape, bits)
        result = np.subtract(result, output_mask, dtype=np.uint32)
        result &= np.uint32(modulus_mask)
    return result


def integer_oracle(weights, inputs, bits, output_mask=None):
    """Python unbounded integers; no digit decomposition or float reduction."""
    mask = (1 << bits) - 1
    return np.asarray([[(sum(int(w) * int(x) for w, x in zip(row, sample))
                         - (int(output_mask[b, j]) if output_mask is not None else 0)) & mask
                        for j, row in enumerate(weights)]
                       for b, sample in enumerate(inputs)], dtype=np.uint32)


def centered(residues, bits):
    """Decode only after modular subtraction; never subtract centered floats."""
    ring_input(residues, residues.shape, bits)
    value = residues.astype(np.int64)
    return np.where(value >= (1 << (bits - 1)), value - (1 << bits), value)


def synthetic():
    rng = np.random.default_rng(20261014)
    fixtures = {
        "all_minus128": np.full((3, 4097), -128, dtype=np.int8),
        "all_one": np.ones((3, 4097), dtype=np.int8),
        "cancellation_not_a_certificate": np.tile(np.array([-128, 127], dtype=np.int8), (3, 2049)),
        "sparse_extremes": np.zeros((3, 4097), dtype=np.int8),
        "mixed_full_i8": rng.integers(-128, 128, (3, 4097), dtype=np.int16).astype(np.int8),
    }
    fixtures["sparse_extremes"][:, ::128] = -128
    reports = []
    checked = 0
    for name, weights in fixtures.items():
        for bits in (16, 24, 32):
            mask = (1 << bits) - 1
            inputs = rng.integers(0, 1 << bits, (2, weights.shape[1]), dtype=np.uint32)
            inputs[0] = np.uint32(128)  # -128 low digit in every product.
            inputs[1, :8] = np.array([0, 127, 128, 255, 32768, 0x7fffffff, 0xffffffff, 0x80808080], dtype=np.uint32) & mask
            output_mask = rng.integers(0, 1 << bits, (2, weights.shape[0]), dtype=np.uint32)
            certificate = plan(weights, 2, bits)
            actual = execute(weights, inputs, bits, certificate, output_mask)
            expected = integer_oracle(weights, inputs, bits, output_mask)
            if not np.array_equal(actual, expected):
                raise AssertionError("independent integer oracle mismatch")
            checked += actual.size
            reports.append({"fixture": name, "bits": bits,
                            "source_i8_sha256": certificate["weight_sha256"],
                            "block_widths": [e - s for _, _, s, e, _ in certificate["records"]],
                            "max_partial_abs_bound": 128 * max(r[4] for r in certificate["records"]),
                            "cost": certificate["cost"],
                            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
    # Larger shape, count-only; no execution/timing/model claim.
    shape_costs = []
    for value in (-128, 32, 1, 0):
        weights = np.full((1024, 4864), value, dtype=np.int8)
        p = plan(weights, 150, 32)
        shape_costs.append({"constant_i8": value, "shape": p["shape"], "batch": 150,
                            "block_widths": [e - s for _, _, s, e, _ in p["records"]], "cost": p["cost"]})
    return {"kind": "synthetic_exactness_and_symbolic_counts_not_native_measurement",
            "seed": 20261014, "oracle_output_values": checked, "fixtures": reports,
            "count_only_shapes": shape_costs,
            "proof": {"product_abs_max": 16384, "l1_limit": L1_LIMIT,
                      "strict_partial_abs_max": 128 * L1_LIMIT,
                      "uniform_minus128_block": L1_LIMIT // 128,
                      "float32_integer_exact_inclusive_limit": LIMIT},
            "no_checkpoint_scan": True, "native_timing": None, "whole_response": None,
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                              (Path(__file__), NATIVE_SOURCE, ROOT / "crates/pllm-core/src/paged.rs",
                               ROOT / "crates/pllm-core/src/kernels.rs",
                               ROOT / "python/pllm/runtime/transformer_engine.py",
                               ROOT / "tests/test_certified_digit_blocks_probe.py", FIXTURE_SOURCE)},
            "native_fixture": {"path": str(FIXTURE_SOURCE.relative_to(ROOT)),
                               "compiled_or_executed_by_worker": False,
                               "parent_only_batches": [1, 150], "all_output_check_required": True}}


def scan(payload):
    """Optional serial gate: explicitly selected public source tensors, no build."""
    if payload.get("admit_public_source_scan") is not True:
        raise ValueError("explicit public-source scan admission required")
    if (type(payload.get("batch")) is not int or not 1 <= payload["batch"] <= 4096
            or type(payload.get("bits")) is not int or payload["bits"] not in (16, 24, 32)):
        raise ValueError("native batch/ring geometry exceeded before source reads")
    native_requested = payload.get("kind") == "native_timing"
    if native_requested:
        native_identity(payload)  # Before checkpoint hashing/loading.
        if len(payload.get("keys", [])) != 1:
            raise ValueError("native timing requires exactly one public down tensor")
    source_wall, source_cpu = time.perf_counter(), time.process_time()
    from safetensors import safe_open
    import ml_dtypes  # noqa: F401 - BF16 dtype registration
    from pllm.runtime import quantization, _native_support

    root = Path(payload["checkpoint"]).resolve()
    lock_path = Path(payload["source_lock"]).resolve()
    lock_hash = file_hash(lock_path)
    lock = json.loads(lock_path.read_text())
    files = lock.get("sha256")
    if not lock.get("revision") or not isinstance(files, dict) or "config.json" not in files:
        raise ValueError("independently retained revision/config/shard SHA256 lock required")
    for name, digest in files.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid locked source path/hash")
    # Admission before hashing large files or reading tensor values.
    mapped = sum((root / name).stat().st_size for name in files)
    allowance = (512 if native_requested else 256) << 20
    peak = mapped + allowance
    if peak > 2 << 30:
        raise MemoryError("mapped source plus scratch/runtime exceeds 2 GiB scan/native gate")
    for name, digest in files.items():
        if file_hash(root / name) != digest:
            raise ValueError("locked source hash mismatch")
    keys = payload["keys"]
    if not 1 <= len(keys) <= 8 or len(set(keys)) != len(keys):
        raise ValueError("select 1..8 distinct output-by-input public tensor keys")
    locations = {}
    for name in files:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(root / name, framework="np") as handle:
            for key in keys:
                if key not in handle.keys():
                    continue
                view = handle.get_slice(key)
                shape = tuple(view.get_shape())
                if (key in locations or len(shape) != 2 or min(shape) < 1 or max(shape) > 16384
                        or shape[0] * shape[1] > MAX_WEIGHTS
                        or payload["batch"] * max(shape) > MAX_ELEMENTS
                        or view.get_dtype() not in ("F32", "F16", "BF16")):
                    raise ValueError("unsupported/duplicate source tensor geometry or dtype")
                if native_requested and (shape != (896, 4864) or payload["batch"] not in (1, 150) or payload["bits"] != 32):
                    raise ValueError("native timing admits only down [896,4864], batch1/150, ring32")
                locations[key] = (name, shape)
    if set(locations) != set(keys) or sum(r * c for _, (r, c) in locations.values()) > 32 << 20:
        raise ValueError("selected keys missing or total source elements exceed 32 Mi")
    extension = _native_support.extension()
    if extension is None:
        raise RuntimeError("installed quantizer required; no build/fallback")
    for module in (quantization, _native_support, extension):
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("installed SDK required; checkout shadowing rejected")
    paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py", NATIVE_SOURCE,
             ROOT / "tests/test_certified_digit_blocks_probe.py", Path(quantization.__file__),
             Path(_native_support.__file__), Path(extension.__file__)]
    if native_requested:
        from pllm.runtime import native as native_module
        if not Path(native_module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("installed control binding shadowed by checkout")
        paths += [FIXTURE_SOURCE, Path(payload["native_library"]), Path(payload["native_build_record"])]
        paths.append(Path(native_module.__file__))
    frozen = {str(path.resolve()): file_hash(path) for path in paths}
    stages = []
    for key in keys:
        name, (rows, columns) = locations[key]
        weights = np.empty((rows, columns), dtype=np.int8)
        scales_hash = hashlib.sha256()
        with safe_open(root / name, framework="np") as handle:
            view = handle.get_slice(key)
            for first in range(0, rows, 32):
                last = min(rows, first + 32)
                raw = np.asarray(view[first:last, :]).astype(np.float32)
                if not np.all(np.isfinite(raw)):
                    raise ValueError("nonfinite public source")
                quantized = quantization.quantize_weight_per_row(raw, bits=8)
                weights[first:last] = quantized.values
                scales_hash.update(quantized.scales.astype("<f4").tobytes())
        certificate = plan(weights, payload["batch"], payload["bits"])
        measurement = None
        source_setup = None
        if native_requested:
            source_setup = {"wall_s": time.perf_counter() - source_wall,
                            "cpu_s": time.process_time() - source_cpu,
                            "scope": "imports, source-lock hashing, header checks, quantization and Python plan; before arm setup"}
            measurement = native_measure(weights, payload, certificate)
        stages.append({"key": key, "orientation": "explicit output-by-input; no transpose",
                       "scales_f32_sha256": scales_hash.hexdigest(), "certificate": certificate,
                       "source_setup": source_setup, "native_measurement": measurement})
        del weights
    for name, digest in files.items():
        if file_hash(root / name) != digest:
            raise RuntimeError("public source changed during scan")
    for path, digest in frozen.items():
        if file_hash(path) != digest:
            raise RuntimeError("scan implementation changed")
    if file_hash(lock_path) != lock_hash:
        raise RuntimeError("source lock changed")
    process_peak = None
    if native_requested:
        import resource
        process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"kind": "source_locked_native_down_fixture" if native_requested else "source_locked_public_weight_block_scan_counts_only", "revision": lock["revision"],
            "source_lock_sha256": lock_hash, "source_sha256": frozen, "checkpoint_files": files,
            "estimated_scan_peak_bytes": peak, "stages": stages,
            "runtime_scratch_allowance_bytes": allowance,
            "worker_peak_through_post_source_validation_bytes_macos": process_peak,
            "native_timing": "per-stage isolated fixture only" if native_requested else None, "whole_response": None,
            "compiled_stage_commitment_match": None, "whole_topology_admitted": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-checkpoint", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--admit-public-source-scan", action="store_true")
    parser.add_argument("--batch", type=int, default=150)
    parser.add_argument("--bits", type=int, choices=(16, 24, 32), default=32)
    parser.add_argument("--compile-native", action="store_true")
    parser.add_argument("--native-check", action="store_true")
    parser.add_argument("--native-timing", action="store_true")
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-build-record", type=Path)
    parser.add_argument("--admit-native-run", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--replicates", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        payload = json.load(sys.stdin)
        dispatch = {"compile": compile_worker, "native_check": native_check_worker,
                    "native_timing": scan, "scan": scan}
        print(json.dumps(dispatch[payload.get("kind", "scan")](payload)))
        return
    if args.output is None:
        parser.error("--output required")
    modes = sum((args.compile_native, args.native_check, args.native_timing))
    if modes > 1:
        parser.error("select only one native mode")
    if modes:
        if not args.native_library or not args.admit_native_run:
            parser.error("native mode requires --native-library and --admit-native-run")
        if not args.compile_native and not args.native_build_record:
            parser.error("native execution requires independently retained --native-build-record")
        if args.native_timing and (not args.scan_checkpoint or not args.source_lock
                                  or len(args.tensor) != 1 or not args.admit_public_source_scan or args.bits != 32):
            parser.error("native timing needs one public down tensor, source lock, source admission and ring32")
        if not args.native_timing and (args.scan_checkpoint or args.source_lock or args.tensor):
            parser.error("compile/check modes do not load checkpoints")
        from frontier_probe_support import fresh_workers
        base = {"native_library": str(args.native_library.resolve()), "admit_native_run": True}
        if args.native_build_record:
            base["native_build_record"] = str(args.native_build_record.resolve())
        with fresh_workers(Path(__file__).resolve(), memory_bytes=3 << 30, timeout=240) as (run, guard, _):
            if args.compile_native:
                report = run(base | {"kind": "compile"})
            elif args.native_check:
                report = run(base | {"kind": "native_check"})
            else:
                base |= {"kind": "native_timing", "checkpoint": str(args.scan_checkpoint.resolve()),
                         "source_lock": str(args.source_lock.resolve()), "keys": args.tensor,
                         "bits": 32, "admit_public_source_scan": True, "repeats": args.repeats}
                runs = []
                arms = ["fixed", "certified", "installed_integer", "installed_digit8"]
                for replicate in range(args.replicates):
                    for batch in (1, 150):
                        # Rotation balances bounded fresh-process ordering.
                        for arm in arms[replicate:] + arms[:replicate]:
                            result = run(base | {"batch": batch, "arm": arm})
                            result["replicate"] = replicate
                            runs.append(result)
                report = {"kind": "serial_fresh_process_native_down_fixture_cohort", "runs": runs,
                          "repeats": args.repeats, "replicates": args.replicates,
                          "whole_response": None, "tier_qualified": False,
                          "native_fixture_source_sha256": file_hash(FIXTURE_SOURCE)}
                validate_matched_native_runs(runs)
            report["installed_build"] = run.build
            report["observed_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    elif args.scan_checkpoint:
        if not args.source_lock or not args.tensor or not args.admit_public_source_scan:
            parser.error("scan requires source lock, explicit tensor keys and admission flag")
        from frontier_probe_support import fresh_workers
        with fresh_workers(Path(__file__).resolve(), memory_bytes=3 << 30, timeout=240) as (run, guard, _):
            report = run({"checkpoint": str(args.scan_checkpoint.resolve()),
                          "source_lock": str(args.source_lock.resolve()), "keys": args.tensor,
                          "batch": args.batch, "bits": args.bits, "admit_public_source_scan": True})
            report["installed_build"] = run.build
            report["observed_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    else:
        if (args.source_lock or args.tensor or args.admit_public_source_scan or args.native_library
                or args.native_build_record or args.admit_native_run):
            parser.error("scan arguments require --scan-checkpoint")
        report = synthetic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
