"""Bounded, read-only ORIGINAL nearest/2048/i16/ring64 checkpoint admission.

This is publication/source work, never inference or a serving benchmark. Only
cached official bytes are read. No tensor mapping, Torch, model execution,
runtime composition, source downloads, or author implementation is used.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import fnmatch
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
GIB = 1 << 30
BUFFER_BYTES = MIB
MAX_HEADER = MIB
MAX_TENSORS = 4096
MAX_SOURCE_BYTES = 2 * GIB
PROCESS_CEILING = 512 * MIB
HOST_RESERVE = 4 * GIB
WALL_SECONDS = 300
CPU_SECONDS = 240
I64_MAX = (1 << 63) - 1
PIN = "7ae557604adf67be50417f59c2c2f167def9a775"
CANDIDATE_BOUNDS = (2048, 32768, 2097152)
SIZES = {"BF16": 2, "F16": 2, "F32": 4}


class Rejected(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejected(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def document(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(Rejected(value)))


def atomic_json(path, value):
    """Atomic exclusive publication: an earlier failure receipt cannot be replaced."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    payload = canonical(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def file_identity(path, fd):
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode):
        raise Rejected("source is not a regular file")
    linked, target = path.lstat(), path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    opened_fields = [getattr(opened, field) for field in fields]
    if opened_fields != [getattr(target, field) for field in fields]:
        raise Rejected(f"source path no longer identifies opened file: {path}")
    return {"opened": opened_fields, "link": [getattr(linked, field) for field in fields],
            "resolved_path": str(path.resolve(strict=True)),
            "symlink": os.readlink(path) if path.is_symlink() else None}


class InputFile:
    def __init__(self, path, io, guard):
        self.path, self.io, self.guard = Path(path), io, guard
        self.fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            self.before = file_identity(self.path, self.fd)
            self.size = os.fstat(self.fd).st_size
        except BaseException:
            os.close(self.fd)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        os.close(self.fd)

    def check_identity(self):
        after = file_identity(self.path, self.fd)
        if after != self.before:
            raise Rejected(f"immutable input changed: {self.path}")
        return after

    def read(self, offset, length, purpose):
        if not 0 <= length <= BUFFER_BYTES or not 0 <= offset <= self.size - length:
            raise Rejected("pread request exceeds admitted slice or source extent")
        self.guard.check()
        raw = os.pread(self.fd, length, offset)
        self.io[purpose] += len(raw)
        if len(raw) != length:
            raise Rejected(f"short pread: {self.path}")
        self.guard.check()
        return raw

    def hashes(self, purpose):
        sha, git = hashlib.sha256(), hashlib.sha1()
        git.update(f"blob {self.size}\0".encode())
        for offset in range(0, self.size, BUFFER_BYTES):
            raw = self.read(offset, min(BUFFER_BYTES, self.size - offset), purpose)
            sha.update(raw)
            git.update(raw)
            del raw
        self.check_identity()
        return {"sha256": sha.hexdigest(), "git_blob_sha1": git.hexdigest()}


def host_snapshot():
    import psutil

    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"total_bytes": memory.total, "available_bytes": memory.available,
            "swap_used_bytes": swap.used, "swapout_bytes": swap.sout,
            "rss_bytes": psutil.Process().memory_info().rss}


def resource_reason(sample, baseline, estimate, *, elapsed, cpu):
    reserve = max(HOST_RESERVE, sample["total_bytes"] // 8)
    if sample["available_bytes"] < reserve:
        return "host_reserve_breach"
    if sample["swap_used_bytes"] > baseline["swap_used_bytes"]:
        return "new_swap_usage"
    if sample["swapout_bytes"] > baseline["swapout_bytes"]:
        return "new_swapout"
    if sample["rss_bytes"] > estimate:
        return "process_estimate_exceeded"
    if elapsed > WALL_SECONDS or cpu > CPU_SECONDS:
        return "publication_scan_time_budget_exceeded"
    return None


class Guard:
    def __init__(self, started, *, sample=host_snapshot, clock=None):
        self.started, self.sample = started, sample
        self.clock = clock or time.monotonic
        self.baseline_started = self.clock()
        self.baseline = sample()
        self.baseline_ended = self.clock()
        self.minimum_swap = self.baseline["swap_used_bytes"]
        self.estimate = PROCESS_CEILING
        self.last_time = 0.0
        self.samples = 0
        self.last = self.baseline
        self.minimum_available = self.baseline["available_bytes"]
        self.peak_rss = self.baseline["rss_bytes"]
        self.max_gap = 0.0
        self.max_observation_duration = self.baseline_ended - self.baseline_started
        self.last_work = None

    def admit_bootstrap(self):
        if self.baseline_ended - self.started > WALL_SECONDS:
            raise Rejected("publication_scan_time_budget_exceeded")
        reserve = max(HOST_RESERVE, self.baseline["total_bytes"] // 8)
        if self.baseline["available_bytes"] < reserve + PROCESS_CEILING:
            raise Rejected("insufficient_host_headroom_before_metadata_import")

    def check(self, *, force=False):
        observation_started = self.clock()
        if not force and observation_started - self.last_time < 0.25:
            return
        current = self.sample()
        # The telemetry read itself can cross the deadline. Clock/usage after
        # observation must decide admission, including the final forced sample.
        self.last_work = work_observation(self.started, clock=self.clock)
        now = self.started + self.last_work["observed_through_elapsed_seconds"]
        if self.last_time:
            self.max_gap = max(self.max_gap, now - self.last_time)
        self.max_observation_duration = max(self.max_observation_duration, now - observation_started)
        self.last_time, self.last = now, current
        self.samples += 1
        self.minimum_available = min(self.minimum_available, current["available_bytes"])
        self.peak_rss = max(self.peak_rss, current["rss_bytes"])
        self.minimum_swap = min(self.minimum_swap, current["swap_used_bytes"])
        reason = resource_reason(current, self.baseline | {"swap_used_bytes": self.minimum_swap},
                                 self.estimate, elapsed=now - self.started,
                                 cpu=self.last_work["scanner_user_cpu_seconds"]
                                     + self.last_work["scanner_system_cpu_seconds"])
        if reason:
            raise Rejected(reason)

    def report(self):
        return {"baseline": self.baseline, "last": self.last, "samples": self.samples,
                "minimum_available_bytes": self.minimum_available,
                "peak_sampled_scanner_rss_bytes": self.peak_rss,
                "maximum_sample_gap_seconds": self.max_gap,
                "maximum_observation_duration_seconds": self.max_observation_duration,
                "baseline_observation_start_elapsed_seconds": self.baseline_started - self.started,
                "baseline_observation_end_elapsed_seconds": self.baseline_ended - self.started,
                "last_observation_end_elapsed_seconds": self.last_time - self.started if self.samples else None,
                "policy": "available memory reserve >= max(4GiB, total/8); any new swap aborts",
                "scope": "single scanner process through last completed observation; not a lifetime or hard-limit guarantee"}


def work_observation(started, *, clock=time.monotonic):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"observed_through_elapsed_seconds": clock() - started,
            "scanner_user_cpu_seconds": usage.ru_utime, "scanner_system_cpu_seconds": usage.ru_stime,
            "child_user_cpu_seconds": children.ru_utime, "child_system_cpu_seconds": children.ru_stime,
            "scanner_high_water_rss_at_observation_bytes": usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024,
            "scope": "process-cumulative CPU/RSS counters at this observation; subsequent work excluded",
            "serving_speed_or_RSS_measurement": False}


def imported_metadata_modules():
    """Use import-system objects, never checkout paths, as executed origins."""
    names = ("pllm", "pllm.runtime", "pllm.modeling", "pllm.model_loader",
             "pllm.runtime.semantic_tensors", "pllm._native", "scripts.gkm_checkpoint_oracle")
    return {name: importlib.import_module(name) for name in names}


def module_origin(module):
    filename = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if not filename or not origin or Path(filename).resolve() != Path(origin).resolve():
        raise Rejected("imported module has no consistent file/spec origin")
    return {"module_file": str(Path(filename).absolute()), "spec_origin": str(Path(origin).absolute()),
            "cached_bytecode": getattr(module, "__cached__", None),
            "package_search_locations": list(getattr(spec, "submodule_search_locations", None) or ()),
            "loader": type(getattr(spec, "loader", None)).__qualname__,
            "version": getattr(module, "__version__", None)}


def bind_imported_modules(modules, open_input, receipt):
    """Hash actual imported package/metadata/native files before calling lowering."""
    receipt.update(scope="selected actual imported module origins and available bytecode; not a complete dependency closure",
                   modules={}, checks={"before_semantic_lowering": "incomplete"})
    bindings = []
    for name, module in modules.items():
        if sys.modules.get(name) is not module:
            raise Rejected(f"imported module object changed: {name}")
        identity = module_origin(module)
        row = receipt["modules"][name] = {**identity, "artifacts": []}
        paths = [Path(identity["module_file"])]
        cached = identity["cached_bytecode"]
        if cached and Path(cached).is_file() and Path(cached).resolve() != paths[0].resolve():
            paths.append(Path(cached))
        artifacts = []
        for path in paths:
            source = open_input(path)
            record = {"path": str(path.absolute()), "bytes": source.size, "identity_before": source.before}
            row["artifacts"].append(record)
            record["sha256_before"] = source.hashes("executed_metadata_hash_before")["sha256"]
            artifacts.append((source, record))
        bindings.append((name, module, identity, artifacts))
    receipt["checks"]["before_semantic_lowering"] = "complete"
    return bindings


def recheck_imported_modules(bindings, receipt, phase):
    receipt["checks"][phase] = "incomplete"
    for name, module, identity, artifacts in bindings:
        if sys.modules.get(name) is not module or module_origin(module) != identity:
            raise Rejected(f"imported module binding changed: {name}")
        for source, record in artifacts:
            source.check_identity()
            after = source.hashes("executed_metadata_hash_" + phase)["sha256"]
            record[phase] = {"sha256": after, "identity": source.check_identity()}
            if after != record["sha256_before"]:
                raise Rejected(f"imported module bytes changed: {name}")
    receipt["checks"][phase] = "complete"


def parse_header(source):
    length = struct.unpack("<Q", source.read(0, 8, "headers"))[0]
    if not 2 <= length <= MAX_HEADER or length % 8:
        raise Rejected("invalid or oversized safetensors header")
    raw = source.read(8, length, "headers")
    header = document(raw)
    if not isinstance(header, dict) or not 1 <= len(header) <= MAX_TENSORS + 1:
        raise Rejected("invalid tensor directory")
    tensors = {}
    for key, row in header.items():
        if key == "__metadata__":
            if not isinstance(row, dict) or any(not isinstance(v, str) for v in row.values()):
                raise Rejected("invalid safetensors metadata")
            continue
        if not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}:
            raise Rejected("invalid tensor metadata")
        shape, offsets, dtype = row["shape"], row["data_offsets"], row["dtype"]
        if (dtype not in SIZES or not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            raise Rejected("unsupported tensor geometry or dtype")
        count = math.prod(shape)
        if count > 1 << 30 or offsets[1] - offsets[0] != count * SIZES[dtype]:
            raise Rejected("tensor size does not match its shape")
        if shape[-1] * SIZES[dtype] > BUFFER_BYTES:
            raise Rejected("one source row exceeds the bounded read buffer")
        tensors[key] = {**row, "elements": count, "absolute_offset": 8 + length + offsets[0]}
    cursor = 0
    for row in sorted(tensors.values(), key=lambda item: item["data_offsets"][0]):
        if row["data_offsets"][0] != cursor:
            raise Rejected("tensor intervals overlap or leave unbound bytes")
        cursor = row["data_offsets"][1]
    if not tensors or 8 + length + cursor != source.size:
        raise Rejected("tensor payload extent differs from source file size")
    return tensors, {"header_bytes": length, "header_sha256": hashlib.sha256(raw).hexdigest(),
                     "payload_bytes": cursor, "tensor_count": len(tensors)}


def decode_round(raw, dtype):
    import numpy as np

    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype("<u4")
        np.left_shift(bits, 16, out=bits)
        values = bits.view("<f4").astype(np.float64)
    else:
        values = np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4"}[dtype]).astype(np.float64)
    finite = np.isfinite(values)
    # Every finite BF16/F16/F32 value and its exact power-of-two scaling fits
    # binary64. No intermediate float32 multiplication or integer cast occurs.
    with np.errstate(invalid="ignore"):
        rounded = np.rint(np.ldexp(values, 11))
    return values, rounded, finite


def compare_oracle(raw, dtype, values, rounded, indices):
    from scripts.gkm_checkpoint_oracle import scalar_bits

    checked = []
    size = SIZES[dtype]
    for index in sorted(set(indices)):
        encoded = raw[index * size:(index + 1) * size]
        oracle = scalar_bits(encoded, dtype)
        value = float(values[index])
        if math.isfinite(value) != oracle["finite"]:
            raise Rejected("scalar oracle finite classification mismatch")
        if oracle["finite"]:
            numerator, denominator = value.as_integer_ratio()
            bits = int.from_bytes(struct.pack("<f", value), "little")
            if (numerator * oracle["denominator"] != oracle["numerator"] * denominator
                    or int(rounded[index]) != oracle["rounded"] or bits != oracle["float32_bits"]):
                raise Rejected(f"scalar integer/bit oracle mismatch at sample {index}")
        checked.append({"index": int(index), "encoded_hex": encoded.hex(), **oracle})
    return checked


def semantic_inventory(plan, tensors, *, tensor_api=None):
    if tensor_api is None:
        tensor_api = importlib.import_module("pllm.runtime.semantic_tensors")
    required = tensor_api.required_checkpoint_tensors(plan)
    local = tensor_api.required_client_tensors(plan)
    for key, shape in required.items():
        if key not in tensors or tuple(tensors[key]["shape"]) != shape:
            raise Rejected(f"semantic source artifact missing or wrong shape: {key}")
    if set(required) != set(tensors):
        raise Rejected("source contains tensors not owned by the semantic lowering")
    phases = []
    for phase in ("prefill", "decode"):
        operations = {row["id"]: row for row in plan.to_dict()[phase]["operations"]}
        rows = []
        for op in operations.values():
            kind = op["operator"]
            if kind not in {"linear", "output_head", "token_lookup"}:
                continue
            weight = op["attributes"]["weight"]
            m, n = tensors[weight]["shape"]
            if kind != "token_lookup":
                if (len(op["inputs"]) != 1 or op["inputs"][0] not in operations
                        or operations[op["inputs"][0]]["output_shape"][-1] != n
                        or op["output_shape"][-1] != m):
                    raise Rejected("semantic matrix orientation is unsupported")
            rows.append({"id": op["id"], "operator": kind, "layer": op["layer"],
                         "input_ids": op["inputs"], "source_key": weight,
                         "bias_key": op["attributes"].get("bias"), "shape_out_in": [m, n]})
        phases.append(rows)
    if phases[0] != phases[1]:
        raise Rejected("linear/table source ownership differs between phases")
    return {"artifacts": phases[0], "local_tensors": {key: list(shape) for key, shape in local.items()},
            "fused_groups": {"status": "blocked", "groups": None,
                "reason": "Existing classify_remote_groups is private Rust; Python runtime_schedule "
                          "requires a runtime composition. No GKM source-numeric composition is "
                          "admitted here. Per-operation source bounds are available; fused GKM "
                          "groups, numeric schedule and final publication layout are unresolved."}}


def process_estimate(tensors):
    # Worst overlap: raw bytes, widened bits, values, scaled/rint values, masks,
    # filtered reductions, int64 q/abs, f32/i16 serializations and row summaries.
    elements = BUFFER_BYTES // min(SIZES.values())
    components = {"python_native_numpy_imports_and_baseline": 256 * MIB,
                  "raw_pread_buffer": BUFFER_BYTES,
                  "bounded_array_overlap_64_bytes_per_element": 64 * elements,
                  "metadata_oracle_report_and_atomic_json_overlap": 96 * MIB,
                  "allocator_slack": 64 * MIB,
                  "telemetry_os_helper_allowance": 16 * MIB}
    total = sum(components.values())
    if total > PROCESS_CEILING or len(tensors) > MAX_TENSORS:
        raise Rejected("process estimate exceeds fixed probe ceiling")
    return {"components_bytes": components, "complete_process_estimate_bytes": total,
            "fixed_ceiling_bytes": PROCESS_CEILING, "max_read_bytes": BUFFER_BYTES,
            "numerical_threads": 1, "persistent_processes": 1,
            "wall_seconds_cap": WALL_SECONDS, "cpu_seconds_cap": CPU_SECONDS,
            "retained_weight_tensor_bytes": 0, "retained_row_L1_array_bytes": 0}


def scan_tensor(source, key, meta, owners, *, buffer_bytes=BUFFER_BYTES):
    import numpy as np

    if not 1 <= buffer_bytes <= BUFFER_BYTES:
        raise Rejected("invalid bounded buffer")
    shape, dtype, count = meta["shape"], meta["dtype"], meta["elements"]
    width, size = shape[-1], SIZES[dtype]
    if buffer_bytes < width * size:
        raise Rejected("buffer cannot hold one source row")
    chunk_elements = (buffer_bytes // (width * size)) * width
    linear = any(owner["operator"] in {"linear", "output_head"} for owner in owners)
    table = any(owner["operator"] == "token_lookup" for owner in owners)
    raw_hash, q_hash, float_hash, rows_hash, l1_hash, oracle_hash = (hashlib.sha256() for _ in range(6))
    result = {"source_key": key, "dtype": dtype, "shape": shape, "elements": count,
              "owners": owners, "linear_i16_owner": linear, "float32_table_owner": table,
              "nonfinite_count": 0, "finite_out_of_i16_count": 0, "negative_zero_count": 0,
              "source_min": None, "source_max": None, "max_abs_source": None,
              "rounded_min": None, "rounded_max": None, "max_abs_rounded": None,
              "max_row_L1": 0 if linear else None, "max_row_L1_index": None,
              "oracle_samples_checked": 0, "oracle_examples": [], "rejection_examples": []}
    for start in range(0, count, chunk_elements):
        take = min(chunk_elements, count - start)
        raw = source.read(meta["absolute_offset"] + start * size, take * size, "source_value_scan")
        raw_hash.update(raw)
        values, rounded, finite = decode_round(raw, dtype)
        bad = finite & ((rounded < -32768) | (rounded > 32767))
        result["nonfinite_count"] += take - int(np.count_nonzero(finite))
        result["finite_out_of_i16_count"] += int(np.count_nonzero(bad))
        result["negative_zero_count"] += int(np.count_nonzero((values == 0) & np.signbit(values)))
        selected = {0, take // 2, take - 1}
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            low = int(finite_indices[np.argmin(values[finite])])
            high = int(finite_indices[np.argmax(values[finite])])
            selected.update((low, high))
            for name, value, reducer in (
                ("source_min", float(values[low]), min), ("source_max", float(values[high]), max),
                ("max_abs_source", max(abs(float(values[low])), abs(float(values[high]))), max),
                ("rounded_min", int(rounded[low]), min), ("rounded_max", int(rounded[high]), max),
                ("max_abs_rounded", max(abs(int(rounded[low])), abs(int(rounded[high]))), max),
            ):
                result[name] = value if result[name] is None else reducer(result[name], value)
            del finite_indices
        checked = compare_oracle(raw, dtype, values, rounded, selected)
        for sample in checked:
            sample["index"] += start
            oracle_hash.update(canonical(sample) + b"\n")
        result["oracle_samples_checked"] += len(checked)
        result["oracle_examples"].extend(checked[:max(0, 8 - len(result["oracle_examples"]))])
        if np.any(bad) or not np.all(finite):
            for index in np.flatnonzero(bad | ~finite)[:max(0, 8 - len(result["rejection_examples"]))]:
                result["rejection_examples"].append({"index": start + int(index),
                    "encoded_hex": raw[int(index) * size:(int(index) + 1) * size].hex(),
                    "finite": bool(finite[index]),
                    "rounded": int(rounded[index]) if finite[index] else None})
        elif linear:
            q = rounded.astype(np.int64).reshape(-1, width)
            q_hash.update(q.astype("<i2").tobytes())
            # width*32768 fits int64 by header admission, so every reduction is exact.
            row_l1 = np.abs(q).sum(axis=1, dtype=np.int64)
            l1_hash.update(row_l1.astype("<i8", copy=False).tobytes())
            index = int(np.argmax(row_l1))
            if result["max_row_L1_index"] is None or int(row_l1[index]) > result["max_row_L1"]:
                result["max_row_L1"] = int(row_l1[index])
                result["max_row_L1_index"] = start // width + index
            del q, row_l1
        if np.all(finite):
            converted = values.astype("<f4").tobytes()
            float_hash.update(converted)
            if table:
                for offset in range(0, len(converted), width * 4):
                    rows_hash.update(hashlib.sha256(memoryview(converted)[offset:offset + width * 4]).digest())
            del converted
        del raw, values, rounded, finite, bad
    source.check_identity()
    valid_q = result["nonfinite_count"] == result["finite_out_of_i16_count"] == 0
    result["raw_source_sha256"] = raw_hash.hexdigest()
    result["oracle_transcript_sha256"] = oracle_hash.hexdigest()
    result["float32_le_sha256"] = float_hash.hexdigest() if result["nonfinite_count"] == 0 else None
    result["table_row_sha256_vector_sha256"] = (rows_hash.hexdigest()
        if table and result["nonfinite_count"] == 0 else None)
    result["i16_le_sha256"] = q_hash.hexdigest() if linear and valid_q else None
    result["row_L1_i64_le_sha256"] = l1_hash.hexdigest() if linear and valid_q else None
    if linear and not valid_q:
        result["max_row_L1"] = result["max_row_L1_index"] = None
    if linear and valid_q:
        l1 = result["max_row_L1"]
        safe = I64_MAX // l1 if l1 else I64_MAX
        result["safe_uniform_encoded_input_max"] = safe
        result["zero_rows_policy"] = "zero L1 is unconstrained by weights; symmetric signed-i64 cap applies"
        result["conditional_public_bounds"] = [
            {"encoded_input_abs_max": bound, "integer_output_abs_max": bound * l1,
             "strictly_below_2_to_63": bound * l1 < 1 << 63} for bound in CANDIDATE_BOUNDS]
    return result


def geometry(inventory, tensors, source_bytes):
    artifacts = inventory["artifacts"]
    linear = [row for row in artifacts if row["operator"] in {"linear", "output_head"}]
    tables = [row for row in artifacts if row["operator"] == "token_lookup"]
    unique_body = {row["source_key"] for row in linear if row["operator"] == "linear"}
    unique_head = {row["source_key"] for row in linear if row["operator"] == "output_head"}
    unique_table = {row["source_key"] for row in tables}
    bias = {row["bias_key"] for row in linear if row["bias_key"] is not None}
    local = set(inventory["local_tensors"]) | bias
    elements = lambda keys: sum(tensors[key]["elements"] for key in keys)
    m_max = max(row["shape_out_in"][0] for row in linear)
    sum_n = sum(row["shape_out_in"][1] for row in linear)
    sum_m = sum(row["shape_out_in"][0] for row in linear)
    table_rows = sum(tensors[key]["shape"][0] for key in unique_table)
    return {"source_files_retained_disk_bytes": source_bytes,
            "source_tensor_payload_bytes": sum(row["elements"] * SIZES[row["dtype"]] for row in tensors.values()),
            "body_linear_artifacts": len(linear) - len(unique_head),
            "body_i16_bytes_per_worker": 2 * elements(unique_body),
            "head_i16_bytes_per_worker": 2 * elements(unique_head),
            "raw_float32_table_bytes_per_worker": 4 * elements(unique_table),
            "tied_source_keys_table_and_head": sorted(unique_table & unique_head),
            "distinct_encoding_policy": "float32 table and i16 head are separate physical owners",
            "linear_i16_bytes_per_worker": 2 * elements(unique_body | unique_head),
            "table_plus_linear_payload_bytes_per_worker": 2 * elements(unique_body | unique_head) + 4 * elements(unique_table),
            "two_workers_table_plus_linear_payload_bytes": 2 * (2 * elements(unique_body | unique_head) + 4 * elements(unique_table)),
            "client_local_float32_coefficients_bytes": 4 * elements(local),
            "client_bias_tensor_count": len(bias), "client_bias_float32_bytes": 4 * elements(bias),
            "publication_embedding_row_hash_bytes": 32 * table_rows,
            "conditional_unfused_publication": {"status": "geometry_only_not_published",
                "matrix_count": len(linear), "global_R_elements": m_max, "S_elements": sum_n,
                "compressed_points_33_bytes_each": 33 * (m_max + sum_n),
                "owner_secret_scalar_payload_32_bytes_each": 32 * m_max,
                "client_accumulator_scalar_payload_32_bytes_each": 32 * (sum_m + sum_n),
                "condition": "one matrix per lowered semantic linear/output-head operation; final groups unresolved"},
            "excluded_from_geometry": ["decoded EC owners and MSM scratch", "runtime copies and allocator overhead",
                "KV, local execution and verifier queues", "source publication generation and delivery scratch"],
            "payloads_are_not_RSS_or_serving_admission": True}


def source_pin():
    tree = ast.parse((ROOT / "benchmarks/research/common.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"MODEL", "REVISION"}}
    if values != {"MODEL": "Qwen/Qwen2.5-0.5B-Instruct", "REVISION": PIN}:
        raise Rejected("research source pin changed")
    return values


def verify_expected(name, source, expected, hashes):
    if source.size != expected["size"]:
        raise Rejected(f"official cached size mismatch: {name}")
    if expected.get("lfs_sha256"):
        if hashes["sha256"] != expected["lfs_sha256"] or source.size != expected["lfs_size"]:
            raise Rejected(f"official LFS SHA-256 mismatch: {name}")
    elif hashes["git_blob_sha1"] != expected["blob_id"]:
        raise Rejected(f"official Git blob SHA-1 mismatch: {name}")


def scan(report, output, guard):
    pin = source_pin()
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    cache = Path(os.environ.get("HF_HUB_CACHE", str(home / "hub"))).expanduser()
    repository = cache / ("models--" + pin["MODEL"].replace("/", "--"))
    snapshot = repository / "snapshots" / pin["REVISION"]
    io = Counter()
    report.update(source={"repo_id": pin["MODEL"], "revision": pin["REVISION"], "snapshot": str(snapshot)},
                  io_bytes=io, tensors=[], executed_metadata={})
    with ExitStack() as stack:
        open_input = lambda path: stack.enter_context(InputFile(path, io, guard))
        modules = imported_metadata_modules()
        bindings = bind_imported_modules(modules, open_input, report["executed_metadata"])
        loader = modules["pllm.model_loader"]
        tree = open_input(repository / "trees" / (pin["REVISION"] + ".json"))
        tree_raw = tree.read(0, tree.size, "source_metadata")
        official = document(tree_raw)
        if official.get("format_version") != 1 or not isinstance(official.get("files"), dict):
            raise Rejected("cached official file tree unavailable")
        report["source"]["official_tree_sha256"] = hashlib.sha256(tree_raw).hexdigest()
        expected = {name: row for name, row in official["files"].items()
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in loader._DIRECTORY_PATTERNS)}
        paths = loader._source_files(snapshot)
        if {path.name for path in paths} != set(expected):
            raise Rejected("cached checkpoint/tokenizer file set differs from official lockable tree")
        if any(Path(name).name != name for name in expected):
            raise Rejected("nested source paths are not supported by this bounded gate")
        sources = {path.name: open_input(path) for path in paths}
        source_bytes = sum(source.size for source in sources.values())
        if not 0 < source_bytes <= MAX_SOURCE_BYTES:
            raise Rejected("cached source exceeds the fixed 2GiB scan scope")
        tensors, headers = {}, {}
        for name, source in sources.items():
            if name.endswith(".safetensors"):
                entries, headers[name] = parse_header(source)
                if set(entries) & set(tensors):
                    raise Rejected("duplicate source tensor across shards")
                tensors.update({key: {**row, "file": name} for key, row in entries.items()})
        if not tensors or len(tensors) > MAX_TENSORS:
            raise Rejected("empty or oversized checkpoint tensor inventory")
        config_raw = sources["config.json"].read(0, sources["config.json"].size, "source_metadata")
        config = document(config_raw)
        plan = modules["pllm.modeling"].lower_model(config, batch=1, max_input_tokens=1, max_new_tokens=1)
        inventory = semantic_inventory(plan, tensors, tensor_api=modules["pllm.runtime.semantic_tensors"])
        recheck_imported_modules(bindings, report["executed_metadata"], "after_semantic_lowering")
        estimate = process_estimate(tensors)
        guard.estimate = estimate["complete_process_estimate_bytes"]
        guard.check(force=True)
        import shutil
        if shutil.disk_usage(output).free < 64 * MIB:
            raise Rejected("insufficient_receipt_disk_reserve")
        report.update(config=config, config_sha256=hashlib.sha256(config_raw).hexdigest(),
                      headers=headers, semantic_plan_digest=plan.digest, semantic_inventory=inventory,
                      geometry=geometry(inventory, tensors, source_bytes), admission=estimate)
        atomic_json(output / "admission.json", {"status": "admitted_before_value_reads", **estimate,
            "source_bytes": source_bytes, "planned_source_hash_and_value_read_bytes": 3 * source_bytes,
            "host": guard.report(), "geometry": report["geometry"], "source": report["source"],
            "executed_metadata": report["executed_metadata"]})
        code_paths = [Path(__file__).resolve(), ROOT / "scripts/gkm_checkpoint_oracle.py",
                      ROOT / "tests/test_gkm_checkpoint_bounds.py", ROOT / "benchmarks/research/common.py",
                      ROOT / "paper/methods/DECISIONS.md", ROOT / "paper/methods/delegation-contracts.md"]
        code = [open_input(path) for path in code_paths]
        report["probe_and_contract_hashes"] = {str(source.path): source.hashes("code_hash_before")["sha256"] for source in code}
        checkout = [open_input(ROOT / path) for path in ("python/pllm/modeling.py", "python/pllm/model_loader.py",
                                                        "python/pllm/runtime/semantic_tensors.py")]
        report["checkout_reference_hashes"] = {
            "scope": "checkout reference files only; not identities of imported/executed modules",
            "files": {str(source.path): source.hashes("checkout_reference_hash_before")["sha256"] for source in checkout}}
        report["source"]["files"] = files = []
        for name, source in sources.items():
            hashes = source.hashes("source_hash_before")
            row = {"path": name, "size": source.size, **hashes,
                   "expected_official_identity": expected[name], "identity_before": source.before}
            files.append(row)
            verify_expected(name, source, expected[name], hashes)
        lock_rows = [{key: row[key] for key in ("path", "size", "sha256")} for row in files]
        checkpoint = loader._digest(loader._CHECKPOINT_DOMAIN, {"files": lock_rows})
        lock = {"schema": "pllm.model_source_lock.v1", "kind": "huggingface",
                "repo_id": pin["MODEL"], "revision": pin["REVISION"], "commit": pin["REVISION"],
                "files": lock_rows, "checkpoint_digest": checkpoint}
        report["source"].update(checkpoint_digest=checkpoint, source_lock=lock,
                                 source_lock_digest=loader._digest(loader._SOURCE_LOCK_DOMAIN, lock))
        owners = {key: [] for key in tensors}
        for artifact in inventory["artifacts"]:
            owners[artifact["source_key"]].append(artifact)
            if artifact["bias_key"] is not None:
                owners[artifact["bias_key"]].append({"operator": "client_local_bias", "id": artifact["id"]})
        for key in inventory["local_tensors"]:
            owners[key].append({"operator": "client_local_coefficient"})
        for key, meta in sorted(tensors.items(), key=lambda pair: (pair[1]["file"], pair[1]["absolute_offset"])):
            report["tensors"].append(scan_tensor(sources[meta["file"]], key, meta, owners[key]))
        for row in files:
            source = sources[row["path"]]
            after = source.hashes("source_hash_after")
            row.update(sha256_after=after["sha256"], identity_after=source.check_identity())
            if any(after[name] != row[name] for name in ("sha256", "git_blob_sha1")):
                raise Rejected("source content changed between full hashes")
        report["source"]["official_tree_identity_before"] = tree.before
        report["source"]["official_tree_identity_after"] = tree.check_identity()
        if tree.hashes("source_metadata_after")["sha256"] != report["source"]["official_tree_sha256"]:
            raise Rejected("official tree changed")
        for source in code:
            if source.hashes("code_hash_after")["sha256"] != report["probe_and_contract_hashes"][str(source.path)]:
                raise Rejected("probe/oracle/contract source changed during scan")
        for source in checkout:
            if source.hashes("checkout_reference_hash_after")["sha256"] != report["checkout_reference_hashes"]["files"][str(source.path)]:
                raise Rejected("checkout reference changed during scan")
        recheck_imported_modules(bindings, report["executed_metadata"], "after_source_scan")
        for source in [*sources.values(), tree, *code, *checkout]:
            source.check_identity()
        guard.check(force=True)
        report["immutable_inputs_verified"] = True
        linear = [row for row in report["tensors"] if row["linear_i16_owner"]]
        failures = sum(row["nonfinite_count"] for row in report["tensors"])
        out_of_range = sum(row["finite_out_of_i16_count"] for row in linear)
        report["summary"] = {"tensors_scanned": len(tensors), "source_elements": sum(row["elements"] for row in tensors.values()),
            "linear_source_tensors": len(linear), "nonfinite_count_all_sources": failures,
            "linear_i16_out_of_range_count": out_of_range,
            "local_only_out_of_i16_count_diagnostic": sum(row["finite_out_of_i16_count"] for row in report["tensors"] if not row["linear_i16_owner"]),
            "max_abs_linear_source": max((row["max_abs_source"] for row in linear if row["max_abs_source"] is not None), default=None),
            "max_abs_linear_rounded": max((row["max_abs_rounded"] for row in linear if row["max_abs_rounded"] is not None), default=None),
            "max_linear_row_L1": max((row["max_row_L1"] for row in linear if row["max_row_L1"] is not None), default=None),
            "global_safe_uniform_encoded_input_max": (min(row["safe_uniform_encoded_input_max"] for row in linear)
                if linear and failures == out_of_range == 0 else None),
            "oracle_samples_checked": sum(row["oracle_samples_checked"] for row in report["tensors"])}
        report["source_numeric_admission"] = "pass" if failures == out_of_range == 0 else "reject"
        report["status"] = "complete" if report["source_numeric_admission"] == "pass" else "rejected_source_values"


def receipt_code_identity():
    """Bind even a failed bootstrap to the small probe/oracle/test sources."""
    result = {}
    for path in (Path(__file__).resolve(), ROOT / "scripts/gkm_checkpoint_oracle.py",
                 ROOT / "tests/test_gkm_checkpoint_bounds.py"):
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            before = file_identity(path, fd)
            size = os.fstat(fd).st_size
            if size > BUFFER_BYTES:
                raise Rejected("probe code exceeds bounded receipt input")
            raw = os.pread(fd, size, 0)
            if len(raw) != size or file_identity(path, fd) != before:
                raise Rejected("probe code changed while binding failure receipt")
            result[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes_read": size}
        finally:
            os.close(fd)
    return result


def finalize_report(report, output, guard, started, *, clock=None):
    """Measure main-report serialization/publication, then force a fresh guard.

    The companion is authoritative for this finite window. Its own write and
    process exit cannot be self-measured and are explicitly outside that window.
    """
    clock = clock or time.monotonic
    scan_status, numeric_result = report["status"], report["source_numeric_admission"]
    report["source_scan_result"] = {"status": scan_status, "numeric_result": numeric_result}
    if scan_status == "complete":
        report.update(status="pending_receipt_finalization", source_numeric_admission="pending_receipt_finalization")
    report["publication_finalization"] = {"status": "pending", "authoritative_companion": "finalization.json"}
    report["publication_work_before_report_serialization"] = work_observation(started, clock=clock)
    if guard is not None:
        report["resource_guard_before_report_serialization"] = guard.report()
    serialization_started = clock()
    publication_error = None
    try:
        artifact = atomic_json(output / "report.json", report)
    except (Exception, KeyboardInterrupt) as exc:
        artifact = None
        publication_error = {"type": type(exc).__name__, "reason": str(exc)}
    serialization_ended = clock()
    observation_error = None
    samples_before = guard.samples if guard is not None else 0
    try:
        if guard is not None:
            guard.check(force=True)
    except (Exception, KeyboardInterrupt) as exc:
        observation_error = {"type": type(exc).__name__, "reason": str(exc)}
    # Never label an earlier successful sample as a post-publication observation.
    after = guard.last_work if guard is not None and guard.samples > samples_before else None
    if guard is None:
        after = work_observation(started, clock=clock)
    finalization_ok = publication_error is observation_error is None and guard is not None
    status = scan_status if finalization_ok else "failed"
    numeric = numeric_result if finalization_ok else "blocked"
    finalization = {
        "schema": "pllm.gkm_checkpoint_finalization.v1", "status": status,
        "source_numeric_admission": numeric, "complete_gkm_admission": "blocked",
        "source_scan_result": {"status": scan_status, "numeric_result": numeric_result},
        "source_scan_failure": report.get("failure"), "report_publication_error": publication_error,
        "final_observation_error": observation_error,
        "guarded_report_window_passed": finalization_ok,
        "report_artifact": artifact,
        "report_serialization_and_atomic_publication": {
            "start_elapsed_seconds": serialization_started - started,
            "end_elapsed_seconds": serialization_ended - started,
            "duration_seconds": serialization_ended - serialization_started},
        "publication_work_after_report_serialization": after,
        "resource_guard_after_report_serialization": guard.report() if guard is not None else None,
        "unobserved_tail": ["finalization.json serialization and atomic publication", "stdout formatting and process exit"],
        "scope": "main report serialization/publication and final observation; not a universal lifetime guarantee",
    }
    atomic_json(output / "finalization.json", finalization)
    return finalization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new receipt directory under target/")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "target").resolve()):
        parser.error("receipts must be written under target/")
    output.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    started = time.monotonic()
    guard = None
    report = {"schema": "pllm.gkm_checkpoint_bounds.v2", "status": "failed", "source_numeric_admission": "blocked",
        "complete_gkm_admission": "blocked", "scope": "offline_public_weight_source_scan_publication_work",
        "source": {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": PIN},
        "bootstrap_process_estimate_bytes": PROCESS_CEILING,
        "numeric_profile": "nearest-even(exact_binary_value*2048), signed-i16 weights, ring64 linear arithmetic",
        "activation_ranges_observed": False, "model_inference_executed": False,
        "quality_privacy_serving_eligibility_established": False,
        "remaining_gates": ["GKM fused-group schedule and publication owner layout",
            "admitted actual encoded-input bounds at every stage", "bias/local float32 operation placement and order",
            "complete original numerical graph, verification and lifecycle", "original-model and common-model quality criteria"],
        "pid": os.getpid(), "python": sys.version, "platform": sys.platform}
    try:
        report["receipt_code_hashes"] = receipt_code_identity()
        if sys.platform != "darwin" or os.uname().machine != "arm64":
            raise Rejected("this public-weight scan profile requires Apple-M CPU")
        guard = Guard(started)
        guard.admit_bootstrap()
        scan(report, output, guard)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", source_numeric_admission="blocked",
                      failure={"type": type(exc).__name__, "reason": str(exc)})
    finally:
        finalization = finalize_report(report, output, guard, started)
    print(json.dumps({key: finalization[key] for key in ("status", "source_numeric_admission", "complete_gkm_admission")}
                     | {"report": str(output / "report.json"), "finalization": str(output / "finalization.json"),
                        "summary": report.get("summary"), "failure": report.get("failure"),
                        "final_observation_error": finalization["final_observation_error"]}, indent=2))
    return 0 if finalization["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
