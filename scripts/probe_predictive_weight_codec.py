"""Bounded lossless public-token predictors; offline serial research only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import struct
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CONTROL = ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"
FRONTIER = ROOT / "docs/evidence/research-all-link-frontier.json"
CONTROL_SHA256 = "9b54ee455e7307429d4cc1bf19c93f1c58ff3ef966c258ddebde2ec8c68c42db"
MODES = ("zlib1", "row-delta", "token-parent", "global-token-dict512")
LOCAL_MODES = MODES[:3]
DICT_LIMIT = 512
DICT_HEADER = struct.Struct(">8sIII")
FRAME_LIMIT = 1 << 20
MAX_ROWS = 1024
MAX_COLUMNS = 4096
TOKEN_LIMIT = 256
PREFIX_LIMIT = 64
HEADER = struct.Struct(">II")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dimensions(rows, columns):
    if (type(rows) is not int or type(columns) is not int or
            not 1 <= rows <= MAX_ROWS or not 1 <= columns <= MAX_COLUMNS or
            rows * columns > FRAME_LIMIT):
        raise ValueError("frame dimensions outside policy")


def token_parents(tokens):
    """Longest earlier exact token prefix/suffix; no numeric fitting/search.

    At most 128 probes per token, at most 1024 dictionary entries. Equal
    lengths prefer prefix, then earliest token id. Long tokens use zero.
    """
    if not 1 <= len(tokens) <= MAX_ROWS:
        raise ValueError("token frame outside policy")
    seen = {}
    parents = np.full(len(tokens), -1, dtype=np.int32)
    for row, token in enumerate(tokens):
        if not isinstance(token, str) or len(token) > TOKEN_LIMIT:
            continue
        best = None
        for length in range(1, min(len(token) - 1, PREFIX_LIMIT) + 1):
            for side, part in enumerate((token[:length], token[-length:])):
                parent = seen.get(part)
                if parent is not None:
                    candidate = (-length, side, parent)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            parents[row] = best[2]
        if token:
            seen.setdefault(token, row)
    return parents


def dictionary_selection(tokens, *, budget=DICT_LIMIT):
    """Top token rows by proper prefix/suffix coverage; ties by public token id.

    Fixed search: <=200000 rows, <=128 lookups/token, <=64-char candidates.
    No weight reads, numerical fitting, or encode-result-based selection.
    """
    if type(budget) is not int or not 1 <= budget <= DICT_LIMIT or not 1 <= len(tokens) <= 200000:
        raise ValueError("dictionary selection budget outside policy")
    eligible = {token: row for row, token in reversed(list(enumerate(tokens)))
                if isinstance(token, str) and 0 < len(token) <= PREFIX_LIMIT}
    if not eligible:
        raise ValueError("no eligible public dictionary tokens")
    coverage = np.zeros(len(tokens), dtype=np.int32)
    for token in tokens:
        if not isinstance(token, str) or not 0 < len(token) <= TOKEN_LIMIT:
            continue
        matches = set()
        for length in range(1, min(len(token) - 1, PREFIX_LIMIT) + 1):
            for part in (token[:length], token[-length:]):
                parent = eligible.get(part)
                if parent is not None:
                    matches.add(parent)
        for parent in matches:
            coverage[parent] += 1
    ranked = sorted(eligible.values(), key=lambda row: (-int(coverage[row]), row))
    ids = ranked[:budget]
    return ids, [int(coverage[row]) for row in ids]


def dictionary_parents(tokens, dictionary_tokens):
    if not 1 <= len(tokens) <= MAX_ROWS or not 1 <= len(dictionary_tokens) <= DICT_LIMIT:
        raise ValueError("dictionary parent budget outside policy")
    if any(not isinstance(token, str) or not 0 < len(token) <= PREFIX_LIMIT for token in dictionary_tokens):
        raise ValueError("dictionary token outside prefix policy")
    seen = {token: index for index, token in reversed(list(enumerate(dictionary_tokens)))}
    result = np.full(len(tokens), -1, dtype=np.int32)
    for row, token in enumerate(tokens):
        if not isinstance(token, str) or not 0 < len(token) <= TOKEN_LIMIT:
            continue
        # Exact selected-token matches are safe: dictionary was delivered first.
        for length in range(min(len(token), PREFIX_LIMIT), 0, -1):
            candidates = [seen[part] for part in (token[:length], token[-length:]) if part in seen]
            if candidates:
                result[row] = min(candidates)
                break
    return result


def dictionary_array(dictionary, columns):
    value = np.asarray(dictionary)
    if (value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != columns or
            not 1 <= value.shape[0] <= DICT_LIMIT):
        raise ValueError("invalid or over-budget dictionary")
    return value


def encode_dictionary(raw, *, rows, columns):
    if type(rows) is not int or not 1 <= rows <= DICT_LIMIT:
        raise ValueError("dictionary rows outside policy")
    if type(columns) is not int or not 1 <= columns <= MAX_COLUMNS or len(raw) != rows * columns:
        raise ValueError("dictionary dimensions/bytes invalid")
    step = min(MAX_ROWS, FRAME_LIMIT // columns)
    frames = [encode_frame(raw[first * columns:min(rows, first + step) * columns],
                           rows=min(step, rows - first), columns=columns, mode="zlib1")
              for first in range(0, rows, step)]
    return DICT_HEADER.pack(b"PWDIC001", rows, columns, len(frames)) + b"".join(frames)


def decode_dictionary(blob, *, columns, expected_sha256):
    if len(blob) < DICT_HEADER.size or len(blob) > (DICT_LIMIT * MAX_COLUMNS + 2 * (64 << 10) + 36):
        raise ValueError("dictionary artifact outside bounds")
    magic, rows, width, count = DICT_HEADER.unpack_from(blob)
    if magic != b"PWDIC001" or not 1 <= rows <= DICT_LIMIT or width != columns or not 1 <= width <= MAX_COLUMNS:
        raise ValueError("invalid dictionary header/budget")
    step = min(MAX_ROWS, FRAME_LIMIT // columns)
    if count != (rows + step - 1) // step:
        raise ValueError("dictionary frame count mismatch")
    offset, output = DICT_HEADER.size, bytearray()
    for first in range(0, rows, step):
        if len(blob) - offset < 8:
            raise ValueError("truncated dictionary frame")
        _, encoded = HEADER.unpack_from(blob, offset)
        frame_rows = min(step, rows - first)
        if not 0 < encoded <= frame_rows * columns + (64 << 10):
            raise ValueError("dictionary encoded frame outside bounds")
        output.extend(decode_frame(blob[offset:offset + 8 + encoded], rows=frame_rows,
                                   columns=columns, mode="zlib1"))
        offset += 8 + encoded
    if offset != len(blob) or hashlib.sha256(output).hexdigest() != expected_sha256:
        raise ValueError("dictionary trailing bytes or trusted digest mismatch")
    return np.frombuffer(bytes(output), np.uint8).reshape(rows, columns)


def encode_frame(raw, *, rows, columns, mode, parents=None, dictionary=None):
    dimensions(rows, columns)
    if mode not in MODES or len(raw) != rows * columns:
        raise ValueError("invalid matrix or mode")
    value = np.frombuffer(raw, np.uint8).reshape(rows, columns)
    bank = dictionary_array(dictionary, columns) if mode == MODES[3] else None
    if mode == "zlib1":
        transformed = raw
    else:
        if mode == "row-delta":
            indices = np.arange(rows, dtype=np.int32) - 1
        else:
            indices = np.asarray(parents)
            if indices.shape != (rows,) or indices.dtype.kind not in "iu":
                raise ValueError("invalid parent index shape/type")
        residual = value.copy()
        for row, parent in enumerate(indices):
            if not -1 <= int(parent) < (len(bank) if bank is not None else row):
                raise ValueError("parent must precede row")
            if parent >= 0:
                residual[row] = value[row] - (bank[int(parent)] if bank is not None else value[int(parent)])
        transformed = residual.tobytes()
        if mode in ("token-parent", MODES[3]):
            transformed = indices.astype("<i4").tobytes() + transformed
    body = zlib.compress(transformed, level=1)
    return HEADER.pack(len(transformed), len(body)) + body


def decode_frame(frame, *, rows, columns, mode, dictionary=None):
    dimensions(rows, columns)
    if mode not in MODES or len(frame) < HEADER.size:
        raise ValueError("invalid frame/mode")
    bank = dictionary_array(dictionary, columns) if mode == MODES[3] else None
    indexed = mode in ("token-parent", MODES[3])
    expected = rows * columns + (4 * rows if indexed else 0)
    size, encoded = HEADER.unpack_from(frame)
    if size != expected or not 0 < encoded <= expected + (64 << 10) or len(frame) != 8 + encoded:
        raise ValueError("frame exceeds bounds or has trailing data")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(memoryview(frame)[8:], expected + 1)
    except zlib.error as error:
        raise ValueError("corrupt compressed frame") from error
    if (len(raw) != expected or not decoder.eof or decoder.unused_data or
            decoder.unconsumed_tail):
        raise ValueError("invalid inflated size or trailing compressed data")
    if mode == "zlib1":
        return raw
    indices = (np.frombuffer(raw, "<i4", count=rows) if indexed
               else np.arange(rows, dtype=np.int32) - 1)
    offset = 4 * rows if indexed else 0
    output = np.frombuffer(raw, np.uint8, offset=offset).reshape(rows, columns).copy()
    for row, parent in enumerate(indices):
        if not -1 <= int(parent) < (len(bank) if bank is not None else row):
            raise ValueError("cyclic/forward/out-of-range parent")
        if parent >= 0:
            output[row] += bank[int(parent)] if bank is not None else output[int(parent)]
    return output.tobytes()


def frozen_sources(directory):
    """Freeze resolved files, not mutable snapshot symlink names alone."""
    from pllm.runtime import quantization, safetensors_store, bundle_compression
    from pllm.runtime._native_support import extension
    native = extension()
    if native is None:
        raise RuntimeError("installed native extension required")
    files = {Path(__file__), ROOT / "scripts/frontier_probe_support.py", CONTROL, FRONTIER,
             ROOT / "scripts/probe_all_link_frontier.py", Path(sys.executable).resolve()}
    files.update((directory / name).resolve() for name in
                 ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"))
    versions = {}
    imported = {}
    # Freeze entire relevant package code/native trees so lazy imports are covered.
    for name, distribution in (("pllm", "pllm.run"), ("numpy", "numpy"),
                               ("safetensors", "safetensors"), ("ml_dtypes", "ml_dtypes"),
                               ("huggingface_hub", "huggingface-hub"), ("filelock", "filelock"),
                               ("psutil", "psutil")):
        module = importlib.import_module(name)
        imported[name] = str(Path(module.__file__).resolve())
        versions[distribution] = importlib.metadata.version(distribution)
        base = Path(module.__file__).parent
        files.update(p.resolve() for p in base.rglob("*")
                     if p.is_file() and p.suffix in (".py", ".so", ".dylib", ".pyd"))
    for module in (quantization, safetensors_store, bundle_compression, zlib, native):
        filename = getattr(module, "__file__", None)
        if filename is not None:
            imported[module.__name__] = str(Path(filename).resolve())
            files.add(Path(filename).resolve())
        elif getattr(module.__spec__, "origin", None) == "built-in":
            # Standalone CPython can link zlib into the already-hashed interpreter.
            imported[module.__name__] = "built-in:" + str(Path(sys.executable).resolve())
        else:
            raise RuntimeError(f"unbound imported implementation: {module.__name__}")
    return {"files": {str(p): sha(p) for p in sorted(files)}, "versions": versions,
            "imported": imported, "python": sys.version, "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "backend": os.environ.get("PLLM_KERNEL_BACKEND", "rust")}


def read_tokens(directory, rows):
    path = directory / "tokenizer.json"
    if path.stat().st_size > 32 << 20:
        raise ValueError("tokenizer exceeds setup bound")
    document = json.loads(path.read_text())
    vocab = document["model"]["vocab"]
    if not isinstance(vocab, dict) or len(vocab) > rows:
        raise ValueError("invalid public tokenizer vocabulary")
    result = [None] * rows
    for token, index in vocab.items():
        if type(index) is not int or not 0 <= index < rows or result[index] is not None:
            raise ValueError("invalid/duplicate token id")
        result[index] = token
    for item in document.get("added_tokens", []):
        index, token = item["id"], item["content"]
        if type(index) is not int or not 0 <= index < rows:
            raise ValueError("invalid added token id")
        if result[index] is not None and result[index] != token:
            raise ValueError("conflicting added token")
        result[index] = token
    return result


def reference_quantize(weight):
    """Independent NumPy formula; never call runtime quantizer here."""
    value = np.asarray(weight, np.float32)
    maximum = np.max(np.abs(value), axis=1)
    scales = np.where(maximum > 0, maximum / 127, 1.0).astype("<f4")
    quantized = np.clip(np.rint(value / scales[:, None]), -127, 127).astype(np.int8)
    return quantized.tobytes(), scales.tobytes()


class BoundedBF16Rows:
    """Independent safetensors header + bounded raw rows; no Torch or BF16 NumPy dtype."""
    def __init__(self, path, key):
        self.handle = Path(path).open("rb")
        try:
            size_bytes = self.handle.read(8)
            if len(size_bytes) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < size <= FRAME_LIMIT:
                raise ValueError("safetensors header outside policy")
            encoded = self.handle.read(size)
            if len(encoded) != size:
                raise ValueError("truncated safetensors JSON")
            metadata = json.loads(encoded)[key]
            self.shape = tuple(metadata["shape"])
            start, end = metadata["data_offsets"]
            if (metadata["dtype"] != "BF16" or len(self.shape) != 2 or
                    any(type(x) is not int or x <= 0 for x in self.shape) or
                    self.shape[0] > 200000 or self.shape[1] > MAX_COLUMNS or
                    type(start) is not int or type(end) is not int or start < 0 or
                    end - start != 2 * self.shape[0] * self.shape[1] or
                    8 + size + end > os.fstat(self.handle.fileno()).st_size):
                raise ValueError("invalid BF16 object bounds")
            self.offset = 8 + size + start
        except BaseException:
            self.handle.close()
            raise

    def read(self, first, rows):
        dimensions(rows, self.shape[1])
        if type(first) is not int or not 0 <= first <= self.shape[0] - rows:
            raise ValueError("BF16 row range outside object")
        self.handle.seek(self.offset + first * self.shape[1] * 2)
        raw = self.handle.read(rows * self.shape[1] * 2)
        if len(raw) != rows * self.shape[1] * 2:
            raise ValueError("truncated BF16 rows")
        bits = np.frombuffer(raw, "<u2").astype("<u4") << 16
        return raw, bits.view("<f4").reshape(rows, self.shape[1])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.close()


def runtime_weight_slice(store, original, key, first, rows):
    try:
        return store.get_slice(key, (slice(first, first + rows), slice(None))), "installed-safetensors-store"
    except (TypeError, ValueError) as error:
        # Some safetensors NumPy adapters cannot express BF16, even with ml_dtypes.
        if not any(word in str(error).lower() for word in ("bfloat", "bf16")):
            raise
        return original.read(first, rows)[1], "bounded-raw-bf16-fallback"


def validated_control(frontier_path=FRONTIER, control_path=CONTROL, expected_sha256=CONTROL_SHA256):
    try:
        from probe_all_link_frontier import conserved_control
    except ModuleNotFoundError as error:
        if error.name != "probe_all_link_frontier":
            raise
        from scripts.probe_all_link_frontier import conserved_control
    if sha(control_path) != expected_sha256:
        raise ValueError("authoritative raw control hash mismatch")
    frontier = json.loads(Path(frontier_path).read_text())
    if not frontier["complete"]:
        raise ValueError("incomplete authoritative frontier")
    control = conserved_control(json.loads(Path(control_path).read_text()))
    if frontier["control"] != control or control["evidence_sha256"] != expected_sha256:
        raise ValueError("authoritative frontier/control reconciliation mismatch")
    return control, {"frontier_path": str(frontier_path), "frontier_sha256": sha(frontier_path),
                     "raw_control_path": str(control_path), "raw_control_sha256": expected_sha256}


def projection(control, *, baseline, candidate):
    pair, bundle = control["pair_bytes"], control["classes"]["public_bundle"]
    if not 0 < bundle <= pair or baseline <= 0 or candidate <= 0:
        raise ValueError("invalid projection ledger/cost")
    saving = baseline - candidate
    return {"matrix_bytes_saved": saving, "complete_bundle_counterfactual_bytes": bundle - saving,
            "cold_repeat_pair_counterfactual_bytes": pair - saving,
            "bundle_fraction_saved": saving / bundle, "pair_fraction_saved": saving / pair,
            "measured_serving_gain": None}


def cohort_schedule():
    return [(repeat, mode) for repeat in range(3) for mode in MODES[repeat:] + MODES[:repeat]]


def codec_worker(payload):
    total_start, cpu_start = time.perf_counter(), time.process_time()
    directory = Path(payload["checkpoint"])
    identity_start = time.process_time()
    if frozen_sources(directory) != payload["identity"]:
        raise ValueError("worker source identity mismatch")
    identity_before_cpu = time.process_time() - identity_start
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.quantization import quantize_weight_per_row
    store = SafeTensorStore(directory)
    key = "model.embed_tokens.weight"
    shape = store.tensor_shape(key)
    config = json.loads((directory / "config.json").read_text())
    if (shape != (151936, 896) or store.tensor_dtype(key) != "BF16" or
            not config["tie_word_embeddings"] or "lm_head.weight" in store.keys):
        raise ValueError("fixture must be pinned W8 tied boundary object")
    setup_start = time.process_time()
    tokens = read_tokens(directory, shape[0])
    setup_cpu = time.process_time() - setup_start
    mode = payload["mode"]
    hashes = {name: hashlib.sha256() for name in
              ("weights", "decoded", "scales", "reference_weights", "reference_scales", "original_bf16", "artifact")}
    counters = {name: 0 for name in ("load_quantize_cpu_seconds", "reference_cpu_seconds",
                "predictor_setup_cpu_seconds", "encode_cpu_seconds", "decode_cpu_seconds")}
    frames = framed = parent_count = index_bytes = 0
    dictionary_ids, selection_coverage, dictionary_tokens = [], [], []
    dictionary = None
    dictionary_blob = b""
    dictionary_hash = None
    dictionary_source_hash, dictionary_scale_hash = hashlib.sha256(), hashlib.sha256()
    dictionary_metrics = {"dictionary_selection_cpu_seconds": 0,
                          "dictionary_source_quantize_reference_cpu_seconds": 0,
                          "dictionary_encode_cpu_seconds": 0, "dictionary_decode_cpu_seconds": 0}
    loader_paths = set()
    with BoundedBF16Rows(directory / "model.safetensors", key) as original:
        if original.shape != shape:
            raise ValueError("independent BF16 header disagrees with installed tensor metadata")
        if mode == MODES[3]:
            start = time.process_time()
            dictionary_ids, selection_coverage = dictionary_selection(tokens)
            dictionary_tokens = [tokens[row] for row in dictionary_ids]
            dictionary_metrics["dictionary_selection_cpu_seconds"] = time.process_time() - start
            start = time.process_time()
            dictionary_raw = bytearray()
            for row in dictionary_ids:
                weight, loader = runtime_weight_slice(store, original, key, row, 1)
                loader_paths.add(loader)
                quantized = quantize_weight_per_row(weight, bits=8)
                source_raw, source_value = original.read(row, 1)
                dictionary_source_hash.update(source_raw)
                ref_raw, ref_scales = reference_quantize(source_value)
                dictionary_scale_hash.update(ref_scales)
                if (quantized.values.tobytes() != ref_raw or
                        quantized.scales.astype("<f4", copy=False).tobytes() != ref_scales):
                    raise AssertionError("dictionary source integers/scales differ independently")
                dictionary_raw.extend(ref_raw)
            dictionary_metrics["dictionary_source_quantize_reference_cpu_seconds"] = time.process_time() - start
            dictionary_hash = hashlib.sha256(dictionary_raw).hexdigest()
            start = time.process_time()
            dictionary_blob = encode_dictionary(bytes(dictionary_raw), rows=len(dictionary_ids), columns=shape[1])
            dictionary_metrics["dictionary_encode_cpu_seconds"] = time.process_time() - start
            start = time.process_time()
            dictionary = decode_dictionary(dictionary_blob, columns=shape[1], expected_sha256=dictionary_hash)
            if dictionary.tobytes() != dictionary_raw:
                raise AssertionError("independent dictionary byte parity failed")
            dictionary_metrics["dictionary_decode_cpu_seconds"] = time.process_time() - start
            hashes["artifact"].update(dictionary_blob)
            del dictionary_raw
        for first in range(0, shape[0], MAX_ROWS):
            rows = min(MAX_ROWS, shape[0] - first)
            start = time.process_time()
            weight, loader = runtime_weight_slice(store, original, key, first, rows)
            loader_paths.add(loader)
            quantized = quantize_weight_per_row(weight, bits=8)
            raw, scales = quantized.values.tobytes(), quantized.scales.astype("<f4", copy=False).tobytes()
            counters["load_quantize_cpu_seconds"] += time.process_time() - start
            start = time.process_time()
            source_raw, source_value = original.read(first, rows)
            hashes["original_bf16"].update(source_raw)
            ref_raw, ref_scales = reference_quantize(source_value)
            if raw != ref_raw or scales != ref_scales:
                raise AssertionError("actual runtime W8 values/scales differ from independent reference")
            hashes["reference_weights"].update(ref_raw)
            hashes["reference_scales"].update(ref_scales)
            counters["reference_cpu_seconds"] += time.process_time() - start
            start = time.process_time()
            parents = token_parents(tokens[first:first + rows]) if mode == "token-parent" else None
            if mode == MODES[3]:
                parents = dictionary_parents(tokens[first:first + rows], dictionary_tokens)
            counters["predictor_setup_cpu_seconds"] += time.process_time() - start
            if parents is not None:
                parent_count += int(np.count_nonzero(parents >= 0))
                index_bytes += parents.nbytes
            start = time.process_time()
            frame = encode_frame(raw, rows=rows, columns=shape[1], mode=mode, parents=parents, dictionary=dictionary)
            counters["encode_cpu_seconds"] += time.process_time() - start
            start = time.process_time()
            decoded = decode_frame(frame, rows=rows, columns=shape[1], mode=mode, dictionary=dictionary)
            counters["decode_cpu_seconds"] += time.process_time() - start
            if decoded != ref_raw:
                raise AssertionError("lossless public matrix roundtrip failed")
            for name, data in (("weights", raw), ("decoded", decoded), ("scales", scales), ("artifact", frame)):
                hashes[name].update(data)
            framed += len(frame)
            frames += 1
    object_hashes = {name: value.hexdigest() for name, value in hashes.items()}
    # Canonical artifact descriptor is charged, including object commitments.
    descriptor = json.dumps({"schema": "predictive-weight-v1", "mode": mode,
                            "shape": shape, "frame_rows": MAX_ROWS,
                            "weight_sha256": object_hashes["weights"],
                            "scales_sha256": object_hashes["scales"],
                            "dictionary_rows": dictionary_ids,
                            "dictionary_raw_sha256": dictionary_hash,
                            "dictionary_layout": "PWDIC001" if dictionary_ids else None},
                           sort_keys=True, separators=(",", ":")).encode()
    identity_start = time.process_time()
    if frozen_sources(directory) != payload["identity"]:
        raise ValueError("worker source identity changed")
    identity_after_cpu = time.process_time() - identity_start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return {"mode": mode, "shape": shape, "checked_values": shape[0] * shape[1],
            "frames": frames, "framed_matrix_bytes": framed,
            "descriptor_bytes": len(descriptor), "candidate_object_bytes": framed + len(descriptor) + len(dictionary_blob),
            "scale_payload_bytes_unchanged": shape[0] * 4,
            "uncompressed_index_bytes_in_frames": index_bytes, "predicted_rows": parent_count,
            "parent_coverage_fraction": parent_count / shape[0],
            "dictionary_selected_self_rows": len(dictionary_ids),
            "structure_parent_rows_excluding_selected_self": (parent_count - len(dictionary_ids)) if dictionary_ids else parent_count,
            "dictionary_or_predictor_distribution_bytes": len(dictionary_blob),
            "dictionary_integer_payload_bytes": len(dictionary_ids) * shape[1],
            "dictionary_sha256": dictionary_hash, "dictionary_selected_rows": dictionary_ids,
            "dictionary_original_bf16_sha256": dictionary_source_hash.hexdigest() if dictionary_ids else None,
            "dictionary_reference_scales_sha256": dictionary_scale_hash.hexdigest() if dictionary_ids else None,
            "dictionary_selection_coverage_counts": selection_coverage,
            "dictionary_source_logical_bf16_bytes": len(dictionary_ids) * shape[1] * 4,
            "matrix_source_logical_bf16_bytes": shape[0] * shape[1] * 4,
            "source_read_accounting": "runtime input plus independent input; logical successful bytes, not OS I/O or failed adapter reads",
            "source_loader_paths": sorted(loader_paths), **dictionary_metrics,
            "artifact_storage_bytes_without_unchanged_scales": framed + len(descriptor) + len(dictionary_blob),
            "artifact_persisted": False, "hashes": object_hashes,
            "tokenizer_setup_cpu_seconds": setup_cpu, **counters,
            "identity_before_cpu_seconds": identity_before_cpu,
            "identity_after_cpu_seconds": identity_after_cpu,
            "module_import_cpu_before_worker_seconds": cpu_start,
            "process_total_cpu_seconds": time.process_time(),
            "worker_total_cpu_seconds": time.process_time() - cpu_start,
            "worker_total_wall_seconds": time.perf_counter() - total_start,
            "fresh_process_peak_rss_bytes": peak,
            "rss_scope": "imports, identity hashing, source mmap, setup, quantization, reference and codec overlap",
            "descriptor_utf8": descriptor.decode()}


def main():
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("unset PYTHONPATH: replay requires installed distribution, not source shadowing")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "target/predictive-weight-codec.json")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(codec_worker(json.load(sys.stdin))))
        return
    if args.processes != 3:
        parser.error("fixed minimal cohort requires exactly three fresh replicates per mode")
    if args.output.exists():
        raise FileExistsError(args.output)
    from frontier_probe_support import fresh_workers
    # Include source hashing/setup in shared lock and host-pressure admission.
    with fresh_workers(__file__, memory_bytes=1024 << 20, timeout=600) as (run, guard, host):
        run_cohort(args, run, guard, host)


def run_cohort(args, run, guard, host):
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]))
    start = time.process_time()
    identity = frozen_sources(directory)
    control, control_identity = validated_control()
    authority_model = json.loads(FRONTIER.read_text())["model"]
    if (authority_model["repo"] != REPO or authority_model["revision"] != REVISION or
            identity["files"][str((directory / "model.safetensors").resolve())] != authority_model["checkpoint_sha256"] or
            identity["files"][str((directory / "config.json").resolve())] != authority_model["config_sha256"]):
        raise ValueError("original checkpoint/config differs from pinned authoritative fixture")
    result = {"schema": "pllm.predictive_weight_codec.v1", "complete": False,
              "model": {"repo": REPO, "revision": REVISION, "quantization": "W8A8"},
              "frozen_modes": list(MODES), "global_dictionary_maximum_rows": DICT_LIMIT,
              "identity": identity, "parent_identity_setup_cpu_seconds": time.process_time() - start,
              "control": control, "control_identity": control_identity, "rows": [],
              "scope": "complete tied integer matrix; unchanged scales",
              "predictor_distribution_horizon": "once per cold/exact-repeat pair; dictionary, indices and descriptor included",
              "remote_queries": 0, "public_numeric_fitting_cost": 0,
              "retained_source_storage_bytes": {
                  name: (directory / name).stat().st_size for name in
                  ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")},
              "limitations": ["no serving integration or measured response savings",
                              "matrix-only counterfactual; other bundle objects unchanged",
                              "source checkpoint retained; no checkpoint bytes replaced",
                              "artifact storage counted but no persisted artifact benchmark",
                              "decode CPU is in-process after encode; RSS covers full fresh worker overlap"]}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".pending")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    result["build"] = run.build
    result["host"] = {"available_bytes": host.available, "reserve_bytes": host.reserve}
    for repeat, mode in cohort_schedule():
        start = time.perf_counter()
        row = run({"checkpoint": str(directory), "identity": identity, "mode": mode})
        row["parent_observed_fresh_process_wall_seconds"] = time.perf_counter() - start
        result["rows"].append(row | {"repeat": repeat})
        save()
    result["swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    rows = result["rows"]
    if len({tuple(r["hashes"][name] for name in
                  ("weights", "decoded", "scales", "reference_weights", "reference_scales", "original_bf16"))
            for r in rows}) != 1:
        raise ValueError("complete object digest mismatch across workers")
    for mode in MODES:
        artifacts = {(r["hashes"]["artifact"], r["candidate_object_bytes"], r["dictionary_sha256"],
                      tuple(r["dictionary_selected_rows"])) for r in rows if r["mode"] == mode}
        if len(artifacts) != 1:
            raise ValueError("fixed per-mode artifact/dictionary changed across replicates")
    result["medians"] = {mode: {field: statistics.median(r[field] for r in rows if r["mode"] == mode)
                                   for field in ("framed_matrix_bytes", "candidate_object_bytes", "encode_cpu_seconds",
                                                 "decode_cpu_seconds", "predictor_setup_cpu_seconds",
                                                 "dictionary_selection_cpu_seconds", "dictionary_source_quantize_reference_cpu_seconds",
                                                 "dictionary_encode_cpu_seconds", "dictionary_decode_cpu_seconds",
                                                 "dictionary_or_predictor_distribution_bytes", "parent_coverage_fraction",
                                                 "worker_total_cpu_seconds", "fresh_process_peak_rss_bytes")}
                         for mode in MODES}
    baseline = result["medians"]["zlib1"]["framed_matrix_bytes"]
    result["projections"] = {}
    for mode in MODES:
        # Stronger plain control needs no new descriptor in existing runtime.
        candidate = (baseline if mode == "zlib1" else result["medians"][mode]["candidate_object_bytes"])
        result["projections"][mode] = projection(control, baseline=baseline, candidate=candidate)
    if frozen_sources(directory) != identity:
        raise ValueError("identity changed across cohort")
    from pllm.metrics import BuildRef
    guard.check()
    if guard.error or BuildRef.capture().to_spec() != run.build:
        raise RuntimeError(guard.error or "frontier build changed before evidence completion")
    result["swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
