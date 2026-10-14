"""Bounded synthetic resident-head reference; no production registration or checkpoints.

Derived shard snapshots compensate for the missing native row-range API. They
are NOT zero-copy imports. Logical payload traffic is not measured disk/network
traffic, and resident payload caps are not whole-client RSS tier qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile

import numpy as np

MAX_SHARD_OWNERS = 8  # Tiny reference only: each fallback creates a file/matrix/executor.


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Shard:
    start: int
    stop: int
    paged: object
    artifact_sha256: str
    weight_sha256: str


class ResidentHead:
    """Single explicit owner, immutable weights, mutable bounded residency.

    Original matrix is borrowed during construction only. Caller must close it.
    Fallback snapshots remain live even for resident shards, permitting eviction.
    No resident weight view escapes this owner. Serial use only.
    """

    def __init__(self, source, root, *, shard_rows=16, cap_bytes=0,
                 workspace_bytes=1 << 20, selected=None, compression="raw"):
        rows, cols = source.shape
        if any(type(n) is not int or n <= 0 for n in (rows, cols)):
            raise ValueError("positive integer source geometry required")
        for value in (shard_rows, cap_bytes, workspace_bytes):
            if type(value) is not int:
                raise ValueError("integer budgets and shard geometry required")
        tile = min(shard_rows, rows) * cols
        if (not 1 <= shard_rows <= 4096 or tile > 4 << 20 or cap_bytes < 0
                or workspace_bytes < 4 * tile or rows * cols > 8 << 20):
            raise ValueError("synthetic geometry or promotion workspace exceeds policy")
        if compression not in ("raw", "zlib"):
            raise ValueError("invalid compression")
        count = (rows + shard_rows - 1) // shard_rows
        if count > MAX_SHARD_OWNERS:
            raise ValueError("shard owner count exceeds tiny reference policy")
        chosen = tuple(range(count) if selected is None else selected)
        if (any(type(i) is not int or not 0 <= i < count for i in chosen)
                or len(set(chosen)) != len(chosen)):
            raise ValueError("invalid or duplicate shard selection")
        self.shape = (rows, cols)
        self.weight_digest = source.weight_digest
        self.cap_bytes = cap_bytes
        self.workspace_bytes = workspace_bytes
        self._kernel = None
        self._resident = {}
        self._shards = ()
        self._closed = False
        self.logical_paged_payload_bytes = 0
        self.logical_resident_payload_bytes = 0
        built = []
        digest = hashlib.sha256()
        try:
            from pllm.native import MaskedGEMM, PagedGEMM

            self._kernel = MaskedGEMM(threads=1, build=False)
            if not self._kernel.native:
                raise RuntimeError("installed native extension required")
            for index, start in enumerate(range(0, rows, shard_rows)):
                stop = min(start + shard_rows, rows)
                values = source.gather_rows(np.arange(start, stop, dtype=np.uint64))
                digest.update(values.tobytes())
                path = Path(root) / f"shard-{index}.paged"
                artifact_hash = PagedGEMM.export(values, path, page_rows=shard_rows,
                                                 compression=compression)
                matrix = PagedGEMM(path, artifact_hash, threads=1)
                built.append(Shard(start, stop, matrix, artifact_hash, sha(values.tobytes())))
                if matrix.weight_digest != built[-1].weight_sha256:
                    raise AssertionError("derived shard numeric identity differs")
            if digest.hexdigest() != self.weight_digest:
                raise AssertionError("full reconstructed original weight hash differs")
            self._shards = tuple(built)
            for index in chosen:
                size = self._size(index)
                if self.resident_bytes + size <= self.cap_bytes:
                    self.promote(index)
        except BaseException:
            self._shards = tuple(built)
            self.close()
            raise

    def _check(self):
        if self._closed:
            raise ValueError("resident head is closed")

    def _index(self, index):
        self._check()
        if type(index) is not int or not 0 <= index < len(self._shards):
            raise ValueError("shard index outside range")

    def _size(self, index):
        shard = self._shards[index]
        return (shard.stop - shard.start) * self.shape[1]

    @property
    def resident_bytes(self):
        return sum(self._size(i) for i in self._resident)

    @property
    def ownership(self):
        self._check()
        return {"resident_i8_payload_bytes": self.resident_bytes,
                "fallback_private_artifact_bytes": sum(s.paged.artifact_bytes for s in self._shards),
                "fallback_metadata_bytes": sum(s.paged.metadata_bytes for s in self._shards),
                "fallback_matrix_and_executor_owners": len(self._shards),
                "resident_shared_executor_owners": 1,
                "maximum_fallback_owner_count": MAX_SHARD_OWNERS,
                "unpriced_overheads": "Python/Rust objects, allocators, executor stacks, inputs/outputs, OS cache; use process RSS",
                "paged_transient_weight_bytes": max(s.paged.transient_weight_bytes for s in self._shards),
                "reserved_promotion_workspace_bytes": self.workspace_bytes,
                "external_original_snapshot_not_owned": True,
                "resident_and_fallback_overlap_explicit": True}

    def promote(self, index):
        self._index(index)
        if index in self._resident:
            return
        if self.resident_bytes + self._size(index) > self.cap_bytes:
            raise ValueError("resident payload cap exceeded")
        shard = self._shards[index]
        values = shard.paged.gather_rows(np.arange(shard.stop - shard.start, dtype=np.uint64))
        if sha(values.tobytes()) != shard.weight_sha256:
            raise AssertionError("promotion actual bytes differ")
        self._resident[index] = self._kernel.compile(values)

    def evict(self, index):
        self._index(index)
        self._resident.pop(index, None)  # Native Matrix has no close; last reference drops owner.

    def execute(self, inputs, operation="clear", modulus=None):
        self._check()
        if operation not in ("clear", "wrap32", "modular"):
            raise ValueError("unsupported exact operation")
        x = np.asarray(inputs)
        if (x.ndim != 2 or x.dtype.kind not in "iu" or x.shape[1] != self.shape[1]
                or not 1 <= len(x) <= 4096 or max(x.size, len(x) * self.shape[0]) > 4 << 20):
            raise ValueError("complete input/output geometry exceeds policy")
        if operation == "modular" and (type(modulus) is not int or not 2 <= modulus < 2**31):
            raise ValueError("invalid modulus")
        result = np.empty((len(x), self.shape[0]), dtype=np.int32 if operation == "clear" else np.uint32)
        for index, shard in enumerate(self._shards):
            matrix = self._resident.get(index, shard.paged)
            args = (x, modulus) if operation == "modular" else (x,)
            result[:, shard.start:shard.stop] = getattr(matrix, operation)(*args)
            if index in self._resident:
                self.logical_resident_payload_bytes += self._size(index)
            else:
                # One exported page per shard. Header/index excluded; not physical I/O.
                self.logical_paged_payload_bytes += shard.paged.artifact_bytes - 88
        return result

    def gather_rows(self, indices):
        self._check()
        ids = np.asarray(indices)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not 1 <= len(ids) <= 4096
                or len(ids) * self.shape[1] > 4 << 20 or int(ids.min()) < 0
                or int(ids.max()) >= self.shape[0]):
            raise ValueError("token indices outside range or workspace")
        result = np.empty((len(ids), self.shape[1]), dtype=np.int8)
        for index, shard in enumerate(self._shards):
            positions = np.flatnonzero((ids >= shard.start) & (ids < shard.stop))
            if not len(positions):
                continue
            local = ids[positions] - shard.start
            if index in self._resident:
                result[positions] = self._resident[index].weight_view()[local]
            else:
                result[positions] = shard.paged.gather_rows(local)
        return result

    def gather_columns(self, indices):
        self._check()
        ids = np.asarray(indices)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not 1 <= len(ids) <= 4096
                or len(ids) * self.shape[0] > 4 << 20 or int(ids.min()) < 0
                or int(ids.max()) >= self.shape[1]):
            raise ValueError("column indices outside range or workspace")
        result = np.empty((len(ids), self.shape[0]), dtype=np.int8)
        for index, shard in enumerate(self._shards):
            if index in self._resident:
                result[:, shard.start:shard.stop] = self._resident[index].weight_view()[:, ids].T
            else:
                result[:, shard.start:shard.stop] = shard.paged.gather_columns(ids)
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._resident.clear()
                for shard in self._shards:
                    shard.paged.close()
            finally:
                self._shards = ()
                self._kernel = None  # Drop shared executor even while wrapper remains reachable.


def run_case(choice="partial", *, rows=67, cols=33, compression="raw"):
    from pllm.native import PagedGEMM
    from pllm.runtime.quantization import dequantize_matmul
    rng = np.random.default_rng(7007)
    weights = rng.integers(-128, 128, (rows, cols), dtype=np.int8)
    inputs = rng.integers(-128, 128, (3, cols), dtype=np.int8)
    shard_rows = 16
    cap = {"paged": 0, "partial": 2 * shard_rows * cols, "full": rows * cols}[choice]
    with tempfile.TemporaryDirectory(prefix="pllm-resident-head-") as directory:
        root = Path(directory)
        original = root / "original.raw"
        original.write_bytes(weights.tobytes())
        original_hash = sha(original.read_bytes())
        source = PagedGEMM.from_raw(original, original_hash, weights.shape, threads=1)
        owner = None
        try:
            owner = ResidentHead(source, root, shard_rows=shard_rows, cap_bytes=cap,
                                 workspace_bytes=4 * shard_rows * cols, compression=compression)
            source.close()
            original.write_bytes(bytes(weights.size))  # Original source mutation cannot alter admitted owners.
            values = owner.execute(inputs)
            expected = (inputs.astype(np.int64) @ weights.astype(np.int64).T).astype(np.int32)
            np.testing.assert_array_equal(values, expected)
            activation_scales = np.array([0.125, 0.03125, 0.7], dtype=np.float32)
            weight_scales = np.linspace(0.01, 0.5, rows, dtype=np.float32)
            bias = np.linspace(-0.5, 0.5, rows, dtype=np.float32)
            # Recombine integers first; preserve original scale/bias ordering across full vocabulary.
            logits = dequantize_matmul(values, activation_scales, weight_scales) + bias
            reference_logits = dequantize_matmul(expected, activation_scales, weight_scales) + bias
            assert logits.tobytes() == reference_logits.tobytes()
            residues = rng.integers(0, 2**32, inputs.shape, dtype=np.uint32)
            totals = residues.astype(np.int64) @ weights.astype(np.int64).T
            np.testing.assert_array_equal(owner.execute(residues, "wrap32"), (totals % 2**32).astype(np.uint32))
            modular = residues % 65521
            np.testing.assert_array_equal(owner.execute(modular, "modular", 65521),
                (modular.astype(np.int64) @ weights.astype(np.int64).T % 65521).astype(np.uint32))
            ids = np.array([rows - 1, 0, 17, 17], dtype=np.int64)
            # Tied aliases point to one owner; untied token bytes stay independent.
            head, token = owner, owner
            assert head is token
            np.testing.assert_array_equal(token.gather_rows(ids), weights[ids])
            token_values = token.gather_rows(ids).astype(np.float32) * weight_scales[ids, None]
            assert token_values.tobytes() == (weights[ids].astype(np.float32) * weight_scales[ids, None]).tobytes()
            columns = np.array([cols - 1, 0, 0], dtype=np.int64)
            np.testing.assert_array_equal(token.gather_columns(columns), weights[:, columns].T)
            untied = rng.integers(-128, 128, weights.shape, dtype=np.int8)
            token_path = root / "untied.raw"
            token_path.write_bytes(untied.tobytes())
            with PagedGEMM.from_raw(token_path, sha(untied.tobytes()), weights.shape, threads=1) as lookup:
                np.testing.assert_array_equal(lookup.gather_rows(ids), untied[ids])
                np.testing.assert_array_equal(head.execute(inputs), expected)
            ownership = owner.ownership
            for index in tuple(owner._resident):
                owner.evict(index)
            np.testing.assert_array_equal(owner.execute(inputs), expected)
            np.testing.assert_array_equal(token.gather_rows(ids), weights[ids])
            return {"choice": choice, "compression": compression, "shape": [rows, cols],
                    "original_weight_sha256": original_hash, "output_sha256": sha(values.tobytes()),
                    "original_object_sha256": original_hash,
                    "synthetic_numeric_metadata_sha256": sha(activation_scales.tobytes() + weight_scales.tobytes() + bias.tobytes()),
                    "full_logits_sha256": sha(logits.tobytes()),
                    "checked_full_output_elements": int(values.size), "ownership": ownership,
                    "logical_paged_payload_bytes_all_checks": owner.logical_paged_payload_bytes,
                    "logical_resident_payload_bytes_all_checks": owner.logical_resident_payload_bytes,
                    "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * (1 if platform.system() == "Darwin" else 1024),
                    "native_module": str(owner._kernel.library_path),
                    "native_sha256": sha(owner._kernel.library_path.read_bytes()),
                    "controls": ["clear", "full_float32_logits_and_bias", "wrap32", "modular", "tied_rows", "tied_columns",
                                 "untied_rows", "source_mutation", "eviction"]}
        finally:
            source.close()
            if owner is not None:
                owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compression", choices=("raw", "zlib"), default="raw")
    args = parser.parse_args()
    if args.output is None or args.output.exists():
        parser.error("new --output required")
    samples = []
    for choice in ("paged", "partial", "full"):
        samples.append(run_case(choice, compression=args.compression))
    assert len({row["output_sha256"] for row in samples}) == 1
    assert len({row["full_logits_sha256"] for row in samples}) == 1
    repo = Path(__file__).resolve().parents[1]
    paths = ("scripts/probe_resident_head_shards.py", "tests/test_resident_head_shards_probe.py",
             "crates/pllm-core/src/paged.rs", "crates/pllm-python/src/paged.rs",
             "crates/pllm-core/src/kernels.rs", "crates/pllm-core/src/public_file.rs",
             "python/pllm/runtime/bundle_storage.py",
             "python/pllm/runtime/paged.py", "python/pllm/runtime/native.py",
             "python/pllm/runtime/transformer_client.py")
    evidence = {"schema": "pllm.resident_head_shards.synthetic.v1", "samples": samples,
                "fresh_process": False, "python": sys.version,
                "benchmark_baseline": "all-paged arm is sharded, not ordinary unsharded PagedGEMM",
                "platform": platform.platform(), "probe_and_inspected_source_sha256":
                    {p: sha((repo / p).read_bytes()) for p in paths},
                "qualification": "synthetic correctness only; no tier, traffic or speedup claim",
                "authentication": "actual-byte SHA-256 at import; no repeated native page MAC/hash",
                "footprint": "67x33 i8 head; <=2211 resident bytes; 2112 reserved promotion bytes; full original and derived disk copies at setup"}
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(samples), "qualification": evidence["qualification"]}))


if __name__ == "__main__":
    main()
