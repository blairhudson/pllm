"""Compiler-bound initial-prefill attention lifetimes, with original reductions."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .transformer_client import TransformerClientError


class AttentionTiling:
    def __init__(self, compiled, runtime, query_rows=16, kv_heads=2):
        from pllm import _native

        if (type(query_rows) is not int or not 1 <= query_rows <= 256
                or type(kv_heads) is not int or not 1 <= kv_heads <= 64
                or runtime.position != 0 or runtime.nonlinear_evaluator is not None
                or any(getattr(runtime, attr) is not None for attr in ("_retained", "_tiered", "_capsule"))
                or runtime.bundle is not compiled._bundle):
            raise TransformerClientError("invalid attention tiling owner or geometry")
        raw = _native.decoder_attention_tiling(compiled._plan.canonical_bytes(),
            compiled._canonical_composition, query_rows, kv_heads)
        contract = json.loads(raw)
        schedule = json.dumps(runtime._schedule, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        digest = hashlib.sha256(b"pllm.decoder_runtime_schedule.v2\0" + schedule).hexdigest()
        if (contract["source_plan_digest"] != runtime._source_plan.digest
                or contract["source_schedule_digest"] != compiled.runtime_schedule_digest
                or digest != compiled.runtime_schedule_digest):
            raise TransformerClientError("attention tiling differs from the compiled binding")
        self.digest = hashlib.sha256(b"pllm.client_attention_tiling.v1\0" + raw).hexdigest()
        self.query_rows, self.kv_heads = query_rows, kv_heads
        self.max_rows = contract["maximum_query_sequence"]
        self.max_keys = contract["maximum_key_sequence"]
        self.regions = {region["first_step"]: region for region in contract["regions"]}
        self.calls = self.tiles = self.peak_score_bytes = 0
        self.failed = False

    def execute(self, runtime, region, operations, values):
        if self.failed or runtime.position != 0:
            raise TransformerClientError("attention tiling owner is unavailable")
        query, keys, cache, positions = (values[region[key]] for key in
            ("query_id", "key_id", "value_id", "positions_id"))
        if (any(type(a) is not np.ndarray or a.dtype != np.float32 or a.ndim != 4
                for a in (query, keys, cache))
                or query.shape[:2] != (1, region["query_heads"])
                or keys.shape[:2] != (1, region["kv_heads"]) or cache.shape != keys.shape
                or query.shape[-1] != region["head_dim"] or keys.shape[-1] != region["head_dim"]
                or not 1 <= query.shape[2] <= self.max_rows
                or not query.shape[2] <= keys.shape[2] <= self.max_keys
                or positions.dtype != np.int64 or positions.shape != (query.shape[2],)
                or not np.array_equal(positions, np.arange(query.shape[2], dtype=np.int64))):
            raise TransformerClientError("attention tile inputs differ from native geometry")
        # The complete output remains live. Q/K/V are borrowed strided views;
        # the key capacity is never shortened, including masked zero slots.
        result = np.empty(query.shape, np.float32)
        try:
            for head in range(0, keys.shape[1], self.kv_heads):
                end = min(head + self.kv_heads, keys.shape[1])
                for row in range(0, query.shape[2], self.query_rows):
                    stop = min(row + self.query_rows, query.shape[2])
                    self._tile(runtime, region, operations, query, keys, cache, positions,
                               result, head, end, row, stop)
                    self.tiles += 1
                    self.peak_score_bytes = max(self.peak_score_bytes,
                        4 * (end - head) * region["group_size"] * (stop - row) * keys.shape[2])
            values[region["output_id"]] = result
            self.calls += 1
        except BaseException:
            result.fill(0)
            self.failed = True
            raise

    @staticmethod
    def _tile(runtime, region, operations, query, keys, cache, positions, result,
              head, end, row, stop):
        group = region["group_size"]
        tile_positions = positions[row:stop]
        local = {region["query_id"]: query[:, head * group:end * group, row:stop],
                 region["key_id"]: keys[:, head:end], region["value_id"]: cache[:, head:end],
                 region["positions_id"]: tile_positions}
        prior = None
        try:
            for op_id in region["operation_ids"]:
                op = operations[op_id]
                local[op_id] = runtime._numeric_output(op, runtime._local(
                    op, local, {}, {}, attention_positions=tile_positions))
                if prior is not None:
                    del local[prior]
                prior = op_id
            result[:, head * group:end * group, row:stop] = local[region["output_id"]]
        finally:
            # No tile-sized tensor escapes, including through a failed frame.
            local.clear()
