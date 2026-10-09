"""Internal compiler-bound MLP working-set experiment using ordinary stage owners."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

from .transformer_client import TransformerClientError


class MlpTiling:
    def __init__(self, compiled, runtime, rows):
        from pllm import _native
        if type(rows) is not int or not 1 <= rows <= 256 or runtime.nonlinear_evaluator is not None:
            raise TransformerClientError("invalid MLP tiling numeric contract")
        raw = _native.decoder_mlp_tiling(compiled._plan.canonical_bytes(), compiled._canonical_composition, rows)
        contract = json.loads(raw)
        if (contract["source_plan_digest"] != compiled.model_plan_digest
                or contract["source_schedule_digest"] != compiled.runtime_schedule_digest):
            raise TransformerClientError("MLP tiling differs from the compiled binding")
        self.digest = hashlib.sha256(b"pllm.client_mlp_tiling.v1\0" + raw).hexdigest()
        self.rows = rows
        self.regions = {r["first_step"]: r for r in contract["regions"]}
        self.calls = self.tiles = self.peak_tile_rows = 0

    def execute(self, runtime, region, phase, steps, operations, ids, values):
        source = values[region["input_id"]]
        if (source.dtype != np.float32 or source.ndim not in (2, 3)
                or source.ndim == 3 and source.shape[0] != 1
                or source.shape[-1] != region["hidden_width"]):
            raise TransformerClientError("MLP tile input differs from native geometry")
        count = source.shape[-2]
        result = np.empty_like(source)
        for start in range(0, count, self.rows):
            end = min(start + self.rows, count)
            self._tile(runtime, region, phase, steps, operations, ids, source, result, start, end)
            self.tiles += 1
            self.peak_tile_rows = max(self.peak_tile_rows, end - start)
        values[region["output_id"]] = result
        self.calls += 1

    @staticmethod
    def _tile(runtime, region, phase, steps, operations, ids, source, result, start, end):
        # Function scope retires every intermediate before the next remote exchange.
        local = {region["input_id"]: source[..., start:end, :]}
        region_steps = steps[region["first_step"]:region["end_step"]]
        remaining = Counter(item for step in region_steps for item in step["input_ids"])
        for step in region_steps:
            if step["executor"] == "client_local":
                op_id, = step["operation_ids"]
                op = operations[op_id]
                local[op_id] = runtime._numeric_output(op, runtime._local(op, local, {}, {}))
            else:
                runtime._linear_step(phase, step, operations, ids, local, continuation=False)
            for item in step["input_ids"]:
                remaining[item] -= 1
                if not remaining[item]:
                    local.pop(item, None)
        output = local[region["output_id"]]
        if output.shape != source[..., start:end, :].shape or output.dtype != np.float32:
            raise TransformerClientError("MLP tile result differs from native geometry")
        result[..., start:end, :] = output
