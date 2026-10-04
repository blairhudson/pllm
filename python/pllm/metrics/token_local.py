"""Token-local projection discovery and exact client-kernel memoization screen."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time


def _token_local_stages(compiled):
    # Only declared row-local operators can propagate token-only dependence.
    safe = {"linear", "rms_norm", "reshape", "scale", "multiply", "silu", "residual_add",
            "sigmoid", "gelu_tanh", "softcap", "permute", "slice"}
    names = set()
    graphs = compiled._plan.to_dict()
    for phase in ("prefill", "decode"):
        local = set()
        for op in graphs[phase]["operations"]:
            if op["operator"] == "token_lookup" or (op["operator"] in safe and op["inputs"]
                    and all(value in local for value in op["inputs"])):
                local.add(op["id"])
                names.add(f"{phase}:{op['id']}")
    return tuple(stage for stage in compiled.stage_bindings
                 if stage.layer_index is not None and all(name in names for name in stage.semantic_operations))


@dataclass(frozen=True, slots=True)
class TokenLocalProjectionProbe:
    """All selected stages are client-local, including misses; hit traces stay local.

    Selective remote omission would reveal token equality. This reference instead
    charges a fixed public ownership set and its extra weights. It does not
    enable a Pipeline placement or bypass any provider's material lifecycle.
    """
    cache_bytes: int = 1 << 20
    decode_steps: int = 4

    def __post_init__(self):
        if type(self.cache_bytes) is not int or not 1 <= self.cache_bytes <= 64 << 20:
            raise ValueError("cache_bytes must be in [1, 64 MiB]")
        if type(self.decode_steps) is not int or not 0 <= self.decode_steps <= 32:
            raise ValueError("decode_steps must be in [0, 32]")

    def run(self, compiled, remote, weights, token_cohort):
        import numpy as np
        from pllm import _native
        from pllm.runtime.native import MaskedGEMM
        from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
        from pllm.metrics.state_reuse import _same_state

        compiled.validate()
        selected = _token_local_stages(compiled)
        if not selected or len(selected) > 16:
            raise ValueError("probe requires one to sixteen token-local projection groups")
        memos, kernels = {}, {}
        client_weights = 0
        for stage in selected:
            w = np.asarray(weights[stage.stage_id])
            if (w.dtype != np.int8 or w.shape != (stage.out_features, stage.in_features)
                    or hashlib.sha256(w.tobytes()).hexdigest() != stage.weight_digest):
                raise ValueError("token-local weights differ from compiled commitment")
            if compiled._bundle.stages[stage.stage_id].input_equalization is not None:
                raise ValueError("token-local probe does not yet bind equalization")
            maximum = max(1, self.cache_bytes // len(selected))
            memos[stage.stage_id] = _native._IntegerRowMemo(w.tobytes(), *w.shape, maximum)
            kernels[stage.stage_id] = MaskedGEMM(threads=1).compile(w)
            client_weights += w.nbytes
        rows_seen = 0
        cpu = {"memo": 0.0, "clear": 0.0}

        def callback(stage_id, activation):
            nonlocal rows_seen
            if stage_id not in memos:
                return remote(stage_id, activation)
            stage = compiled._bundle.stages[stage_id]
            q = quantize_activation_per_row(activation, bits=stage.activation_bits)
            rows = q.values.size // stage.in_features
            rows_seen += rows
            started = time.process_time_ns()
            result = np.frombuffer(memos[stage_id].evaluate(q.values.tobytes(), rows), dtype="<i4").reshape(rows, stage.out_features)
            cpu["memo"] += (time.process_time_ns() - started) / 1e9
            started = time.process_time_ns()
            oracle = kernels[stage_id].clear(q.values)
            cpu["clear"] += (time.process_time_ns() - started) / 1e9
            if not np.array_equal(result, oracle):
                raise AssertionError("memo changed exact integer projection")
            output = dequantize_matmul(result, q.scales, stage.weight_scales,
                output_shape=q.original_shape[:-1] + (stage.out_features,))
            if stage.bias is not None:
                output += stage.bias
            return np.ascontiguousarray(output, dtype=np.float32)

        checks = []
        cohorts = tuple(tuple(ids) for ids in token_cohort)
        if not 1 <= len(cohorts) <= 16:
            raise ValueError("expected one to sixteen public token cohorts")
        for ids in cohorts:
            if not 1 <= len(ids) <= compiled._plan.to_dict()["prefill"]["query_sequence"]:
                raise ValueError("token cohort exceeds compiled prefill bound")
            a, b = compiled.runtime(remote), compiled.runtime(callback)
            x, y = a.prepare_ids(ids)[1], b.prepare_ids(ids)[1]
            exact = bool(np.array_equal(x.view(np.uint32), y.view(np.uint32)))
            for _ in range(self.decode_steps):
                token = int(np.argmax(x))
                x, y = a.decode_step(token, a.caches)[0], b.decode_step(token, b.caches)[0]
                exact &= bool(np.array_equal(x.view(np.uint32), y.view(np.uint32)))
            checks.append(exact and _same_state(a.snapshot(), b.snapshot()))
        body = [s for s in compiled.stage_bindings if s.layer_index is not None]
        body_bytes = lambda stage: (stage.in_features + 2 * stage.out_features) * (stage.wire_bits // 8)
        selected_bodies = sum(body_bytes(s) for s in selected)
        all_bodies = sum(body_bytes(s) for s in body)
        selected_macs = sum(s.in_features * s.out_features for s in selected)
        all_macs = sum(s.in_features * s.out_features for s in body)
        stats = [memo.stats() for memo in memos.values()]
        return {"schema": "pllm.token_local_projection_probe.v1", "complete": False, "executable": False,
            "model_plan_digest": compiled._plan.digest, "groups": len(selected),
            "exact_logits_and_all_kv": checks, "projection_rows": rows_seen,
            "cache_hits": sum(s[0] for s in stats), "cache_misses": sum(s[1] for s in stats),
            "accounted_cache_bytes": sum(s[3] for s in stats), "cache_capacity_bytes": self.cache_bytes,
            "new_client_i8_weights_bytes": client_weights, "new_client_native_snapshot_bytes": client_weights,
            "raw_cold_weight_transfer_bytes": client_weights, "client_peak_memory_bytes": None,
            "memo_kernel_cpu_seconds": cpu["memo"], "clear_kernel_cpu_seconds": cpu["clear"],
            "optimistic_prepared_arithmetic_body_reduction": selected_bodies / all_bodies,
            "remaining_remote_body_linear_mac_fraction": 1 - selected_macs / all_macs,
            "fixed_public_ownership_required": True, "conditional_remote_skip_privacy_admitted": False,
            "full_wire_bytes": None, "tenfold_body_gate_passed": selected_bodies / all_bodies >= 0.9}
