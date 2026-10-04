"""No-training public rotation/precision gate on a checked decoder binding."""
from dataclasses import dataclass
import hashlib
import secrets
import time


@dataclass(frozen=True, slots=True)
class OrthogonalActivationProbe:
    activation_bits: int = 6
    block: int = 128
    decode_steps: int = 3

    def __post_init__(self):
        if type(self.activation_bits) is not int or self.activation_bits not in (4, 6, 8):
            raise ValueError("activation bits must be 4, 6 or 8")
        if type(self.block) is not int or self.block not in (8, 16, 32, 64, 128):
            raise ValueError("orthogonal block must be a public power of two from 8 to 128")
        if type(self.decode_steps) is not int or not 0 <= self.decode_steps <= 8:
            raise ValueError("decode steps must be 0..8")

    def run(self, compiled, remote, weights, token_cohort) -> dict:
        import numpy as np
        from pllm import _native
        from pllm.native import MaskedGEMM
        from pllm.runtime.quantization import quantize_weight_per_row, quantize_activation_per_row, dequantize_matmul
        compiled.validate()
        cohorts = tuple(tuple(ids) for ids in token_cohort)
        if not 1 <= len(cohorts) <= 16 or any(not 1 <= len(ids) <= compiled._plan.prefill["query_sequence"] for ids in cohorts):
            raise ValueError("orthogonal probe requires bounded public token cohorts")
        stages = [s for s in compiled.stage_bindings if s.layer_index is not None]
        if not stages or sum(s.in_features * s.out_features for s in stages) > 1 << 30:
            raise ValueError("orthogonal probe body exceeds 1 GiB i8 policy")
        bound = int(compiled._bundle.cfg["vocab_size"])
        if any(type(token) is not int or not 0 <= token < bound for ids in cohorts for token in ids):
            raise ValueError("public token cohort is invalid")
        for stage in stages:
            w = np.asarray(weights[stage.stage_id])
            if w.dtype != np.int8 or w.shape != (stage.out_features, stage.in_features) or hashlib.sha256(w.tobytes()).hexdigest() != stage.weight_digest:
                raise ValueError("orthogonal source weights differ from compiled commitment")
            if stage.in_features % self.block or compiled._bundle.stages[stage.stage_id].input_equalization is not None:
                raise ValueError("orthogonal probe cannot bind this input layout or equalization")
        references = []
        for ids in cohorts:
            runtime = compiled.runtime(remote)
            logits = runtime.prepare_ids(ids)[1]
            rows, selected = [logits.copy()], []
            for _ in range(self.decode_steps):
                token = int(np.argmax(logits))
                selected.append(token)
                logits = runtime.decode_step(token, runtime.caches)[0]
                rows.append(logits.copy())
            references.append((selected, rows))
        records = []
        owner = MaskedGEMM(threads=1)
        for rotated in (False, True):
            started = time.process_time_ns()
            kernels, scales, seeds, widths = {}, {}, {}, {}
            for stage in stages:
                spec = compiled._bundle.stages[stage.stage_id]
                w = np.asarray(weights[stage.stage_id])
                seed = int.from_bytes(hashlib.sha256(stage.stage_id.encode()).digest()[:8], "little")
                seeds[stage.stage_id] = seed
                if rotated:
                    full = w.astype(np.float32) * spec.weight_scales[:, None]
                    data = _native.public_hadamard(full.astype("<f4").tobytes(), stage.in_features, self.block, seed)
                    q = quantize_weight_per_row(np.frombuffer(data, "<f4").reshape(w.shape), bits=8)
                    w, scale = q.values, q.scales
                else:
                    scale = spec.weight_scales
                kernels[stage.stage_id] = owner.compile(w)
                scales[stage.stage_id] = scale
                widths[stage.stage_id] = _native.offset_row_bits(w.tobytes(), stage.in_features, (1 << (self.activation_bits - 1)) - 1)
            setup_cpu = (time.process_time_ns() - started) / 1e9
            client_transform_cpu = 0

            def candidate(stage_id, activation):
                nonlocal client_transform_cpu
                spec = compiled._bundle.stages[stage_id]
                original_shape = activation.shape
                if rotated:
                    start = time.process_time_ns()
                    data = _native.public_hadamard(np.asarray(activation, dtype="<f4").tobytes(), spec.in_features, self.block, seeds[stage_id])
                    activation = np.frombuffer(data, "<f4").reshape(original_shape)
                    client_transform_cpu += time.process_time_ns() - start
                q = quantize_activation_per_row(activation, bits=self.activation_bits)
                result = kernels[stage_id].clear(q.values)
                output = dequantize_matmul(result, q.scales, scales[stage_id], output_shape=original_shape[:-1] + (spec.out_features,))
                if spec.bias is not None:
                    output += spec.bias
                return np.ascontiguousarray(output, dtype=np.float32)

            prefill, decode, worst = 0, 0, 0.0
            for ids, (selected, golden) in zip(cohorts, references, strict=True):
                runtime = compiled.runtime(candidate)
                logits = runtime.prepare_ids(ids)[1]
                prefill += int(np.argmax(logits) == np.argmax(golden[0]))
                worst = max(worst, float(np.max(np.abs(logits - golden[0]))))
                for i, token in enumerate(selected):
                    logits = runtime.decode_step(token, runtime.caches)[0]
                    decode += int(np.argmax(logits) == np.argmax(golden[i + 1]))
                    worst = max(worst, float(np.max(np.abs(logits - golden[i + 1]))))
            word = lambda s: next(b for b in (16, 24, 32) if b >= max(widths[s.stage_id])) // 8
            dense = sum((s.in_features + 2 * s.out_features) * word(s) for s in stages)
            packed = sum(s.in_features * word(s) + 2 * ((sum(widths[s.stage_id]) + 7) // 8) for s in stages)
            records.append({"public_rotation": rotated, "activation_bits": self.activation_bits,
                "prefill_top1_matches": prefill, "prefill_cases": len(cohorts), "same_token_decode_top1_matches": decode,
                "same_token_decode_cases": len(cohorts) * self.decode_steps, "worst_absolute_logit_error": worst,
                "provider_transform_and_kernel_setup_cpu_seconds": setup_cpu,
                "extra_client_transform_cpu_seconds": client_transform_cpu / 1e9,
                "dense_prepared_arithmetic_bytes_per_executed_row": dense,
                "packed_prepared_arithmetic_bytes_per_executed_row": packed,
                "all_checked_selections_match": prefill == len(cohorts) and decode == len(cohorts) * self.decode_steps})
            kernels.clear()
        return {"schema": "pllm.orthogonal_activation_probe.v1", "model_plan_digest": compiled.model_plan_digest,
                "binding_digest": compiled.digest, "cohort_digest": hashlib.sha256(secrets.token_bytes(32) + repr(cohorts).encode()).hexdigest(),
                "cohort_digest_salted": True, "public_block": self.block, "configurations": records,
                "scope": "local clear-kernel precision/rotation ablation against locked W8A8 execution",
                "client_body_weights_required": 0, "whole_decoder_executable": False,
                "limitations": ["Transforms dequantized W8 source weights then re-quantizes; no checkpoint training",
                    "Client transformation CPU measured separately; fixture host also owns provider weights",
                    "Body estimates exclude masks/ticket/control/transport metadata and token boundary",
                    "No Pipeline numeric contract, broad quality, protected transport or peak-memory admission"]}
