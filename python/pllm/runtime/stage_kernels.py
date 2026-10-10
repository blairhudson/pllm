"""Internal measured, phase-aware stage placement reference for existing models.

Only the fixed remote linear groups choose CPU or Metal. Target arithmetic,
ownership, weights and the client semantic schedule retain their identities.
This local-clear callback is not a negotiated provider execution protocol.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from pllm import _native
from pllm.metrics import BuildRef
from .metal import MetalGEMM
from .quantization import quantize_activation_per_row, dequantize_matmul


def stage_groups(compiled, engine):
    schedule = compiled._validated_schedule()
    model = engine.models[compiled._bundle.model_id]
    metadata, privacy = model.manifest.metadata, compiled._bundle.privacy
    if (metadata.get("body_fingerprint") != privacy.get("body_fingerprint")
            or metadata.get("seeded_stage_commitment") != privacy.get("stage_commitment")):
        raise ValueError("stage kernel checkpoint differs from bound source")
    bindings = {op: stage for stage in compiled.stage_bindings for op in stage.semantic_operations}
    groups = {}
    for step in schedule["prefill"]["steps"]:
        if step["operators"] != ["linear"] * len(step["operators"]):
            continue
        if step["executor"] != "remote_stage" or step.get("terminal_row_only"):
            raise ValueError("stage kernel reference requires ordinary remote linear groups")
        row = bindings[f"prefill:{step['operation_ids'][0]}"]
        stage, bound = model.stages[row.stage_id], compiled._bundle.stages[row.stage_id]
        if (stage.compiled_weight is None or stage.weight_digest != row.weight_digest
                or stage.spec.in_features != row.in_features or stage.spec.out_features != row.out_features
                or stage.spec.activation_bits != row.activation_bits or stage.spec.weight_bits != row.weight_bits
                or stage.modulus != row.modulus or stage.wire_bits != row.wire_bits
                or not np.array_equal(stage.weight.scales, bound.weight_scales)
                or (stage.bias is None) != (bound.bias is None)
                or (stage.bias is not None and not np.array_equal(stage.bias, bound.bias))
                or stage.input_equalization is not None or bound.input_equalization is not None):
            raise ValueError("stage kernel weights or numeric boundaries differ")
        groups[tuple(step["weight_ids"])] = (row.stage_id, stage)
    return groups


def compile_stage_kernels(compiled, profile, *, memory_bytes):
    if (type(memory_bytes) is not int or not 0 <= memory_bytes <= 2 << 30
            or profile.get("binding_digest") != compiled.digest
            or profile.get("build_digest") != BuildRef.capture().expected_digest):
        raise ValueError("stage kernel build, binding or resource contract differs")
    return json.loads(_native.stage_kernel_plan(compiled._plan.canonical_bytes(),
        compiled._canonical_composition, json.dumps(profile, sort_keys=True).encode(), memory_bytes))


class StageKernelExecutor:
    def __init__(self, compiled, engine, profile, *, memory_bytes):
        self.plan = compile_stage_kernels(compiled, profile, memory_bytes=memory_bytes)
        groups = stage_groups(compiled, engine)
        self.stages, self.choices, self.gpu = {}, {}, {}
        self.counts = Counter()
        self.closed = False
        weights = {}
        for choice in self.plan["choices"]:
            stage_id, stage = groups[tuple(choice["weights"])]
            self.stages[stage_id] = (stage.compiled_weight, stage.weight.scales.copy(),
                None if stage.bias is None else stage.bias.copy(), stage.spec.activation_bits,
                stage.spec.in_features, stage.spec.out_features)
            self.choices[stage_id] = choice
            if "metal" in (choice["prefill"], choice["decode"]):
                weights[stage_id] = stage.compiled_weight.weight_view()
        try:
            if weights:
                self.gpu = MetalGEMM().bind_stages(weights)
        except BaseException:
            self.close()
            raise

    def __call__(self, stage_id, activation):
        try:
            if self.closed or stage_id not in self.stages:
                raise ValueError("stage kernel owner is unavailable")
            matrix, scales, bias, bits, width, output = self.stages[stage_id]
            values = np.asarray(activation)
            if (values.ndim != 2 or values.dtype != np.float32 or values.shape[1] != width
                    or values.shape[0] not in {1, self.plan["prefill_rows"]} or not np.all(np.isfinite(values))):
                raise ValueError("stage kernel query exceeds its measured public geometry")
            phase = "decode" if values.shape[0] == 1 else "prefill"
            backend = self.choices[stage_id][phase]
            quantized = quantize_activation_per_row(values, bits=bits)
            kernel = self.gpu[stage_id] if backend == "metal" else matrix
            integer = kernel.clear(quantized.values)
            result = dequantize_matmul(integer, quantized.scales, scales,
                                      output_shape=quantized.original_shape[:-1] + (output,))
            if bias is not None:
                result = result + bias
            self.counts[f"{phase}_{backend}_calls"] += 1
            return np.ascontiguousarray(result, dtype=np.float32)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        self.gpu.clear()
        self.stages.clear()
