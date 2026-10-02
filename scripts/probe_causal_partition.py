"""Numeric gate only: canonical causal-prefix reductions on an unchanged W8A8 body."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pllm import Model, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime import causal_reduction
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@contextmanager
def canonical_reductions():
    original = SemanticDecoderRuntime._local

    def local(runtime, operation, tensors, *args):
        kind = operation["operator"]
        inputs = operation["inputs"]
        if kind in {"attention_scores", "attention_values"}:
            if "key_layout" in operation["attributes"] or "value_layout" in operation["attributes"]:
                raise ValueError("numeric gate supports full dense causal attention only")
            fn = causal_reduction.scores if kind == "attention_scores" else causal_reduction.values
            queries = tensors[inputs[0]]
            positions = np.arange(runtime.position, runtime.position + queries.shape[2], dtype=np.int64)
            return fn(queries, tensors[inputs[1]], positions,
                      int(operation["attributes"]["group_size"]))
        if kind == "softmax":
            return causal_reduction.softmax(tensors[inputs[0]])
        return original(runtime, operation, tensors, *args)

    with patch.object(SemanticDecoderRuntime, "_local", local):
        yield


def run(root):
    source = resolve_model(Model.path(str(root), model_id="partition-gate"))
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(source.manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("partition-gate"))
    graph = lower_model(json.loads((root / "config.json").read_bytes()), batch=1,
                        max_input_tokens=64, max_new_tokens=4)
    pipeline = MaskedLinearCpu(Model("partition-gate"), quantization=SymmetricPerRow())
    prompt = "Describe what makes a clear explanation useful. Give one example."
    ids = bundle.tokenizer().encode(prompt, add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True)))
    ids = ids[:31]

    def remote(stage_id, activation):
        stage = engine.models[bundle.model_id].stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        output = dequantize_matmul(stage.compiled_weight.clear(quantized.values), quantized.scales,
            stage.weight.scales, output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,))
        if stage.bias is not None:
            output += stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    reports = []
    for mode in ("legacy", "canonical_prefix_f32_reference"):
        selected = pipeline if mode == "legacy" else MaskedLinearCpu(Model("partition-gate"),
            quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
        compiled = compile_runtime_model(graph, bundle, composition=selected)
        continuation = graph.continuation_schedule(selected)
        from contextlib import nullcontext
        with nullcontext():
            fresh = compiled.runtime(remote)
            reference = fresh.prepare_ids(ids)[1]
            golden = fresh.snapshot()
            for prefix in sorted({1, 4, max(1, len(ids) // 2), len(ids) - 1}):
                split = compiled.runtime(remote)
                split.prepare_ids(ids[:prefix])
                split.install_continuation(continuation)
                actual = split.continue_ids(ids[prefix:])[-1]
                snapshot = split.snapshot()
                equal_kv = all(np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)
                    for a, b in zip(snapshot.caches, golden.caches, strict=True))
                reports.append({"mode": mode, "partition": prefix,
                    "array_equal_logits": bool(np.array_equal(reference, actual)),
                    "array_equal_all_kv": equal_kv,
                    "max_abs_logit_difference": float(np.max(np.abs(reference - actual)))})
            folded = compiled.runtime(remote)
            folded.prepare_ids(ids[:1])
            for token in ids[1:]:
                actual = folded.decode_step(token, folded.caches)[0]
            snapshot = folded.snapshot()
            reports.append({"mode": mode, "partition": "teacher_fold",
                "array_equal_logits": bool(np.array_equal(reference, actual)),
                "array_equal_all_kv": all(np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)
                    for a, b in zip(snapshot.caches, golden.caches, strict=True)),
                "max_abs_logit_difference": float(np.max(np.abs(reference - actual)))})
    return {"schema": "pllm.causal_partition_gate.v1", "source_lock_digest": source.source_lock_digest,
        "body_fingerprint": bundle.privacy["body_fingerprint"], "input_tokens": len(ids),
        "cohort_digest": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "scope": "local W8A8 numeric gate with explicit compiled prefix_f32 contract; no cache promotion",
        "samples": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
