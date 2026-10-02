"""Numeric gate only: canonical causal-prefix reductions on an unchanged W8A8 body."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from pllm import Model, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


CONTEXTS = (
    "Describe what makes a clear explanation useful. Give one example.",
    "A public benchmark must count preparation, online traffic, cache misses and cancellation. "
    "Explain why reusing a conversation prefix does not reduce unrelated fresh requests.",
    "Summarize this public note: inference runs on separate roles. The client keeps private state. "
    "Public model objects may be cached, but cryptographic masks must be fresh for each execution. "
    "Compare a repeated question with a new question and explain which costs remain.",
)


def run(root):
    source = resolve_model(Model.path(str(root), model_id="partition-gate"))
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(source.manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("partition-gate"))
    graph = lower_model(json.loads((root / "config.json").read_bytes()), batch=1,
                         max_input_tokens=128, max_new_tokens=4)
    pipeline = MaskedLinearCpu(Model("partition-gate"), quantization=SymmetricPerRow())
    cohorts = [bundle.tokenizer().encode(prompt,
        add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True))) for prompt in CONTEXTS]

    def remote(stage_id, activation):
        stage = engine.models[bundle.model_id].stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        output = dequantize_matmul(stage.compiled_weight.clear(quantized.values), quantized.scales,
            stage.weight.scales, output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,))
        if stage.bias is not None:
            output += stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    reports = []
    for mode in ("legacy", "prefix_f32"):
        selected = pipeline if mode == "legacy" else MaskedLinearCpu(Model("partition-gate"),
            quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
        compiled = compile_runtime_model(graph, bundle, composition=selected)
        continuation = graph.continuation_schedule(selected)
        for index, ids in enumerate(cohorts):
            fresh = compiled.runtime(remote)
            reference = fresh.prepare_ids(ids)[1]
            golden = fresh.snapshot()
            for prefix in range(1, len(ids)):
                split = compiled.runtime(remote)
                split.prepare_ids(ids[:prefix])
                split.install_continuation(continuation)
                actual = split.continue_ids(ids[prefix:])[-1]
                snapshot = split.snapshot()
                equal_kv = all(np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)
                    for a, b in zip(snapshot.caches, golden.caches, strict=True))
                reports.append({"mode": mode, "context_index": index, "partition": prefix,
                    "array_equal_logits": bool(np.array_equal(reference, actual)),
                    "array_equal_all_kv": equal_kv,
                    "max_abs_logit_difference": float(np.max(np.abs(reference - actual)))})
            folded = compiled.runtime(remote)
            actual = folded.prepare_ids(ids[:1])[1]
            for token in ids[1:]:
                actual = folded.decode_step(token, folded.caches)[0]
            snapshot = folded.snapshot()
            reports.append({"mode": mode, "context_index": index, "partition": "teacher_fold",
                "array_equal_logits": bool(np.array_equal(reference, actual)),
                "array_equal_all_kv": all(np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)
                    for a, b in zip(snapshot.caches, golden.caches, strict=True)),
                "max_abs_logit_difference": float(np.max(np.abs(reference - actual)))})
    asyncio.run(engine.unload(bundle.model_id))
    return {"schema": "pllm.causal_partition_gate.v2", "source_lock_digest": source.source_lock_digest,
        "body_fingerprint": bundle.privacy["body_fingerprint"], "input_tokens": [len(ids) for ids in cohorts],
        "cohort_digest": hashlib.sha256(json.dumps(cohorts, separators=(",", ":")).encode()).hexdigest(),
        "canonical_passed": all(row["array_equal_logits"] and row["array_equal_all_kv"]
                                for row in reports if row["mode"] == "prefix_f32"),
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
    if not result["canonical_passed"]:
        raise SystemExit("canonical numeric gate failed")
