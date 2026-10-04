"""Bounded SDK research gates; real weights require the explicit --real switch."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from pllm import Model, lower_model
from pllm.metrics import GeneratedStateReuseProbe, StateCompatibilityProbe
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


CONTEXTS = (
    "A telescope helps astronomers observe distant objects. Explain why.",
    "List two ways to check whether a short calculation is correct.",
    "Describe a practical use for a map and a compass.",
)
SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")


def load(source, *, placement=False):
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
        client_linear_roles=("qkv_projection", "attention_output") if placement else ())
    asyncio.run(engine.load(source.manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))

    def remote(stage_id, activation):
        stage = engine.models[bundle.model_id].stages[stage_id]
        q = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        output = dequantize_matmul(stage.compiled_weight.clear(q.values), q.scales, stage.weight.scales,
            output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
        if stage.bias is not None:
            output += stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)
    return engine, bundle, remote


def state_gates(source):
    engine, bundle, remote = load(source)
    other_engine = None
    try:
        plan = lower_model(json.loads((Path(source.manifest.source) / "config.json").read_bytes()),
            batch=1, max_input_tokens=128, max_new_tokens=16)
        ids = [bundle.tokenizer().encode(text,
            add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True))) for text in CONTEXTS]
        pipeline = MaskedLinearCpu(Model(source.manifest.id), quantization=SymmetricPerRow(
            weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
        compiled = compile_runtime_model(plan, bundle, composition=pipeline)
        generated = [GeneratedStateReuseProbe(steps=8).run(compiled, remote, tokens) for tokens in ids]
        transport = compile_runtime_model(plan, bundle, composition=pipeline.with_params(
            delivery=ClientBundleTransport("artifacts", compression="zlib")))
        other_engine, placed_bundle, placed_remote = load(source, placement=True)
        placed = compile_runtime_model(plan, placed_bundle, composition=pipeline.with_params(
            placement=ClientLinearRoles(["qkv_projection", "attention_output"])))
        pairs = []
        for name, target, target_remote in (("compressed_delivery", transport, remote),
                                           ("client_attention", placed, placed_remote)):
            record = StateCompatibilityProbe().compare(compiled, target)
            record["candidate"] = name
            record["same_token_prefill_and_decode"] = []
            record["transferred_suffixes"] = []
            for tokens in ids:
                a, b = compiled.runtime(remote), target.runtime(target_remote)
                la, lb = a.prepare_ids(tokens)[1], b.prepare_ids(tokens)[1]
                equal = bool(np.array_equal(la.view(np.uint32), lb.view(np.uint32)))
                for _ in range(4):
                    token = int(np.argmax(la))
                    la = a.decode_step(token, a.caches)[0]
                    lb = b.decode_step(token, b.caches)[0]
                    equal &= bool(np.array_equal(la.view(np.uint32), lb.view(np.uint32)))
                from pllm.metrics.state_reuse import _same_state
                record["same_token_prefill_and_decode"].append(equal and _same_state(a.snapshot(), b.snapshot()))
                cut = max(1, len(tokens) - 2)
                source_runtime = compiled.runtime(remote)
                source_runtime.prepare_ids(tokens[:cut])
                started = time.process_time_ns()
                transferred = compiled.transfer_snapshot(source_runtime.snapshot(), target)
                transfer_cpu = (time.process_time_ns() - started) / 1e9
                resumed = target.runtime(target_remote)
                resumed.restore(transferred)
                from pllm.configuration import Pipeline
                resumed.install_continuation(plan.continuation_schedule(
                    Pipeline.from_spec(json.loads(target._canonical_composition))))
                actual = resumed.continue_ids(tokens[cut:])[-1]
                fresh = target.runtime(target_remote)
                expected = fresh.prepare_ids(tokens)[1]
                record["transferred_suffixes"].append({"reused_tokens": cut, "new_tokens": len(tokens) - cut,
                    "bit_equal_logits": bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
                    "bit_equal_all_kv": _same_state(resumed.snapshot(), fresh.snapshot()),
                    "client_transfer_cpu_seconds": transfer_cpu,
                    "copied_state_payload_bytes": sum(c.key.nbytes + c.value.nbytes for c in transferred.caches),
                    "provider_transfer_bytes": 0, "network_cost_measured": False})
            pairs.append(record)
        return {"generated": generated, "compatibility": pairs}
    finally:
        asyncio.run(engine.unload(bundle.model_id))
        if other_engine is not None:
            asyncio.run(other_engine.unload(bundle.model_id))


def aggregation_gate(source):
    import hashlib
    from pllm import _native
    from pllm.metrics import MaskedAggregationProbe

    engine, bundle, _ = load(source)
    try:
        plan = lower_model(json.loads((Path(source.manifest.source) / "config.json").read_bytes()),
            batch=1, max_input_tokens=128, max_new_tokens=32)
        compiled = compile_runtime_model(plan, bundle, composition=MaskedLinearCpu(Model(source.manifest.id)))
        binding = max((s for s in compiled.stage_bindings if s.layer_index is not None),
            key=lambda s: s.out_features / s.in_features)
        stage = engine.models[bundle.model_id].stages[binding.stage_id]
        weights = stage.weight.values
        widths = _native.offset_row_bits(weights.tobytes(), binding.in_features, 127)
        result = []
        for rows in (1, 39):
            rng = np.random.default_rng(340 + rows)
            x = rng.integers(-127, 128, size=(rows, binding.in_features), dtype=np.int8)
            share_b = rng.integers(0, 2**32, size=x.shape, dtype=np.uint32)
            share_a = ((x.astype(np.int64) - share_b) & 0xffffffff).astype(np.uint32)
            a, b = stage.compiled_weight.wrap32(share_a), stage.compiled_weight.wrap32(share_b)
            expected = stage.compiled_weight.clear(x)
            raw = _native.offset_reconstruct_rows(
                _native.offset_pack_rows(a.astype("<u4").tobytes(), widths, rows),
                _native.offset_pack_rows(b.astype("<u4").tobytes(), widths, rows), widths, rows)
            assert np.array_equal(np.frombuffer(raw, dtype="<i8").reshape(expected.shape), expected)
            report = MaskedAggregationProbe(repetitions=7).run(a, b, widths)
            report.update(model_plan_digest=plan.digest, weight_digest=hashlib.sha256(weights.tobytes()).hexdigest(),
                input_columns=binding.in_features, numeric_fixture="public seeded i8 inputs; native real-weight wrap32 products")
            result.append(report)
        return result
    finally:
        asyncio.run(engine.unload(bundle.model_id))


def token_local_gate(source):
    from pllm.metrics import TokenLocalProjectionProbe
    engine, bundle, remote = load(source)
    try:
        plan = lower_model(json.loads((Path(source.manifest.source) / "config.json").read_bytes()),
            batch=1, max_input_tokens=128, max_new_tokens=16)
        compiled = compile_runtime_model(plan, bundle, composition=MaskedLinearCpu(Model(source.manifest.id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32")))
        ids = [bundle.tokenizer().encode(text,
            add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True))) for text in CONTEXTS]
        weights = {key: stage.weight.values for key, stage in engine.models[bundle.model_id].stages.items()}
        return TokenLocalProjectionProbe(cache_bytes=1 << 20, decode_steps=8).run(compiled, remote, weights, ids)
    finally:
        asyncio.run(engine.unload(bundle.model_id))


def head_gate(source):
    from unittest.mock import patch
    from pllm.metrics import ProgressiveHeadProbe
    engine, bundle, remote = load(source)
    try:
        plan = lower_model(json.loads((Path(source.manifest.source) / "config.json").read_bytes()),
            batch=1, max_input_tokens=128, max_new_tokens=16)
        compiled = compile_runtime_model(plan, bundle, composition=MaskedLinearCpu(Model(source.manifest.id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32")))
        # A canonical public token-boundary stage, not a model-family dispatch.
        head_id = "lm_head"
        stage = engine.models[bundle.model_id].stages[head_id]
        if stage.bias is not None:
            raise ValueError("progressive reference currently requires a bias-free head")
        queries, scales = [], []
        original = ClientBundle.local_linear
        def capture(owner, stage_id, values):
            if owner is bundle and stage_id == head_id:
                q = quantize_activation_per_row(values, bits=8)
                queries.append(q.values.reshape(-1, stage.spec.in_features)[-1].copy())
                scales.append(float(q.scales.reshape(-1)[-1]))
            return original(owner, stage_id, values)
        with patch.object(ClientBundle, "local_linear", capture):
            for text in CONTEXTS[:2]:
                runtime = compiled.runtime(remote)
                logits = runtime.prepare_ids(bundle.tokenizer().encode(text,
                    add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True))))[1]
                for _ in range(7):
                    logits = runtime.decode_step(int(np.argmax(logits)), runtime.caches)[0]
        reports = [ProgressiveHeadProbe(prefix_bits=bits).run(stage.weight.values, stage.weight.scales,
            np.asarray(queries, np.int8), np.asarray(scales, np.float32)) for bits in (4, 5, 6)]
        return {"schema": "pllm.progressive_head_cohort.v1", "model_plan_digest": plan.digest,
            "body_fingerprint": bundle.privacy["body_fingerprint"], "configurations": reports}
    finally:
        asyncio.run(engine.unload(bundle.model_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--method", choices=("state", "aggregation", "token-local", "head"), default="state")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pllm-next-five-") as temp:
        if args.real:
            source = resolve_model(SOURCE)
        else:
            from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
            root = create_tiny_llama_checkpoint(Path(temp) / "model")
            source = resolve_model(Model.path(str(root), model_id="next-five-tiny"))
        data = {"state": state_gates, "aggregation": aggregation_gate,
                "token-local": token_local_gate, "head": head_gate}[args.method](source)
        import hashlib
        import platform
        import pllm
        from pllm import _native
        environment = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "pllm": pllm.__version__,
            "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()}
        result = {"schema": "pllm.next_five_research.v1", "source_lock_digest": source.source_lock_digest,
            "environment": environment,
            "environment_digest": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
            "real_checkpoint": args.real, args.method: data}
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if args.method == "head":
            keys = ("prefix_bits", "exact_greedy_agreements", "observed_maximum_refinement_rows",
                "control_head_cpu_seconds_median", "progressive_head_cpu_seconds_median",
                "projected_maximum_sparse_pir_bytes", "projected_maximum_sparse_pir_worker_cpu_seconds")
            print(json.dumps([{key: row[key] for key in keys} for row in data["configurations"]], indent=2))
            return
        if args.method != "state":
            print(json.dumps(data, indent=2))
            return
        print(json.dumps({"state_numeric_passed": all(row["numeric_gate_passed"]
            for row in result["state"]["generated"]), "compatibility": [
                {"candidate": row["candidate"], "contract_equal": row["contract_equal"],
                 "different_fields": row["different_contract_fields"],
                 "exact_outputs": all(row["same_token_prefill_and_decode"])}
                for row in result["state"]["compatibility"]]}, indent=2))


if __name__ == "__main__":
    main()
