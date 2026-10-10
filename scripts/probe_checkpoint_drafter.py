"""Fresh-process existing-model drafter gate; target remains pinned Qwen W8A8."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROMPTS = (
    "Name two tools a carpenter uses, in one sentence.",
    "Why does a shadow change length during a day? Answer briefly.",
    "Finish this Python expression: sorted([4, 1, 3]) returns",
    "Write one polite sentence declining a meeting on Tuesday.",
    "What happens when water freezes? Give one sentence.",
    "Give three different words meaning happy.",
    "Explain how a dictionary differs from a list in Python.",
    "Convert three hours into minutes. Show the multiplication.",
    "Describe the purpose of a library in one short sentence.",
    "Write a short sentence with the words moon and river.",
    "What is the next number after 2, 4, 6, 8? Explain briefly.",
    "Give a practical tip for keeping a desk tidy.",
)


def worker(mode):
    import asyncio
    import numpy as np
    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.checkpoint_drafter import CheckpointDrafter
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.greedy_verification_reference import verify_batched_greedy
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    from scripts.decoder_probe_support import DecoderFixture, stage_output
    from scripts.probe_task_qualified_models import MODELS

    fixture = DecoderFixture(inputs=256, outputs=32)
    quantization = SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32")
    fixture.composition = fixture.composition.with_params(quantization=quantization)
    fixture.compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=fixture.composition)
    contract = fixture.plan.continuation_schedule(fixture.composition)
    small_source = small_engine = small_bundle = small_compiled = small_contract = None
    small_import_cpu = 0.0
    if mode != "serial":
        model, revision = MODELS["smol"]
        start = time.process_time()
        small_source = resolve_model(Model.hf(model, revision=revision, local_files_only=True))
        small_engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
        asyncio.run(small_engine.load(small_source.manifest))
        small_bundle = ClientBundle.unpack(small_engine.client_bundle(small_source.manifest.id))
        small_plan = lower_model((small_source.path / "config.json").read_bytes(),
                                batch=1, max_input_tokens=256, max_new_tokens=32)
        small_pipeline = MaskedLinearCpu(Model.hf(model, revision=revision, local_files_only=True), quantization=quantization)
        small_compiled = compile_runtime_model(small_plan, small_bundle, composition=small_pipeline)
        small_contract = small_plan.continuation_schedule(small_pipeline)
        small_import_cpu = time.process_time() - start

    def small_remote(stage_id, activation):
        stage = small_engine.models[small_source.manifest.id].stages[stage_id]
        return stage_output(stage, activation, 8)

    records = []
    for prompt in PROMPTS:
        ledger = {"calls": 0, "rows": 0, "ring_body_bytes": 0}
        def remote(stage_id, activation):
            stage = fixture.model.stages[stage_id]
            rows = int(np.prod(activation.shape[:-1]))
            ledger["calls"] += 1
            ledger["rows"] += rows
            ledger["ring_body_bytes"] += rows * (stage.spec.in_features + stage.spec.out_features) * stage.seeded_profile.wire_bits // 8
            return fixture.remote(stage_id, activation)

        runtime = fixture.compiled.runtime(remote)
        ids = runtime.encode_prompt(fixture.bundle.render_prompt([{"role": "user", "content": prompt}]))
        hashes, values, selected = hashlib.sha256(), 0, []
        def observe(scores):
            nonlocal values
            values += scores.size
            hashes.update(scores.astype("<f4", copy=False).tobytes())

        drafter = None
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if mode == "serial":
            _, scores, _ = runtime.prepare_ids(ids)
            for index in range(16):
                observe(scores)
                selected.append(int(np.argmax(scores)))
                if index < 15:
                    scores = runtime.forward_ids([selected[-1]])[-1]
            result = {"tokens": selected, "draft_offered": 0, "draft_accepted": 0,
                      "target_decode_rows": 15, "target_batches": 15}
        else:
            drafter = CheckpointDrafter(small_compiled.runtime(small_remote), small_contract,
                runtime.tokenizer, ids, small_bundle.render_prompt([{"role": "user", "content": prompt}]))
            result = verify_batched_greedy(runtime, contract, ids, drafter,
                                          outputs=16, width=int(mode), on_output=observe)
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        state = hashlib.sha256()
        state_values = 0
        for cache in runtime.caches:
            for array in (cache.key[:cache.length], cache.value[:cache.length]):
                state.update(array.astype("<f4", copy=False).tobytes())
                state_values += array.size
        records.append({"prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": len(ids),
            "output_digest": hashlib.sha256(json.dumps(result.pop("tokens")).encode()).hexdigest(),
            "logits_digest": hashes.hexdigest(), "kv_digest": state.hexdigest(),
            "checked_values": values + state_values, "wall_seconds": wall, "cpu_seconds": cpu,
            "drafter": dict(drafter.counts) if drafter else None, "target": result, **ledger})
        if drafter is not None:
            drafter.close()
        runtime.reset()
    return {"mode": mode, "target_source": fixture.lock(),
            "draft_source_lock": small_source.source_lock_digest if small_source else None,
            "draft_checkpoint": small_source.checkpoint_digest if small_source else None,
            "draft_import_cpu_seconds": small_import_cpu,
            "lifetime_cpu_seconds": time.process_time(),
            "lifetime_peak_rss_bytes": process_memory()["lifetime_peak_rss_bytes"],
            "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin)["mode"])))
        return
    if args.output is None or args.output.exists() or not 1 <= args.processes <= 5:
        parser.error("new output and 1..5 processes required")
    from frontier_probe_support import fresh_workers
    samples = []
    with fresh_workers(__file__, memory_bytes=4 << 30, timeout=1200) as (run, guard, host):
        for replicate in range(args.processes):
            modes = ("serial", "2", "4") if replicate % 2 == 0 else ("4", "2", "serial")
            for mode in modes:
                samples.append(run({"mode": mode}))
    control = next(row for row in samples if row["mode"] == "serial")
    for sample in samples:
        for expected, actual in zip(control["records"], sample["records"], strict=True):
            for key in ("prompt_digest", "input_tokens", "output_digest", "logits_digest", "kv_digest", "checked_values"):
                if actual[key] != expected[key]:
                    raise ValueError(f"target parity failed: {sample['mode']} {key}")
    summary = {}
    for mode in ("serial", "2", "4"):
        rows = [row for row in samples if row["mode"] == mode]
        summary[mode] = {
            "median_cpu_seconds": statistics.median(sum(record["cpu_seconds"] for record in row["records"]) for row in rows),
            "median_wall_seconds": statistics.median(sum(record["wall_seconds"] for record in row["records"]) for row in rows),
            "median_peak_rss_bytes": statistics.median(row["lifetime_peak_rss_bytes"] for row in rows),
            "target_calls": sum(record["calls"] for record in rows[0]["records"]),
            "target_rows": sum(record["rows"] for record in rows[0]["records"]),
            "target_ring_body_bytes": sum(record["ring_body_bytes"] for record in rows[0]["records"]),
            "proposed": sum(record["target"]["draft_offered"] for record in rows[0]["records"]),
            "accepted": sum(record["target"]["draft_accepted"] for record in rows[0]["records"]),
        }
    document = {"schema": "pllm.research.checkpoint_drafter.v1", "build": run.build,
        "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processes_per_mode": args.processes, "samples": samples, "summary": summary,
        "scope": "Compiled local-clear exact greedy target; both checkpoint owners charged; no SDK transport or WAN timing",
        "prompts": list(PROMPTS), "checked_values": sum(row["checked_values"] for sample in samples for row in sample["records"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
