"""Offline-only public checkpoint teacher pilot. Prepare manifest before fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preflight(data):
    from huggingface_hub import try_to_load_from_cache
    from benchmarks.research.checkpoint_tiny_model import PILOT, file_hash
    from pllm.runtime.benchmark_memory import host_memory
    host = host_memory()  # Invalid/unknown telemetry raises, never assume headroom.
    memory = 4 << 30
    if host.available - host.reserve < memory or host.disk_free < 1 << 30:
        raise RuntimeError("pilot requires 4GiB post-reserve RAM and 1GiB free disk")
    files = {}
    checkpoint_header = None
    for name in (*PILOT["files"], "tokenizer.json", "tokenizer_config.json"):
        path = try_to_load_from_cache(PILOT["model"], name, revision=PILOT["revision"])
        if not isinstance(path, str) or not Path(path).is_file():
            raise RuntimeError(f"pre-position pinned public artifact: {name}; downloads forbidden")
        actual = file_hash(path)
        if name in PILOT["files"] and actual != PILOT["files"][name]:
            raise ValueError(f"checkpoint source mismatch: {name}")
        files[name] = {"sha256": actual, "bytes": Path(path).stat().st_size}
        if name.endswith(".safetensors"):
            # Public shape metadata only, never checkpoint tensor values.
            with Path(path).open("rb") as stream:
                length = int.from_bytes(stream.read(8), "little")
                if not 0 < length <= 8 << 20:
                    raise ValueError("unbounded checkpoint metadata")
                checkpoint_header = json.loads(stream.read(length))
    import math
    from benchmarks.research.checkpoint_tiny_model import FEATURES, runtime_parameters
    target_parameters = sum(math.prod(spec["shape"]) for name, spec in checkpoint_header.items()
                            if name != "__metadata__")
    helper_parameters = runtime_parameters({"ranker_including_bias": FEATURES,
                                            "backbone": 0, "embedding": 0, "head": 0})
    return {"pilot": PILOT, "files": files, "host": asdict(host), "reserve_bytes": host.reserve,
            "memory_budget_bytes": memory, "data_sha256": data["sha256"],
            "parameter_inventory": {"helper_total": helper_parameters,
                                    "target_all_checkpoint_tensors_including_embedding_head_norm_bias": target_parameters,
                                    "pilot_local_target_plus_helper": target_parameters + helper_parameters},
            "helper_resource_bound": {"runtime_weight_bytes": FEATURES * 8,
                                      "training_and_feature_workspace_allowance_bytes": 4 << 20},
            "target_placement": "local-clear pilot only; provider-owned target in proposed serving contract"}


class Teacher:
    def __init__(self, expected):
        import asyncio
        from pllm import Model
        from pllm.model_loader import resolve_model
        from pllm.runtime.transformer_client import ClientBundle
        from pllm.runtime.transformer_engine import MaskedTransformerEngine
        from benchmarks.research.checkpoint_tiny_model import file_hash
        source_spec = expected["pilot"]
        self.choice = Model.hf(source_spec["model"], revision=source_spec["revision"], local_files_only=True)
        self.source = resolve_model(self.choice)
        for name, lock in expected["files"].items():
            if file_hash(self.source.path / name) != lock["sha256"]:
                raise ValueError("source changed after frozen manifest")
        self.engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
        asyncio.run(self.engine.load(self.source.manifest))
        self.bundle = ClientBundle.unpack(self.engine.client_bundle(self.source.manifest.id))
        self.body = self.engine.models[self.source.manifest.id].stages
        self.stages = [{"inputs": s.spec.in_features, "outputs": s.spec.out_features,
                        "bits": s.seeded_profile.wire_bits} for s in self.body.values()
                       if s.spec.role not in {"token_lookup", "lm_head"}]
        self.lock = {"checkpoint_digest": self.source.checkpoint_digest,
                     "source_lock_digest": self.source.source_lock_digest, "files": expected["files"],
                     "numeric": "W8A8-prefix_f32-local-clear", "pilot": source_spec}

    def execute(self, text, *, answer=None):
        import numpy as np
        from pllm import lower_model
        from pllm.profiles import MaskedLinearCpu
        from pllm.quantization import SymmetricPerRow
        from pllm.runtime.model_binding import compile_runtime_model
        from scripts.decoder_probe_support import stage_output
        from benchmarks.research.checkpoint_tiny_model import POLICY, digest, projected_cost
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        rendered = self.bundle.render_prompt([{"role": "user", "content": text}])
        tokenizer = self.bundle.tokenizer()
        ids = tokenizer.encode(rendered, add_bos=bool(self.bundle.tokenizer_descriptor.get("add_bos_token", True)))
        if not 1 <= len(ids) <= POLICY["max_input_tokens"]:
            raise ValueError("prompt exceeds predeclared token bound; never silently truncate")
        plan = lower_model((self.source.path / "config.json").read_bytes(), batch=1,
                           max_input_tokens=len(ids), max_new_tokens=POLICY["max_output_tokens"])
        composition = MaskedLinearCpu(self.choice, quantization=SymmetricPerRow(
            weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
        compiled = compile_runtime_model(plan, self.bundle, composition=composition)
        stage_rows, calls = 0, 0

        def remote(key, activation):
            nonlocal stage_rows, calls
            stage_rows += int(np.prod(activation.shape[:-1]))
            calls += 1
            return stage_output(self.body[key], activation, 8)

        runtime = compiled.runtime(remote)
        if runtime.encode_prompt(rendered) != ids:
            raise ValueError("tokenizer identity mismatch")
        _, logits, _ = runtime.prepare_ids(ids)
        eos, generated, selected, losses = False, [], [], []
        forced = tokenizer.encode(answer, add_bos=False) if answer is not None else None
        # Reject gold-token BPE boundary mismatch; do not silently optimize a
        # different string than prompt+answer continuation.
        if answer is not None:
            combined = runtime.encode_prompt(rendered + answer)
            if combined != ids + forced:
                raise ValueError("answer BPE boundary differs from causal prompt continuation")
        if forced is not None and not 1 <= len(forced) <= POLICY["max_answer_tokens"]:
            raise ValueError("gold answer exceeds bounded teacher-forcing length")
        steps = len(forced) if forced is not None else POLICY["max_output_tokens"]
        for index in range(steps):
            token = forced[index] if forced is not None else int(np.argmax(logits))
            selected.append(token)
            if forced is not None:
                shifted = logits.astype(np.float64) - float(np.max(logits))
                losses.append(float(np.log(np.exp(shifted).sum()) - shifted[token]))
            elif token == int(runtime.cfg["eos_token_id"]):
                eos = True
                break
            generated.append(token)
            if index + 1 < steps:
                logits = runtime.forward_ids([token])[-1]
        rows = len(ids) + len(selected) - 1
        result = {"text": tokenizer.decode(generated), "eos": eos, "input_tokens": len(ids),
                  "selected_tokens": len(selected), "executed_sequence_rows": rows,
                  "stage_rows": stage_rows, "stage_calls": calls,
                  "projected_arithmetic_bytes": projected_cost(self.stages, rows, len(ids) + steps - 1),
                  "input_digest": digest(ids), "plan_digest": plan.digest,
                  "cpu_seconds": time.process_time() - start_cpu,
                  "wall_seconds": time.perf_counter() - start_wall,
                  "answer_nll": float(np.mean(losses)) if losses else None}
        runtime.reset()
        return result


def worker(manifest, artifact):
    from benchmarks.research.checkpoint_tiny_model import (
        POLICY, Ranker, candidates, digest, exact_match, fit, needs_fallback, prepare_data, prompt)
    from pllm.runtime.benchmark_memory import process_memory
    # A digest-bound manifest is not a signature. Re-derive rows from the locked
    # public source so edited manifests cannot introduce private training text.
    if digest(prepare_data(manifest["data_path"])) != digest(manifest["data"]):
        raise ValueError("manifest rows differ from locked public corpus")
    started = time.process_time(), time.perf_counter()
    teacher = Teacher(manifest["preflight"])
    binding = digest(teacher.lock)
    rows = manifest["data"]["rows"]
    train = [row for row in rows if row["split"] == "train"]
    teacher_records, losses = [], []
    for row in train:
        records = [teacher.execute(prompt(row, c["text"]), answer=row["answers"][0]) for c in candidates(row)]
        losses.append([record["answer_nll"] for record in records])
        teacher_records.append({"id": row["id"], "candidates": records})
    model = fit(train, losses, binding, epochs=POLICY["epochs"])
    model.save(artifact)
    model = Ranker.load(artifact, binding)
    evaluation = []
    for row in rows:
        if row["split"] == "train":
            continue
        start = time.process_time(), time.perf_counter()
        retained = model.choose(row)["text"]
        helper_work = {"cpu_seconds": time.process_time() - start[0], "wall_seconds": time.perf_counter() - start[1]}
        choices = candidates(row)
        contexts = {"full": row["context"], "head": choices[0]["text"], "tail": choices[1]["text"],
                    "lexical": choices[2]["text"], "learned": retained}
        results = {}
        for mode, context in contexts.items():
            record = teacher.execute(prompt(row, context))
            record.update(em=exact_match(record["text"], row["answers"]),
                          literal_exact=record["text"].strip() in row["answers"],
                          answer_retention=any(answer in context for answer in row["answers"]))
            results[mode] = record
        fallback = needs_fallback(results["learned"]["text"], results["learned"]["eos"], retained)
        # Fresh rerun: failed compressed work AND full retry both charged. No oracle router.
        retry = teacher.execute(prompt(row, row["context"])) if fallback else None
        final = retry or results["learned"]
        results["learned_then_full"] = {
            "fallback": fallback, "retry": retry, "em": exact_match(final["text"], row["answers"]),
            "eos": final["eos"], "literal_exact": final["text"].strip() in row["answers"],
            "cpu_seconds": helper_work["cpu_seconds"] + results["learned"]["cpu_seconds"] + (retry["cpu_seconds"] if retry else 0),
            "wall_seconds": helper_work["wall_seconds"] + results["learned"]["wall_seconds"] + (retry["wall_seconds"] if retry else 0),
            "projected_arithmetic_bytes": {k: v + (retry["projected_arithmetic_bytes"][k] if retry else 0)
                                           for k, v in results["learned"]["projected_arithmetic_bytes"].items()}}
        evaluation.append({"id": row["id"], "split": row["split"], "kind": row["kind"],
                           "helper_work": helper_work, "results": results})
    summaries = {}
    for split in ("dev", "heldout"):
        cohort = [r for r in evaluation if r["split"] == split]
        summaries[split] = {}
        for mode in ("full", "head", "tail", "lexical", "learned", "learned_then_full"):
            qa = [r["results"][mode] for r in cohort if r["kind"] == "public_qa"]
            stress = [r["results"][mode] for r in cohort if r["kind"] != "public_qa"]
            summaries[split][mode] = {"qa_em": sum(r["em"] for r in qa) / len(qa),
                                     "qa_count": len(qa), "qa_correct": sum(r["em"] for r in qa),
                                     "stress_count": len(stress),
                                     "stress_literal_correct": sum(r["literal_exact"] for r in stress),
                                     "stress_literal_exact": sum(r["literal_exact"] for r in stress) / len(stress),
                                     "all_eos": all(r["results"][mode]["eos"] for r in cohort),
                                     "projected_arithmetic_bytes": sum(sum(r["results"][mode]["projected_arithmetic_bytes"].values()) for r in cohort)}
        full, candidate = summaries[split]["full"], summaries[split]["learned_then_full"]
        ratio = candidate["projected_arithmetic_bytes"] / full["projected_arithmetic_bytes"]
        summaries[split]["gate"] = {"arithmetic_ratio": ratio, "screen_pass": (
            candidate["qa_em"] >= max(POLICY["gates"]["minimum_qa_em"], full["qa_em"])
            and candidate["stress_literal_exact"] >= POLICY["gates"]["minimum_stress_em"]
            and candidate["all_eos"] and ratio <= POLICY["gates"]["max_projected_arithmetic_ratio"])}
        summaries[split]["gate"]["learned_adds_value_over_lexical"] = (
            summaries[split]["learned"]["qa_em"] > summaries[split]["lexical"]["qa_em"]
            or (summaries[split]["learned"]["qa_em"] == summaries[split]["lexical"]["qa_em"]
                and summaries[split]["learned"]["projected_arithmetic_bytes"] < summaries[split]["lexical"]["projected_arithmetic_bytes"]))
    return {"status": "public_checkpoint_pilot_not_serving_qualification", "source": teacher.lock,
            "parameters": model.parameter_inventory, "artifact_bytes": Path(artifact).stat().st_size,
            "cold_additional_client_artifact_bytes": Path(artifact).stat().st_size,
            "teacher_training": teacher_records, "evaluation": evaluation, "summary": summaries,
            "lifetime_cpu_seconds": time.process_time(), "worker_cpu_seconds": time.process_time() - started[0],
            "worker_wall_seconds": time.perf_counter() - started[1], "memory": process_memory(),
            "cost_scope": "local-clear + three directed arithmetic projections; NOT full transport",
            "unpriced": ["cold target client bundle/tokenizer delivery", "publisher/provider weights if cold",
                         "seed requests/tickets/control/TLS", "artifact import/serialization peak buffers",
                         "private transport and preparation CPU/RSS", "padding/discarded correlation renewal"],
            "identity": "changed-input: no original logits/KV/sampling equivalence; no protected execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="new manifest path; metadata/hash/preflight only")
    parser.add_argument("--data", type=Path, help="pre-positioned public SQuAD dev-v1.1.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--mechanics", action="store_true", help="synthetic bounded fit only; no checkpoint/data load")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # CLI-only offline policy, set before any source resolution.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from benchmarks.research.checkpoint_tiny_model import SCHEMA, digest, file_hash, prepare_data, support_table
    if args.mechanics:
        from benchmarks.research.checkpoint_tiny_model import mechanics_report
        if not args.output or args.output.exists():
            parser.error("--mechanics requires fresh --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(mechanics_report(ROOT), indent=2) + "\n")
        print("Synthetic mechanics only; real checkpoint remains untrained.")
        return
    if args.worker:
        payload = json.load(sys.stdin)
        print(json.dumps(worker(payload["manifest"], payload["artifact"])))
        return
    if args.prepare:
        if not args.data or args.prepare.exists():
            parser.error("--prepare requires --data and new manifest path")
        data = prepare_data(args.data)
        document = {"schema": SCHEMA, "data": data, "data_path": str(args.data.resolve()), "preflight": preflight(data),
                    "support": support_table(ROOT),
                    "code": {str(p.relative_to(ROOT)): file_hash(p) for p in (
                        Path(__file__).resolve(), ROOT / "benchmarks/research/checkpoint_tiny_model.py")}}
        document["manifest_digest"] = digest(document)
        args.prepare.parent.mkdir(parents=True, exist_ok=True)
        args.prepare.write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps({"prepared_only": True, "split_digest": data["split_digest"], "rows": len(data["rows"])}))
        return
    if not args.manifest or not args.output or not args.artifact or args.output.exists() or args.artifact.exists():
        parser.error("--manifest and fresh --output/--artifact required")
    manifest = json.loads(args.manifest.read_text())
    identity = manifest.pop("manifest_digest")
    if digest(manifest) != identity:
        raise ValueError("manifest tampered")
    if digest(prepare_data(manifest["data_path"])) != digest(manifest["data"]):
        raise ValueError("manifest rows differ from locked public corpus")
    for name, expected in manifest["code"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("training code changed after split freeze")
    if manifest["preflight"]["files"] != preflight(manifest["data"])["files"]:
        raise ValueError("source changed after split freeze")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    from frontier_probe_support import fresh_workers
    with fresh_workers(__file__, memory_bytes=4 << 30, timeout=1800) as (run, guard, host):
        result = run({"manifest": manifest, "artifact": str(args.artifact.resolve())})
    result.update(schema=SCHEMA, manifest_digest=identity, host=asdict(host), build=run.build,
                  maximum_swap_growth_bytes=guard.maximum_swap_growth_bytes,
                  artifact_sha256=file_hash(args.artifact))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
