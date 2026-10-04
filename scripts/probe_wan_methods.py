"""Five bounded WAN method screens; real checkpoints require --real."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
from pllm import Model, lower_model, _native
from pllm.metrics import (ArtifactEntropyProbe, ArtifactPlaneProbe, BatchedPrivateLookupProbe, PreparedDuplexProbe, PreparedResidueProbe,
                          ProjectedResharingProbe, OrthogonalActivationProbe)
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.semantic_stages import scheduled_stage_specs
from probe_next_five import load, SOURCE

TEXTS = (
    "A cyclist travels twelve kilometres in one hour. What is the average speed?",
    "Explain one difference between a lake and a river.",
    "Write a short sentence using the words quiet and garden.",
    "Why should a scientific experiment include a control group?",
    "Name a tool that could measure the temperature of water.",
    "A box contains three red balls and two blue balls. How many balls are there?",
    "Describe a safe way to store a digital backup.",
    "What happens to a shadow when a light source changes position?",
)


def run(source, method, real):
    engine, bundle, remote = load(source)
    try:
        cfg = json.loads((Path(source.manifest.source) / "config.json").read_bytes())
        pipeline = MaskedLinearCpu(Model(source.manifest.id), quantization=SymmetricPerRow(
            weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
        plan = lower_model(cfg, batch=1, max_input_tokens=128, max_new_tokens=32)
        compiled = compile_runtime_model(plan, bundle, composition=pipeline)
        stages = engine.models[bundle.model_id].stages
        specs = scheduled_stage_specs(plan, pipeline)
        down = next(s for s in specs if s.role == "mlp_down")
        expanded = max((s for s in compiled.stage_bindings if s.layer_index is not None),
                       key=lambda s: s.out_features / s.in_features)
        if method == "resharing":
            probe = ProjectedResharingProbe()
            data = probe.run(stages[down.id].weight.values)
            cost_plan = lower_model(cfg, batch=1, max_input_tokens=39, max_new_tokens=32)
            data["response_projection"] = probe.project(cost_plan, pipeline, response_new_tokens=32)
        elif method == "residues":
            data = {"kernels": [PreparedResidueProbe(rows=rows, repetitions=5).run(
                stages[expanded.stage_id].weight.values, dense_wire_bits=stages[expanded.stage_id].seeded_profile.wire_bits) for rows in (1, 39)]}
            raw, packed, metadata = 0, 0, 0
            for s in compiled.stage_bindings:
                if s.layer_index is None:
                    continue
                widths = _native.offset_row_bits(stages[s.stage_id].weight.values.tobytes(), s.in_features, 127)
                word = stages[s.stage_id].seeded_profile.wire_bits // 8
                raw += (s.in_features + 2 * s.out_features) * word
                packed += s.in_features * word + 2 * ((sum(widths) + 7) // 8)
                metadata += len(widths)
            data["body_projection"] = {"executed_rows": 70, "dense_integer_body_bytes": raw * 70,
                "packed_integer_body_bytes": packed * 70, "width_manifest_bytes": metadata,
                "scope": "39+32 arithmetic bodies only; native public bounds, no controls or transport"}
        elif method == "duplex":
            selected = {}
            remote_ids = {s.stage_id for s in compiled.stage_bindings if s.layer_index is not None}
            for stage in specs:
                if stage.id in remote_ids and stage.role not in selected:
                    selected[stage.role] = stage
            data = {"kernels": [{"role": role, **PreparedDuplexProbe(rows=39, repetitions=3).run(
                stages[stage.id].weight.values, dense_wire_bits=stages[stage.id].seeded_profile.wire_bits)}
                for role, stage in selected.items()]}
        elif method in {"artifacts", "entropy"}:
            probe = ArtifactEntropyProbe if method == "entropy" else ArtifactPlaneProbe
            data = probe(repetitions=2).run(engine.client_bundle(bundle.model_id))
        elif method == "retrieval":
            # Public, fixed-size source records; no candidate decisions leaked.
            table = stages["lm_head"].weight.values
            data = BatchedPrivateLookupProbe(records=table.shape[0], record_bytes=table.shape[1],
                batch_size=8).run(table.tobytes())
        else:
            ids = [bundle.tokenizer().encode(text, add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True)))
                   for text in (TEXTS if real else TEXTS[:2])]
            weights = {s.stage_id: stages[s.stage_id].weight.values for s in compiled.stage_bindings if s.layer_index is not None}
            data = OrthogonalActivationProbe(activation_bits=6, block=128 if real else 8, decode_steps=3).run(
                compiled, remote, weights, ids)
        return {"model_plan_digest": plan.digest, "body_fingerprint": bundle.privacy["body_fingerprint"], "result": data}
    finally:
        asyncio.run(engine.unload(bundle.model_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--method", choices=("resharing", "residues", "artifacts", "entropy", "duplex", "retrieval", "orthogonal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pllm-wan-methods-") as temp:
        if args.real:
            source = resolve_model(SOURCE)
        else:
            from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
            source = resolve_model(Model.path(str(create_tiny_llama_checkpoint(Path(temp) / "model")), model_id="wan-method-tiny"))
        print(f"Running bounded {args.method} screen", flush=True)
        data = run(source, args.method, args.real)
        report = {"schema": "pllm.wan_method_screen.v1", "method": args.method, "real_checkpoint": args.real,
            "source_lock_digest": source.source_lock_digest, **data,
            "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                            "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()}}
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(data["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
