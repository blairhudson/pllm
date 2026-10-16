"""Admitted offline publication and independent native pins for row-codec cohorts.

This source-owning publisher is outside measured serving clients. Head-only
publication and a separate complete source import bind each frozen recipe. No
private inputs are emitted. A generated fixture is a local test, not assurance.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.research import private_boundary_row_codec as study
from benchmarks.research.sota_combinations import atomic_json


def offline_model(model):
    from pllm import Model

    if model.kind not in {"huggingface", "safetensors", "vllm"}:
        raise ValueError("publisher requires a prepositioned Hugging Face-compatible source")
    return Model.from_spec({**model.to_spec(), "local_files_only": True})


def source_identity(source):
    if source.path is None or not source.checkpoint_digest or not source.source_lock_digest:
        raise ValueError("publisher requires a complete local source lock")
    return source.source_lock_digest, source.checkpoint_digest


def source_price(source):
    from pllm.runtime.safetensors_store import SafeTensorStore

    store = SafeTensorStore(source.path)
    source_bytes = sum(path.stat().st_size for path in {value.path for value in store._locations.values()})
    tokenizer_bytes = sum(path.stat().st_size for path in source.path.glob("*tokenizer*") if path.is_file())
    if not source_bytes:
        raise ValueError("publisher requires actual checkpoint shards")
    return 6 * source_bytes + 32 * tokenizer_bytes + (512 << 20)


def worker(payload):
    import msgpack
    from pllm import Experiment, Pipeline, lower_model
    from pllm.model_loader import resolve_model
    from pllm.roles import PrivateLookupAndFullHead
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.private_boundary_certificate import import_certificate
    from pllm.runtime.private_boundary_loader import LoadedPrivateBoundary, build_private_boundary_contract
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    started, cpu = time.perf_counter(), time.process_time()
    controls = [Experiment.from_spec(spec) for spec in payload["controls"]]
    first = controls[0]
    source_spec = offline_model(first.pipeline.model)
    source = resolve_model(source_spec)
    identity = source_identity(source)
    approved = payload["approved_head"]
    if not payload["tiny"] and (source.source_lock_digest != approved["certificate"]["source_lock_sha256"]
            or source.checkpoint_digest != approved["certificate"]["checkpoint_sha256"]):
        raise ValueError("source differs from independent approved checkpoint before import")
    # Full source plus float import, quantized/native owners, tokenizer expansion
    # and runtime allowance. This offline owner is additional to serving costs.
    price = source_price(source)
    if price > payload["memory_bytes"]:
        raise ValueError("publisher full-source allocation exceeds reserved capacity")
    if any(exp.pipeline.model != first.pipeline.model for exp in controls):
        raise ValueError("publisher controls must share one exact source")
    folder = Path(payload["output"])
    publication_start, publication_cpu = time.perf_counter(), time.process_time()
    publisher = first.with_params(pipeline__model=source_spec)
    pin = PrivateLookupAndFullHead.publish_head_certificate(publisher, folder / "head.cert")
    if source_identity(resolve_model(source_spec)) != identity:
        raise ValueError("checkpoint changed during certificate publication")
    publication = {"elapsed_seconds": time.perf_counter() - publication_start,
        "process_cpu_seconds": time.process_time() - publication_cpu, "artifact": pin}
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
        private_boundary=True, prepared_output_encoding="row_residues")
    try:
        asyncio.run(engine.load(source.manifest))
        model = engine.models[source.manifest.id]
        raw = engine.client_bundle(source.manifest.id)
        config = json.loads((source.path / "config.json").read_bytes())
        records = []
        for control in controls:
            components = dict(control.pipeline.components)
            options = dict(components["boundary"].params)
            options.update(head_output_encoding="row_residues", head_certificate=pin)
            components["boundary"] = PrivateLookupAndFullHead(**options)
            candidate = control.with_params(pipeline=Pipeline.from_profile(control.pipeline.profile,
                model=control.pipeline.model, components=components))
            tiny = payload["tiny"]
            cap = 8 if tiny else control.budget.max_new_tokens
            study.validate_pair_specs(control.to_spec(), candidate.to_spec(), max_output_tokens=cap, tiny=tiny)
            plan = lower_model(config, batch=1, max_input_tokens=control.budget.max_input_tokens,
                max_new_tokens=control.budget.max_new_tokens)
            commitments = {}
            for name, exp in zip(study.NAMES, (control, candidate)):
                schedule = plan.runtime_schedule(exp.pipeline)
                loaded = LoadedPrivateBoundary(source, plan, schedule, model.stages["lm_head"],
                    model.tokenizer, config)
                descriptor, _ = build_private_boundary_contract(loaded, exp)
                bundle = ClientBundle._from_document(msgpack.unpackb(raw, raw=False, strict_map_key=False),
                    storage=exp.pipeline.components["delivery"].params.get("storage", "memory"), threads=1)
                compiled = compile_runtime_model(plan, bundle, composition=exp.pipeline)
                if compiled.private_boundary_descriptor() != descriptor:
                    raise ValueError("independent source and compiled head descriptors differ")
                commitments[name] = {"pipeline_digest": exp.pipeline.digest(),
                    "composition_digest": descriptor.composition_digest,
                    "schedule_digest": descriptor.schedule_digest, "descriptor_digest": descriptor.digest,
                    "numeric_body_fingerprint": compiled.to_spec()["body_fingerprint"]}
            widths, metadata = import_certificate(pin, descriptor, weights=model.stages["lm_head"].compiled_weight.weight_view(),
                with_metadata=True)
            if (metadata["source"]["source_digest"], metadata["source"]["checkpoint_digest"]) != identity:
                raise ValueError("publisher source and admitted source differ")
            if not tiny:
                expected = approved["certificate"]
                fields = {"source_digest": "source_lock_sha256", "checkpoint_digest": "checkpoint_sha256",
                    "model_config_digest": "model_config_sha256", "head_weight_digest": "weight_sha256",
                    "head_scales_digest": "scales_sha256", "vocabulary": "vocabulary", "hidden": "hidden"}
                if any(metadata["source"][key] != expected[value] for key, value in fields.items()):
                    raise ValueError("publisher differs from independently pinned checkpoint scan")
                if widths.hex() != approved["widths_hex"]:
                    raise ValueError("publisher row widths differ from independent complete scan")
            stem = f'{control.budget.max_input_tokens}x{control.budget.max_new_tokens}'
            for suffix, value in (("control", control.to_spec()), ("candidate", candidate.to_spec()),
                                  ("native-head-commitments", commitments)):
                atomic_json(folder / f"{stem}-{suffix}.json", value)
            records.append({"workload": stem, "native_commitments": commitments,
                "certificate_metadata": metadata, "width_histogram": {str(w): widths.count(w) for w in sorted(set(widths))},
                "payload_bytes_per_role": (sum(widths) + 7) // 8})
        if source_identity(resolve_model(source_spec)) != identity:
            raise ValueError("checkpoint changed during independent source binding")
        atomic_json(folder / "certificate-accounting.json", {"missing_owners": ["client", "boundary_a", "boundary_b"],
            "delivery_locations": {owner: pin["path"] for owner in ("client", "boundary_a", "boundary_b")},
            "derivation": "Local offline publisher; actual native head/source/scale binding and full row scan. "
                + ("Generated fixture only." if tiny else "All widths/source pins match independent archived checkpoint scan.")
                + " See publication.json; three-owner delivery is scenario arithmetic, not measured network."})
        return {"schema": "pllm.private_boundary_row_codec_publication.v1", "complete": True,
            "tiny": tiny, "source_lock_digest": source.source_lock_digest,
            "publication": publication, "independent_source_binding": records,
            "source_allocation_estimate_bytes": price, "worker_measured_window_seconds": time.perf_counter() - started,
            "worker_measured_window_cpu_seconds": time.process_time() - cpu,
            "worker_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "scope": "offline publisher and independent source binding; timings exclude child import/startup; no serving or network claim"}
    finally:
        asyncio.run(engine.unload(source.manifest.id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--control", type=Path, action="append")
    parser.add_argument("--approved-head-evidence", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    if not args.output or not args.control or len(args.control) > 2 or (not args.tiny and not args.approved_head_evidence):
        parser.error("require fresh output directory, one/two controls, and independent head evidence unless --tiny")
    args.output.mkdir()
    source_files = [Path(__file__), Path(__file__).with_name("frontier_probe_support.py"), *args.control]
    if args.approved_head_evidence:
        source_files.append(args.approved_head_evidence)
    sources = study.hashes(source_files)
    payload = {"controls": [study.bounded_json(p, 1 << 20) for p in args.control], "tiny": args.tiny,
        "output": str(args.output.resolve()),
        "approved_head": study.bounded_json(args.approved_head_evidence, 16 << 20) if args.approved_head_evidence else None}
    progress = {"complete": False, "sources": sources}
    atomic_json(args.output / "publication.json", progress)
    from frontier_probe_support import fresh_workers
    try:
        from pllm import Experiment
        from pllm.model_loader import resolve_model
        source = resolve_model(offline_model(Experiment.from_spec(payload["controls"][0]).pipeline.model))
        source_identity(source)
        payload["memory_bytes"] = source_price(source)
        if payload["memory_bytes"] > 8 << 30:
            raise ValueError("publisher metadata price exceeds 8 GiB ceiling")
        with fresh_workers(Path(__file__).resolve(), memory_bytes=payload["memory_bytes"], timeout=600) as (run, guard, host):
            report = run(payload)
            if study.hashes(source_files) != sources:
                raise ValueError("publisher inputs changed")
            report.update(sources=sources, build=run.build, host=asdict(host), guard_error=guard.error,
                maximum_observed_swap_growth_bytes=guard.maximum_swap_growth_bytes)
        atomic_json(args.output / "publication.json", report)
        print(json.dumps({"complete": True, "output": str(args.output)}, indent=2))
    except BaseException as exc:
        progress.update(failure_type=type(exc).__name__)
        atomic_json(args.output / "publication.json", progress)
        raise


if __name__ == "__main__":
    main()
