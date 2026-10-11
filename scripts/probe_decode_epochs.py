"""Bounded decode epochs: exact decoder/one-use stage reference, not an SDK mode.

Only launcher owns real-checkpoint cohorts. Serialized stage bodies are measured;
SDK admissions, public delivery, wire, CPU and independent process RSS are not.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EPOCHS = (1, 4, 8, 16, 32, 64)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_digest(root):
    """Hash actual imported/source files, not git state or an asserted wheel ID."""
    root = Path(root)
    rows = [(str(p.relative_to(root)), sha(p)) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix not in {".pyc"} and "__pycache__" not in p.parts]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def freeze(checkpoint):
    import pllm
    import numpy
    import msgpack
    from pllm import _native
    from pllm.metrics import BuildRef
    distributions = sorted((d.metadata["Name"], d.version,
                            hashlib.sha256((d.read_text("RECORD") or "").encode()).hexdigest())
                           for d in importlib.metadata.distributions())
    return {"build": BuildRef.capture().to_spec(),
            "imported_pllm_tree_sha256": tree_digest(Path(pllm.__file__).parent),
            "native_sha256": sha(_native.__file__),
            "numpy_tree_sha256": tree_digest(Path(numpy.__file__).parent),
            "msgpack_tree_sha256": tree_digest(Path(msgpack.__file__).parent),
            "dependency_records_sha256": hashlib.sha256(json.dumps(distributions).encode()).hexdigest(),
            "checkpoint_tree_sha256": tree_digest(checkpoint),
            "driver_sha256": sha(__file__),
            "tests_sha256": sha(ROOT / "tests/test_decode_epochs_probe.py"),
            "support_sha256": sha(ROOT / "scripts/frontier_probe_support.py"),
            "control_reader_sha256": sha(ROOT / "scripts/probe_all_link_frontier.py"),
            "control_sha256": sha(ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json")}


def reference_pipeline(source, bound):
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.preparation import PreparedInventory
    from pllm.protocols import MaskedLinear
    from pllm.state import ClientPrefixReuse
    return MaskedLinearCpu(source,
        linear=MaskedLinear(prefill_pruning="terminal", request_encoding="stage_packed",
                            output_encoding="row_residues"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(allocation="demand", refill="on-demand"),
        cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=bound))


def memory_admission(checkpoint, *, input_tokens, cap, bound):
    """Header/config-only bound; no tensor values loaded. One resident engine.

    Reuse allocation components, not whole three-process topology peaks. Loading,
    bundle import and execution are serial phases. Retained source mapping, engine,
    bundle and launcher remain charged throughout; largest temporary phase wins.
    """
    from pllm import Model
    from pllm.runtime.benchmark_memory import estimate_memory
    if not 1 <= input_tokens <= 16 or not 1 <= cap <= 128 or not input_tokens + cap - 1 <= bound <= 256:
        raise ValueError("reference workload requires input<=16, cap<=128, input+cap-1<=bound<=256")
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    source_bytes = sum(p.stat().st_size for p in Path(checkpoint).glob("*.safetensors"))
    if not source_bytes:
        raise ValueError("existing safetensors checkpoint required for source mapping admission")
    pipeline = reference_pipeline(Model.path(str(checkpoint), model_id="decode-epochs-reference"), bound)
    estimate = estimate_memory(config, pipeline, max_input_tokens=input_tokens, max_output_tokens=cap,
        inventory_rows=input_tokens + cap - 1, cache_bytes=32 << 20,
        cache_bound_tokens=bound, checkpoint_bytes=source_bytes)
    c = estimate["components"]
    owners = {"resident_engine_i8_native_bytes": c["per_engine_i8_and_native_bytes"],
              "resident_bundle_weights_metadata_bytes": c["raw_client_bundle_bytes"],
              "worker_import_baseline_bytes": 256 << 20,
              "checkpoint_mapping_bytes": source_bytes,
              "launcher_import_hash_buffers_bytes": 256 << 20}
    phases = {"weight_loading": c["largest_loading_temporary_bytes"],
              "bundle_import": c["client_bundle_work_bytes"],
              "execution": c["client_tensor_work_bytes"] + 2 * c["client_state_work_bytes"]
                  + c["client_mask_bytes"] + c["provider_correction_bytes"]
                  + c["client_cache_bytes"] + (64 << 20)}
    unadjusted = sum(owners.values()) + max(phases.values())
    required = (5 * unadjusted + 3) // 4
    return {"scope": "conservative config/header allocation bound, not RSS; one in-process engine",
            "required_bytes": required, "owners": owners, "mutually_exclusive_temporary_phases": phases,
            "allocation_slack_percent": 25, "shared_estimator": estimate}


def historical_control():
    import importlib.util
    spec = importlib.util.spec_from_file_location("decode_epochs_historical_control",
        ROOT / "scripts/probe_all_link_frontier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = module.conserved_control(json.loads(module.CONTROL.read_bytes()))
    return {"evidence_sha256": control["evidence_sha256"], "pair_bytes": control["pair_bytes"],
            "classes": control["classes"], "scope": control["scope"]}


def load_reference(checkpoint, *, input_tokens, cap, bound):
    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.model_binding import compile_runtime_model
    source = Model.path(str(checkpoint), model_id="decode-epochs-reference")
    resolved = resolve_model(source)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1,
                                     prepared_output_encoding="row_residues")
    asyncio.run(engine.load(resolved.manifest))
    try:
        bundle = ClientBundle.unpack(engine.client_bundle(resolved.manifest.id))
        pipeline = reference_pipeline(source, bound)
        plan = lower_model(json.loads((Path(checkpoint) / "config.json").read_text()),
                           batch=1, max_input_tokens=bound, max_new_tokens=cap)
        return engine, compile_runtime_model(plan, bundle, composition=pipeline), pipeline
    except BaseException:
        asyncio.run(engine.unload(resolved.manifest.id))
        raise


class EpochMaterial:
    """In-process three-owner algebra reference; real native masks and lease ledger.

    No network authorization claim: provider correction acceptance is a separate
    SDK gate. Every correction is packed, decoded, and counted once here.
    """

    def __init__(self, engine, compiled):
        self.engine, self.compiled = engine, compiled
        self.active = False
        self.lease = self._inventory = self.remote = None
        self.corrections = {}
        self.tickets = set()
        self.audit = Counter()
        self.bytes = Counter()
        self.epochs = []

    @property
    def inventory(self):
        return self.lease

    def prepare(self, prefill_rows, decode_steps, *, continuation=False):
        if self.active:
            raise RuntimeError("Preparation must remain idle during execution")
        self._unadmitted_rows = {}
        try:
            self._prepare(prefill_rows, decode_steps, continuation=continuation)
        except BaseException:
            issued = sum(stage.request.rows for stage in self._unadmitted_rows.values())
            self.audit.update({"prepared_stage_rows_issued": issued,
                               "prepared_stage_rows_discarded": issued})
            for prepared in self._unadmitted_rows.values():
                prepared.cancel()
            self.finish()
            raise
        finally:
            self._unadmitted_rows = {}

    def _prepare(self, prefill_rows, decode_steps, *, continuation=False):
        from pllm.runtime.preparation_protocol import PreparationRequest, CorrectionPush
        from pllm.runtime.transformer_client import PreparedStageRows, PreparedInventory, PreparedRemoteLinear
        demand = self.compiled.prepared_stage_rows(prefill_rows, decode_steps, continuation=continuation)
        bundle = self.compiled._bundle
        session = secrets.token_hex(16)
        rows = self._unadmitted_rows
        self.corrections = {}
        for sid, count in demand.items():
            stage = bundle.stages[sid]
            profile = stage.seeded_profile
            request = PreparationRequest(secrets.token_hex(16), session, bundle.model_id,
                bundle.privacy["body_fingerprint"], sid, stage.weight_digest, count,
                stage.in_features, stage.out_features, stage.weight_bits, stage.activation_bits,
                profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
                secrets.token_bytes(32))
            payload = request.pack()
            self.bytes["client_to_preparation_requests"] += len(payload)
            request = PreparationRequest.unpack(payload, context=request.context)
            push = asyncio.run(self.engine.prepare_seeded_stage(request))
            payload = push.pack()
            self.bytes["preparation_to_inference_corrections"] += len(payload)
            push = CorrectionPush.unpack(payload)
            self.corrections[sid] = [push.correction, 0]
            rows[sid] = PreparedStageRows(request)
        capacity = max(1, prefill_rows + decode_steps)
        self._inventory = PreparedInventory(session, capacity, rows, demand=True,
            _audit=lambda key, count: self.audit.update({key: count}))
        self._unadmitted_rows = {}
        self.lease = self._inventory.reserve(capacity)
        self.remote = PreparedRemoteLinear(bundle.model_id, bundle.privacy["body_fingerprint"],
            bundle.stages, self.lease, self.inference, request_encoding="stage_packed")
        self.epochs.append({"stage_rows_reserved": sum(demand.values()),
                            "prefill_rows": prefill_rows, "decode_steps_allowance": decode_steps,
                            "continuation": continuation})
        self.active = True

    def inference(self, sid, payloads):
        if not self.active:
            raise RuntimeError("no active bounded execution")
        import numpy as np
        from pllm.runtime.stage_protocol import PreparedStageBatchRequest, MaskedStageResponse
        from pllm.runtime.residue_codec import pack_row_response
        stage = self.engine.models[self.compiled._bundle.model_id].stages[sid]
        replies = []
        for payload in payloads:
            metadata = self.compiled._bundle.stages[sid]
            request = PreparedStageBatchRequest.unpack(payload,
                expected_input_bits=max(metadata.output_residue_bits))
            for ticket in request.correlation_ids:
                if ticket in self.tickets:
                    raise RuntimeError("one-use ticket reused")
                self.tickets.add(ticket)
            correction, offset = self.corrections[sid]
            n = len(request.correlation_ids)
            if offset + n > len(correction):
                raise RuntimeError("correction reservation exhausted")
            profile = self.compiled._bundle.stages[sid].seeded_profile
            transformed = (stage.compiled_weight.wrap32(request.masked_input)
                           if profile.wire_bits == 32 else
                           stage.compiled_weight.modular(request.masked_input, profile.modulus))
            output = np.add(transformed, correction[offset:offset + n], dtype=np.uint32)
            if profile.wire_bits < 32:
                output &= np.uint32(profile.modulus - 1)
            self.corrections[sid][1] += n
            reply = pack_row_response(MaskedStageResponse(request.batch_id, output, profile.modulus,
                profile.wire_bits, stage_id=sid, ring=profile.ring),
                metadata.output_residue_bits, namespace="prepared")
            self.bytes["client_to_inference_online"] += len(payload)
            self.bytes["inference_to_client_online"] += len(reply)
            replies.append(reply)
        return replies

    def __call__(self, sid, activation):
        if not self.active:
            raise RuntimeError("online fallback forbidden")
        return self.remote(sid, activation)

    def finish(self):
        if self.lease is not None:
            self.lease.close()
        if self._inventory is not None:
            self._inventory.cancel()
        self.active = False
        self.corrections.clear()


def state_bits(runtime, logits, salt):
    import numpy as np
    values = [np.asarray(logits)]
    for cache in runtime.caches:
        values.extend((cache.key[:cache.length], cache.value[:cache.length]))
    h = hashlib.sha256(salt)
    for value in values:
        h.update(value.view(np.uint32).tobytes())
    return {"bits": h.hexdigest(), "position": runtime.position,
            "float32_values": sum(v.size for v in values)}


def geometry_forecast(compiled, *, input_rows=150, cap=512, used=80, epoch=16):
    """Compiler rows and committed per-column widths; not admitted network bodies."""
    demands = []
    emitted = 0
    while emitted < used:
        allowance = min(epoch, cap - emitted)
        demands.append(compiled.prepared_stage_rows(input_rows if emitted == 0 else 1,
            allowance - 1, continuation=emitted != 0))
        emitted += min(allowance, used - emitted)
    control = compiled.prepared_stage_rows(input_rows, cap - 1)
    total = Counter()
    for demand in demands:
        total.update(demand)
    def size(demand):
        return sum((n * sum(compiled._bundle.stages[sid].output_residue_bits) + 7) // 8
                   for sid, n in demand.items())
    return {"scope": "geometry forecast: packed correction residues only; excludes framing and admission",
            "admitted_execution": False,
            "input_rows": input_rows, "cap": cap, "used": used, "epoch": epoch,
            "execution_count": len(demands), "control_reserved_stage_rows": sum(control.values()),
            "epoch_reserved_stage_rows": sum(total.values()),
            "control_packed_correction_bytes": size(control), "epoch_packed_correction_bytes": size(total),
            "packed_correction_saving_bytes": size(control) - size(total),
            "whole_request_saving_fraction": None}


def run_cell(engine, compiled, pipeline, ids, *, cap, epoch, used=None,
             termination="explicit", salt=b"test-only", fail_after=None, capture=None):
    """Same greedy samples/work in each matched cell; EOS predicate is controlled.

    explicit: final emitted token pending. eos: execute it, then stop at next
    selection (injected public stop predicate, NOT natural checkpoint EOS).
    full-cap: treat sampled EOS as ordinary output consistently in both controls.
    """
    if any(type(n) is not int or n < 1 for n in (cap, epoch)):
        raise ValueError("cap and epoch must be positive integers")
    if termination not in {"explicit", "eos"}:
        raise ValueError("unknown termination")
    used = cap if used is None else used
    if type(used) is not int or not 0 <= used <= cap:
        raise ValueError("used output count exceeds cap")
    contract = compiled._plan.continuation_schedule(pipeline)
    # Admission before any reservation, including pending final generated token.
    contract.admit(len(ids), max(1, cap - 1))
    material = EpochMaterial(engine, compiled)
    runtime = compiled.runtime(material)
    output, pending, logits = [], None, None
    snapshot = owner = None
    status = "completed"
    try:
        runtime.install_continuation(contract)
        while True:
            allowance = min(epoch, cap - len(output))
            if allowance <= 0:
                break
            if pending is not None:
                contract.admit(runtime.position, allowance)
            material.prepare(len(ids) if logits is None else 1,
                             allowance - 1, continuation=logits is not None)
            if logits is None:
                _, logits, _ = runtime.prepare_ids(ids)
            else:
                runtime.restore_for_response(snapshot, owner)
                logits = runtime.continue_ids([pending])[-1]
                pending = None
            if capture:
                capture(runtime, logits)
            for step in range(allowance):
                if fail_after is not None and len(output) == fail_after:
                    raise RuntimeError("injected failure/cancellation")
                if len(output) == used:
                    break
                token = runtime.sample(logits, temperature=0)
                output.append(int(token))
                pending = int(token)
                if len(output) == cap or (len(output) == used and termination == "explicit"):
                    break
                if step + 1 == allowance:
                    break
                logits, _ = runtime.decode_step(token, runtime.caches, runtime.position)
                pending = None
                if capture:
                    capture(runtime, logits)
            material.finish()
            owner = secrets.token_hex(16)
            snapshot = runtime.snapshot_for_response(owner)
            if len(output) == cap or (len(output) == used and (termination == "explicit" or pending is None)):
                break
            # EOS selection at an epoch boundary needs one fresh executed pending
            # row even though zero further emitted output is requested.
            if len(output) == used and termination == "eos":
                contract.admit(runtime.position, 1)
                material.prepare(1, 0, continuation=True)
                runtime.restore_for_response(snapshot, owner)
                logits = runtime.continue_ids([pending])[-1]
                pending = None
                if capture:
                    capture(runtime, logits)
                material.finish()
                break
    except RuntimeError as error:
        if str(error) != "injected failure/cancellation":
            raise
        status = "failed_or_cancelled"
    finally:
        try:
            material.finish()
        finally:
            if sys.exc_info()[0] is not None:
                runtime.reset()
    try:
        return cell_result(runtime, logits, salt, output, pending, material,
                           epoch=epoch, cap=cap, status=status, termination=termination, input_rows=len(ids))
    finally:
        runtime.reset()


def cell_result(runtime, logits, salt, output, pending, material, *, epoch, cap, status, termination, input_rows):
    numeric = state_bits(runtime, logits, salt) if logits is not None else None
    result = {"epoch": epoch, "cap": cap, "emitted_count": len(output), "status": status,
              "termination": termination, "executed_generated_rows": runtime.position - input_rows,
              "pending_count": int(pending is not None), "state": numeric,
              "output_commitment": hashlib.sha256(salt + json.dumps(output).encode()).hexdigest(),
              "stage_rows": dict(material.audit), "epochs": material.epochs,
              "execution_count": len(material.epochs), "added_idle_pauses": max(0, len(material.epochs) - 1),
              "measured_serialized_stage_body_bytes": dict(material.bytes),
              "measured_covered_stage_body_total": sum(material.bytes.values()),
              "sdk_admission_control_bytes": None, "public_delivery_bytes": None,
              "all_link_tcp_stream_bytes": None, "client_cpu_seconds": None,
              "client_peak_rss_bytes": None, "pause_seconds": None,
              "client_active_kv_logits_logical_bytes": None if numeric is None else 4 * numeric["float32_values"],
              "remote_body_mac_fraction_proposed": 1.0}
    if material.audit["prepared_stage_rows_reserved"] != (
        material.audit["prepared_stage_rows_claimed"] + material.audit["prepared_stage_rows_burned"]):
        raise RuntimeError("reserved stage rows do not reconcile with claimed + burned")
    return result


def worker(payload):
    checkpoint = Path(payload["checkpoint"])
    before = freeze(checkpoint)
    if before != payload["frozen"]:
        raise RuntimeError("source/dependency/checkpoint/control changed before worker")
    cap, count = payload["cap"], payload["input_tokens"]
    admission = memory_admission(checkpoint, input_tokens=count, cap=cap, bound=payload["bound"])
    if admission != payload["admission"]:
        raise RuntimeError("allocation admission changed before worker")
    engine, compiled, pipeline = load_reference(checkpoint, input_tokens=count, cap=cap,
                                               bound=payload["bound"])
    # Public synthetic token workload: no private prompts or state in evidence.
    ids = [3 + i % 97 for i in range(count)]
    salt = bytes.fromhex(payload["salt"])
    cells = []
    try:
        for used, termination in ((cap, "explicit"), (payload["used"], "explicit"),
                                  (payload["used"], "eos")):
            control = run_cell(engine, compiled, pipeline, ids, cap=cap, epoch=cap,
                               used=used, termination=termination, salt=salt)
            for epoch in payload["epochs"]:
                cell = run_cell(engine, compiled, pipeline, ids, cap=cap, epoch=epoch,
                                used=used, termination=termination, salt=salt)
                if any(cell[key] != control[key] for key in
                       ("state", "output_commitment", "executed_generated_rows", "pending_count")):
                    raise RuntimeError("exact target logits/KV/pending/output mismatch")
                cells.append({"used": used, "termination": termination, "control": control,
                              "candidate": cell, "exact_gate_passed": True,
                              "covered_stage_body_saving_bytes": control["measured_covered_stage_body_total"]
                                  - cell["measured_covered_stage_body_total"]})
        geometry = [{"stage_id": s.stage_id, "role": s.role,
                     "in_features": compiled._bundle.stages[s.stage_id].in_features,
                     "out_features": compiled._bundle.stages[s.stage_id].out_features,
                     "output_residue_width_sum": sum(compiled._bundle.stages[s.stage_id].output_residue_bits),
                     "wire_bits": compiled._bundle.stages[s.stage_id].seeded_profile.wire_bits}
                    for s in compiled.stage_bindings
                    if compiled._bundle.stages[s.stage_id].client_weight is None]
    finally:
        asyncio.run(engine.unload(compiled._bundle.model_id))
    if freeze(checkpoint) != before:
        raise RuntimeError("source/dependency/checkpoint/control changed during worker")
    forecasts = None
    # Geometry is shape arithmetic, not an executable large-cap resume: the
    # packed decoder here is deliberately capped at a 256-token continuation.
    forecasts = [geometry_forecast(compiled, epoch=e) for e in payload["epochs"]]
    return {"frozen": before, "compiled_digest": compiled.digest,
            "illustrative_workload_geometry_forecasts": forecasts,
            "compiler_stage_geometry": geometry, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--used", type=int, default=17)
    parser.add_argument("--input-tokens", type=int, default=16)
    parser.add_argument("--bound", type=int, default=256)
    parser.add_argument("--epochs", type=int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--memory-bytes", type=int, default=4 << 30,
                        help="allocation ceiling; calculated admission must fit")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if not args.checkpoint or not args.output:
        parser.error("--checkpoint and --output required; existing local checkpoint only")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.used < args.cap or any(e not in PUBLIC_EPOCHS for e in args.epochs):
        parser.error("require 0 <= used < cap and public epochs 1/4/8/16/32/64")
    from frontier_probe_support import fresh_workers
    admission = memory_admission(args.checkpoint, input_tokens=args.input_tokens, cap=args.cap, bound=args.bound)
    if admission["required_bytes"] > args.memory_bytes:
        raise ValueError("calculated allocation bound exceeds explicit memory ceiling")
    frozen = freeze(args.checkpoint)
    payload = {"checkpoint": str(args.checkpoint.resolve()), "frozen": frozen,
               "cap": args.cap, "used": args.used, "input_tokens": args.input_tokens,
               "admission": admission,
               "bound": args.bound, "epochs": args.epochs, "salt": secrets.token_hex(32)}
    with fresh_workers(__file__, memory_bytes=admission["required_bytes"], timeout=1800) as (run, guard, host):
        result = run(payload)
        if freeze(args.checkpoint) != frozen:
            raise RuntimeError("frozen cohort identity changed")
        report = {"schema": "pllm.research.decode_epochs.v1", "complete": True,
                  "scope": "exact decoder + in-process one-use prepared stages; not executable SDK choice",
                  "workload": "public synthetic token IDs; controlled EOS predicate, greedy samples unchanged",
                  "measurements": result, "memory_guard_passed": guard.error is None,
                  "allocation_admission": admission, "historical_control": historical_control(),
                  "host_guard": {"initial_available_bytes": host.available,
                    "initial_total_bytes": host.total, "initial_swap_used_bytes": host.swap_used,
                    "reserve_bytes": host.reserve, "minimum_available_bytes": guard.minimum_available_bytes,
                    "maximum_host_used_bytes": host.total - guard.minimum_available_bytes,
                    "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                    "maximum_swap_used_bytes_upper_bound": host.swap_used + guard.maximum_swap_growth_bytes},
                  "cost_owners": ["checkpoint/import/source hashing", "client logits and full KV",
                    "old/new state overlap if SDK snapshots used", "mask expansion and corrections",
                    "provider matrices", "idle preparation pauses", "SDK control/admission not measured"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["host_guard"].update(
        minimum_available_bytes=guard.minimum_available_bytes,
        maximum_host_used_bytes=host.total - guard.minimum_available_bytes,
        maximum_swap_growth_bytes=guard.maximum_swap_growth_bytes,
        maximum_swap_used_bytes_upper_bound=host.swap_used + guard.maximum_swap_growth_bytes)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
