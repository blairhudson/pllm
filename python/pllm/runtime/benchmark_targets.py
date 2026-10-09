"""Compiler-bound local benchmark executors. No benchmark-only serving branches."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import secrets
import time

import numpy as np

from pllm._benchmark_contract import TargetUnavailable, digest
from pllm._benchmark_worker import peak_rss


def model_context(parameters, m):
    from pllm import lower_model
    from pllm.configuration import Experiment
    from pllm.profiles import resolve_runtime_composition
    from .semantic_stages import scheduled_stage_specs
    m.admit(128 << 20)
    declared = parameters["model"]
    experiment = Experiment.from_spec(declared["experiment"]).with_params(pipeline__profile=declared["profile"])
    options = resolve_runtime_composition(experiment.pipeline)
    if options is None:
        raise TargetUnavailable("unimplemented_composition")
    config, resolved = declared["config"], None
    if config is None:
        if experiment.pipeline.model.kind == "tiny" and parameters.get("weights") != "checkpoint":
            from pllm.sources import _tiny_model_config
            config = _tiny_model_config()
        else:
            from pllm.model_loader import resolve_model
            resolved = resolve_model(experiment.pipeline.model)
            if resolved.path is None:
                raise TargetUnavailable("local_configuration_required")
            path = resolved.path / "config.json"
            if path.stat().st_size > 2 << 20:
                raise TargetUnavailable("configuration_limit", status="blocked")
            config = json.loads(path.read_bytes())
    if parameters.get("weights") == "checkpoint" and resolved is None:
        # A caller-supplied configuration cannot self-assert checkpoint identity.
        raise TargetUnavailable("checkpoint_requires_resolved_source")
    plan = lower_model(config, batch=1, max_input_tokens=experiment.budget.max_input_tokens,
                       max_new_tokens=experiment.budget.max_new_tokens)
    schedule = plan.runtime_schedule(experiment.pipeline)
    stages = [replace(s, weight_bits=options.weight_bits, activation_bits=options.activation_bits)
              for s in scheduled_stage_specs(plan, experiment.pipeline)]
    source = digest(config) if resolved is None else resolved.source_lock_digest
    m.details.update(plan_digest=plan.digest, schedule_digest=schedule.digest,
        component_ids=sorted({c.component for c in experiment.pipeline.components.values()}),
        model={"id": str(experiment.pipeline.model.model_id or experiment.pipeline.model.source),
               "checkpoint_digest": None if resolved is None else resolved.checkpoint_digest,
               "source_lock_digest": None if resolved is None else source})
    m.comparison.update(source_digest=source,
        numeric_digest=digest({"config": config, "weight_bits": options.weight_bits,
                               "activation_bits": options.activation_bits}),
        privacy_digest=digest({"scope": "local_reference_no_role_transport"}),
        state_digest=digest({"state": "none"}))
    return experiment, config, plan, stages, options, resolved


def state_layout(plan, tokens):
    states = []
    for row in plan.prefill["state_outputs"]:
        if row["kind"] not in {"key", "value"} or len(row["shape"]) != 4:
            raise TargetUnavailable("exclusive_dense_kv_required")
        shape = list(row["shape"])
        if tokens > row["maximum_sequence"]:
            raise TargetUnavailable("state_exceeds_compiled_bound", status="blocked")
        shape[-2] = tokens
        states.append({"layer": row["layer"], "kind": row["kind"], "shape": shape})
    if not states or len({(s["layer"], s["kind"]) for s in states}) != len(states):
        raise TargetUnavailable("exclusive_dense_kv_required")
    return states


def _select(stages, parameters):
    selected = [s for s in stages if s.role == parameters["role"] and s.layer_index == parameters["layer"]]
    if len(selected) != 1:
        raise TargetUnavailable("semantic_stage_not_unique", details={"available_stage_roles": sorted({s.role for s in stages})})
    return selected[0]


def _chunks(path):
    with Path(path).open("rb") as source:
        while block := source.read(1 << 20):
            yield block


def _weight_file(stage, parameters, context, m):
    _, config, _, _, options, resolved = context
    path = m.directory / "public-weight.i8"
    state, bound = hashlib.sha256(), 0
    count = stage.in_features * stage.out_features
    rows = max(1, min(256, (1 << 20) // stage.in_features))
    cpu, wall = time.process_time(), time.perf_counter()
    if options.public_equalization_digest is not None:
        raise TargetUnavailable("equalized_stage_fixture_not_implemented")
    if parameters["weights"] == "synthetic":
        rng = np.random.default_rng(parameters["seed"])
        def weights():
            high = (1 << (stage.weight_bits - 1)) - 1
            for start in range(0, stage.out_features, rows):
                yield rng.integers(-high, high + 1, (min(rows, stage.out_features - start), stage.in_features), dtype=np.int8)
    else:
        from .safetensors_store import SafeTensorStore
        from .transformer_engine import MaskedTransformerEngine
        manifest = replace(resolved.manifest, metadata={**resolved.manifest.metadata,
                                                        "stage_origin": "semantic_schedule_v1"})
        engine = MaskedTransformerEngine(weight_bits=options.weight_bits, activation_bits=options.activation_bits)
        store = SafeTensorStore(resolved.path)
        runtime = engine._load_stage(store, stage, manifest)
        def weights():
            for start in range(0, stage.out_features, rows):
                yield runtime.quantized_weight.values[start:start + rows]
    with path.open("xb") as out:
        for array in weights():
            bound = max(bound, int(np.max(np.abs(array.astype(np.int64)).sum(axis=1))))
            block = array.tobytes()
            state.update(block)
            out.write(block)
    if path.stat().st_size != count:
        raise ValueError("selected stage weight size differs")
    m.details["fixture_loading"] = {"cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall, "public_weight_bytes": count}
    m.fixture = {"kind": parameters["weights"], "digest": state.hexdigest(), "metadata": {
        "stage_id": stage.id, "role": stage.role, "layer": stage.layer_index,
        "shape": [stage.out_features, stage.in_features], "weight_bits": stage.weight_bits,
        "public_seed": parameters["seed"] if parameters["weights"] == "synthetic" else None}}
    m.comparison["source_digest"] = digest({"source": m.comparison["source_digest"], "weights": state.hexdigest()})
    return path, state.hexdigest(), bound * ((1 << (stage.activation_bits - 1)) - 1)


def _oracle(weights, inputs, modulus):
    # Signed int64 is independent of native unsigned kernels and floating GEMM.
    if weights.shape[1] * 128 * ((1 << 32) - 1) >= 1 << 63:
        raise ValueError("integer oracle accumulation exceeds i64")
    result = np.empty((inputs.shape[0], weights.shape[0]), dtype=np.uint32)
    for start in range(0, weights.shape[0], 64):
        result[:, start:start + 64] = (inputs.astype(np.int64) @ weights[start:start + 64].astype(np.int64).T) % modulus
    return result


def _stage(target, p, m):
    from pllm.native import MaskedGEMM, PagedGEMM
    from .preparation_protocol import (PreparationRequest, expand_preparation_mask, prepared_mask_rows,
                                       seeded_ring_profile)
    from .stage_protocol import pack_residues, unpack_residues
    context = model_context(p, m)
    stage = _select(context[3], p)
    if context[4].verification_component is not None:
        raise TargetUnavailable("verified_stage_target_not_implemented")
    if target.kind == "stage" and p["execution"] != "protocol_local":
        raise TargetUnavailable("stage_execution_scope_not_implemented")
    rows, width, out = p["rows"], stage.in_features, stage.out_features
    if rows > context[0].budget.max_input_tokens or rows * max(width, out) > 4 << 20:
        raise TargetUnavailable("stage_workload_bound", status="blocked")
    if rows * width * out * (m.sampling.repetitions + m.sampling.warmups) > 2_000_000_000:
        raise TargetUnavailable("integer_work_limit", status="blocked")
    m.admit((160 << 20) + 8 * width * out + 32 * rows * (width + out)
            + ((224 << 20) if p["kernel"] == "digit8_native" else 0))
    path, weight_hash, bound = _weight_file(stage, p, context, m)
    weights = np.memmap(path, dtype=np.int8, mode="r", shape=(out, width))
    executor = MaskedGEMM(threads=m.sampling.threads)
    if not executor.native:
        raise TargetUnavailable("native_extension_required")
    if p["kernel"] == "digit8_native":
        from pllm.native import capabilities
        if not capabilities().get("digit8_cpu"):
            raise TargetUnavailable("digit8_cpu_unavailable")
    if p["kernel"] in {"digit4", "digit8"} and p["storage"] == "paged":
        raise TargetUnavailable("digit_paged_kernel_not_implemented")
    matrix = (PagedGEMM.from_raw(path, weight_hash, (out, width), threads=m.sampling.threads)
               if p["storage"] == "paged" else executor.compile(weights) if p["kernel"] in {"native", "digit8_native"} else None)
    if p["kernel"] in {"digit4", "digit8"}:
        from .benchmark_digit_gemm import DigitGEMM
        matrix = DigitGEMM(weights, digit_bits=int(p["kernel"].removeprefix("digit")))
    profile = seeded_ring_profile(bound)
    def multiply(x):
        values = np.ascontiguousarray(x)
        if p["kernel"] == "digit8_native":
            return matrix.digit8(values, profile.wire_bits)
        return matrix.wrap32(values) if profile.modulus == 1 << 32 else matrix.modular(values, profile.modulus)
    rng = np.random.default_rng(p["seed"] + 1)
    high = (1 << (stage.activation_bits - 1)) - 1
    clear = rng.integers(-high, high + 1, (rows, width), dtype=np.int8)
    def request():
        return PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), "benchmark-stage",
            weight_hash, stage.id, weight_hash, rows, width, out, stage.weight_bits, stage.activation_bits,
            bound, profile.ring, profile.modulus, profile.wire_bits, secrets.token_bytes(32))
    request_bytes, reply_bytes = [], []
    def execute(index):
        req = request()
        destination = m.directory / f"correction-{index}.bin"
        step = p.get("chunk_rows", 0) or rows
        cursor = prepared_mask_rows(req) if p.get("chunk_rows", 0) else None
        try:
            if cursor is None:
                full_r = expand_preparation_mask(req)
                full_s = expand_preparation_mask(req, "output-s")
            with destination.open("xb") as sink:
                for begin in range(0, rows, step):
                    count = min(step, rows - begin)
                    if cursor is not None:
                        r, s = cursor.take(begin, count)
                        r = np.frombuffer(r, "<u4").reshape(count, width)
                        s = np.frombuffer(s, "<u4").reshape(count, out)
                    else:
                        r, s = full_r[begin:begin + count], full_s[begin:begin + count]
                    correction = (multiply(r) - s) & np.uint32(profile.modulus - 1)
                    if target.kind == "stage":
                        masked = (clear[begin:begin + count].astype(np.uint32) - r) & np.uint32(profile.modulus - 1)
                        frame = pack_residues(masked, profile.wire_bits)
                        masked = unpack_residues(frame, masked.shape, profile.wire_bits)
                        response = (multiply(masked) + correction) & np.uint32(profile.modulus - 1)
                        encoded = pack_residues(response, profile.wire_bits)
                        actual = (unpack_residues(encoded, response.shape, profile.wire_bits) + s) & np.uint32(profile.modulus - 1)
                        request_bytes.append(len(frame))
                        reply_bytes.append(len(encoded))
                        sink.write(actual.astype("<u4", copy=False).tobytes())
                    else:
                        sink.write(correction.astype("<u4", copy=False).tobytes())
            return destination, req
        finally:
            if cursor is not None:
                cursor.cancel()
    values = m.run(execute)
    m.details["target_peak_rss_bytes"], m.details["target_cpu_from_birth_seconds"] = peak_rss(), time.process_time()
    for destination, req in values:
        actual = np.fromfile(destination, dtype="<u4").reshape(rows, out)
        if target.kind == "stage":
            expected = _oracle(weights, clear.astype(np.uint32), profile.modulus)
        else:
            expected = (_oracle(weights, expand_preparation_mask(req), profile.modulus)
                        - expand_preparation_mask(req, "output-s")) & np.uint32(profile.modulus - 1)
        if not np.array_equal(actual, expected):
            m.oracle = {"kind": "independent_signed_i64", "outcome": "failed", "checked_values": actual.size}
            raise TargetUnavailable("integer_parity_failed", status="failed")
    m.oracle = {"kind": "independent_signed_i64", "outcome": "passed", "checked_values": rows * out * len(values)}
    # Preparation outputs are fresh random material; compare the public algebra
    # and oracle success, never randomized masks or secret material digests.
    m.comparison.update(workload_digest=digest({"shape": [rows, width, out], "phase": p.get("phase"), "seed": p["seed"]}),
        output_digest=m.commitment({"algebra": "W-r-minus-s" if target.kind == "preparation" else "W-x",
            "weight": weight_hash, "ring": profile.to_dict(), "shape": [rows, out]}),
        measurement_digest=digest({"kind": target.kind, "scope": "local_expand_compute_encode_file", "oracle_outside_timing": True}))
    m.details.update(ring=profile.to_dict(), integer_macs=rows * width * out,
        chunk_rows=p.get("chunk_rows", 0), local_file_payload_bytes=rows * out * 4,
        numeric_backend={"numpy": np.__version__, "float32_exact_partial_sum_limit": (1 << 24) - 1})
    m.metric("local.arithmetic_bodies", rows * (width + out) * (profile.wire_bits // 8) if target.kind == "stage" else rows * out * (profile.wire_bits // 8),
             kind="bytes", origin="deterministic", scope="unframed_arithmetic_payload")
    m.limitations += ["Single compiler-selected stage; synthetic weights are not checkpoint values.",
        "Local mask/codec reference excludes provider transport, acknowledgement, inventory admission and full-response work.",
        "This isolated target does not exercise serving dispatch or its shape threshold.",
        "Process peak ends before independent integer validation; file payload and filesystem cache are separate owners."]


def _artifacts(p, m):
    from pllm import _native
    from .bundle_storage import FilePayload
    from .paged import PagedGEMM
    context = model_context(p, m)
    stage = _select(context[3], p)
    count = stage.in_features * stage.out_features
    m.admit((128 << 20) + 8 * count)
    path, weight_hash, _ = _weight_file(stage, p, context, m)
    def execute(_):
        payload = FilePayload(_chunks(path), size=count, digest=weight_hash, native=p["storage"] == "shared")
        try:
            matrix = PagedGEMM._from_payload_executor(payload, (stage.out_features, stage.in_features), _native.Executor(m.sampling.threads, True))
            # Retiring delivery cannot invalidate the independent execution owner.
            payload.close()
            if p["operation"] == "bind":
                return hashlib.sha256(matrix.clear(np.ones((1, stage.in_features), np.int8)).tobytes()).hexdigest()
            return matrix.weight_digest
        finally:
            payload.close()
    values = m.run(execute)
    m.details["target_peak_rss_bytes"], m.details["target_cpu_from_birth_seconds"] = peak_rss(), time.process_time()
    if p["operation"] == "bind":
        weights = np.memmap(path, dtype=np.int8, mode="r", shape=(stage.out_features, stage.in_features))
        expected = hashlib.sha256(weights.sum(axis=1, dtype=np.int64).astype("<i4").tobytes()).hexdigest()
    else:
        expected = weight_hash
    if any(value != expected for value in values):
        raise ValueError("imported artifact changed")
    m.oracle = {"kind": "signed_integer_or_content_hash", "outcome": "passed", "checked_values": stage.out_features if p["operation"] == "bind" else count}
    m.comparison.update(workload_digest=digest({"shape": [stage.out_features, stage.in_features], "operation": p["operation"]}),
        output_digest=m.commitment(values[0]), measurement_digest=digest({"scope": "local_file_import", "operation": p["operation"]}))
    m.details["private_disk_payload_bytes"] = count * (1 if p["storage"] == "shared" else 2)
    m.limitations += ["Selected immutable weight object; excludes tokenizer, complete bundle validation, network and filesystem-cache costs."]


def execute_target(target, m):
    p = target.to_spec()["parameters"]
    if m.inputs and target.kind not in {"response", "operator"}:
        raise TargetUnavailable("private_inputs_not_supported_for_target")
    if target.kind in {"stage", "preparation"}:
        if target.kind == "preparation" and "output_columns" in p:
            from .benchmark_correction_stripes import execute_stripes
            return execute_stripes(p, m)
        return _stage(target, p, m)
    if target.kind == "artifacts":
        return _artifacts(p, m)
    if target.kind in {"state", "projection"}:
        from .benchmark_state_targets import execute_state
        return execute_state(target, p, m)
    if target.kind == "response":
        from .benchmark_response_target import execute_response
        return execute_response(p, m)
    if target.kind == "operator":
        from .benchmark_response_target import execute_operator
        return execute_operator(p, m)
    context = m.run(lambda _: model_context(p, m))[-1]
    m.details["stages"] = [s.to_dict() for s in context[3]]
    m.fixture = {"kind": "projection", "digest": context[2].digest, "metadata": {"values_loaded": False}}
    m.oracle = {"kind": "native_semantic_schedule", "outcome": "passed", "checked_values": len(context[3])}
    m.comparison.update(workload_digest=digest(context[0].budget.to_spec()), output_digest=context[2].digest,
                        measurement_digest=digest({"scope": "geometry_only"}))
