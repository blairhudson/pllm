"""Integer residual delegation research; clear arithmetic, never private serving.

Only the parent-run cohort imports the compiler or reads checkpoint tensors.
Search is public-weight-only and bounded; no SVD, activation fitting or clipping.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CONTROL = ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"
MAX_ELEMENTS = 12_000_000


def residual_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def residual_bounds(weight):
    """Exact public signed-x bound, row residue widths and actual provider ring."""
    if weight.ndim != 2 or weight.dtype.kind != "i" or not weight.size:
        raise ValueError("nonempty signed integer matrix required")
    l1 = np.sum(np.abs(weight.astype(np.int64)), axis=1, dtype=np.int64)
    bounds = l1 * 127
    widths = np.array([max(1, int(b).bit_length() + 1) for b in bounds], np.uint8)
    ring = next((b for b in (16, 24, 32) if int(bounds.max()) < 1 << (b - 1)), None)
    if ring is None:
        raise ValueError("residual exceeds bounded native ring")
    return l1, widths, ring


def residual_body_bytes(weight, rows=1):
    _, widths, ring = residual_bounds(weight)
    # Stage-packed ingress; correction and output each row-residue packed.
    # Tensor payload only. Actual protocol/framing/scale deltas are unknown.
    inputs = (rows * weight.shape[1] * int(widths.max()) + 7) // 8
    result = (rows * int(widths.sum()) + 7) // 8
    return {"inputs": inputs, "outputs": result, "corrections": result,
            "total": inputs + 2 * result, "provider_ring_bits": ring}


def residual_search(weight, *, rank=2, seeds=2, native_i8=True):
    """Greedy signed integer outer products; complete E retained at every step.

    Each B row is 4 or 16 times sign of a public residual row, NOT a block
    mean/anchor. A coefficients in {-2,-1,0,1,2}. Per-row ties use L1 after
    encoded-width objective. Candidate selection uses total body bytes then L1.
    Rank zero is always retained; wider residuals are recorded, never narrowed.
    """
    if weight.dtype != np.int8 or weight.ndim != 2 or not weight.size:
        raise ValueError("original W must be nonempty int8")
    if weight.size > MAX_ELEMENTS or not 0 <= rank <= 4 or not 1 <= seeds <= 2:
        raise ValueError("bounded search admission failed")
    m, n = weight.shape
    if rank * (m + n) * 10 > m * n:
        raise ValueError("rank exceeds 10% per-stage client MAC budget")
    e = weight.astype(np.int16)
    a = np.empty((m, 0), np.int8)
    b = np.empty((0, n), np.int8)
    yield a.copy(), b.copy(), e.copy()
    for _ in range(rank):
        l1, _, _ring = residual_bounds(e)
        indices = np.argsort(-l1, kind="stable")[:seeds]
        best = None
        for index in indices:
            for magnitude in (4, 16):
                vector = (np.sign(e[index]) * magnitude).astype(np.int8)
                candidate = np.empty_like(e)
                coefficients = np.zeros(m, np.int8)
                for start in range(0, m, 128):
                    end = min(m, start + 128)
                    source = e[start:end].astype(np.int64)
                    picked = source.copy()
                    best_l1 = np.abs(source).sum(axis=1)
                    best_width = np.array([max(1, (int(x) * 127).bit_length() + 1)
                                           for x in best_l1])
                    for coefficient in (-2, -1, 1, 2):
                        trial = source - coefficient * vector.astype(np.int64)
                        trial_l1 = np.abs(trial).sum(axis=1)
                        trial_width = np.array([max(1, (int(x) * 127).bit_length() + 1)
                                                for x in trial_l1])
                        take = (trial_width < best_width) | ((trial_width == best_width) & (trial_l1 < best_l1))
                        if native_i8:
                            take &= (trial.min(axis=1) >= -128) & (trial.max(axis=1) <= 127)
                        picked[take] = trial[take]
                        coefficients[start:end][take] = coefficient
                        best_l1[take], best_width[take] = trial_l1[take], trial_width[take]
                    if np.abs(picked).max() > 32767:
                        raise ValueError("residual construction exceeds int16")
                    candidate[start:end] = picked.astype(np.int16)
                score = (residual_body_bytes(candidate)["total"], int(np.abs(candidate.astype(np.int64)).sum()))
                if best is None or score < best[0]:
                    best = score, coefficients, vector, candidate
        _, coefficients, vector, e = best
        a = np.column_stack((a, coefficients))
        b = np.vstack((b, vector))
        if not np.array_equal(weight.astype(np.int64), a.astype(np.int64) @ b.astype(np.int64) + e):
            raise AssertionError("factor identity changed W")
        yield a.copy(), b.copy(), e.copy()


def residual_oracle(weight, inputs, outputs):
    """Independent Python arbitrary-precision dot products, no NumPy matmul."""
    return [[sum(int(weight[j, k]) * int(row[k]) for k in range(weight.shape[1]))
             for j in outputs] for row in inputs]


def residual_exact(weight, a, b, e, inputs, *, native=False):
    if any(v.dtype.kind != "i" for v in (weight, a, b, e, inputs)):
        raise ValueError("signed integer operands required")
    if (weight.dtype != np.int8 or a.dtype != np.int8 or b.dtype != np.int8
            or inputs.dtype != np.int8 or e.dtype not in (np.dtype("int8"), np.dtype("int16"))):
        raise ValueError("W/A/B/x require int8; retained E requires int8/int16")
    if (weight.ndim != 2 or weight.size > MAX_ELEMENTS or min(weight.shape) <= 0
            or inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]
            or e.shape != weight.shape or a.ndim != 2 or b.ndim != 2
            or a.shape[0] != weight.shape[0] or b.shape[1] != weight.shape[1]
            or a.shape[1] != b.shape[0] or b.shape[0] > 4):
        raise ValueError("bounded residual geometry required")
    if np.max(np.abs(inputs.astype(np.int64))) > 127:
        raise ValueError("W8A8 input outside public bound")
    if not np.array_equal(weight.astype(np.int64), a.astype(np.int64) @ b.astype(np.int64) + e):
        raise ValueError("E must remain exact")
    # Conservative, explicit bound on intermediates before using int64 arithmetic.
    intermediate = 127 * np.abs(b.astype(np.int64)).sum(axis=1)
    if sum(int(x) for x in intermediate) * max(1, int(np.abs(a.astype(np.int64)).max(initial=0))) >= 2**63:
        raise ValueError("client intermediate exceeds int64")
    x = inputs.astype(np.int64)
    local = (x @ b.astype(np.int64).T) @ a.astype(np.int64).T
    delegated = x @ e.astype(np.int64).T
    admission = bool(int(e.min()) >= -128 and int(e.max()) <= 127)
    if native and admission:
        from pllm.runtime.native import MaskedGEMM
        from pllm.runtime.preparation_protocol import seeded_ring_profile
        from pllm import _native
        owner = MaskedGEMM(threads=1, build=False)
        if not owner.native:
            raise RuntimeError("installed native kernel required")
        matrix = owner.compile(e.astype(np.int8))
        l1, widths, bits = residual_bounds(e)
        if seeded_ring_profile(int(l1.max()) * 127).wire_bits != bits:
            raise AssertionError("provider ring mismatch")
        actual_widths = _native.offset_row_bits(e.astype(np.int8).tobytes(), e.shape[1], 127)
        if bytes(widths) != actual_widths:
            raise AssertionError("native row-width mismatch")
        modulus = 1 << bits
        rng = np.random.default_rng(5105)
        masks = rng.integers(0, modulus, size=x.shape, dtype=np.uint32)
        masked = ((x - masks.astype(np.int64)) % modulus).astype(np.uint32)
        output = matrix.wrap32(masked).astype(np.uint64)
        correction = matrix.wrap32(masks).astype(np.uint64)
        residues = ((output + correction) % modulus).astype(np.int64)
        centered = residues.copy()
        centered[centered >= modulus // 2] -= modulus
        np.testing.assert_array_equal(centered, delegated)
        # Exercise installed prepared correction/output row-residue codecs, too.
        output_mask = rng.integers(0, modulus, size=delegated.shape, dtype=np.uint32)
        input_widths = bytes([int(widths.max())]) * x.shape[1]
        packed_input = _native.offset_pack_rows(masked.astype("<u4").tobytes(), input_widths, len(x))
        projected = np.frombuffer(_native.unpack_residue_rows(packed_input, input_widths, len(x)), "<u4").reshape(x.shape)
        correction_body = _native.prepared_pack_correction(correction.astype("<u4").tobytes(),
            output_mask.astype("<u4").tobytes(), bytes(widths), len(x))
        output_body = _native.prepared_pack_output(matrix.wrap32(projected).astype("<u4").tobytes(),
            correction_body, bytes(widths), len(x))
        recovered = np.frombuffer(_native.prepared_unpack_output(output_body,
            output_mask.astype("<u4").tobytes(), bytes(widths), len(x)), "<i8").reshape(delegated.shape)
        np.testing.assert_array_equal(recovered, delegated)
        np.testing.assert_array_equal(matrix.clear(inputs.astype(np.int8)), delegated)
        original = owner.compile(weight)
        np.testing.assert_array_equal(original.clear(inputs.astype(np.int8)), local + delegated)
    result = local + delegated  # BEFORE original float32 dequantization.
    np.testing.assert_array_equal(result, x @ weight.astype(np.int64).T)
    selected = sorted(set((0, weight.shape[0] // 2, weight.shape[0] - 1)))
    if result[:, selected].tolist() != residual_oracle(weight, inputs, selected):
        raise AssertionError("independent Python integer oracle mismatch")
    return result, admission


def residual_frozen():
    """Hash actual installed files, dependencies, source and shared controls."""
    import pllm
    from pllm.runtime._native_support import extension
    paths = {Path(__file__), ROOT / "scripts/frontier_probe_support.py", ROOT / "scripts/probe_all_link_frontier.py", CONTROL, GEOMETRY,
             Path(sys.executable).resolve()}
    for name in ("Cargo.lock", "Cargo.toml", "pyproject.toml"):
        if (ROOT / name).exists():
            paths.add(ROOT / name)
    for root in (ROOT / "python/pllm", ROOT / "crates", Path(pllm.__file__).parent):
        if root.exists():
            paths.update(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".rs", ".so", ".toml"))
    module = extension()
    if module is None:
        raise RuntimeError("installed native extension required")
    paths.add(Path(module.__file__))
    dependencies = {}
    for dist in importlib.metadata.distributions():
        dependencies[dist.metadata["Name"]] = dist.version
        for file in dist.files or ():
            path = Path(dist.locate_file(file))
            if path.is_file() and (path.suffix in (".py", ".so", ".dylib") or path.name in ("METADATA", "RECORD")):
                paths.add(path)
    return {"files": {str(p): residual_digest(p) for p in sorted(paths)},
            "dependencies": dependencies, "python": sys.version}


def residual_compile(directory):
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import MaskedLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role
    config = json.loads((directory / "config.json").read_bytes())
    pipeline = MaskedLinearCpu(Model.hf(REPO, revision=REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues", prefill_pruning="terminal"))
    plan = lower_model(config, batch=1, max_input_tokens=150, max_new_tokens=8)
    schedule = plan.runtime_schedule(pipeline).to_dict()
    stages = [s for s in scheduled_stage_specs(plan, pipeline) if s.layer_index is not None]
    operations = {op["id"]: op for op in plan.prefill["operations"]}
    steps = {(s["layer"], semantic_stage_role(s, operations)): s for s in schedule["prefill"]["steps"]
             if s["weight_ids"] and s["layer"] is not None}
    geometry = [{"id": s.id, "role": s.role, "layer": s.layer_index,
                 "shape": [s.out_features, s.in_features],
                 "prefill_rows": 1 if steps[s.layer_index, s.role].get("terminal_row_only", False) else 150,
                 "decode_rows": 1} for s in stages]
    return stages, geometry, {"plan_digest": plan.digest, "pipeline_digest": pipeline.digest(), "schedule": schedule}


def residual_worker(payload):
    identity_start = time.process_time()
    if payload["frozen"] != residual_frozen():
        raise RuntimeError("worker runtime/source identity changed")
    identity_cpu = time.process_time() - identity_start
    if payload.get("generated"):
        result = residual_generated_callback()
        if payload["frozen"] != residual_frozen():
            raise RuntimeError("generated callback identities changed")
        return result
    directory = Path(payload["checkpoint"])
    for name, value in payload["checkpoint_files"].items():
        if residual_digest(directory / name) != value:
            raise RuntimeError("checkpoint identity changed")
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.quantization import quantize_weight_per_row, dequantize_matmul
    from probe_all_link_frontier import conserved_control
    stages, geometry, compiler = residual_compile(directory)
    stage = next(s for s in stages if s.role == payload["role"] and s.layer_index == payload["layer"])
    prior = json.loads(GEOMETRY.read_bytes())
    control = conserved_control(json.loads(CONTROL.read_bytes()))
    if prior["source"]["body_fingerprint"] != control["body_fingerprint"]:
        raise ValueError("control/geometry body identity differs")
    if not prior["complete"] or {r["stage"]: r["shape"] for r in prior["stages"]} != {r["id"]: r["shape"] for r in geometry}:
        raise ValueError("compiler-derived geometry differs from frozen evidence")
    store = SafeTensorStore(directory)
    start = time.process_time()
    quantized = [quantize_weight_per_row(store.get(key), bits=8) for key in stage.weight_keys]
    weight = np.concatenate([q.values for q in quantized])
    scales = np.concatenate([q.scales for q in quantized])
    source_cpu = time.process_time() - start
    store.clear_cache()
    del quantized
    if weight.shape != (stage.out_features, stage.in_features):
        raise AssertionError("compiled fusion shape mismatch")
    historical_stage = next(r for r in prior["stages"] if r["stage"] == stage.id)
    if hashlib.sha256(weight.tobytes()).hexdigest() != historical_stage["weight_sha256"]:
        raise AssertionError("actual W8 compiler stage differs from historical control weights")
    g = next(r for r in geometry if r["id"] == stage.id)
    rng = np.random.default_rng(5105)
    inputs = rng.integers(-127, 128, (3, weight.shape[1]), dtype=np.int8)
    inputs[0].fill(127)
    inputs[1].fill(-127)
    baseline = residual_body_bytes(weight)
    rows = []
    search = residual_search(weight, rank=payload["rank"], seeds=payload["seeds"], native_i8=payload["native_i8"])
    fitting_cpu = 0.0
    while True:
        start = time.process_time()
        try:
            a, b, e = next(search)
        except StopIteration:
            break
        fitting_cpu += time.process_time() - start
        start = time.process_time()
        result, admitted = residual_exact(weight, a, b, e, inputs, native=True)
        expected = inputs.astype(np.int64) @ weight.astype(np.int64).T
        activation_scales = np.ones(3, np.float32)
        np.testing.assert_array_equal(dequantize_matmul(result.astype(np.int32), activation_scales, scales),
                                      dequantize_matmul(expected.astype(np.int32), activation_scales, scales))
        parity_cpu = time.process_time() - start
        cost = residual_body_bytes(e)
        rank = b.shape[0]
        factor_bytes = a.nbytes + b.nbytes
        saved = baseline["total"] - cost["total"] if admitted else 0
        l1, widths, ring = residual_bounds(e)
        # Both factors delivered once, plus byte-per-output public width map.
        extra = factor_bytes + (len(widths) if rank else 0)
        rows.append({"rank": rank, "client_macs_per_row": rank * sum(weight.shape),
            "client_mac_fraction": rank * sum(weight.shape) / weight.size,
            "residual_dtype": "int8" if admitted else "int16", "native_residual_admitted": admitted,
            "residual_min": int(e.min()), "residual_max": int(e.max()),
            "kernel": "native wrap32 reduced to seeded ring; clear parity" if admitted else "blocked: no native int16 residual kernel",
            "row_l1_max": int(l1.max()), "row_l1_sum": int(l1.sum()),
            "row_width_histogram": {str(int(v)): int((widths == v).sum()) for v in np.unique(widths)},
            "body_payload_per_row": cost, "baseline_body_payload_per_row": baseline,
            "factor_delivery_payload_bytes": factor_bytes, "factor_storage_payload_bytes": factor_bytes,
            "research_int64_factor_conversion_bytes": 8 * factor_bytes,
            "factor_dtype": "int8", "local_intermediate_dtype": "int64",
            "nonzero_factor_coefficients": int(np.count_nonzero(a)),
            "client_native_factor_snapshot_bytes": 0, "client_native_factor_kernel": "not supported for widened Bx; research int64 only",
            "residual_provider_snapshot_bytes": weight.size if admitted else None,
            "residual_provider_storage_bytes": weight.size if admitted else 2 * weight.size,
            "original_provider_weight_bytes_retained_during_probe": weight.nbytes,
            "scales_bytes_unchanged": scales.nbytes, "new_layout_map_bytes": len(widths) if rank else 0,
            "minimum_extra_artifact_payload_bytes": extra,
            "saved_payload_per_executed_row_if_admitted": saved,
            "break_even_executed_rows_lower_bound": (extra // saved + 1) if saved > 0 else None,
            "one_150_plus_8_request_net_tensor_payload_projection": saved * (g["prefill_rows"] + 7) - extra,
            "horizon_request_payload_projections": {str(h): h * saved * (g["prefill_rows"] + 7) - extra for h in (1, 4, 16)},
            "fitting_cpu_seconds_cumulative": fitting_cpu, "parity_cpu_seconds": parity_cpu,
            "exact_integer_oracle": True, "original_stage_dequantization_exact": True,
            "commitments": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in (("A", a), ("B", b), ("E", e))}})
    identity_start = time.process_time()
    if payload["frozen"] != residual_frozen():
        raise RuntimeError("runtime/source changed during worker")
    identity_cpu += time.process_time() - identity_start
    for name, value in payload["checkpoint_files"].items():
        if residual_digest(directory / name) != value:
            raise RuntimeError("checkpoint changed during worker")
    return {"stage": g, "compiler": compiler, "whole_body_geometry": geometry,
            "control": control, "residual_search_policy": "native-i8" if payload["native_i8"] else "reference-i16",
            "quantization": "explicit W8A8 symmetric per-row; original scales unchanged",
            "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
            "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
            "source_and_quantization_cpu_seconds": source_cpu,
            "runtime_identity_hash_cpu_seconds": identity_cpu,
            "candidates": rows}


def residual_generated_callback():
    """Parent-only optional fresh worker: generated decoder, no serving protocol."""
    import asyncio
    import tempfile
    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import MaskedLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    with tempfile.TemporaryDirectory(prefix="residual-generated-") as temp:
        root = create_tiny_llama_checkpoint(Path(temp) / "tiny")
        source_model = Model.path(str(root), model_id="residual-generated")
        source = resolve_model(source_model)
        engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
                                        prepared_output_encoding="row_residues")
        asyncio.run(engine.load(source.manifest))
        try:
            model = engine.models[source.manifest.id]
            plan = lower_model(json.loads((root / "config.json").read_bytes()), batch=1,
                               max_input_tokens=8, max_new_tokens=3)
            pipeline = MaskedLinearCpu(source_model, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
                linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues", prefill_pruning="terminal"))
            compiled = compile_runtime_model(plan, ClientBundle.unpack(engine.client_bundle(source.manifest.id)), composition=pipeline)
            factors = {key: list(residual_search(stage.weight.values, rank=1, seeds=1))[-1]
                       for key, stage in model.stages.items() if stage.spec.layer_index is not None}
            trajectories = []
            budgets = {p: {"body_macs": 0, "client_macs": 0} for p in ("prefill", "decode")}
            for variant in ("original", "residual"):
                phase = "prefill"
                def callback(key, activation):
                    stage = model.stages[key]
                    q = quantize_activation_per_row(activation, bits=8)
                    if variant == "residual" and key in factors:
                        a, b, e = factors[key]
                        integer, admitted = residual_exact(stage.weight.values, a, b, e, q.values, native=True)
                        if not admitted:
                            raise RuntimeError("generated E kernel admission failed")
                        budgets[phase]["body_macs"] += len(q.values) * stage.weight.values.size
                        budgets[phase]["client_macs"] += len(q.values) * b.shape[0] * sum(stage.weight.values.shape)
                    else:
                        integer = stage.compiled_weight.clear(q.values)
                    output = dequantize_matmul(integer.astype(np.int32), q.scales, stage.weight.scales,
                        output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
                    if stage.bias is not None:
                        output += stage.bias
                    return np.ascontiguousarray(output, np.float32)
                runtime = compiled.runtime(callback)
                observed = []
                def capture(logits):
                    observed.append([logits.copy()] + [value.copy() for cache in runtime.caches
                        for value in (cache.key[:cache.length], cache.value[:cache.length])])
                _, logits, _ = runtime.prepare_ids([2, 3, 4, 5])
                capture(logits)
                phase = "decode"
                for _ in range(2):
                    logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
                    capture(logits)
                trajectories.append(observed)
            for before, after in zip(*trajectories, strict=True):
                for left, right in zip(before, after, strict=True):
                    np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
            for value in budgets.values():
                if value["client_macs"] * 10 > value["body_macs"]:
                    raise ValueError("generated executed phase exceeds client MAC budget")
            return {"scope": "generated seed23 two-layer Qwen2 hidden32/intermediate64 W8A8; clear research callback, NO private serving",
                    "logits_and_active_kv_bit_exact": True, "states_compared": 3,
                    "phase_actual_callback_macs": budgets, "body_groups_with_rank1": len(factors),
                    "body_groups_with_nonzero_factors": sum(bool(np.any(a)) for a, b, e in factors.values()),
                    "native_factor_snapshots": "none; local A(Bx) is research int64",
                    "trajectory_sha256": hashlib.sha256(b"".join(v.tobytes() for state in trajectories[1] for v in state)).hexdigest()}
        finally:
            asyncio.run(engine.unload(source.manifest.id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--generated-only", action="store_true", help="optional parent-run generated logits/KV callback worker")
    parser.add_argument("--rank", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--seeds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--residual-policy", choices=("native-i8", "reference-i16"), default="native-i8")
    parser.add_argument("--layer", type=int, choices=range(24), default=0)
    parser.add_argument("--role", choices=("all", "qkv_projection", "attention_output", "mlp_gate_up", "mlp_down"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "target/residual-delegation.json")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(residual_worker(json.load(sys.stdin))))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    from frontier_probe_support import fresh_workers
    if args.generated_only:
        with fresh_workers(__file__, memory_bytes=1 << 30, timeout=600) as (run, guard, _host):
            frozen = residual_frozen()
            result = run({"frozen": frozen, "generated": True})
            if frozen != residual_frozen():
                raise RuntimeError("generated cohort identities changed")
            result.update(frozen=frozen, build=run.build, swap_growth_bytes=guard.maximum_swap_growth_bytes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    result = {"schema": "pllm.residual_delegation.v1", "complete": False,
              "scope": "representative public compiled matrices; clear research arithmetic, NOT private serving",
              "model": {"repo": REPO, "revision": REVISION, "weight_bits": 8, "activation_bits": 8},
              "workers": [], "whole_body_savings": None,
              "unknowns": ["live masked protocol admission and authentication", "framed all-link bodies and factor manifests",
                           "factor artifact compression/distribution/publication CPU", "native widened factor kernel and snapshots",
                           "client serving CPU/RSS and provider preparation CPU/RSS", "all 96 matrix fitting coverage",
                           "whole-decoder logits/KV callback parity not run", "transport wire and source checkpoint distribution"]}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    # Discovery, hashes, imports and fitting all serialized inside same lock.
    with fresh_workers(__file__, memory_bytes=768 << 20, timeout=600) as (run, guard, _host):
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                            allow_patterns=["config.json", "model.safetensors"]))
        frozen = residual_frozen()
        checkpoint_files = {name: residual_digest(directory / name) for name in ("config.json", "model.safetensors")}
        result.update(build=run.build, frozen=frozen, checkpoint_files=checkpoint_files)
        roles = ("qkv_projection", "attention_output", "mlp_gate_up", "mlp_down") if args.role == "all" else (args.role,)
        for role in roles:
            if frozen != residual_frozen():
                raise RuntimeError("identity changed before worker")
            row = run({"frozen": frozen, "checkpoint": str(directory), "checkpoint_files": checkpoint_files,
                       "role": role, "layer": args.layer, "rank": args.rank, "seeds": args.seeds,
                       "native_i8": args.residual_policy == "native-i8"})
            result["workers"].append(row)
            save()
            if frozen != residual_frozen():
                raise RuntimeError("identity changed after worker")
        result["swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    result["complete"] = True
    result["coverage"] = {"measured_stage_groups": len(result["workers"]), "total_compiled_body_groups": 96,
                          "extrapolation": "none; unsampled layers may differ"}
    geometry = result["workers"][0]["whole_body_geometry"]
    result["whole_body_resource_forecast"] = {phase: {
        "body_macs": sum(np.prod(g["shape"]).item() * g[f"{phase}_rows"] for g in geometry),
        "all_stage_rank_client_macs": sum(args.rank * sum(g["shape"]) * g[f"{phase}_rows"] for g in geometry)}
        for phase in ("prefill", "decode")}
    for phase, value in result["whole_body_resource_forecast"].items():
        value["client_mac_fraction"] = value["all_stage_rank_client_macs"] / value["body_macs"]
        value["within_10_percent"] = value["all_stage_rank_client_macs"] * 10 <= value["body_macs"]
    result["whole_body_resource_forecast"]["raw_factors_if_every_stage_same_rank_bytes"] = sum(args.rank * sum(g["shape"]) for g in geometry)
    result["whole_body_resource_forecast"]["note"] = "geometry only, no fitted whole-body byte or native resource claim; E remains dense remote"
    save()


if __name__ == "__main__":
    main()
