"""Five fresh client-work offload screens, with pinned 4B geometry and public weights.

Each worker is a bounded, in-process reference with synthetic private-value stand-ins.
Role phase CPU and exact query/reply bodies are isolated; RSS covers the entire worker,
not a separately deployed client. No Pipeline choice, live protocol or quality claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

from client_offload_numeric import (
    PublicRope, digest, export_rope, export_sqrt, folding_probe, rope_contract,
    sqrt_index, sqrt_restore,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
INPUTS, OUTPUTS = 16, 8


def memory():
    import psutil
    import resource
    return {"rss_bytes": psutil.Process().memory_info().rss,
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if sys.platform == "darwin" else 1024)}


def geometry(snapshot: Path):
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.semantic_stages import scheduled_stage_specs, provider_owns_linear

    config = json.loads((snapshot / "config.json").read_bytes())
    plan = lower_model(config, batch=1, max_input_tokens=INPUTS, max_new_tokens=OUTPUTS)
    pipeline = MaskedLinearCpu(Model.hf(MODEL, revision=REVISION),
                              quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    graph = plan.to_dict()
    stages = [s for s in scheduled_stage_specs(plan, pipeline) if provider_owns_linear(s)]
    roots = sum((1 if phase == "prefill" else OUTPUTS - 1) * math.prod(op["output_shape"][:-1])
                for phase in ("prefill", "decode") for op in graph[phase]["operations"]
                if op["operator"] == "rms_norm")
    rope = [op for op in graph["prefill"]["operations"] if op["operator"] == "rotary_embedding"]
    if not config["tie_word_embeddings"] or any(op["attributes"] != rope[0]["attributes"] for op in rope):
        raise ValueError("round-two probe needs one public rotary contract and a tied boundary")
    return {"model": MODEL, "revision": REVISION, "config_sha256": digest(snapshot / "config.json"),
            "plan_digest": plan.digest, "input_tokens": INPUTS, "output_tokens": OUTPUTS,
            "executed_rows": INPUTS + OUTPUTS - 1, "hidden": config["hidden_size"],
            "vocabulary": config["vocab_size"], "layers": config["num_hidden_layers"],
            "heads": config["num_attention_heads"], "kv_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"], "rmsnorm_sqrt_elements": roots,
            "rope_contract": rope_contract(plan.digest, rope[0]["attributes"], INPUTS + OUTPUTS),
            "stages": [{"id": s.id, "input": s.in_features, "output": s.out_features} for s in stages]}


def publish(snapshot: Path, root: Path, g: dict):
    """All input is public. Stream quantization; no complete float embedding array."""
    import numpy as np
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore

    started = time.process_time()
    store = SafeTensorStore(snapshot)
    key = "model.embed_tokens.weight"
    n, d = g["vocabulary"], g["hidden"]
    if store.tensor_shape(key) != (n, d) or store.tensor_dtype(key) != "BF16":
        raise ValueError("public boundary source shape/dtype differs")
    scales = np.empty(n, dtype="<f4")
    source_values = hashlib.sha256()
    selections = (slice(i, min(i + 512, n)) for i in range(0, n, 512))
    with (root / "head.i8").open("xb") as head, (root / "embedding.records").open("xb") as records:
        offset = 0
        for chunk in store.iter_slices(key, selections):
            source_values.update(chunk.astype("<f4", copy=False).tobytes())
            q = quantize_weight_per_row(chunk, bits=8)
            head.write(q.values.tobytes())
            scales[offset:offset + len(chunk)] = q.scales
            packed = np.empty((len(chunk), d + 4), dtype=np.uint8)
            packed[:, :d] = q.values.view(np.uint8)
            packed[:, d:] = np.frombuffer(q.scales.astype("<f4").tobytes(), np.uint8).reshape(-1, 4)
            records.write(packed.tobytes())
            offset += len(chunk)
    scales.tofile(root / "head.scales")
    for source, filename in (("model.layers.0.input_layernorm.weight", "gamma.f32"),
                             ("model.layers.0.self_attn.q_proj.weight", "q_weight.f32")):
        value = store.get(source)
        value.astype("<f4", copy=False).tofile(root / filename)
    rope_sha = export_rope(root / "rope.bin", g["rope_contract"])
    sqrt_sha = export_sqrt(root / "sqrt.f32")
    files = {name: {"sha256": digest(root / name), "bytes": (root / name).stat().st_size} for name in
             ("head.i8", "head.scales", "embedding.records", "gamma.f32", "q_weight.f32", "rope.bin", "sqrt.f32")}
    assert files["rope.bin"]["sha256"] == rope_sha and files["sqrt.f32"]["sha256"] == sqrt_sha
    return {"files": files, "public_embedding_f32_sha256": source_values.hexdigest(),
            "q_weight_shape": list(store.tensor_shape("model.layers.0.self_attn.q_proj.weight")),
            "public_compiler_cpu_seconds": time.process_time() - started, "memory": memory(),
            "scope": "public cached-source compilation and derived artifacts; no model download"}


def rope_probe(g: dict, root: Path, public: dict):
    import numpy as np
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime

    start = time.process_time()
    candidate = PublicRope(root / "rope.bin", public["files"]["rope.bin"]["sha256"], g["rope_contract"])
    import_cpu = time.process_time() - start
    rng = np.random.default_rng(61021)
    samples = []
    for mode in ("ordinary", "precompiled", "precompiled", "ordinary"):
        start = time.process_time()
        checked = 0
        # Synthetic Q and K, exact semantic dimensions and ordinary prefill/decode positions.
        for first, rows in [(0, INPUTS), *((INPUTS + step, 1) for step in range(OUTPUTS - 1))]:
            positions = np.arange(first, first + rows, dtype=np.int64)
            for _ in range(g["layers"]):
                for heads in (g["heads"], g["kv_heads"]):
                    x = rng.standard_normal((1, heads, rows, g["head_dim"]), dtype=np.float32)
                    # Oracle and PRNG are excluded from timed operation samples below.
                    expected = SemanticDecoderRuntime._rotary(x, positions, g["rope_contract"]["attributes"])
                    actual = candidate.apply(x, positions)
                    if not np.array_equal(expected.view("u4"), actual.view("u4")):
                        raise RuntimeError("public rotary artifact changes private activation bits")
                    checked += x.size
        validation_cpu = time.process_time() - start
        # Repeat identical fixed tensors for measured work; no oracle in this window.
        tensors = [(np.ones((1, h, r, g["head_dim"]), np.float32), np.arange(p, p + r, dtype=np.int64))
                   for p, r in [(0, INPUTS), *((INPUTS + s, 1) for s in range(OUTPUTS - 1))]
                   for h in (g["heads"], g["kv_heads"])]
        start = time.process_time()
        for _ in range(5):
            for _ in range(g["layers"]):
                for x, positions in tensors:
                    if mode == "ordinary":
                        SemanticDecoderRuntime._rotary(x, positions, g["rope_contract"]["attributes"])
                    else:
                        candidate.apply(x, positions)
        samples.append({"mode": mode, "response_shaped_rotary_cpu_seconds": (time.process_time() - start) / 5,
                        "bitwise_checked_elements": checked, "separate_validation_cpu_seconds": validation_cpu})
    return {"samples": samples, "client_import_cpu_seconds": import_cpu,
            "additional_public_artifact_bytes": public["files"]["rope.bin"]["bytes"],
            "exact_operator_bits": True, "privacy": "public tables fetched in full; positions and activations stay local",
            "scope": "all response-shaped Q/K rotary operators only; no decoder or provider transport"}


def private_lookup(servers, descriptor, index):
    from pllm import _native
    start = time.process_time()
    a, b, decoder = _native.private_page_issue(*descriptor, index)
    client = time.process_time() - start
    replies, workers = [], []
    for server, query in zip(servers, (a, b), strict=True):
        start = time.process_time()
        replies.append(server.evaluate(query))
        workers.append(time.process_time() - start)
    start = time.process_time()
    result = decoder.decode(*replies)
    client += time.process_time() - start
    return result, {"client_cpu_seconds": client, "worker_cpu_seconds": workers,
                    "query_and_reply_body_bytes": sum(map(len, (a, b, *replies)))}


def sqrt_probe(g: dict, root: Path, public: dict):
    import numpy as np
    from pllm import _native

    data = (root / "sqrt.f32").read_bytes()
    if hashlib.sha256(data).hexdigest() != public["files"]["sqrt.f32"]["sha256"]:
        raise ValueError("public sqrt compiler identity differs")
    # Pack 64 results per public page, hiding the page and retaining the offset locally.
    servers = [_native.PrivatePageServer(data, 64 * 4, party) for party in (0, 1)]
    descriptor = servers[0].descriptor()
    if servers[1].descriptor() != descriptor:
        raise ValueError("private sqrt worker descriptors differ")
    table = np.frombuffer(data, "<f4")
    rng = np.random.default_rng(61022)
    # Independent broad bit-pattern oracle before paying for PIR queries.
    words = rng.integers(0x00800000, 0x7f800000, 10000, dtype=np.uint32)
    values = np.concatenate((words.view(np.float32), np.asarray([
        np.finfo(np.float32).tiny, np.finfo(np.float32).max, 1e-6, 1, 2, 3, 4, 0.25], np.float32)))
    for x in values:
        index, power = sqrt_index(x)
        actual = sqrt_restore(table[index].tobytes(), power)
        if actual.view("u4") != np.sqrt(x).view("u4"):
            raise RuntimeError("normalized public sqrt table changes IEEE float32 result")
    measurements = []
    for x in values[-8:]:
        start = time.process_time()
        index, power = sqrt_index(x)
        client = time.process_time() - start
        page, measured = private_lookup(servers, descriptor, index // 64)
        start = time.process_time()
        actual = sqrt_restore(page[index % 64 * 4:index % 64 * 4 + 4], power)
        measured["client_cpu_seconds"] += client + time.process_time() - start
        if actual.view("u4") != np.sqrt(x).view("u4"):
            raise RuntimeError("private sqrt lookup differs from clear float32 sqrt")
        measurements.append(measured)
    control = np.ones(g["rmsnorm_sqrt_elements"], dtype=np.float32)
    start = time.process_time()
    for _ in range(100):
        np.sqrt(control)
    clear_cpu = (time.process_time() - start) / 100
    body = measurements[0]["query_and_reply_body_bytes"]
    if len({m["query_and_reply_body_bytes"] for m in measurements}) != 1:
        raise RuntimeError("private sqrt query sizes depend on secret value")
    return {"public_table_bytes_each_worker": len(data), "page_bytes": 256,
            "exact_float32_cases": len(values), "exact_private_queries": len(measurements),
            "samples": measurements, "clear_response_sqrt_cpu_seconds": clear_cpu,
            "projected_16_plus_8_body_bytes": body * g["rmsnorm_sqrt_elements"],
            "projected_lookup_count": g["rmsnorm_sqrt_elements"],
            "projection_scope": "one fresh page query per RMSNorm scalar; no normalization sums or divisions",
            "native_server_query_capacity": 4096,
            "requires_additional_independent_server_instances": g["rmsnorm_sqrt_elements"] > 4096,
            "privacy": "fresh two-server DPF keys; exponent and page offset remain client-local",
            "decision": "cost gate only; no full-normalization or serving activation"}


def boundary_probe(g: dict, root: Path, public: dict):
    import numpy as np
    from pllm import _native
    from pllm.native import PagedGEMM
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

    n, d = g["vocabulary"], g["hidden"]
    data = (root / "embedding.records").read_bytes()
    if hashlib.sha256(data).hexdigest() != public["files"]["embedding.records"]["sha256"]:
        raise ValueError("public scaled embedding identity differs")
    servers = [_native.PrivatePageServer(data, d + 4, party) for party in (0, 1)]
    descriptor = servers[0].descriptor()
    if servers[1].descriptor() != descriptor:
        raise ValueError("scaled token worker descriptors differ")
    table = np.frombuffer(data, np.uint8).reshape(n, d + 4)
    scales = np.fromfile(root / "head.scales", "<f4")
    if digest(root / "head.scales") != public["files"]["head.scales"]["sha256"]:
        raise ValueError("head scales differ from public compiler identity")
    # Independent native snapshots stand in for the two online workers.
    kernels = [PagedGEMM.from_raw(root / "head.i8", public["files"]["head.i8"]["sha256"], (n, d))
               for _ in range(2)]
    rng = np.random.default_rng(61023)
    lookups, heads = [], []
    try:
        for index in (0, n // 2, n - 1):
            record, measurement = private_lookup(servers, descriptor, index)
            if record != table[index].tobytes():
                raise RuntimeError("private embedding code/scale bytes differ")
            recovered = np.frombuffer(record[:d], np.int8).astype(np.float32) * np.frombuffer(record[d:], "<f4")[0]
            expected = table[index, :d].view(np.int8).astype(np.float32) * scales[index]
            if not np.array_equal(recovered.view("u4"), expected.view("u4")):
                raise RuntimeError("private scaled embedding changes float32 bits")
            lookups.append(measurement)
        for _ in range(3):
            x = rng.standard_normal((1, d), dtype=np.float32)
            quantized = quantize_activation_per_row(x, bits=8)
            start = time.process_time()
            clear = kernels[0].clear(quantized.values)
            clear_logits = dequantize_matmul(clear, quantized.scales, scales)
            clear_cpu = time.process_time() - start
            start = time.process_time()
            first = np.frombuffer(os.urandom(d * 4), "<u4").reshape(1, d)
            second = np.subtract(quantized.values.astype(np.uint32), first, dtype=np.uint32)
            client_cpu = time.process_time() - start
            outputs, worker_cpu = [], []
            for kernel, share in zip(kernels, (first, second), strict=True):
                start = time.process_time()
                outputs.append(kernel.wrap32(share))
                worker_cpu.append(time.process_time() - start)
            start = time.process_time()
            integer = np.add(outputs[0], outputs[1], dtype=np.uint32).view(np.int32)
            logits = dequantize_matmul(integer, quantized.scales, scales)
            client_cpu += time.process_time() - start
            if not np.array_equal(clear, integer) or not np.array_equal(clear_logits.view("u4"), logits.view("u4")):
                raise RuntimeError("two-worker head changes exact integers or float32 logits")
            heads.append({"client_cpu_seconds": client_cpu, "worker_cpu_seconds": worker_cpu,
                          "local_paged_head_cpu_seconds": clear_cpu,
                          "arithmetic_body_bytes": 2 * (d + n) * 4})
    finally:
        for kernel in kernels:
            kernel.close()
    body = (lookups[0]["query_and_reply_body_bytes"] * g["executed_rows"]
            + heads[0]["arithmetic_body_bytes"] * OUTPUTS)
    return {"embedding_samples": lookups, "head_samples": heads,
            "exact_scaled_embedding_bits": True, "exact_head_integer_and_logit_bits": True,
            "source_embedding_f32_sha256": public["public_embedding_f32_sha256"],
            "head_i8_sha256": public["files"]["head.i8"]["sha256"],
            "client_weight_artifact_bytes_potentially_removed": n * d,
            "client_scales_retained_bytes": n * 4,
            "public_embedding_record_bytes_each_worker": len(data),
            "additional_head_snapshot_disk_bytes_each_worker": n * d,
            "projected_added_online_16_plus_8_bodies": body,
            "projected_two_worker_embedding_scan_bytes": 2 * len(data) * g["executed_rows"],
            "privacy": "two non-colluding online workers; DPF token keys and fresh exact-ring head shares",
            "scope": "real quantized public boundary, synthetic token/hidden queries; no body, network or generation",
            "client_peak_memory_saving_bytes": None,
            "decision": "requires new tied-boundary role graph and full-response compute/transport admission"}


def absmax_probe(g: dict):
    executable = ROOT / "target/release/examples/client_absmax_probe"
    completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=180, check=True)
    result = json.loads(completed.stdout)
    # Counts include every grouped remote stage plus the local head quantizer.
    rows = g["executed_rows"]
    widths = [stage["input"] for stage in g["stages"]]
    input_bits = 31 * (rows * sum(widths) + OUTPUTS * g["hidden"])
    output_bits = 31 * (rows * len(widths) + OUTPUTS)
    comparisons = rows * sum(width - 1 for width in widths) + OUTPUTS * (g["hidden"] - 1)
    result["projection"] = {"online_input_and_output_label_bytes": 16 * (input_bits + output_bits),
                            "one_use_half_gate_ciphertext_bytes": comparisons * 31 * 4 * 32,
                            "uncompressed_helper_to_client_encoding_bytes": 32 * (input_bits + output_bits),
                            "scope": "absmax only; excludes circuit instructions, transport, division and quantization",
                            "largest_stage_width": max(widths), "reference_maximum_width": 2560,
                            "complete_tensor_schedule_admitted": False}
    result["privacy"] = "offline trusted helper issues one-use encodings; evaluator sees labels, client decodes max"
    result["scope"] = "native one-use finite-f32 magnitude circuit; wall time, no network or helper transport"
    return result


def folding(g: dict, root: Path, public: dict):
    import numpy as np
    for name in ("q_weight.f32", "gamma.f32"):
        if digest(root / name) != public["files"][name]["sha256"]:
            raise ValueError("public folding source differs")
    weights = np.fromfile(root / "q_weight.f32", "<f4").reshape(public["q_weight_shape"])
    gamma = np.fromfile(root / "gamma.f32", "<f4")
    inputs = np.random.default_rng(61024).standard_normal((23, g["hidden"]), dtype=np.float32)
    result = folding_probe(weights, gamma, inputs)
    result["source"] = "pinned Qwen3-4B layer-0 input norm and Q projection; synthetic normalized rows"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--worker", choices=("publish", "rope", "sqrt", "boundary", "absmax", "folding"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.worker:
        manifest = json.loads(args.manifest.read_bytes())
        root, g = args.manifest.parent, manifest["geometry"]
        if args.worker == "publish":
            result = publish(args.snapshot, root, g)
        elif args.worker == "absmax":
            result = absmax_probe(g)
        else:
            result = {"rope": rope_probe, "sqrt": sqrt_probe, "boundary": boundary_probe,
                      "folding": folding}[args.worker](g, root, manifest["public"])
        if args.worker == "absmax":
            result["python_launcher_memory"] = memory()
            result["native_process_peak_rss_bytes"] = None
        else:
            result["worker_memory_all_roles"] = memory()
        print(json.dumps(result))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    if not (ROOT / "target/release/examples/client_absmax_probe").exists():
        parser.error("build cargo --release -p pllm-garble --example client_absmax_probe first")
    from huggingface_hub import hf_hub_download
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    host = host_memory()
    if host.available - host.reserve < 4 * GiB or host.disk_free < 4 * GiB:
        raise RuntimeError("round-two references require 4 GiB physical headroom and disk")
    config = Path(hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True))
    snapshot = config.parent
    active = []

    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    report = {"schema": "pllm.client_offload_round2.v1", "status": "running", "scope": __doc__,
              "measured_at": datetime.now(timezone.utc).isoformat(), "geometry": geometry(snapshot),
              "platform": platform.platform(), "python": platform.python_version(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
              "implementation_sha256": {path: digest(ROOT / path) for path in (
                  "scripts/probe_client_offload_round2.py", "scripts/client_offload_numeric.py",
                  "crates/pllm-garble/examples/client_absmax_probe.rs",
                  "crates/pllm-garble/src/boolean.rs", "crates/pllm-garble/src/private_pages.rs",
                  "python/pllm/runtime/semantic_executor.py", "python/pllm/runtime/quantization.py")},
              "native_extension_sha256": None, "host_before": asdict(host)}
    from pllm import _native
    report["native_extension_sha256"] = digest(Path(_native.__file__))
    report["absmax_executable_sha256"] = digest(ROOT / "target/release/examples/client_absmax_probe")

    def checkpoint(output):
        output.seek(0)
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.truncate()
        output.flush()

    guard.start()
    try:
        with args.output.open("x") as output, tempfile.TemporaryDirectory(
                prefix="pllm-offload-round2-", dir=args.scratch_dir) as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest = {"geometry": report["geometry"]}

            def child(mode):
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                manifest_path.write_text(json.dumps(manifest))
                command = [sys.executable, __file__, "--worker", mode, "--manifest", str(manifest_path),
                           "--snapshot", str(snapshot)]
                process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=dict(os.environ, HF_HUB_OFFLINE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                             VECLIB_MAXIMUM_THREADS="1"), start_new_session=True)
                active.append(process)
                try:
                    stdout, stderr = process.communicate(timeout=240)
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    active.remove(process)
                if process.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                return json.loads(stdout)

            checkpoint(output)
            try:
                print("public compiler", flush=True)
                report["public"] = manifest["public"] = child("publish")
                checkpoint(output)
                for name in ("rope", "sqrt", "boundary", "absmax", "folding"):
                    print(name, flush=True)
                    report[name] = child(name)
                    checkpoint(output)
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                report["status"] = "complete"
            except BaseException as exc:
                report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                report["failure_type"] = type(exc).__name__
                raise
            finally:
                report["maximum_host_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
                checkpoint(output)
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
