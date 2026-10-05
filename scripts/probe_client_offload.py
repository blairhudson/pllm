"""Five bounded client-offload screens; no Pipeline or serving-path activation.

Public tokenizer compilation is isolated from fresh client processes. Other
screens use synthetic private-value stand-ins and pinned semantic geometry, not
checkpoint tensor values. AEAD/HE timings include their native library calls.
Results are isolated CPU/RSS/body measurements, not complete-response evidence.
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
import statistics
import subprocess
import sys
import tempfile
import time

from client_offload_tokenizer import IndexedTokenizer, compile_public_tokenizer, file_digest


MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ROOT = Path(__file__).resolve().parents[1]
# Public, fixed predeclared tokenizer cases. No private text enters the report.
TEXTS = (
    "", "Hello, world!", " leading  spaces\n\n\tand tabs", "can't I'M we've isn't",
    "1234567890 3.14159 -42", "中文分词测试。日本語 한국어", "café cafe\u0301 naïve",
    "🙂🚀👩🏽‍💻", "\x00\x01\x7f", "a" * 256, "🦀" * 32,
    "<|im_start|>user\nExplain private inference.<|im_end|>\n<|im_start|>assistant\n",
    "<think>Bounded public research.</think>", "<|endoftext|><|im_start|>",
    "Public weights, private activations: preserve every token and every boundary.",
    "def f(x):\n    return x * (x + 1)  # exact integer arithmetic\n",
)


def memory() -> dict:
    import psutil
    import resource

    return {"rss_bytes": psutil.Process().memory_info().rss,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if platform.system() == "Darwin" else 1024)}


def tokenizer_worker(mode: str, source: Path, artifact: Path, digest: str) -> dict:
    from tokenizers import Tokenizer

    before = memory()
    started = time.process_time()
    tokenizer = (Tokenizer.from_file(str(source)) if mode == "resident" else
                 IndexedTokenizer(artifact, digest))
    setup = time.process_time() - started
    after_load = memory()
    outcome = hashlib.sha256()
    count = 0
    started = time.process_time()
    for text in TEXTS:
        ids = (tokenizer.encode(text, add_special_tokens=False).ids if mode == "resident"
               else tokenizer.encode(text))
        decoded = (tokenizer.decode(ids, skip_special_tokens=False) if mode == "resident"
                   else tokenizer.decode(ids))
        outcome.update(json.dumps([ids, decoded], ensure_ascii=False).encode())
        count += len(ids)
    execution = time.process_time() - started
    result = {"mode": mode, "before": before, "after_load": after_load, "after_work": memory(),
              "setup_cpu_seconds": setup, "encode_decode_cpu_seconds": execution,
              "combined_cpu_seconds": setup + execution, "cases": len(TEXTS),
              "token_count": count, "public_result_sha256": outcome.hexdigest()}
    if mode == "indexed":
        tokenizer.close()
    return result


def geometry(config_path: Path) -> dict:
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.semantic_stages import scheduled_stage_specs, provider_owns_linear

    config = json.loads(config_path.read_bytes())
    plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
    pipeline = MaskedLinearCpu(Model.hf(MODEL, revision=REVISION),
                              quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    stages = [s for s in scheduled_stage_specs(plan, pipeline) if provider_owns_linear(s)]
    return {"model": MODEL, "revision": REVISION, "config_sha256": file_digest(config_path),
            "plan_digest_64_plus_8": plan.digest,
            "layers": config["num_hidden_layers"], "hidden": config["hidden_size"],
            "intermediate": config["intermediate_size"], "heads": config["num_attention_heads"],
            "kv_heads": config["num_key_value_heads"], "head_dim": config["head_dim"],
            "max_context": config["max_position_embeddings"],
            "stages": [{"id": s.id, "input": s.in_features, "output": s.out_features}
                       for s in stages]}


def kv_projection(g: dict, inputs: int, outputs: int = 8) -> dict:
    """No resampling/reduction change: reload one whole layer for ordinary attention."""
    if not 1 <= inputs or outputs < 1 or inputs + outputs > g["max_context"]:
        raise ValueError("KV projection exceeds the public source context")
    layers = g["layers"]
    row = 2 * g["kv_heads"] * g["head_dim"] * 4
    # Public-size tiles cap every AEAD blob at 8 MiB; append one slot per token.
    prefix_slots = math.ceil(inputs * row / (8 << 20))
    writes = layers * (row * (inputs + outputs - 1) + 28 * (prefix_slots + outputs - 1))
    reads = layers * sum(row * (inputs + step) + 28 * (step + prefix_slots)
                         for step in range(1, outputs))
    return {"input_tokens": inputs, "output_tokens": outputs, "dtype": "float32",
            "full_kv_logical_bytes": row * (inputs + outputs - 1) * layers,
            "one_layer_plaintext_bytes": row * (inputs + outputs - 1),
            "sealed_upload_body_bytes": writes, "sealed_decode_download_body_bytes": reads,
            "added_all_link_bodies": writes + reads,
            "100_mbps_download_floor_seconds": reads * 8 / 100_000_000,
            "40_mbps_upload_floor_seconds": writes * 8 / 40_000_000,
            "slot_count": layers * (prefix_slots + outputs - 1),
            "excludes": ["request framing", "HTTP/TLS", "prefill working set", "attention scratch",
                         "geometric capacity", "snapshots", "model and tokenizer memory"]}


def sealed_kv_probe(g: dict) -> dict:
    import numpy as np
    from client_offload_sealed import SealedSlots

    rng = np.random.default_rng(601)
    rows, heads, width = 32, 2, 16
    kv = rng.normal(size=(2, heads, rows, width)).astype("<f4")
    q = rng.normal(size=(heads, 1, width)).astype("<f4")

    def attention(state):
        scores = q @ state[0].swapaxes(-1, -2) / np.float32(math.sqrt(width))
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities @ state[1]

    owner = SealedSlots(b"synthetic/plan-A/response-A/layer-0/kv-f32")
    start = time.process_time()
    frame = owner.seal(kv.tobytes())
    seal_cpu = time.process_time() - start
    started = time.process_time()
    recovered = np.frombuffer(owner.open(0, frame), "<f4").reshape(kv.shape)
    open_cpu = time.process_time() - started
    expected, actual = attention(kv), attention(recovered)
    if not np.array_equal(kv.view("u4"), recovered.view("u4")) or not np.array_equal(expected, actual):
        raise RuntimeError("sealed KV changes attention inputs or outputs")
    owner.close()
    # Isolate realistic per-layer AEAD costs, without allocating a full decoder.
    samples = []
    for inputs in (16, 64, 4096):
        projection = kv_projection(g, inputs)
        size = projection["one_layer_plaintext_bytes"]
        if size > SealedSlots.MAX_BYTES:
            # Tile storage only; ordinary attention still needs the full layer.
            chunk = SealedSlots.MAX_BYTES
        else:
            chunk = size
        enc, dec = [], []
        for _ in range(3):
            owner = SealedSlots(b"synthetic/AEAD-throughput/f32")
            data = rng.bytes(chunk)
            started = time.process_time()
            frame = owner.seal(data)
            enc.append(time.process_time() - started)
            started = time.process_time()
            opened = owner.open(0, frame)
            dec.append(time.process_time() - started)
            if opened != data:
                raise RuntimeError("sealed slot differs")
            owner.close()
        samples.append(projection | {"measured_chunk_bytes": chunk,
                       "seal_cpu_seconds_median": statistics.median(enc),
                       "open_cpu_seconds_median": statistics.median(dec)})
    return {"exact_kv_and_attention": True, "toy_shape": list(kv.shape),
            "toy_seal_cpu_seconds": seal_cpu, "toy_open_cpu_seconds": open_cpu,
            "projections": samples, "provider_receives": "AEAD frames and public ordered slot IDs",
            "scope": "blob/attention oracle and per-layer crypto, no remote transport or decoder RSS",
            "decision": "memory/traffic tradeoff; short 4B response has little KV to offload"}


def mask_tape_probe(g: dict) -> dict:
    import numpy as np
    from client_offload_sealed import SealedSlots
    from pllm.runtime.preparation_protocol import PreparationRequest, seeded_ring_profile
    from pllm.runtime.transformer_client import PreparedStageRows

    stage = max(g["stages"], key=lambda s: s["input"] + s["output"])
    cases = []
    for rows in (1, 16, 64):
        control, preparation, decrypt = [], [], []
        for _ in range(3):
            profile = seeded_ring_profile(127 * 127 * stage["input"])
            request = PreparationRequest(os.urandom(16).hex(), "offload-synthetic", MODEL,
                g["plan_digest_64_plus_8"], stage["id"], "shape-only-no-weights", rows,
                stage["input"], stage["output"], 8, 8, profile.signed_output_bound,
                profile.ring, profile.modulus, profile.wire_bits, os.urandom(32))
            started = time.process_time()
            original = PreparedStageRows(request)
            r, s = original.take_masks(0, rows)
            control.append(time.process_time() - started)
            expected = r.tobytes() + s.tobytes()
            owner = SealedSlots(request.pack() + stage["id"].encode())
            started = time.process_time()
            issuer = PreparedStageRows(request)
            r2, s2 = issuer.take_masks(0, rows)
            frame = owner.seal(r2.tobytes() + s2.tobytes())
            preparation.append(time.process_time() - started)
            started = time.process_time()
            recovered = owner.open(0, frame, consume=True)
            r3 = np.frombuffer(recovered, "<u4", count=r.size).reshape(r.shape)
            s3 = np.frombuffer(recovered, "<u4", offset=r.nbytes).reshape(s.shape)
            decrypt.append(time.process_time() - started)
            if recovered != expected or not np.array_equal(r3, r) or not np.array_equal(s3, s):
                raise RuntimeError("mask tape differs from native one-use expansion")
            original.cancel()
            issuer.cancel()
            owner.close()
        cases.append({"rows": rows, "input": stage["input"], "output": stage["output"],
                      "mask_bytes": len(expected), "frame_bytes": len(frame),
                      "client_expansion_cpu_median": statistics.median(control),
                      "client_open_cpu_median": statistics.median(decrypt),
                      "preparation_expand_seal_cpu_median": statistics.median(preparation),
                      "exact_mask_parity": True})
    projections = []
    width = sum(s["input"] + s["output"] for s in g["stages"])
    for inputs in (16, 64):
        raw = (inputs + 7) * width * 4
        framed = raw + len(g["stages"]) * 8 * 28
        projections.append({"input_tokens": inputs, "output_tokens": 8,
                            "mask_plain_bytes": raw, "encrypted_tape_bytes": framed,
                            "additional_all_link_bodies": 2 * framed,
                            "100_mbps_client_download_floor_seconds": framed * 8 / 100_000_000,
                            "scope": "u32 shape bound; Preparation->Inference->Client; framing floor"})
    return {"cases": cases, "projections": projections,
            "privacy_contract": "Preparation knows masks offline; Inference has neither seed nor AEAD key",
            "scope": "existing native mask bytes plus one-use AEAD oracle; no live inventory adapter",
            "decision": "test client crypto saving against fresh incompressible tape traffic"}


def he_probes(g: dict) -> dict:
    import numpy as np
    import tenseal as ts

    started = time.process_time()
    client = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                        coeff_mod_bit_sizes=[40, 30, 30, 30, 30, 40], n_threads=1)
    client.global_scale = 2**30
    client.generate_galois_keys()
    public = client.serialize(save_public_key=True, save_secret_key=False,
                              save_galois_keys=True, save_relin_keys=True)
    provider = ts.context_from(public, n_threads=1)
    if provider.has_secret_key():
        raise RuntimeError("HE provider received secret key")
    setup = {"degree": 8192, "slots": 4096, "coeff_modulus_bits": [40, 30, 30, 30, 30, 40],
             "scale_bits": 30, "public_context_bytes": len(public), "provider_has_secret_key": False,
             "shared_setup_cpu_seconds": time.process_time() - started, "tenseal": ts.__version__}
    rng = np.random.default_rng(602)
    n, d = 8, 16
    k, v = (rng.uniform(-1, 1, (n, d)) for _ in range(2))
    q = rng.uniform(-1, 1, d)

    def upload(values):
        body = ts.ckks_vector(client, values.tolist()).serialize()
        return ts.ckks_vector_from(provider, body), len(body)

    started = time.process_time()
    uploaded = [upload(row) for row in k] + [upload(column) for column in v.T]
    kv_upload_cpu = time.process_time() - started
    keys = [item[0] for item in uploaded[:n]]
    values = [item[0] for item in uploaded[n:]]
    started = time.process_time()
    encrypted_q, q_bytes = upload(q)
    client_online_cpu = time.process_time() - started
    started = time.process_time()
    score_bodies = [row.dot(encrypted_q).serialize() for row in keys]
    provider_cpu = time.process_time() - started
    started = time.process_time()
    scores = np.array([ts.ckks_vector_from(client, body).decrypt()[0] for body in score_bodies])
    scores /= math.sqrt(d)
    probabilities = np.exp(scores - max(scores))
    probabilities /= sum(probabilities)
    encrypted_p, p_bytes = upload(probabilities)
    client_online_cpu += time.process_time() - started
    started = time.process_time()
    output_bodies = [column.dot(encrypted_p).serialize() for column in values]
    provider_cpu += time.process_time() - started
    started = time.process_time()
    output = np.array([ts.ckks_vector_from(client, body).decrypt()[0] for body in output_bodies])
    client_online_cpu += time.process_time() - started
    expected_scores = k @ q / math.sqrt(d)
    p = np.exp(expected_scores - max(expected_scores))
    p /= sum(p)
    expected = p @ v
    if not np.isfinite(output).all():
        raise RuntimeError("HE attention returned nonfinite values")
    body_min = min(q_bytes, p_bytes, *(len(b) for b in score_bodies + output_bodies))
    started = time.process_time()
    for _ in range(100):
        clear_scores = k @ q / math.sqrt(d)
        clear_p = np.exp(clear_scores - max(clear_scores))
        clear_result = (clear_p / sum(clear_p)) @ v
    clear_cpu = (time.process_time() - started) / 100
    attention = {"rows": n, "width": d, "kv_upload_bytes": sum(b for _, b in uploaded),
                 "kv_upload_cpu_seconds": kv_upload_cpu, "client_online_cpu_seconds": client_online_cpu,
                 "provider_online_cpu_seconds": provider_cpu, "clear_cpu_seconds_per_call": clear_cpu,
                 "online_body_bytes": q_bytes + p_bytes + sum(map(len, score_bodies + output_bodies)),
                 "max_absolute_error": float(np.max(np.abs(output - expected))),
                 "float32_bitwise_equal": bool(np.array_equal(output.astype("f4"), clear_result.astype("f4"))),
                 "optimistic_4b_decode_7_rows_bodies": 4 * body_min * g["layers"] * 7,
                 "projection_scope": "four minimum measured ciphertexts per layer/row; packing unimplemented",
                 "scope": "one encrypted QK/AV island; softmax remains client-local; no complete decoder"}

    # Fixed public [-8,8] profile. A private prompt never chooses this interval.
    def silu(x):
        return x / (1 + np.exp(-x))

    fit = np.polynomial.Chebyshev.interpolate(lambda u: silu(8 * u), 15, domain=[-1, 1])
    coefficients = fit.convert(kind=np.polynomial.Polynomial).coef
    grid = np.linspace(-8, 8, 65537)
    profile_error = float(np.max(np.abs(fit(grid / 8) - silu(grid))))
    x = np.linspace(-8, 8, 64)
    started = time.process_time()
    encrypted, input_bytes = upload(x / 8)
    client_cpu = time.process_time() - started
    started = time.process_time()
    result_body = encrypted.polyval(coefficients.tolist()).serialize()
    evaluation_cpu = time.process_time() - started
    started = time.process_time()
    result = np.array(ts.ckks_vector_from(client, result_body).decrypt())
    client_cpu += time.process_time() - started
    if not np.isfinite(result).all():
        raise RuntimeError("HE SiLU returned nonfinite values")
    silu_result = {"public_range": [-8, 8], "degree": 15, "checked_values": len(x),
                   "profile_grid_points": len(grid), "profile_grid_max_error": profile_error,
                   "coefficient_sha256": hashlib.sha256(coefficients.astype("<f8").tobytes()).hexdigest(),
                   "encrypted_max_error_vs_polynomial": float(np.max(np.abs(result - fit(x / 8)))),
                   "encrypted_max_error_vs_silu": float(np.max(np.abs(result - silu(x)))),
                   "float32_bitwise_equal": bool(np.array_equal(result.astype("f4"), silu(x).astype("f4"))),
                   "client_cpu_seconds": client_cpu, "provider_cpu_seconds": evaluation_cpu,
                   "input_ciphertext_bytes": input_bytes, "output_ciphertext_bytes": len(result_body),
                   "optimistic_4b_16_plus_8_bodies": math.ceil(g["intermediate"] / 4096) *
                       g["layers"] * 23 * (input_bytes + len(result_body)),
                   "projection_scope": "one freshly encrypted input/output per SIMD SiLU chunk, no gate multiply",
                   "scope": "approximate encrypted activation island, no domain certificate or model-quality evidence"}
    return {"setup": setup, "attention": attention, "silu": silu_result, "process_memory": memory()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--worker", choices=("compile", "resident", "indexed", "sealed-kv", "mask-tape", "he"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--contract-digest")
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker == "compile":
            started = time.process_time()
            result = compile_public_tokenizer(args.source, args.artifact)
            result |= {"compile_cpu_seconds": time.process_time() - started, "memory": memory()}
        elif args.worker in {"resident", "indexed"}:
            result = tokenizer_worker(args.worker, args.source, args.artifact, args.contract_digest)
        else:
            g = json.loads(args.geometry.read_bytes())
            result = {"sealed-kv": sealed_kv_probe, "mask-tape": mask_tape_probe, "he": he_probes}[args.worker](g)
        print(json.dumps(result))
        return
    if not args.output or args.output.exists():
        parser.error("a new --output path is required")
    from huggingface_hub import hf_hub_download
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    host = host_memory()
    if host.available - host.reserve < 2 * GiB or host.disk_free < GiB:
        raise RuntimeError("bounded offload screens need 2 GiB admitted RAM and 1 GiB disk")
    source, config = (Path(hf_hub_download(MODEL, name, revision=REVISION, local_files_only=True))
                      for name in ("tokenizer.json", "config.json"))
    active = []

    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    report = {"schema": "pllm.client_offload_five.v1", "status": "running", "scope": __doc__,
              "measured_at": datetime.now(timezone.utc).isoformat(), "geometry": geometry(config),
              "python": platform.python_version(), "platform": platform.platform(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "driver_sha256": file_digest(Path(__file__)),
              "support_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                 ("client_offload_tokenizer.py", "client_offload_sealed.py")}}

    def checkpoint(stream):
        stream.seek(0)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.truncate()
        stream.flush()

    guard.start()
    try:
        with args.output.open("x") as output, tempfile.TemporaryDirectory(
                prefix="pllm-client-offload-", dir=args.scratch_dir) as temporary:
            root = Path(temporary)
            artifact = root / "public-tokenizer"
            geometry_path = root / "geometry.json"
            geometry_path.write_text(json.dumps(report["geometry"]))

            def child(mode, *, digest=""):
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                command = [sys.executable, __file__, "--worker", mode, "--source", str(source),
                           "--artifact", str(artifact), "--geometry", str(geometry_path),
                           "--contract-digest", digest]
                process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                             OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
                active.append(process)
                try:
                    stdout, stderr = process.communicate(timeout=180)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    active.remove(process)
                if process.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                return json.loads(stdout)

            checkpoint(output)
            report["tokenizer_compilation"] = child("compile")
            digest = file_digest(artifact / "contract.json")
            samples = []
            for mode in ("resident", "indexed", "indexed", "resident"):
                print(f"tokenizer {mode}", flush=True)
                samples.append(child(mode, digest=digest))
            if len({s["public_result_sha256"] for s in samples}) != 1:
                raise RuntimeError("tokenizer offload fails exact IDs/text parity")
            report["tokenizer"] = {"parity": True, "samples": samples,
                "scope": "fresh-process public BPE setup/encode/decode; trusted precompiled artifact digest required"}
            checkpoint(output)
            for name in ("sealed-kv", "mask-tape", "he"):
                print(name, flush=True)
                report[name] = child(name)
                checkpoint(output)
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            report["maximum_host_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
            report["status"] = "complete"
            checkpoint(output)
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
