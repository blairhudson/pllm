"""Bounded synthetic shared-depth decoder; not a quality or hardware benchmark.

Independent full-sequence oracle versus incremental, depth-indexed KV execution.
No upstream code is imported or executed. Default run performs no network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
BASE = f"https://huggingface.co/tomg-group-umd/huginn-0125/resolve/{REVISION}/"
SOURCES = {
    "config.json": "e9fe79df06a783ca33a76038c59a715b6a62f15c4a5b17b9681e697fea46c79c",
    "raven_modeling_minimal.py": "a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa",
    "README.md": "66a7c428fc8aa7dd63ec0682009c1eee099fe86d251e704918bef642f4e755fa",
    "model.safetensors.index.json": "a28061573e97ba3645b482c11d82ba903b40942de0113472b30afb5a3cf4c1eb",
}


def model(tied=True, width=8, depth=4, vocab=19, seed=61):
    if not (1 <= width <= 64 and 1 <= depth <= 32 and 2 <= vocab <= 256):
        raise ValueError("bounded geometry exceeded")
    rng = np.random.default_rng(seed)
    def block():
        return {k: rng.normal(0, .16, (width, width)) for k in ("q", "k", "v", "o", "up", "down")}
    core = block()
    return {"width": width, "depth": depth, "vocab": vocab,
            "embedding": rng.normal(0, .3, (vocab, width)),
            "head": rng.normal(0, .3, (width, vocab)),
            "norms": 1 + rng.normal(0, .03, (depth, 2, width)),
            "adapters": rng.normal(0, .02, (depth, width)),
            "blocks": [core] * depth if tied else [core] + [block() for _ in range(depth - 1)]}


def expanded(m):
    """Exact materialized unrolling of model DEFINED with shared weights."""
    return {**m, "blocks": [{k: v.copy() for k, v in b.items()} for b in m["blocks"]]}


def posthoc(m, average):
    core = {k: np.mean([b[k] for b in m["blocks"]], axis=0) if average else m["blocks"][0][k].copy()
            for k in m["blocks"][0]}
    return {**m, "blocks": [core] * m["depth"]}


def tensors(m):
    """Deduplicate only explicit object ownership; equal bytes alone do not tie."""
    result = {k: m[k] for k in ("embedding", "head", "norms", "adapters")}
    seen = set()
    for i, b in enumerate(m["blocks"]):
        for k, a in b.items():
            if id(a) not in seen:
                result[f"block.{i}.{k}"] = a
                seen.add(id(a))
    return result


def artifact(m):
    arrays = tensors(m)
    manifest = {"format": "synthetic-shared-core-v1", "dtype": "<f8",
                "width": m["width"], "depth": m["depth"], "vocab": m["vocab"],
                "core_owners": [next(j for j, b in enumerate(m["blocks"]) if b is core) for core in m["blocks"]],
                "tokenizer": {"type": "integer-id-only", "size": m["vocab"]},
                "tensors": {k: list(v.shape) for k, v in arrays.items()}}
    header = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    payload = b"".join(np.asarray(v, dtype="<f8").tobytes() for v in arrays.values())
    execution_source = Path(__file__).read_bytes()
    complete = (b"SCORE001" + struct.pack("<QQ", len(header), len(execution_source))
                + header + execution_source + payload)
    return {"parameter_payload_bytes": len(payload), "manifest_and_framing_bytes": 16 + len(header),
            "execution_source_bytes": len(execution_source), "extra_source_length_framing_bytes": 8,
            "preinstalled_dependencies": "Python and NumPy; their provisioning not included",
            "complete_download_bytes": len(complete), "sha256": hashlib.sha256(complete).hexdigest(),
            "distinct_parameters": len(payload) // 8}


def oracle(m, tokens):
    """Full-sequence causal computation; no decoder helpers or cache reuse."""
    tokens = np.asarray(tokens)
    if (tokens.ndim != 1 or not 1 <= len(tokens) <= 512
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0) or np.any(tokens >= m["vocab"])):
        raise ValueError("invalid tokens or bounded oracle horizon")
    x = m["embedding"][tokens].copy()
    n = len(tokens)
    for i, b in enumerate(m["blocks"]):
        z = x / np.sqrt(np.sum(x*x, axis=1, keepdims=True) / m["width"] + 1e-6) * m["norms"][i, 0]
        q, k, v = (z @ b[name] for name in ("q", "k", "v"))
        scores = q @ k.T / math.sqrt(m["width"])
        scores[np.triu_indices(n, 1)] = -np.inf
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        x = x + ((exp / exp.sum(axis=1, keepdims=True)) @ v) @ b["o"]
        z = x / np.sqrt(np.sum(x*x, axis=1, keepdims=True) / m["width"] + 1e-6) * m["norms"][i, 1]
        x = x + np.tanh(z @ b["up"]) @ b["down"] + z * m["adapters"][i]
    return x @ m["head"]


class Decoder:
    def __init__(self, m, horizon=128):
        if not 1 <= horizon <= 512:
            raise ValueError("bounded horizon exceeded")
        self.m, self.position, self.calls = m, 0, 0
        self.keys = np.zeros((m["depth"], horizon, m["width"]))
        self.values = np.zeros_like(self.keys)

    def step(self, token):
        m, p = self.m, self.position
        if not isinstance(token, (int, np.integer)) or not 0 <= token < m["vocab"] or p >= self.keys.shape[1]:
            raise ValueError("invalid token or horizon")
        x = m["embedding"][token].copy()
        for i, b in enumerate(m["blocks"]):
            z = x * (1 / np.sqrt(np.mean(x*x) + 1e-6)) * m["norms"][i, 0]
            q = z @ b["q"]
            self.keys[i, p] = z @ b["k"]
            self.values[i, p] = z @ b["v"]
            score = self.keys[i, :p+1] @ q / math.sqrt(m["width"])
            prob = np.exp(score - score.max())
            prob /= prob.sum()
            x += (prob @ self.values[i, :p+1]) @ b["o"]
            z = x * (1 / np.sqrt(np.mean(x*x) + 1e-6)) * m["norms"][i, 1]
            x += np.tanh(z @ b["up"]) @ b["down"] + z * m["adapters"][i]
            self.calls += 1
        self.position += 1
        return x @ m["head"]


def resources(m, tokens):
    w, d = m["width"], m["depth"]
    payload = artifact(m)["parameter_payload_bytes"]
    core = 6*w*w*8
    extra = (2*w*m["vocab"] + 3*d*w)*8
    # All matrices touched at every depth even if same matrices remain resident.
    return {"resident_parameter_payload_bytes": payload,
            "kv_live_bytes": 2*d*tokens*w*8,
            "block_invocations": d*tokens,
            "dense_matmul_flops": tokens*(12*d*w*w + 2*w*m["vocab"]),
            "attention_matmul_flops": 4*d*w*tokens*(tokens+1)//2,
            "matrix_touch_bytes_no_cache_assumption": d*tokens*6*w*w*8,
            "paged_core_window_bytes": 6*w*w*8,
            "paged_parameter_window_plus_depth_extras_and_head_bytes": core + extra,
            "resident_parameters_plus_live_kv_lower_bound_bytes": payload + 2*d*tokens*w*8,
            "paged_window_plus_live_kv_lower_bound_bytes": core + extra + 2*d*tokens*w*8,
            "paging_contract": "one core resident, all extras/head resident; untied may reload next block; OS page-cache/RSS not bounded here",
            "rss_measured": False,
            "exclusions": "Python/BLAS/runtime, allocator, activation temporaries, page cache; not RSS"}


def cold_case(prompt, output, mbps, local_tps, prepared_tps, prepared_base, prepared_per_token, download):
    """Scenario assumptions, not measured hardware or PLLM graph performance."""
    total = prompt + output
    local_compute = total / local_tps
    remote_compute = total / prepared_tps
    graph_bytes = prepared_base + prepared_per_token * total
    local = download / mbps + local_compute
    prepared = graph_bytes / mbps + remote_compute
    delta_slope = prepared_per_token / mbps + 1/prepared_tps - 1/local_tps
    crossing = (download - prepared_base) / mbps / delta_slope if delta_slope else None
    return {"prompt": prompt, "output": output, "network_bytes_per_second": mbps,
            "local_effective_tokens_per_second": local_tps, "prepared_effective_tokens_per_second": prepared_tps,
            "local_complete_download_bytes": download, "prepared_complete_bytes": graph_bytes,
            "local_seconds": local, "prepared_seconds": prepared,
            "local_faster": local < prepared,
            "equal_cost_total_tokens_linear_model": crossing,
            "nonnegative_break_even_exists": crossing is not None and crossing >= 0,
            "local_advantage_slope_seconds_per_token": delta_slope,
            "prepared_base_bytes": prepared_base, "prepared_bytes_per_input_or_output_token": prepared_per_token}


def verify_sources():
    for name, expected in SOURCES.items():
        with urllib.request.urlopen(BASE + name, timeout=30) as response:
            data = response.read(1 << 20)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
    inventory_url = f"https://huggingface.co/api/models/tomg-group-umd/huginn-0125/tree/{REVISION}?recursive=false&expand=false"
    with urllib.request.urlopen(inventory_url, timeout=30) as response:
        inventory = response.read(1 << 20)
    if hashlib.sha256(inventory).hexdigest() != "ecd964fc17b510c9b57bfcf27128535ce581f35b0dba1d474be2b94b3d8d4d87":
        raise ValueError("source inventory hash mismatch")


def evidence():
    tied, untied = model(), model(False)
    tokens = np.array([1, 7, 2, 4, 8, 3, 9, 6])
    dec = Decoder(tied, len(tokens))
    incremental = np.array([dec.step(t) for t in tokens])
    expected = oracle(tied, tokens)
    np.testing.assert_allclose(incremental, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(expected, oracle(expanded(tied), tokens))
    changes = {name: float(np.max(np.abs(oracle(untied, tokens)-oracle(posthoc(untied, avg), tokens))))
               for name, avg in (("average", True), ("replace_first", False))}
    # Compact boundary is lossy unless model contract already fixes representation.
    hidden = tied["embedding"][tokens].astype(np.float32)
    scale = float(np.max(np.abs(hidden)) / 127)
    quant = np.rint(hidden / scale).astype(np.int8)
    boundary = {"token_id_bytes_u16": len(tokens)*2, "fp32_hidden_bytes": hidden.nbytes,
                "int8_hidden_plus_f32_scale_bytes": quant.nbytes+4,
                "max_abs_reconstruction_error": float(np.max(np.abs(hidden-quant.astype(np.float32)*scale))),
                "lossless_float_boundary": False, "fully_local_inference_boundary_bytes": 0,
                "privacy": "int8/latent/token IDs are not encryption; no prompt leaves fully local execution"}
    # All checkpoint estimates are scalar arithmetic; no weights allocated/downloaded.
    actual_checkpoint_download = 15648442384
    scenarios = [cold_case(p, o, bw, lt, 100, 64_000_000, 8_000_000, actual_checkpoint_download)
                 for p, o in ((32, 8), (512, 128), (4096, 512))
                 for bw in (12_500_000, 125_000_000) for lt in (2, 20)]
    return {"schema": "cold-shared-core-synthetic-v1", "synthetic_only": True,
            "claim": "exact tied definition preserves unrolling; posthoc tying does not preserve untied outputs",
            "geometry": {"width": 8, "depth": 4, "vocab": 19, "tokens": tokens.tolist(), "seed": 61},
            "oracle_max_abs_error": float(np.max(np.abs(incremental-expected))),
            "defined_tied_unrolling_bit_exact": True, "posthoc_output_change": changes,
            "tied_artifact": artifact(tied), "untied_artifact": artifact(untied),
            "expanded_tied_artifact": artifact(expanded(tied)),
            "tied_resources": resources(tied, len(tokens)), "untied_resources": resources(untied, len(tokens)),
            "allocated_kv_bytes": dec.keys.nbytes + dec.values.nbytes,
            "boundary": boundary,
            "scenario_status": "unmeasured throughput and prepared graph assumptions; uploaded complete runtime-file download counted, dependencies preinstalled",
            "cold_scenarios": scenarios,
            "huginn_estimates": {"uploaded_tensor_bytes": 15645600384,
                "complete_runtime_file_download_bytes": actual_checkpoint_download,
                "safetensors_files_including_headers_bytes": 15645609096,
                "runtime_metadata_tokenizer_code_bytes": 2833288,
                "uploaded_parameter_count": 3911400096, "embedding_duplicate_parameters": 65536*5280,
                "deduplicated_parameters_estimate": 3911400096-65536*5280,
                "deduplicated_bf16_payload_bytes_estimate": (3911400096-65536*5280)*2,
                "hypothetical_4bit_payload_lower_bound_bytes": (3911400096-65536*5280)//2,
                "full_kv_bf16_at_4096_tokens_32_loops_bytes": 2*(2+4*32+2)*4096*5280*2,
                "full_kv_bf16_at_4096_tokens_8_loops_bytes": 2*(2+4*8+2)*4096*5280*2,
                "warning": "4bit excludes scale/zero-point/pack overhead and is NOT a verified compatible checkpoint"},
            "provenance": {"checkpoint_revision": REVISION, "source_sha256": SOURCES,
                "source_base_url": BASE, "paper": "https://arxiv.org/html/2502.05171v2",
                "runtime_file_inventory_url": f"https://huggingface.co/api/models/tomg-group-umd/huginn-0125/tree/{REVISION}?recursive=false&expand=false",
                "runtime_file_inventory_response_sha256": "ecd964fc17b510c9b57bfcf27128535ce581f35b0dba1d474be2b94b3d8d4d87",
                "universal_transformers": "https://arxiv.org/abs/1807.03819v3",
                "implementation": "independent NumPy fixture; not Huginn architecture reproduction"},
            "replay_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), ROOT/"tests/test_cold_shared_core.py")},
            "limits": {"added_array_budget_bytes": 128*1024*1024, "checkpoint_downloaded": False,
                "timing_measured": False, "quality_measured": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-sources", action="store_true", help="fetch only four small provenance files, never weights")
    args = parser.parse_args()
    if args.verify_sources:
        verify_sources()
    text = json.dumps(evidence(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
