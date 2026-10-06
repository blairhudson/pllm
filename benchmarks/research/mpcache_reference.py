"""Source-locked local KV-selection diagnostic; no protected Pipeline choice."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

import numpy as np

from benchmarks.research.nonlinear_block_reference import ROOT, sha, state_digest

CONFIGURATION = {
    "source": "MPCache arXiv:2501.06807v2 sections 4.2-4.4, equations 2 and 3",
    "max_input_tokens": 256, "selected_positions": 4,
    "observation_rows": 16, "recent_rows": 8,
    "always_fraction": 0.125, "static_fraction": 0.75,
    "cluster_rows": [16, 4], "cluster_keep_fraction": 0.5, "alpha": 0.6,
    "candidates": ["full_control", "static", "dynamic_bound", "dynamic_affine", "combined", "combined_index_share"],
    "deviations": [
        "public fixed short-context policy, not original LongBench or MPC benchmark",
        "static scores average query heads; cluster queries average grouped-query heads",
        "cross-layer reuse intersects original positions with each layer's retained rows, then includes its always/recent rows",
        "client-local plaintext selection and cache gathers; protected top-k and oblivious gather absent",
        "float32 inputs, float64 cluster accumulation; no protected fixed-point representation",
        "full runtime KV is retained; static eviction is emulated with selected views, so memory savings are projections",
    ],
}
PUBLIC_PROMPTS = tuple(
    "A fictional village keeps these records. "
    + " ".join(f"House {j} stores {(j * 7 + i * 3) % 41 + 1} blue jars." for j in range(1, 17))
    + question
    for i, question in enumerate((
        " Explain why a written record helps the village.",
        " Which house is described first? Answer briefly.",
        " Give one reason to count the jars again tomorrow.",
        " Explain the difference between a jar and its contents.",
        " Describe one way to protect these records from rain.",
        " Does the record state the weight of each jar?",
        " How many houses appear in the record?",
        " Suggest a useful heading for this list.",
        " What colour are the jars?",
        " Explain why the house number is useful.",
        " Is a stored count necessarily the current count?",
        " Suggest a way to check a copied record for mistakes.",
    ))
)


class Oracle:
    def __init__(self):
        subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-core", "--example", "cache_selection_probe"], cwd=ROOT, check=True)
        target = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))["target_directory"]
        self.child = subprocess.Popen([str(Path(target) / "release/examples/cache_selection_probe")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.checked = 0

    def request(self, **values):
        self.child.stdin.write(json.dumps(values, allow_nan=False) + "\n")
        self.child.stdin.flush()
        line = self.child.stdout.readline(1 << 20)
        if not line.endswith("\n"):
            raise RuntimeError("native selection oracle failed or exceeded response bound")
        answer = json.loads(line)
        indices = answer["indices"]
        if indices != sorted(set(indices)):
            raise ValueError("selection indices must be ordered and unique")
        return answer

    def retain(self, scores, keep, recent=0):
        scores = np.asarray(scores, dtype=np.float32)
        answer = self.request(scores=scores.tolist(), keep=keep, recent=recent)["indices"]
        end = len(scores) - recent
        expected = sorted(sorted(range(end), key=lambda i: (-float(scores[i]), i))[:keep - recent] + list(range(end, len(scores))))
        if answer != expected:
            raise ValueError("native static retention differs from independent ordering")
        self.checked += 1
        return np.array(answer, dtype=np.int64)

    def clusters(self, keys, query, candidates, width, alpha):
        groups = [candidates[candidates // width == c] for c in sorted(set(candidates // width))]
        keep = max(1, math.ceil(len(groups) * CONFIGURATION["cluster_keep_fraction"]))
        answer = self.request(keys=keys.reshape(-1).tolist(), query=query.tolist(), candidates=candidates.tolist(), cluster_rows=width, keep=keep, alpha=alpha)
        q = query.astype(np.float64)
        scores = []
        for rows in groups:
            hi = keys[rows].max(axis=0).astype(np.float64)
            lo = keys[rows].min(axis=0).astype(np.float64)
            if alpha is None:
                scores.append(float(np.sum(np.maximum(q * hi, q * lo))))
            else:
                a = float(np.float32(alpha))
                scores.append(float(np.sum(q * (a * hi + (1 - a) * lo))))
        selected = sorted(sorted(range(len(groups)), key=lambda i: (-scores[i], i))[:keep])
        expected = np.concatenate([groups[i] for i in selected])
        if not np.allclose(answer["scores"], scores, rtol=1e-12, atol=1e-10) or answer["indices"] != expected.tolist():
            raise ValueError("native cluster selection differs from independent equation oracle")
        self.checked += 1
        return expected

    def close(self):
        if self.child.poll() is None:
            self.child.stdin.close()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        self.child.stdout.close()


class Selection:
    def __init__(self, runtime, oracle, candidate):
        self.runtime, self.oracle, self.candidate = runtime, oracle, candidate
        self.original = runtime._local
        self.importance, self.positions, self.always, self.selected = {}, {}, {}, {}
        self.selection_rows = []
        self.retained_bytes = 0
        runtime._local = self.local

    def local(self, op, values, *state):
        kind, layer, inputs = op["operator"], op.get("layer"), op["inputs"]
        if kind == "attention_scores" and self.runtime.position > 0:
            query, keys = values[inputs[0]], values[inputs[1]]
            full_count = keys.shape[2]
            positions = self.positions[layer]
            if full_count != self.runtime.position + 1 or positions[-1] != self.runtime.position - 1:
                raise ValueError("retained cache lost logical token identity")
            positions = np.append(positions, self.runtime.position)
            self.positions[layer] = positions
            keys = keys[:, :, positions, :]
            count = len(positions)
            if self.candidate in {"full_control", "static"}:
                selected = np.arange(count)
            elif self.candidate == "combined_index_share" and layer >= 3 and layer % 2:
                previous = set(self.selected[layer - 1].tolist())
                selected = np.array([i for i, position in enumerate(positions) if position in previous], dtype=np.int64)
            else:
                flat = np.ascontiguousarray(keys[0].transpose(1, 0, 2)).reshape(count, -1)
                kv_heads, features = keys.shape[1], keys.shape[-1]
                q = query.reshape(kv_heads, -1, features).mean(axis=1).reshape(-1).astype(np.float32)
                selected = np.arange(count)
                alpha = None if self.candidate == "dynamic_bound" else CONFIGURATION["alpha"]
                for width in CONFIGURATION["cluster_rows"]:
                    selected = self.oracle.clusters(flat, q, selected, width, alpha)
            mandatory = np.array([i for i, position in enumerate(positions) if position in self.always[layer]], dtype=np.int64)
            selected = np.unique(np.concatenate((selected, mandatory, np.arange(max(0, count - CONFIGURATION["recent_rows"]), count))))
            if selected[-1] != count - 1:
                raise ValueError("current token missing from dynamic cache view")
            self.selected[layer] = positions[selected]
            self.selection_rows.append({"layer": layer, "available": count, "selected": len(selected), "original_index_sha256": sha(positions[selected].astype("<i8").tobytes())})
            if len(selected) != full_count:
                values = dict(values)
                values[inputs[1]] = np.ascontiguousarray(keys[:, :, selected, :])
        elif kind == "attention_values" and self.runtime.position > 0:
            if len(self.selected[layer]) != values[inputs[1]].shape[2]:
                values = dict(values)
                values[inputs[1]] = np.ascontiguousarray(values[inputs[1]][:, :, self.selected[layer], :])
        answer = self.original(op, values, *state)
        if kind == "softmax" and self.runtime.position == 0:
            self.importance[layer] = np.asarray(answer)[0, :, -CONFIGURATION["observation_rows"]:, :].mean(axis=(0, 1))
        return answer

    def evict(self):
        for layer, cache in enumerate(self.runtime.caches):
            n = cache.length
            importance = self.importance[layer]
            if len(importance) != n:
                raise ValueError("importance does not cover complete prefill state")
            always = self.oracle.retain(importance, max(1, math.ceil(n * CONFIGURATION["always_fraction"])))
            self.always[layer] = set(always.tolist())
            if self.candidate in {"static", "combined", "combined_index_share"}:
                retained = self.oracle.retain(importance, max(CONFIGURATION["recent_rows"], math.ceil(n * CONFIGURATION["static_fraction"])), CONFIGURATION["recent_rows"])
            else:
                retained = np.arange(n)
            self.positions[layer] = retained
            # Do not forge the full-KV runtime's state contract. Attention views
            # emulate retained rows; the backing cache stays complete in this oracle.
            self.retained_bytes += len(retained) * (cache.key[0].nbytes + cache.value[0].nbytes)
        self.importance.clear()


def run(output):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture

    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("MPCache diagnostic needs 4 GiB beyond host reserve")
    oracle = None
    def abort(_reason):
        if oracle is not None and oracle.child.poll() is None:
            os.killpg(oracle.child.pid, signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        oracle = Oracle()
        fixture = DecoderFixture(inputs=CONFIGURATION["max_input_tokens"], outputs=CONFIGURATION["selected_positions"])
        rows = []
        for prompt in PUBLIC_PROMPTS:
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            ids = fixture.tokens(prompt)
            baseline = fixture.compiled.runtime(fixture.remote)
            _, logits, _ = baseline.prepare_ids(ids)
            scores, forcing = [logits.copy()], [int(np.argmax(logits))]
            for _ in range(CONFIGURATION["selected_positions"] - 1):
                logits = baseline.forward_ids([forcing[-1]])[-1]
                scores.append(logits.copy())
                forcing.append(int(np.argmax(logits)))
            full_kv_digest = state_digest(baseline)
            del baseline
            for candidate in CONFIGURATION["candidates"]:
                runtime = fixture.compiled.runtime(fixture.remote)
                selected = Selection(runtime, oracle, candidate)
                started = time.perf_counter()
                _, logits, _ = runtime.prepare_ids(ids)
                if not np.array_equal(logits.view(np.uint32), scores[0].view(np.uint32)):
                    raise ValueError("selection changed prefill before its declared boundary")
                selected.evict()
                observed = [logits.copy()]
                for token in forcing[:-1]:
                    observed.append(runtime.forward_ids([token])[-1].copy())
                errors = [float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(observed, scores, strict=True)]
                if candidate == "full_control" and (any(errors) or state_digest(runtime) != full_kv_digest):
                    raise ValueError("all-row state/selection control changed logits or KV")
                rows.append({
                    "candidate": candidate, "input_tokens": len(ids), "token_cohort_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
                    "same_token_top1_matches": [int(np.argmax(a)) == b for a, b in zip(observed, forcing, strict=True)],
                    "worst_abs_logit_error": errors, "projected_prefill_active_kv_bytes": selected.retained_bytes,
                    "full_prefill_active_kv_bytes": 2 * len(ids) * sum(cache.key.shape[1] * cache.key.shape[2] * 4 for cache in runtime.caches),
                    "selection": selected.selection_rows, "diagnostic_elapsed_seconds": time.perf_counter() - started,
                    "final_kv_sha256": state_digest(runtime),
                })
                runtime._local = selected.original
                del selected, runtime
        summary = {}
        for candidate in CONFIGURATION["candidates"]:
            data = [r for r in rows if r["candidate"] == candidate]
            summary[candidate] = {
                "prefill_matches": sum(r["same_token_top1_matches"][0] for r in data),
                "decode_matches": sum(sum(r["same_token_top1_matches"][1:]) for r in data),
                "prefill_count": len(data), "decode_count": len(data) * (CONFIGURATION["selected_positions"] - 1),
                "worst_abs_logit_error": max(max(r["worst_abs_logit_error"]) for r in data),
                "projected_prefill_active_kv_bytes": sum(r["projected_prefill_active_kv_bytes"] for r in data),
                "full_prefill_active_kv_bytes": sum(r["full_prefill_active_kv_bytes"] for r in data),
                "selected_attention_rows": sum(s["selected"] for r in data for s in r["selection"]),
                "available_attention_rows": sum(s["available"] for r in data for s in r["selection"]),
            }
        files = ["benchmarks/research/mpcache_reference.py", "scripts/decoder_probe_support.py", "crates/pllm-core/src/cache_selection_reference.rs", "crates/pllm-core/examples/cache_selection_probe.rs", "crates/pllm-models/src/cache.rs"]
        report = {
            "schema": "pllm.mpcache_reference.v1", "configuration": CONFIGURATION,
            "source": fixture.lock(), "public_prompt_sha256": sha(json.dumps(PUBLIC_PROMPTS).encode()),
            "paper_sha256": sha((ROOT / "papers/r23-mpcache.pdf").read_bytes()),
            "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in files},
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "summary": summary, "observations": rows, "independent_native_oracle_checks": oracle.checked,
            "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "reference": "same-token pinned W8A8 full-cache, not upstream float32 or LongBench quality",
            "executable_sdk": False, "protected_selection": False, "whole_generation_quality": False,
            "whole_response_network_bytes": None, "process_peak_rss_bytes": None,
            "limitations": CONFIGURATION["deviations"] + ["diagnostic elapsed time includes JSON oracle IPC; not a runtime speed measurement", "KV counts are active payloads, not process peaks or geometric allocation", "selected addresses and counts depend on private values; no remote index disclosure is authorized"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"summary": summary, "oracle_checks": oracle.checked, "executable_sdk": False}))
    finally:
        if oracle is not None:
            oracle.close()
        guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
