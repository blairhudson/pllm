"""Optimistic plaintext NEXUS/THOR attention gate; no HE inference benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import signal
import subprocess

import numpy as np

from benchmarks.research.nonlinear_block_reference import PUBLIC_PROMPTS, ROOT, sha

CONFIGURATION = {
    "methods": ["nexus_exp", "thor_exp"],
    "public_score_bound": 16,
    "nexus": {"public_shift": 16, "squarings": 8},
    "thor": {"delta1": 1, "delta2": 32, "coefficient_source": "Appendix C printed degree-15 exponential"},
    "normalization": "exact f64 reference division; encrypted reciprocal omitted",
    "input_bound": 128, "output_positions": 3,
    "calibration": "fixed public parameters; no prompt-dependent fitting or clipping",
}


class DomainRejected(ValueError):
    pass


class Oracle:
    def __init__(self):
        subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-core", "--example", "polynomial_softmax_probe"], cwd=ROOT, check=True)
        target = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))["target_directory"]
        self.child = subprocess.Popen([str(Path(target) / "release/examples/polynomial_softmax_probe")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)

    def evaluate(self, method, scores, query_offset=0):
        if (scores.dtype != np.float32 or scores.ndim != 4 or scores.shape[0] != 1
                or not all(1 <= size <= 512 for size in scores.shape[1:]) or scores.size > 1 << 20
                or type(query_offset) is not int or query_offset < 0
                or query_offset + scores.shape[2] != scores.shape[3]):
            raise ValueError("invalid reference attention scores")
        positions = query_offset + np.arange(scores.shape[2])
        public_valid = np.broadcast_to(np.arange(scores.shape[3]) <= positions[:, None], scores.shape)
        if not np.isfinite(scores[public_valid]).all() or not np.isneginf(scores[~public_valid]).all():
            raise ValueError("nonfinite score is not a publicly masked causal slot")
        matrix = scores.reshape(-1, scores.shape[-1])
        allowed = np.isfinite(matrix)
        rows = [row[mask].tolist() for row, mask in zip(matrix, allowed, strict=True)]
        request = json.dumps({"method": method, "rows": rows}, allow_nan=False) + "\n"
        if len(request.encode()) > 16 << 20:
            raise ValueError("reference tensor exceeds oracle frame bound")
        self.child.stdin.write(request)
        self.child.stdin.flush()
        line = self.child.stdout.readline(16 << 20)
        if not line.endswith("\n"):
            raise RuntimeError("native softmax oracle failed")
        answer = json.loads(line)
        if answer["error"] is not None:
            raise DomainRejected(answer["error"])
        if len(answer["rows"]) != len(rows):
            raise ValueError("native softmax row count changed")
        result = np.zeros_like(matrix)
        for output, mask, values in zip(result, allowed, answer["rows"], strict=True):
            if len(values) != np.count_nonzero(mask):
                raise ValueError("native softmax causal extent changed")
            output[mask] = values
        if not np.isfinite(result).all() or np.any(result < 0) or not np.allclose(result.sum(axis=-1), 1, atol=1e-6):
            raise ValueError("native softmax probability invariant failed")
        return result.reshape(scores.shape)

    def close(self):
        self.child.stdin.close()
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()
        self.child.stdout.close()


def run(output):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture

    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("checkpoint diagnostic needs 4 GiB beyond host reserve")
    oracle = None

    def abort(_reason):
        if oracle is not None and oracle.child.poll() is None:
            try:
                os.killpg(oracle.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os.kill(os.getpid(), signal.SIGTERM)

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        oracle = Oracle()
        fixture = DecoderFixture(inputs=CONFIGURATION["input_bound"], outputs=CONFIGURATION["output_positions"])
        observations = []
        for prompt in PUBLIC_PROMPTS:
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            ids = fixture.tokens(prompt)
            baseline = fixture.compiled.runtime(fixture.remote)
            _, logits, _ = baseline.prepare_ids(ids)
            expected = [logits.copy()]
            forcing = [int(np.argmax(logits))]
            for _ in range(CONFIGURATION["output_positions"] - 1):
                logits = baseline.forward_ids([forcing[-1]])[-1]
                expected.append(logits.copy())
                forcing.append(int(np.argmax(logits)))
            del baseline
            for method in CONFIGURATION["methods"]:
                runtime = fixture.compiled.runtime(fixture.remote)
                original = runtime._local
                counters = {"operators": 0, "valid_scores": 0, "outside_public_range": 0}

                def local(op, values, *state, _original=original, _runtime=runtime):
                    if op["operator"] != "softmax":
                        return _original(op, values, *state)
                    scores = values[op["inputs"][0]]
                    valid = scores[np.isfinite(scores)]
                    counters["operators"] += 1
                    counters["valid_scores"] += valid.size
                    counters["outside_public_range"] += int(np.count_nonzero(np.abs(valid) > CONFIGURATION["public_score_bound"]))
                    return oracle.evaluate(method, scores, _runtime.position)

                runtime._local = local
                actual, rejection = [], None
                try:
                    _, logits, _ = runtime.prepare_ids(ids)
                    actual.append(logits.copy())
                    for token in forcing[:-1]:
                        actual.append(runtime.forward_ids([token])[-1].copy())
                except DomainRejected as exc:
                    rejection = str(exc)
                finally:
                    runtime._local = original
                observations.append({
                    "method": method, "input_tokens": len(ids),
                    "token_cohort_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
                    "completed_positions": len(actual), "rejection": rejection,
                    "top1_matches": [int(np.argmax(a)) == forcing[i] for i, a in enumerate(actual)],
                    "worst_abs_logit_error": [float(np.max(np.abs(a.astype(np.float64) - expected[i]))) for i, a in enumerate(actual)],
                    **counters,
                })
                del runtime, original, local
        summaries = {}
        for method in CONFIGURATION["methods"]:
            rows = [r for r in observations if r["method"] == method]
            summaries[method] = {
                "attempted_prompts": len(rows), "rejected_prompts": sum(r["rejection"] is not None for r in rows),
                "completed_positions": sum(r["completed_positions"] for r in rows),
                "matching_positions": sum(sum(r["top1_matches"]) for r in rows),
                "worst_abs_logit_error": max((e for r in rows for e in r["worst_abs_logit_error"]), default=None),
                "outside_public_range": sum(r["outside_public_range"] for r in rows),
            }
        files = ["benchmarks/research/he_softmax_reference.py", "benchmarks/research/nonlinear_block_reference.py", "scripts/decoder_probe_support.py", "crates/pllm-core/src/polynomial_softmax_reference.rs", "crates/pllm-core/examples/polynomial_softmax_probe.rs"]
        report = {
            "schema": "pllm.he_softmax_reference.v1", "configuration": CONFIGURATION,
            "source": fixture.lock(), "public_prompt_sha256": sha(json.dumps(PUBLIC_PROMPTS).encode()),
            "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in files},
            "paper_sha256": {p: sha((ROOT / p).read_bytes()) for p in ["papers/r21-nexus.pdf", "papers/thor.pdf"]},
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "summary": summaries, "observations": observations,
            "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "reference": "same-token pinned W8A8 clear-kernel; all other operators unchanged",
            "executable_sdk": False, "ciphertext_execution": False, "whole_generation_quality": False,
            "whole_response_network_bytes": None, "process_peak_rss_bytes": None,
            "limitations": [
                "optimistic plaintext exponent/softmax gate, not a reproduction of either HE system",
                "exact f64 normalization replaces Goldschmidt/aSOR reciprocal; CKKS errors and bootstrapping absent",
                "fixed [-16,16] bound rejects without clipping or fallback; public causal slots only",
                "Qwen uses RMSNorm, SiLU, GQA, rotary KV and autoregressive feedback requiring separate contracts",
                "no packed ciphertext matrix schedule, encrypted persistent state or protected token feedback",
                "numeric oracles do not supply ciphertext costs or provider performance scores",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"summary": summaries, "ciphertext_execution": False}))
    finally:
        if oracle is not None:
            oracle.close()
        guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
