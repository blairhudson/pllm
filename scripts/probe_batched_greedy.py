"""Pinned clear-target block gate: exact logits/KV and fully charged attempts.

The real drafter copies bounded n-grams from client-owned history. The separately
labelled oracle knows future target tokens and is only an optimistic batching
control. Stage bytes are arithmetic projections; provider reservations, transport,
WAN timing and process peak memory are not measured by this local diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from pllm import lower_model
from pllm.quantization import SymmetricPerRow
from pllm.runtime.greedy_verification_reference import verify_batched_greedy
from pllm.runtime.model_binding import compile_runtime_model

from decoder_probe_support import DecoderFixture, PROMPT, digest, stage_output


def ngram(history, cap):
    """Copy the latest previous continuation; fall back to repeating the last token."""
    working = list(history)
    proposed = []
    for _ in range(cap):
        token = working[-1]
        found = False
        for size in range(min(4, len(working) - 1), 0, -1):
            for start in range(len(working) - size - 1, -1, -1):
                if working[start:start + size] == working[-size:]:
                    token = working[start + size]
                    found = True
                    break
            if found:
                break
        working.append(token)
        proposed.append(token)
    return proposed


def run(outputs):
    fixture = DecoderFixture(inputs=128, outputs=32)
    fixture.composition = fixture.composition.with_params(
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    fixture.plan = lower_model((fixture.source.path / "config.json").read_bytes(),
                              batch=1, max_input_tokens=128, max_new_tokens=32)
    fixture.compiled = compile_runtime_model(fixture.plan, fixture.bundle,
                                             composition=fixture.composition)
    contract = fixture.plan.continuation_schedule(fixture.composition)
    tokens = fixture.tokens(PROMPT)
    if len(tokens) != 39:
        raise ValueError("pinned prompt token count changed")
    records = []

    def callback(counters):
        def remote(key, activation):
            stage = fixture.body[key]
            rows = activation.size // stage.spec.in_features
            counters["stage_calls"] += 1
            counters["stage_rows"] += rows
            counters["body_integer_macs"] += rows * stage.spec.in_features * stage.spec.out_features
            counters["projected_online_arithmetic_bytes"] += (
                rows * (stage.spec.in_features + stage.spec.out_features)
                * stage.seeded_profile.wire_bits // 8)
            counters["projected_correction_arithmetic_bytes"] += (
                rows * stage.spec.out_features * stage.seeded_profile.wire_bits // 8)
            return stage_output(stage, activation, 8)
        return remote

    for count in outputs:
        counters = Counter()
        reference = fixture.compiled.runtime(callback(counters))
        start = time.process_time()
        _, logits, _ = reference.prepare_ids(tokens)
        expected, golden = [], []
        for index in range(count):
            expected.append(int(np.argmax(logits)))
            golden.append(logits.copy())
            if index < count - 1:
                logits = reference.forward_ids([expected[-1]])[-1]
        serial_cpu = time.process_time() - start
        serial_counts = dict(counters)

        for name in ("ngram", "oracle"):
            for width in (2, 4):
                counters = Counter()
                target = fixture.compiled.runtime(callback(counters))
                checked = []

                def propose(history, cap):
                    start = time.process_time()
                    offset = len(history) - len(tokens)
                    result = (ngram(history, cap) if name == "ngram"
                              else expected[offset:offset + cap])
                    counters["draft_cpu_seconds"] += time.process_time() - start
                    return result

                start = time.process_time()
                result = verify_batched_greedy(target, contract, tokens, propose,
                    outputs=count, width=width, on_output=checked.append)
                counters["aggregate_local_cpu_seconds"] = time.process_time() - start
                if result["tokens"] != expected or target.position != reference.position:
                    raise AssertionError("batched verification changed selected tokens or position")
                np.testing.assert_array_equal(np.asarray(checked).view(np.uint32),
                                              np.asarray(golden).view(np.uint32))
                kv_words = 0
                for actual, original in zip(target.caches, reference.caches, strict=True):
                    if actual.length != original.length:
                        raise AssertionError("retained KV length differs")
                    for kind in ("key", "value"):
                        values = getattr(actual, kind)[:actual.length]
                        np.testing.assert_array_equal(values.view(np.uint32),
                            getattr(original, kind)[:original.length].view(np.uint32))
                        if np.any(getattr(actual, kind)[actual.length:result["maximum_queried_position"]]):
                            raise AssertionError("rejected KV was not erased")
                        kv_words += values.size
                if result["target_decode_rows"] != count - 1 + result["rejected_target_rows"]:
                    raise AssertionError("rejected target work was not charged")
                head_rows = 1 + result["target_decode_rows"]
                counters["client_head_rows"] = head_rows
                counters["client_head_integer_macs"] = (
                    head_rows * fixture.bundle.cfg["hidden_size"] * fixture.bundle.cfg["vocab_size"])
                result["output_digest"] = digest(result.pop("tokens"))
                records.append({"draft": name, "width": width, "outputs": count,
                    "result": result, "counts": dict(counters),
                    "exact_logit_values": sum(row.size for row in golden),
                    "exact_retained_kv_values": kv_words,
                    "serial_counts": serial_counts, "serial_head_rows": count,
                    "serial_local_cpu_seconds": serial_cpu})
                target.reset()
        reference.reset()
    return {"schema": "pllm.batched_greedy_probe.v1", "scope": __doc__,
        "source": fixture.lock(), "continuation_digest": contract.digest,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__},
        "input_tokens": len(tokens), "sampling": "greedy, fixed cap, no EOS stop",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__import__(verify_batched_greedy.__module__, fromlist=["__file__"]).__file__).read_bytes()
        ).hexdigest(), "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", type=int, nargs="+", default=[8, 32], choices=range(2, 33))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = run(args.outputs)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(result["records"])}))
