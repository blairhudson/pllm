"""Fresh checkpoint quality for the installed masked-wire numeric contract.

This clear diagnostic measures prefill, same-token decode and free-running
greedy trajectories. Protected worker execution is measured separately through
the ordinary Experiment benchmark; no protocol timings come from this screen.
"""
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

import numpy as np

from benchmarks.research.gated_numeric_screen import screen
from benchmarks.research.masked_wire_reference import integer_oracle

ROOT = Path(__file__).resolve().parents[2]

# Frozen before execution, independently of the older helper/joint/Q12 screens.
PUBLIC_HELDOUT = (
    "Why does a wet towel feel cooler in a breeze?",
    "Explain why a magnifying glass can make a small letter look larger.",
    "A library has seven shelves with twelve books each. How many books is that?",
    "Why does an empty metal saucepan heat faster than one filled with water?",
    "How can tree rings help estimate the age of a tree?",
    "What is the purpose of a fuse in an electrical circuit?",
    "Why is it harder to walk on loose sand than on a firm path?",
    "Explain how clouds can form when warm air rises.",
    "A shop sells three pencils for six dollars. What do nine pencils cost?",
    "Why can a sealed balloon grow larger when it is warmed?",
    "How does a pulley change the direction of an applied force?",
    "What is one reason to repeat an experiment several times?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class NativeMapping:
    """Installed Rust numeric mapping, checked against an independent integer formula."""

    def __init__(self):
        self.checked = 0
        self.profile = {"table": [math.floor(4096 * (i / 64) / (1 + math.exp(i / 64)))
                                  for i in range(1024)]}

    def evaluate(self, gate, up):
        from pllm import _native

        if (gate.shape != up.shape or gate.dtype != np.float32 or up.dtype != np.float32
                or not 0 < gate.size <= 1 << 20):
            raise ValueError("numeric frame shape or representation differs")
        if (not np.all(np.isfinite(gate)) or not np.all(np.isfinite(up))
                or np.any(np.abs(gate) > 48) or np.any(np.abs(up) > 128)):
            raise ValueError("profile domain rejection")
        g = np.floor(gate.astype(np.float64) * (1 << 24)).astype("<i8")
        u = np.floor(up.astype(np.float64) * (1 << 24)).astype("<i8")
        actual = np.frombuffer(_native._masked_wire_clear(g.tobytes(), u.tobytes()), dtype="<i8").reshape(g.shape)
        expected = integer_oracle(self.profile, g, u)
        if not np.array_equal(actual, expected):
            raise ValueError("installed native mapping differs from independent integer oracle")
        self.checked += gate.size
        return actual.astype(np.float32) / np.float32(4096)


def generation(fixture, bridge, prompts, check, *, outputs=8):
    trials = []
    layers = {op["layer"] for op in fixture.plan.prefill["operations"] if op["operator"] == "silu"}
    for text in prompts:
        ids = fixture.tokens(text)
        reference = []
        for name in ("reference", "control", "masked-wire-q12"):
            runtime = fixture.compiled.runtime(fixture.remote)
            original = runtime._local
            pending, seen = {}, set()

            def local(operation, values, *state):
                check()
                value = original(operation, values, *state)
                if operation["operator"] == "silu":
                    pending[operation["id"]] = values[operation["inputs"][0]]
                elif operation["operator"] == "multiply":
                    keys = [key for key in operation["inputs"] if key in pending]
                    if len(keys) == 1:
                        gate = pending.pop(keys[0])
                        up = values[next(key for key in operation["inputs"] if key != keys[0])]
                        seen.add(operation["layer"])
                        if name == "masked-wire-q12":
                            return bridge.evaluate(gate, up)
                return value

            runtime._local = local
            selected, rejection = [], None
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for index in range(outputs):
                    check()
                    if index:
                        logits = runtime.forward_ids([selected[-1]])[-1]
                    selected.append(int(np.argmax(logits)))
                    if selected[-1] == int(fixture.bundle.cfg["eos_token_id"]):
                        break
                if pending or seen != layers:
                    raise ValueError("generation does not cover every semantic gated region")
            except ValueError as error:
                if str(error) != "profile domain rejection" or name != "masked-wire-q12":
                    raise
                rejection = str(error)
            finally:
                del runtime._local
            if name == "reference":
                reference = selected
            else:
                prefix = 0
                for a, b in zip(reference, selected):
                    if a != b:
                        break
                    prefix += 1
                trials.append({
                    "candidate": name, "input_tokens": len(ids),
                    "public_input_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
                    "reference_output_tokens": len(reference), "generated_tokens": len(selected),
                    "matching_prefix_tokens": prefix,
                    "matching_positions": sum(a == b for a, b in zip(reference, selected)),
                    "sequence_match": rejection is None and selected == reference,
                    "output_sha256": sha(np.asarray(selected, dtype="<i8").tobytes()),
                    "rejection": rejection,
                })
            original = None
            del runtime, local
    summary = {}
    for name in ("control", "masked-wire-q12"):
        rows = [row for row in trials if row["candidate"] == name]
        summary[name] = {
            "attempted_prompts": len(rows),
            "rejected_prompts": sum(row["rejection"] is not None for row in rows),
            "exact_trajectories": sum(row["sequence_match"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "reference_output_tokens": sum(row["reference_output_tokens"] for row in rows),
            "matching_positions": sum(row["matching_positions"] for row in rows),
            "matching_prefix_tokens": sum(row["matching_prefix_tokens"] for row in rows),
        }
    return {"summary": summary, "trials": trials, "maximum_generated_tokens": outputs,
            "sampling": "greedy; each candidate feeds back its own selected token; stop on EOS"}


def run(output):
    from pllm import _native
    from pllm.metrics import benchmark_operator
    from pllm.nonlinear import MaskedWireGatedMultiplyQ12
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture

    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("quality screen needs 4 GiB beyond the host reserve")
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
                           lambda _: os.kill(os.getpid(), signal.SIGTERM))

    def check():
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)

    guard.start()
    try:
        metadata = benchmark_operator(MaskedWireGatedMultiplyQ12(), lanes=1, samples=1)
        fixture = DecoderFixture(inputs=64, outputs=8)
        bridge = NativeMapping()
        teacher_forced = screen(fixture, {"masked-wire-q12": bridge}, PUBLIC_HELDOUT, check)
        greedy = generation(fixture, bridge, PUBLIC_HELDOUT, check)
        if (teacher_forced["summary"]["control"]["bit_exact_positions"] != 48
                or teacher_forced["summary"]["control"]["exact_kv_positions"] != 48
                or greedy["summary"]["control"]["exact_trajectories"] != 12):
            raise ValueError("fresh-runtime control is not exact")
        check()
    finally:
        guard.close()
    source_files = ("benchmarks/research/nonlinear_decoder_quality.py",
                    "benchmarks/research/gated_numeric_screen.py", "benchmarks/research/masked_wire_reference.py",
                    "scripts/decoder_probe_support.py", "python/pllm/runtime/masked_wire.py",
                    "crates/pllm-core/src/residual_silu_reference.rs")
    report = {
        "schema": "pllm.nonlinear_decoder_quality.v1", "source": fixture.lock(),
        "configuration": {"component": metadata.configuration["component"],
                          "input_conversion": "float32-to-signed-q24-floor-reject",
                          "maximum_input_tokens": 64, "same_token_positions": 4, "greedy_output_cap": 8},
        "numeric_contract_digest": metadata.numeric_contract_digest,
        "native_implementation_digest": metadata.native_implementation_digest,
        "installed_native_sha256": sha(Path(_native.__file__).read_bytes()),
        "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
        "teacher_forced": teacher_forced, "greedy_generation": greedy,
        "independent_integer_checked_elements": bridge.checked,
        "source_sha256": {path: sha((ROOT / path).read_bytes()) for path in source_files},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "scope": "clear replay of the installed nonlinear numeric contract on the locked W8A8 decoder",
        "protected_checkpoint_execution": False,
        "limitations": ["twelve public prompts are not representative task-quality evidence",
                        "reference is the locked W8A8 baseline, not upstream floating point",
                        "free-generation token agreement is not a semantic-answer quality score",
                        "protected worker costs and end-to-end transport are measured separately"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"teacher_forced": teacher_forced["summary"], "generation": greedy["summary"],
                      "checked_elements": bridge.checked}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new evidence output path")
    run(args.output)
