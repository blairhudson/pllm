"""Complete shared Q7 block costs and source-locked checkpoint domain rejection."""

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
import statistics
import subprocess
import sys

import numpy as np

from benchmarks.research.shared_rescale_reference import ROOT, measure_child

CONFIGURATION = {
    "papers": ["fusefss", "sigma", "fss-mixed"],
    "source_sections": {
        "fusefss": ["Appendix N", "Appendix O"],
        "fss-mixed": ["Figure 1", "Figure 14"],
    },
    "profile": "pllm.numeric.gated_multiply.q7.v1",
    "input": "gate/up signed Q14 in [-1,1], caller-enforced public bound",
    "steps": [
        "two Q14-to-Q7 rescalings",
        "Beaver square",
        "quadratic SiLU ties-even /512",
        "Beaver gated product",
        "Q7 output ties-even /128",
    ],
    "ring_bits": 24,
    "backends": ["compact_dcf", "interval_dcf"],
    "layouts": ["unfused", "fused"],
    "lanes": [1, 64, 512],
    "samples": 5,
    "workloads": [
        {"input_tokens": 39, "output_tokens": 8},
        {"input_tokens": 39, "output_tokens": 32},
    ],
    "checkpoint_diagnostic": {"maximum_input_tokens": 64, "selected_positions": 3},
    "source_deviations": [
        "PLLM quadratic SiLU profile; no reproduction of the paper's spline coefficients or coefficient-lookup compiler",
        "arithmetic shared predicates, 24-bit ring with four-byte words and PLLM ties-to-even extension",
        "test-only dealer supplies independent Beaver triples and FSS keys; no distributed preprocessing or malicious security",
    ],
}
PUBLIC_PROMPTS = (
    "Explain how a compass helps a walker when a trail is hidden by snow.",
    "Why can two equal buckets feel different in weight when one contains sand?",
    "Describe the difference between saving a file and making a backup of it.",
    "Four friends share twenty strawberries equally. How many does each get?",
    "Explain why a shadow changes length during the day.",
    "Suggest a simple experiment to compare paper towel absorbency.",
    "What does a library catalogue tell a reader about finding a book?",
    "Why should bread dough be covered while it rises?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_values(lanes):
    fixed = [-16384, -16320, -8256, -192, -64, 0, 64, 192, 8256, 16320, 16384]
    return [fixed[i] if i < len(fixed) else (i * 7919) % 32769 - 16384 for i in range(lanes)]


def round_even(x, denominator):
    q, r = divmod(x, denominator)
    return q + int(2 * r > denominator or (2 * r == denominator and q % 2 != 0))


def oracle_digest(lanes):
    inputs = public_values(lanes)
    output = bytearray()
    for g, u in zip(inputs, reversed(inputs), strict=True):
        g, u = round_even(g, 128), round_even(u, 128)
        activated = round_even(g * g + 256 * g, 512)
        result = round_even(activated * u, 128) & ((1 << 24) - 1)
        output.extend(result.to_bytes(4, "little"))
    return sha(output)


def native_cases(check, active):
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "pllm-garble",
            "--example",
            "shared_gated_block_probe",
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT
        )
    )
    binary = Path(metadata["target_directory"]) / "release/examples/shared_gated_block_probe"
    cases = []
    for i, lanes in enumerate(CONFIGURATION["lanes"]):
        for layout in CONFIGURATION["layouts"]:
            for backend in CONFIGURATION["backends"][:: 1 if i % 2 == 0 else -1]:
                check()
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.research.nonlinear_block_reference",
                        "--child",
                        str(binary),
                        backend,
                        layout,
                        str(lanes),
                        str(CONFIGURATION["samples"]),
                    ],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                active.append(child)
                try:
                    stdout, stderr = child.communicate(timeout=180)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    active.remove(child)
                check()
                if child.returncode:
                    raise RuntimeError(stderr)
                case = json.loads(stdout)
                if (
                    case["output_sha256"] != oracle_digest(lanes)
                    or case["executable_sdk"] is not False
                ):
                    raise ValueError("complete block differs from independent exact integer oracle")
                if [case[k] for k in ("backend", "layout", "lanes", "samples")] != [
                    backend,
                    layout,
                    lanes,
                    CONFIGURATION["samples"],
                ]:
                    raise ValueError("native probe identity differs")
                case["median_seconds"] = {
                    k: statistics.median(case[k + "_ns"]) / 1e9
                    for k in ("issuance", "online", "clear")
                }
                cases.append(case)
    return cases


def domain_diagnostic(check):
    # This fixture owns only public diagnostic values. No private trace is saved.
    from scripts.decoder_probe_support import DecoderFixture

    fixture = DecoderFixture(inputs=64, outputs=3)
    check()
    layers = {}
    traces = []
    for text in PUBLIC_PROMPTS:
        ids = fixture.tokens(text)
        runtime = fixture.compiled.runtime(fixture.remote)
        original = runtime._local
        pending = {}

        def observe(operation, values, *state):
            check()
            answer = original(operation, values, *state)
            if operation["operator"] == "silu":
                pending[operation["id"]] = np.array(values[operation["inputs"][0]], copy=True)
            elif operation["operator"] == "multiply":
                inputs = operation["inputs"]
                keys = [key for key in inputs if key in pending]
                if len(keys) == 1:
                    g = pending.pop(keys[0])
                    u = np.asarray(values[next(key for key in inputs if key != keys[0])])
                    if g.size != u.size:
                        raise ValueError("gated semantic shapes differ")
                    g, u, target = g.reshape(-1), u.reshape(-1), np.asarray(answer).reshape(-1)
                    row = layers.setdefault(
                        str(operation["layer"]),
                        {
                            "elements": 0,
                            "gate_outside_q7": 0,
                            "up_outside_q7": 0,
                            "either_outside_q7": 0,
                            "gate_outside_scaled16": 0,
                            "maximum_abs_gate": 0.0,
                            "maximum_abs_up": 0.0,
                            "in_domain_bit_exact_outputs": 0,
                            "in_domain_worst_absolute_error": 0.0,
                        },
                    )
                    valid = (np.abs(g) <= 1) & (np.abs(u) <= 1)
                    row["elements"] += g.size
                    row["gate_outside_q7"] += int(np.count_nonzero(np.abs(g) > 1))
                    row["up_outside_q7"] += int(np.count_nonzero(np.abs(u) > 1))
                    row["either_outside_q7"] += int(np.count_nonzero(~valid))
                    row["gate_outside_scaled16"] += int(np.count_nonzero(np.abs(g) > 16))
                    row["maximum_abs_gate"] = max(row["maximum_abs_gate"], float(np.max(np.abs(g))))
                    row["maximum_abs_up"] = max(row["maximum_abs_up"], float(np.max(np.abs(u))))
                    if np.any(valid):
                        gg = np.rint(np.rint(g[valid].astype(np.float64) * 16384) / 128).astype(
                            np.int64
                        )
                        uu = np.rint(np.rint(u[valid].astype(np.float64) * 16384) / 128).astype(
                            np.int64
                        )
                        activated = np.rint((gg * gg + 256 * gg) / 512).astype(np.int64)
                        result = (np.rint(activated * uu / 128) / 128).astype(np.float32)
                        row["in_domain_bit_exact_outputs"] += int(
                            np.count_nonzero(
                                result.view(np.uint32) == target[valid].view(np.uint32)
                            )
                        )
                        row["in_domain_worst_absolute_error"] = max(
                            row["in_domain_worst_absolute_error"],
                            float(np.max(np.abs(result.astype(np.float64) - target[valid]))),
                        )
            return answer

        runtime._local = observe
        _, logits, _ = runtime.prepare_ids(ids)
        positions = []
        chosen = int(np.argmax(logits))
        for index in range(3):
            if index:
                logits = runtime.forward_ids([chosen])[-1]
            chosen = int(np.argmax(logits))
            positions.append(
                {
                    "logits_sha256": sha(np.ascontiguousarray(logits).tobytes()),
                    "kv_sha256": state_digest(runtime),
                }
            )
        if pending:
            raise ValueError("unconsumed nonlinear source")
        traces.append(
            {
                "input_tokens": len(ids),
                "token_cohort_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
                "reference_positions": positions,
            }
        )
        del runtime
    if len(layers) != 24:
        raise ValueError("expected every compiled Qwen2.5 gated layer")
    aggregate = {
        key: sum(v[key] for v in layers.values())
        for key in (
            "elements",
            "gate_outside_q7",
            "up_outside_q7",
            "either_outside_q7",
            "gate_outside_scaled16",
            "in_domain_bit_exact_outputs",
        )
    }
    return fixture, {
        "source": fixture.lock(),
        "public_prompt_sha256": sha(json.dumps(PUBLIC_PROMPTS).encode()),
        "traces": traces,
        "layers": layers,
        "aggregate": aggregate,
        "q7_domain_admitted": aggregate["either_outside_q7"] == 0,
        "scaled16_gate_domain_admitted": aggregate["gate_outside_scaled16"] == 0,
        "reference": "pinned compiled W8A8 clear-kernel trajectory; not upstream FP32 or broad task quality",
        "candidate_decoder_executed": False,
        "protected_checkpoint_execution": False,
    }


def geometry(config, cases):
    from pllm import lower_model

    result = []
    for workload in CONFIGURATION["workloads"]:
        plan = lower_model(
            config,
            batch=1,
            max_input_tokens=workload["input_tokens"],
            max_new_tokens=workload["output_tokens"],
        )
        operators = {
            phase: [op for op in plan.to_dict()[phase]["operations"] if op["operator"] == "silu"]
            for phase in ("prefill", "decode")
        }
        count = {
            phase: sum(math.prod(op["output_shape"]) for op in ops)
            for phase, ops in operators.items()
        }
        elements = count["prefill"] + (workload["output_tokens"] - 1) * count["decode"]
        rows = []
        for case in cases:
            lanes = case["lanes"]
            batches = {
                phase: sum(math.ceil(math.prod(op["output_shape"]) / lanes) for op in ops)
                for phase, ops in operators.items()
            }
            batches = batches["prefill"] + (workload["output_tokens"] - 1) * batches["decode"]
            header = 2 * 78 * case["rounds"]
            bodies = (case["peer_frame_bytes"] - header) // lanes
            rows.append(
                {
                    "backend": case["backend"],
                    "layout": case["layout"],
                    "batch_lanes": lanes,
                    "both_party_key_payload_bytes": 2
                    * case["party_key_payload_bytes"]
                    * elements
                    // lanes,
                    "peer_frame_bytes": elements * bodies + batches * header,
                    "client_input_output_share_payload_bytes_if_not_resident": 24 * elements,
                    "issuance_seconds_linear_projection": case["median_seconds"]["issuance"]
                    * elements
                    / lanes,
                    "local_online_seconds_linear_projection": case["median_seconds"]["online"]
                    * elements
                    / lanes,
                }
            )
        result.append(
            {
                **workload,
                "plan_digest": plan.digest,
                "gated_elements": elements,
                "cases": rows,
                "executable_decoder": False,
            }
        )
    return result


def state_digest(runtime):
    digest = hashlib.sha256()
    for layer, cache in enumerate(runtime.caches):
        digest.update(str((layer, cache.length)).encode())
        for array in (cache.key, cache.value):
            if array is None:
                raise ValueError("missing full-KV diagnostic state")
            active = np.ascontiguousarray(array[: cache.length])
            digest.update(str((active.shape, str(active.dtype))).encode())
            digest.update(active.tobytes())
    return digest.hexdigest()


def run(output):
    from pllm.runtime.benchmark_memory import host_memory, GiB, MemoryWatchdog

    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("nonlinear checkpoint diagnostic requires 4 GiB beyond host reserve")
    active = []

    # Watchdog exits this owned probe on sustained host pressure, including child work.
    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        os.kill(os.getpid(), signal.SIGTERM)

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)

    def check():
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)

    guard.start()
    try:
        cases = native_cases(check, active)
        fixture, numeric = domain_diagnostic(check)
        projected = geometry((fixture.source.path / "config.json").read_bytes(), cases)
        check()
    finally:
        guard.close()
    files = [
        "benchmarks/research/nonlinear_block_reference.py",
        "scripts/decoder_probe_support.py",
        "crates/pllm-garble/src/shared_arithmetic_reference.rs",
        "crates/pllm-garble/src/shared_gated_block_reference.rs",
        "crates/pllm-garble/src/shared_rescale_reference.rs",
        "crates/pllm-garble/src/interval_fss.rs",
        "crates/pllm-garble/src/compact_dcf.rs",
        "crates/pllm-garble/examples/shared_gated_block_probe.rs",
        "crates/pllm-core/src/activation.rs",
        "crates/pllm-core/src/fixed_point.rs",
        "Cargo.lock",
    ]
    report = {
        "schema": "pllm.nonlinear_block_reference.v1",
        "configuration": CONFIGURATION,
        "cases": cases,
        "numeric_domain_diagnostic": numeric,
        "geometry_projections": projected,
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in files},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {
            "host": asdict(host),
            "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        },
        "executable_sdk": False,
        "privacy_reviewed": False,
        "full_wire_bytes": None,
        "whole_generation_quality": False,
        "limitations": [
            "native measurements contain dealer and both evaluators in one process",
            "no secret input range proof or float-to-fixed bridge",
            "public checkpoint trace rejects unsupported ranges; no clipping or silently widened profile",
            "Qwen costs are hypothetical dimensions, not admitted numerical mapping",
            "all triple and FSS payloads and peer frames counted; offline framing and distribution unmeasured",
            "linear projections omit decoder attention, normalization, linear work, feedback and checkpoint distribution",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "checked_outputs": sum(c["checked_outputs"] for c in cases),
                "domain": numeric["aggregate"],
                "sdk_admitted": False,
            }
        )
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(measure_child(sys.argv[2:])))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new reproduction path")
    run(args.output)


if __name__ == "__main__":
    main()
