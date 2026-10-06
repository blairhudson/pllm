"""Paper-derived masked-wire block: exact local protocol and fresh numeric gate."""
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

from benchmarks.research.gated_numeric_screen import screen
from benchmarks.research.piecewise_gated_reference import NumericBridge
from benchmarks.research.shared_rescale_reference import ROOT, measure_child

CONFIGURATION = {
    "fraction": 12, "input_fraction": 24, "ring_bits": 40, "effective_gate_bits": 19,
    "table_entries": 1024, "index_fraction": 6, "gate_bound": 48, "up_bound": 128,
    "rounding": "floor for trusted float bridge, both input shifts, residual table and final product",
    "numeric_profile": "SIGMA Figure 7 GPU order / Appendix F; abs after floor TR then residual clip",
    "protocol": "early-terminated Boolean DPF, Grotto comparison, GapARS/TR, masked-wire select/lookup/multiply",
    "secret_seed_bits": 128, "maximum_lanes": 256, "lanes": [1, 64, 256], "repetitions": 4, "samples": 4,
    "maximum_input_tokens": 64, "selected_positions": 4,
    "numeric_gate": "all twelve prefill and 36 same-token decode selections agree without range rejection",
    "cost_gate": "raw two-party complete-block keys plus measured peer-frame geometry below historical prepared whole response",
    "baseline": "docs/evidence/latent-response-network-qwen25-2026-09-28.json",
    "previous_screen": "docs/evidence/research-piecewise-gated-reference-qwen25.json",
    "source_deviations": [
        "CPU-only independent construction; separate 128-bit seeds and controls instead of SIGMA's 127-bit seed packing",
        "two AES calls per half expansion; reference does not reproduce GPU kernels or half-PRG throughput",
        "Pika-style LUT sign derived from local full-domain bit counts; explicit mask/correction shares, no seeded distribution",
        "trusted clear float-to-Q24 boundary; no protected IEEE conversion or fixed-point linear decoder",
        "public Qwen-shaped range and effective width are PLLM mapping choices, not SIGMA model parameters",
        "in-process dealer and party-local state; no authenticated transport, malicious-output checks or independent security review",
        "FuseFSS general fusion and semantic scheduling remain unimplemented",
    ],
}

# Frozen before this screen; disjoint from the previous joint/piecewise cohorts.
PUBLIC_HELDOUT = (
    "Why do bicycle brakes become warm when a rider slows down?",
    "How does a rain gauge measure the amount of rainfall?",
    "What makes a compass needle point in a particular direction?",
    "A gardener plants five rows of nine seeds. How many seeds are planted?",
    "Why does a spinning top remain upright longer than a stationary one?",
    "How does insulation help a thermos keep a drink warm?",
    "What is the difference between mass and volume?",
    "Why do railway tracks use many sleepers instead of a few?",
    "Describe how yeast helps bread dough rise.",
    "What happens to the water level when ice floating in a glass melts?",
    "Why might two measurements of the same length differ slightly?",
    "A train covers sixty kilometres in half an hour. What is its average speed?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer_oracle(profile, gate_q24, up_q24):
    g, u = np.asarray(gate_q24, dtype=np.int64), np.asarray(up_q24, dtype=np.int64)
    if (g.shape != u.shape or np.any(g < -(48 << 24)) or np.any(g > 48 << 24)
            or np.any(u < -(128 << 24)) or np.any(u > 128 << 24)):
        raise ValueError("profile domain")
    gate, up = np.floor_divide(g, 4096), np.floor_divide(u, 4096)
    index = np.minimum(np.abs(np.floor_divide(gate, 64)), 1023)
    table = np.asarray(profile["table"], dtype=np.int64)
    return np.floor_divide((np.maximum(gate, 0) - table[index]) * up, 4096)


class Bridge(NumericBridge):
    def evaluate(self, gate, up):
        if gate.shape != up.shape or not 0 < gate.size <= 1 << 20:
            raise ValueError("native numeric frame bound")
        self.child.stdin.write(int(gate.size).to_bytes(4, "little"))
        self.child.stdin.write(np.ascontiguousarray(gate, dtype="<f4").tobytes())
        self.child.stdin.write(np.ascontiguousarray(up, dtype="<f4").tobytes())
        self.child.stdin.flush()
        status = self.child.stdout.read(1)
        if status == b"\x01":
            raise ValueError("profile domain rejection")
        if status != b"\x00":
            raise RuntimeError("native numeric process exited before reply")
        body = self.child.stdout.read(4 * gate.size)
        if len(body) != 4 * gate.size:
            raise RuntimeError("incomplete native numeric reply")
        result = np.frombuffer(body, dtype="<f4").reshape(gate.shape)
        expected = integer_oracle(self.profile, np.floor(gate.astype(np.float64)*2**24).astype(np.int64),
                                  np.floor(up.astype(np.float64)*2**24).astype(np.int64))
        if not np.array_equal(result.view(np.uint32), (expected/4096).astype(np.float32).view(np.uint32)):
            raise ValueError("native residual block differs from independent integer oracle")
        self.checked += gate.size
        return result


def public_values(lanes, bound):
    fixed = [-bound*2**24, -16*2**24-1, -16*2**24, -2**24, -4097, -4096, -1, 0, 1,
             4095, 4096, 2**24, 16*2**24-1, 16*2**24, bound*2**24]
    return np.array([fixed[i] if i < len(fixed) else i*982451653 % (2*bound*2**24+1)-bound*2**24
                     for i in range(lanes)], dtype=np.int64)


def paper_sources():
    library = json.loads((ROOT/"docs/data/research/paper-library.json").read_text())
    records = library["papers"]
    wanted = {"sigma", "grotto", "pika"}
    result = {r["id"]: {k: r[k] for k in ("source_url", "file", "sha256", "bytes")} for r in records if r["id"] in wanted}
    if set(result) != wanted:
        raise ValueError("paper provenance incomplete")
    for record in result.values():
        if sha((ROOT/record["file"]).read_bytes()) != record["sha256"]:
            raise ValueError("primary PDF digest mismatch")
    return result


def projections(config, cases, source):
    from pllm import lower_model
    baseline = json.loads((ROOT/CONFIGURATION["baseline"]).read_text())
    if any(source[k] != baseline["source"][k] for k in ("model", "revision", "body_fingerprint")):
        raise ValueError("historical source mismatch")
    case = next(c for c in cases if c["lanes"] == 256)
    bits = case["party_key_bits"] // 256
    peer_bits = case["peer_payload_bits"] // 256
    previous = json.loads((ROOT/CONFIGURATION["previous_screen"]).read_text())
    result = []
    for cohort in baseline["cohorts"]:
        outputs = cohort["output_tokens"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        sizes = {phase: [math.prod(op["output_shape"]) for op in plan.to_dict()[phase]["operations"] if op["operator"] == "silu"]
                 for phase in ("prefill", "decode")}
        count = sum(sizes["prefill"]) + (outputs-1)*sum(sizes["decode"])
        batches = sum(math.ceil(n/256) for n in sizes["prefill"]) + (outputs-1)*sum(math.ceil(n/256) for n in sizes["decode"])
        material = 2 * math.ceil(bits * count / 8)
        peer = peer_bits*count//8 + 2*case["rounds"]*case["header_bytes"]*batches
        old = next(x for x in previous["geometry_projections"] if x["output_tokens"] == outputs)
        result.append({"input_tokens": 39, "output_tokens": outputs, "semantic_plan_digest": plan.digest,
            "gated_elements": count, "bounded_256_lane_batches": batches,
            "both_party_raw_key_payload_bytes": material, "per_party_raw_key_payload_bytes": material//2,
            "peer_frame_bytes": peer, "peer_payload_bytes": peer_bits*count//8,
            "public_table_bytes_per_party": case["public_table_bytes"],
            "client_entry_exit_payload_bytes_if_not_resident": 25*count,
            "historical_prepared_covered_bytes": cohort["prepared_control"]["covered_all_link_body_bytes"],
            "cost_gate_pass": material+peer < cohort["prepared_control"]["covered_all_link_body_bytes"],
            "previous_numeric_screen_sha256": sha(json.dumps(old, sort_keys=True).encode()),
            "scope": "hypothetical complete Q24-to-Q12 gated blocks; changes numerics; not a matched decoder ranking",
            "upstream_protected_float_conversion_bytes": None, "whole_decoder_bytes": None,
            "executable_decoder": False})
    return result


def run(output):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    if host.available-host.reserve < 4*GiB:
        raise RuntimeError("numeric diagnostic needs 4 GiB headroom beyond host reserve")
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-garble", "--example", "masked_wire_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"])/"release/examples/masked_wire_probe"
    active = []

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
    bridge = None
    try:
        bridge = Bridge(binary, 12, 1024, active)
        profile = bridge.profile
        # Independent table formula; no model traces enter coefficient generation.
        table = [math.floor(4096*(i/64)/(1+math.exp(i/64))) for i in range(1024)]
        if profile["table"] != table:
            raise ValueError("residual table differs from independent formula")
        cases = []
        for repetition in range(CONFIGURATION["repetitions"]):
            for lanes in CONFIGURATION["lanes"][::1 if repetition % 2 == 0 else -1]:
                check()
                child = subprocess.Popen([sys.executable, "-m", "benchmarks.research.masked_wire_reference", "--child", str(binary), str(lanes)],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                active.append(child)
                try:
                    stdout, stderr = child.communicate(timeout=120)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    active.remove(child)
                if child.returncode:
                    raise RuntimeError(stderr)
                case = json.loads(stdout)
                expected = integer_oracle(profile, public_values(lanes, 48), public_values(lanes, 128)[::-1])
                if case["output_sha256"] != sha(expected.astype("<i8").tobytes()) or case["profile_digest"] != profile["digest"]:
                    raise ValueError("protected block oracle or profile mismatch")
                case["median_seconds"] = {k: statistics.median(case[k+"_ns"])/1e9 for k in ("issuance", "online", "clear")}
                cases.append(case)
        from scripts.decoder_probe_support import DecoderFixture, BODY, MODEL, REVISION
        fixture = DecoderFixture(inputs=64, outputs=4)
        projected = projections((fixture.source.path/"config.json").read_bytes(), cases, {"model": MODEL, "revision": REVISION, "body_fingerprint": BODY})
        numeric = screen(fixture, {"sigma-residual-q12": bridge}, PUBLIC_HELDOUT, check)
        check()
    finally:
        if bridge is not None:
            bridge.close()
        guard.close()
    files = ["benchmarks/research/masked_wire_reference.py", "benchmarks/research/gated_numeric_screen.py",
        "benchmarks/research/piecewise_gated_reference.py", "benchmarks/research/shared_rescale_reference.py",
        "scripts/decoder_probe_support.py", "crates/pllm-core/src/residual_silu_reference.rs",
        "crates/pllm-garble/src/compact_dpf.rs", "crates/pllm-garble/src/masked_wire_reference.rs",
        "crates/pllm-garble/examples/masked_wire_probe.rs", "Cargo.lock"]
    report = {"schema": "pllm.masked_wire_reference.v1", "configuration": CONFIGURATION, "papers": paper_sources(),
        "profile": profile, "native_cases": cases, "numeric_gate": numeric, "geometry_projections": projected,
        "source_sha256": {p: sha((ROOT/p).read_bytes()) for p in files},
        "control_sha256": {p: sha((ROOT/p).read_bytes()) for p in (CONFIGURATION["baseline"], CONFIGURATION["previous_screen"])},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "decision": "no-go" if not numeric["summary"]["sigma-residual-q12"]["screen_pass"] or not all(p["cost_gate_pass"] for p in projected) else "further-review",
        "executable_sdk": False, "privacy_reviewed": False, "protected_checkpoint_execution": False,
        "full_wire_bytes": None, "distributed_dealer": False,
        "limitations": CONFIGURATION["source_deviations"] + ["complete decoder operators, source import, KV, token feedback and representative quality remain unimplemented",
            "key payload is real encoded reference material; offline network framing, transport metadata and GPU storage are extra",
            "native local online time includes both evaluator kernels and trusted client boundary; no WAN or paper speed comparison"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"numeric": numeric["summary"], "checked_outputs": sum(c["checked_outputs"] for c in cases),
                      "projections": projected, "decision": report["decision"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", nargs=2, metavar=("BINARY", "LANES"))
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure_child([args.child[0], "--measure", args.child[1], str(CONFIGURATION["samples"])])))
        return
    if not args.output or args.output.exists():
        parser.error("choose a new --output reproduction path")
    run(args.output)


if __name__ == "__main__":
    main()
