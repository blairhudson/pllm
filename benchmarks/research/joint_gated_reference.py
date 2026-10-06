"""Frozen high-precision joint numeric gate and pre-issuance protocol rejection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess

import numpy as np

from benchmarks.research.piecewise_gated_reference import NumericBridge, snapshot, state_digest
from benchmarks.research.shared_rescale_reference import ROOT

# Freeze all four policies and the decision rule before evaluating the checkpoint.
# Uniform mathematical secants; no public or private model trace fits coefficients.
CONFIGURATION = {
    "fractions": [12, 16], "pieces": [512, 2048], "coefficient_fraction": 28,
    "gate_bound": 48, "up_bound": 128, "tail_boundary": 16,
    "tail_policy": "zero below -16; identity at/above +16; reject outside public bounds",
    "fit": "uniform mathematical SiLU secants; no checkpoint or prompt calibration",
    "rounding": "float-to-Qf ties-even, then one ties-even rounding of complete affine-times-up numerator to Qf",
    "maximum_input_tokens": 64, "selected_positions": 4,
    "numeric_gate": "all 12 prefill and 36 same-token decode selections match without domain rejection",
    "cost_gate": "complete MLP one-use material plus peer bodies below historical prepared whole-response covered bodies",
    "baseline": "docs/evidence/latent-response-network-qwen25-2026-09-28.json",
    "previous_screen": "docs/evidence/research-piecewise-gated-reference-qwen25.json",
    "candidate": "one-use dense two-input function shares include signed decoding, piece selection and final rounding",
    "alternatives": ["uncompressed half-gates with 16-byte labels", "public shifted-polynomial coefficients"],
    "protocol_word_bits": 32,
    "scope": "public clear numeric experiment and exact layout-specific algebra/cost gate; no protected checkpoint execution",
}

PUBLIC_HELDOUT = (
    "How does a thermostat keep the temperature in a room nearly constant?",
    "Why are expansion gaps left between sections of a concrete bridge?",
    "A shop packs seven pencils in each of eight boxes. How many pencils are packed?",
    "Explain why stars appear to move across the night sky.",
    "What is the purpose of a fuse in an electrical circuit?",
    "How can rotating crops help a farmer maintain healthy soil?",
    "Describe a way to check whether a measuring jug is accurate.",
    "Why does a wet shirt dry faster on a windy day?",
    "What does the scale on a map tell someone planning a journey?",
    "Explain how a pulley can change the direction of a force.",
    "Why should a scientific experiment include a control group?",
    "A recipe needs three cups of flour for two loaves. How much is needed for six loaves?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer_oracle(profile, gate, up):
    """Exact segmented integer arithmetic, independent of Rust's i128 product.

    Decompose the affine value before multiplication. Every NumPy intermediate
    fits signed i64, although the unrounded numerator can exceed 64 bits.
    """
    f, c = profile["fraction"], profile["coefficient_fraction"]
    gate, up = np.asarray(gate, dtype=np.int64), np.asarray(up, dtype=np.int64)
    if (gate.shape != up.shape or np.any(gate < -(48 << f)) or np.any(gate > 48 << f)
            or np.any(up < -(128 << f)) or np.any(up > 128 << f)):
        raise ValueError("profile domain")
    starts = np.array([row[0] for row in profile["intervals"]], dtype=np.int64)
    coefficients = np.array([row[1] for row in profile["intervals"]], dtype=np.int64)
    chosen = coefficients[np.searchsorted(starts, gate, side="right") - 1]
    affine = chosen[..., 0] * gate + chosen[..., 1]
    high, low = np.divmod(affine, 1 << c)
    q, r = np.divmod(high * up, 1 << f)
    correction, remainder = np.divmod(r * (1 << c) + low * up, 1 << (c + f))
    q = q + correction
    half = 1 << (c + f - 1)
    return q + ((remainder > half) | ((remainder == half) & (q % 2 != 0)))


def scalar_oracle(profile, gate, up):
    """Unbounded Python integers for public boundary checks of both native paths."""
    _, (a, b) = next(row for row in reversed(profile["intervals"]) if gate >= row[0])
    q, r = divmod((a * gate + b) * up, 1 << (profile["coefficient_fraction"] + profile["fraction"]))
    half = 1 << (profile["coefficient_fraction"] + profile["fraction"] - 1)
    return q + int(r > half or (r == half and q % 2 != 0))


class JointBridge(NumericBridge):
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
        raw = self.child.stdout.read(4 * gate.size)
        if len(raw) != 4 * gate.size:
            raise RuntimeError("incomplete native numeric reply")
        result = np.frombuffer(raw, dtype="<f4").reshape(gate.shape)
        scale = 1 << self.profile["fraction"]
        expected = integer_oracle(self.profile, np.rint(gate.astype(np.float64) * scale).astype(np.int64),
                                  np.rint(up.astype(np.float64) * scale).astype(np.int64))
        expected = (expected.astype(np.float64) / scale).astype(np.float32)
        if not np.array_equal(result.view(np.uint32), expected.view(np.uint32)):
            raise ValueError("native joint output differs from segmented integer oracle")
        self.checked += gate.size
        return result


def boundary_checks(bridge):
    p = bridge.profile
    scale = 1 << p["fraction"]
    pairs = [(g, u) for start, _ in p["intervals"]
             for g in (start - 1, start, start + 1, -48 * scale, 48 * scale)
             if abs(g) <= 48 * scale for u in (-128 * scale, -1, 0, 1, 128 * scale)]
    g, u = np.asarray(pairs, dtype=np.int64).T
    expected = np.array([scalar_oracle(p, int(a), int(b)) for a, b in pairs], dtype=np.int64)
    if not np.array_equal(integer_oracle(p, g, u), expected):
        raise ValueError("segmented oracle differs from unbounded integers")
    result = bridge.evaluate((g / scale).astype(np.float32), (u / scale).astype(np.float32))
    if not np.array_equal(result, (expected / scale).astype(np.float32)):
        raise ValueError("boundary native result differs")
    # A Boolean-affinity witness: disjoint input bits individually produce zero,
    # together a nonzero output. XOR/NOT alone cannot implement this function.
    witnesses = ((0, 0), (scale, 0), (0, scale), (scale, scale))
    values = [scalar_oracle(p, g, u) for g, u in witnesses]
    wg, wu = np.asarray(witnesses, dtype=np.int64).T
    native = bridge.evaluate((wg / scale).astype(np.float32), (wu / scale).astype(np.float32))
    if not np.array_equal(native, (np.asarray(values) / scale).astype(np.float32)):
        raise ValueError("native non-affinity witness differs")
    xor = values[0] ^ values[1] ^ values[2] ^ values[3]
    if xor == 0:
        raise ValueError("non-affinity witness missing")
    return {"checked_outputs": len(pairs) + len(witnesses), "output_sha256": sha(result.tobytes() + native.tobytes()),
            "non_affinity_values": values, "non_affinity_xor": xor}


def numeric_gate(fixture, bridges, check):
    trials = []
    layers = {op["layer"] for op in fixture.plan.prefill["operations"] if op["operator"] == "silu"}
    for text in PUBLIC_HELDOUT:
        check()
        ids = fixture.tokens(text)
        reference = []
        for name, bridge in [("control", None), *bridges.items()]:
            check()
            runtime = fixture.compiled.runtime(fixture.remote)
            original = runtime._local
            pending, seen = {}, set()
            stats = {"elements": 0, "gate_outside": 0, "up_outside": 0, "tail_elements": 0,
                     "max_abs_gate": 0.0, "max_abs_up": 0.0, "worst_block_error": 0.0}

            def local(operation, values, *state):
                check()
                result = original(operation, values, *state)
                if operation["operator"] == "silu":
                    pending[operation["id"]] = np.array(values[operation["inputs"][0]], copy=True)
                elif operation["operator"] == "multiply":
                    keys = [key for key in operation["inputs"] if key in pending]
                    if len(keys) == 1:
                        gate = pending.pop(keys[0])
                        up = np.asarray(values[next(key for key in operation["inputs"] if key != keys[0])])
                        seen.add(operation["layer"])
                        stats["elements"] += gate.size
                        stats["gate_outside"] += int(np.count_nonzero(np.abs(gate) > 48))
                        stats["up_outside"] += int(np.count_nonzero(np.abs(up) > 128))
                        stats["tail_elements"] += int(np.count_nonzero(np.abs(gate) >= 16))
                        stats["max_abs_gate"] = max(stats["max_abs_gate"], float(np.max(np.abs(gate))))
                        stats["max_abs_up"] = max(stats["max_abs_up"], float(np.max(np.abs(up))))
                        if bridge:
                            candidate = bridge.evaluate(gate, up)
                            stats["worst_block_error"] = max(stats["worst_block_error"], float(np.max(np.abs(result - candidate))))
                            return candidate
                return result

            runtime._local = local
            positions, rejected = [], None
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for position in range(CONFIGURATION["selected_positions"]):
                    if position:
                        logits = runtime.forward_ids([reference[position - 1][2]])[-1]
                    states = snapshot(runtime)
                    selected = int(np.argmax(logits))
                    if bridge is None:
                        reference.append((logits.copy(), states, selected))
                    expected, expected_states, expected_token = reference[position]
                    if len(states) != len(expected_states) or any(a.shape != b.shape for a, b in zip(states, expected_states, strict=True)):
                        raise ValueError("full-KV layout differs")
                    positions.append({"position": position, "selection_match": selected == expected_token,
                        "all_logits_bit_exact": np.array_equal(logits.view(np.uint32), expected.view(np.uint32)),
                        "worst_abs_logit_error": float(np.max(np.abs(logits - expected))),
                        "mean_abs_logit_error": float(np.mean(np.abs(logits.astype(np.float64) - expected))),
                        "top5_recall": len(set(np.argsort(logits)[-5:]) & set(np.argsort(expected)[-5:])) / 5,
                        "logits_sha256": sha(logits.tobytes()), "kv_sha256": state_digest(states),
                        "kv_bit_exact": all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(states, expected_states, strict=True)),
                        "kv_worst_abs_error": max(float(np.max(np.abs(a-b))) for a, b in zip(states, expected_states, strict=True))})
                if pending or seen != layers:
                    raise ValueError("joint override lacks full semantic MLP coverage")
            except ValueError as error:
                if str(error) != "profile domain rejection":
                    raise
                rejected = str(error)
            finally:
                del runtime._local
            trials.append({"candidate": name, "input_tokens": len(ids), "public_tokens_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
                           "positions": positions, "domain": stats, "rejection": rejected})
            original = None
            del runtime, local
    summary = {}
    for name in ["control", *bridges]:
        rows = [row for row in trials if row["candidate"] == name]
        positions = [p for row in rows for p in row["positions"]]
        summary[name] = {"attempted_prompts": len(rows), "rejected_prompts": sum(r["rejection"] is not None for r in rows),
            "prefill_matches": sum(p["selection_match"] for p in positions if p["position"] == 0),
            "decode_matches": sum(p["selection_match"] for p in positions if p["position"] > 0),
            "completed_positions": len(positions), "bit_exact_positions": sum(p["all_logits_bit_exact"] for p in positions),
            "exact_kv_positions": sum(p["kv_bit_exact"] for p in positions),
            "worst_abs_logit_error": max((p["worst_abs_logit_error"] for p in positions), default=None),
            "mean_top5_recall": statistics.mean(p["top5_recall"] for p in positions) if positions else None,
            "screen_pass": len(positions) == 4 * len(PUBLIC_HELDOUT) and all(p["selection_match"] for p in positions),
            "native_oracle_checked_elements": bridges[name].checked if name in bridges else 0}
    return {"source": fixture.lock(), "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
            "reference": "pinned compiled W8A8 same-token clear trajectory, not upstream FP32 or broad task quality",
            "summary": summary, "trials": trials, "protected_checkpoint_execution": False}


def protocol_costs(profile, count, prepared_bytes):
    # Signed inclusive bounds need f+7 gate bits and f+9 up bits: +128 is
    # excluded by an f+8-bit signed domain. Never price a smaller-than-valid ring.
    f = profile["fraction"]
    gate_bits = (48 << f).bit_length() + 1
    up_bits = (128 << f).bit_length() + 1
    index_bits = gate_bits + up_bits
    mask_bytes = math.ceil(gate_bits / 8) + math.ceil(up_bits / 8)
    entries = 1 << index_bits
    dense_material = (8 * entries + 2 * mask_bytes) * count
    peer = 2 * mask_bytes * count
    # One AND is a conservative floor, not a full-circuit gate-count estimate.
    half_gate_floor = 2 * 16 * count
    return {
        "gate_ring_bits": gate_bits, "up_ring_bits": up_bits, "table_entries_per_lane": entries,
        "dense_function_shares": {
            "both_party_table_and_mask_payload_bytes": dense_material,
            "peer_opening_payload_bytes": peer,
            "material_plus_peer_payload_bytes": dense_material + peer,
            "input_output_sharing_payload_bytes_if_not_resident": (2 * mask_bytes + 8) * count,
            "peer_rounds": 1,
            "cost_gate_pass": dense_material + peer < prepared_bytes,
            "issuance_cpu_seconds": None, "online_cpu_seconds": None, "full_wire_bytes": None,
            "allocated": False,
        },
        "uncompressed_half_gates": {
            "minimum_and_gates_per_lane": 1, "ciphertext_bytes_per_and": 32,
            "ciphertext_body_floor_bytes": half_gate_floor,
            "cost_gate_ruled_out_by_floor": half_gate_floor >= prepared_bytes,
            "complete_material_and_peer_bytes": None, "input_transfer_bytes": None,
            "circuit_implemented": False,
        },
        "hypothetical_compressed_function_shares": {
            "peer_opening_payload_bytes": peer,
            "remaining_material_budget_bytes": max(0, prepared_bytes - peer),
            "remaining_bytes_per_lane": max(0, prepared_bytes - peer) / count,
            "complete_material_and_peer_bytes": None,
            "cost_gate_pass": None, "construction_available": False,
        },
    }


def geometry(config, profiles, source):
    from pllm import lower_model
    baseline = json.loads((ROOT / CONFIGURATION["baseline"]).read_text())
    if any(baseline["source"][k] != source[k] for k in ("model", "revision", "body_fingerprint")):
        raise ValueError("historical comparator source differs")
    result = []
    for cohort in baseline["cohorts"]:
        outputs = cohort["output_tokens"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        counts = {phase: sum(math.prod(op["output_shape"]) for op in plan.to_dict()[phase]["operations"] if op["operator"] == "silu")
                  for phase in ("prefill", "decode")}
        count = counts["prefill"] + (outputs - 1) * counts["decode"]
        covered = cohort["prepared_control"]["covered_all_link_body_bytes"]
        result.append({"input_tokens": 39, "output_tokens": outputs, "semantic_plan_digest": plan.digest,
            "gated_elements": count, "historical_prepared_covered_bytes": covered,
            "total_bytes_per_lane_budget": covered / count,
            "scope": "raw MLP layout admission versus historical whole-response bodies, not matched runtime ranking",
            "profiles": {name: protocol_costs(profile, count, covered) for name, profile in profiles.items()},
            "executable_decoder": False})
    return result


def run(output):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("joint checkpoint diagnostic requires 4 GiB beyond host reserve")
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-core", "--example", "joint_gated_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"]) / "release/examples/joint_gated_probe"
    audit = json.loads(subprocess.check_output([str(binary), "--audit"]))
    active, bridges = [], {}

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
        for fraction, pieces in itertools.product(CONFIGURATION["fractions"], CONFIGURATION["pieces"]):
            bridges[f"q{fraction}-{pieces}"] = JointBridge(binary, fraction, pieces, active)
        profiles = {name: bridge.profile for name, bridge in bridges.items()}
        boundaries = {name: boundary_checks(bridge) for name, bridge in bridges.items()}
        for bridge in bridges.values():
            bridge.checked = 0
        from scripts.decoder_probe_support import DecoderFixture
        # Cost admission precedes checkpoint loading and any protected material.
        from pllm import Model
        from pllm.model_loader import resolve_model
        from scripts.decoder_probe_support import BODY, MODEL, REVISION
        source = resolve_model(Model.hf(MODEL, revision=REVISION))
        projected = geometry((source.path / "config.json").read_bytes(), profiles,
                             {"model": MODEL, "revision": REVISION, "body_fingerprint": BODY})
        check()
        fixture = DecoderFixture(inputs=CONFIGURATION["maximum_input_tokens"], outputs=CONFIGURATION["selected_positions"])
        numeric = numeric_gate(fixture, bridges, check)
        check()
    finally:
        for bridge in bridges.values():
            bridge.close()
        guard.close()
    decisions = {}
    for name in profiles:
        numeric_ok = numeric["summary"][name]["screen_pass"]
        dense_ok = all(p["profiles"][name]["dense_function_shares"]["cost_gate_pass"] for p in projected)
        decisions[name] = {"numeric_screen_pass": numeric_ok, "dense_cost_screen_pass": dense_ok,
            "half_gates_ruled_out_by_floor": all(p["profiles"][name]["uncompressed_half_gates"]["cost_gate_ruled_out_by_floor"] for p in projected),
            "compressed_joint_construction_available": False,
            "further_integration_justified": numeric_ok and dense_ok}
    files = ["benchmarks/research/joint_gated_reference.py", "benchmarks/research/piecewise_gated_reference.py",
             "benchmarks/research/shared_rescale_reference.py", "scripts/decoder_probe_support.py",
             "crates/pllm-core/src/joint_gated_reference.rs", "crates/pllm-core/examples/joint_gated_probe.rs",
             "crates/pllm-garble/src/boolean.rs", "Cargo.lock"]
    report = {"schema": "pllm.joint_gated_reference.v1", "configuration": CONFIGURATION,
        "profiles": profiles, "boundary_checks": boundaries, "algebra_audit": audit,
        "numeric_gate": numeric, "geometry_projections": projected, "candidate_decisions": decisions,
        "decision": "go-to-bounded-prototype" if any(d["further_integration_justified"] for d in decisions.values()) else "no-go",
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in files},
        "control_sha256": {p: sha((ROOT / p).read_bytes()) for p in (CONFIGURATION["baseline"], CONFIGURATION["previous_screen"])},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "executable_sdk": False, "privacy_reviewed": False, "protected_prototype_issued": False,
        "whole_generation_quality": False, "full_wire_bytes": None,
        "limitations": ["new numeric identity; one final rounding cannot inherit previous piecewise or W8A8 parity",
            "uniform high precision profiles frozen before fresh held-out evaluation; no post-result fitting",
            "dense table shares are a fully specified mathematical candidate rejected before allocation, not measured FSS",
            "half-gates floor applies only to the stated uncompressed layout, not all secure computation",
            "narrow rings assume valid source-bound fixed-point shares; upstream conversion/range enforcement remains unimplemented",
            "raw material and peer bodies omit framing, other operators, issuance, distributed dealer, model delivery and full wire",
            "free compressed joint keys are an unavailable hypothetical; unknown costs never pass admission"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"numeric": numeric["summary"], "algebra": audit, "decision": report["decision"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new reproduction path")
    run(args.output)


if __name__ == "__main__":
    main()
