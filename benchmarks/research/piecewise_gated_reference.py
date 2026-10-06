"""Public SiLU numeric policies, complete FSS block costs and held-out Qwen gate."""
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
import sys

import numpy as np

from benchmarks.research.shared_rescale_reference import ROOT, measure_child

# Frozen before the first checkpoint evaluation. No prompts fit these splines.
CONFIGURATION = {
    "fractions": [8, 9], "pieces": [16, 64], "layouts": ["separate", "vector"],
    "lanes": [1, 64], "samples": 3, "ring_bits": 32, "input_fraction": 16,
    "coefficient_fraction": 14, "gate_bound": 48, "up_bound": 128,
    "tail_boundary": 8, "tail_policy": "zero below -8; identity at/above +8; no clipping",
    "fit": "uniform mathematical SiLU secants; no checkpoint/calibration/held-out values",
    "rounding": "ties-to-even at float-to-Q16, both input rescalings, affine output and gated product",
    "maximum_input_tokens": 64, "selected_positions": 4,
    "numeric_screen_gate": "every held-out prefill and same-token decode selection matches; no domain rejection",
    "cost_gate": "uncompressed one-use material plus peer bodies for MLP alone below historical prepared whole-response covered bodies",
    "baseline": "docs/evidence/latent-response-network-qwen25-2026-09-28.json",
    "historical_block": "docs/evidence/research-nonlinear-block-reference-qwen25.json",
    "papers": {
        "fusefss": {"url": "https://arxiv.org/abs/2606.09551", "sha256": "6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0", "sections": ["I.2", "N", "O"]},
        "fss-mixed": {"url": "https://eprint.iacr.org/2020/1392", "sections": ["Figure 1", "Figure 14"]},
    },
    "source_deviations": [
        "PLLM affine SiLU coefficients and explicit tails, not paper-supplied SiLU spline parameters",
        "universal arithmetic DCF implements public-coefficient vector lookup; no packed Boolean predicates required",
        "separate control uses independent scalar coefficient masks/keys in one batched round, not a SIGMA system reproduction",
        "32-bit CPU reference with ties-even; no GPU, distributed issuance, authenticated transport or malicious security",
    ],
}
PUBLIC_HELDOUT = (
    "Why does a metal spoon feel colder than a wooden spoon in the same room?",
    "Describe how a lock in a canal helps a boat travel uphill.",
    "A gardener plants six rows of nine onions. How many onions were planted?",
    "Explain why a bicycle tyre gets harder when air is pumped into it.",
    "What is the difference between evaporation and boiling?",
    "How can a reader tell whether a news report is quoting an eyewitness?",
    "Give two reasons to label containers before storing them in a freezer.",
    "Why can a rainbow appear after a short rain shower?",
    "Describe a fair way to compare how quickly two brands of seed germinate.",
    "What happens to the perimeter of a square when its side length doubles?",
    "Explain how tree roots can help keep soil from washing away.",
    "Why might a railway timetable use a twenty-four-hour clock?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rounded(values, shift):
    q, r = np.divmod(values, 1 << shift)
    return q + ((r > 1 << (shift - 1)) | ((r == 1 << (shift - 1)) & (q % 2 != 0)))


def integer_oracle(profile, gate_q16, up_q16):
    """Independent NumPy integer path, without masked comparisons or triples."""
    if np.any(np.abs(gate_q16) > 48 << 16) or np.any(np.abs(up_q16) > 128 << 16):
        raise ValueError("profile domain")
    fraction = profile["fraction"]
    gate, up = rounded(gate_q16, 16 - fraction), rounded(up_q16, 16 - fraction)
    starts = np.array([x[0] for x in profile["intervals"]], dtype=np.int64)
    payloads = np.array([x[1] for x in profile["intervals"]], dtype=np.uint32).view(np.int32).astype(np.int64)
    coefficients = payloads[np.searchsorted(starts, gate & 0xFFFFFFFF, side="right") - 1]
    affine = coefficients[..., 0] * gate + coefficients[..., 1]
    activated = rounded(affine, 14)
    product = activated * up
    if np.any(np.abs(affine) >= 1 << 31) or np.any(np.abs(product) >= 1 << 31):
        raise ValueError("signed product overflow")
    return rounded(product, fraction)


def public_values(lanes, bound):
    fixed = [-bound*65536, -8*65536-1, -8*65536, -65536, -192, -128, -64,
             0, 64, 128, 192, 65536, 8*65536-1, 8*65536, bound*65536]
    return np.array([fixed[i] if i < len(fixed) else i*7919 % (bound*131072+1)-bound*65536
                     for i in range(lanes)], dtype=np.int64)


def oracle_digest(profile, lanes):
    result = integer_oracle(profile, public_values(lanes, 48), public_values(lanes, 128)[::-1])
    return sha((result & 0xFFFFFFFF).astype("<u4").tobytes())


class NumericBridge:
    def __init__(self, binary, fraction, pieces, active):
        self.active = active
        self.child = subprocess.Popen([str(binary), str(fraction), str(pieces)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        active.append(self.child)
        try:
            self.profile = json.loads(self.child.stdout.readline(1 << 20))
            if (self.profile["fraction"], self.profile["pieces"]) != (fraction, pieces):
                raise ValueError("native profile identity differs")
        except BaseException:
            self.close()
            raise
        self.checked = 0

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
        expected = integer_oracle(self.profile,
            np.rint(gate.astype(np.float64)*65536).astype(np.int64),
            np.rint(up.astype(np.float64)*65536).astype(np.int64))
        expected = (expected.astype(np.float64)/(1 << self.profile["fraction"])).astype(np.float32)
        if not np.array_equal(result.view(np.uint32), expected.view(np.uint32)):
            raise ValueError("native numeric output differs from independent integer oracle")
        self.checked += gate.size
        return result

    def close(self):
        if self.child.stdin:
            self.child.stdin.close()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()
        self.child.stdout.close()
        self.child.stderr.close()
        self.active.remove(self.child)


def native_cases(binary, profiles, active, check):
    cases = []
    for fraction, pieces, lanes in itertools.product(CONFIGURATION["fractions"], CONFIGURATION["pieces"], CONFIGURATION["lanes"]):
        for layout in CONFIGURATION["layouts"][::1 if pieces == 16 else -1]:
            check()
            child = subprocess.Popen([sys.executable, "-m", "benchmarks.research.piecewise_gated_reference", "--child",
                str(binary), str(fraction), str(pieces), layout, str(lanes), str(CONFIGURATION["samples"])],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            active.append(child)
            try:
                stdout, stderr = child.communicate(timeout=180)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                active.remove(child)
            if child.returncode:
                raise RuntimeError(stderr)
            case = json.loads(stdout)
            p = profiles[f"q{fraction}-{pieces}"]
            if case["profile_digest"] != p["digest"] or case["output_sha256"] != oracle_digest(p, lanes):
                raise ValueError("shared block profile or integer result differs")
            if [case[k] for k in ("fraction", "pieces", "layout", "lanes", "samples")] != [fraction, pieces, layout, lanes, CONFIGURATION["samples"]]:
                raise ValueError("shared block identity differs")
            case["median_seconds"] = {k: statistics.median(case[k+"_ns"])/1e9 for k in ("issuance", "online", "clear")}
            cases.append(case)
    return cases


def snapshot(runtime):
    return [np.array(a[:cache.length], copy=True) for cache in runtime.caches for a in (cache.key, cache.value)]


def state_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


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
                        stats["tail_elements"] += int(np.count_nonzero(np.abs(gate) >= 8))
                        stats["max_abs_gate"] = max(stats["max_abs_gate"], float(np.max(np.abs(gate))))
                        stats["max_abs_up"] = max(stats["max_abs_up"], float(np.max(np.abs(up))))
                        if bridge:
                            candidate = bridge.evaluate(gate, up)
                            stats["worst_block_error"] = max(stats["worst_block_error"], float(np.max(np.abs(result-candidate))))
                            return candidate
                return result

            runtime._local = local
            positions = []
            rejected = None
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for position in range(CONFIGURATION["selected_positions"]):
                    if position:
                        logits = runtime.forward_ids([reference[position-1][2]])[-1]
                    states = snapshot(runtime)
                    selected = int(np.argmax(logits))
                    if bridge is None:
                        reference.append((logits.copy(), states, selected))
                    expected, expected_states, expected_token = reference[position]
                    if len(states) != len(expected_states) or any(a.shape != b.shape for a, b in zip(states, expected_states, strict=True)):
                        raise ValueError("candidate full-KV shape differs")
                    positions.append({"position": position, "selection_match": selected == expected_token,
                        "all_logits_bit_exact": np.array_equal(logits.view(np.uint32), expected.view(np.uint32)),
                        "worst_abs_logit_error": float(np.max(np.abs(logits-expected))),
                        "mean_abs_logit_error": float(np.mean(np.abs(logits.astype(np.float64)-expected))),
                        "top5_recall": len(set(np.argsort(logits)[-5:]) & set(np.argsort(expected)[-5:]))/5,
                        "logits_sha256": sha(logits.tobytes()), "kv_sha256": state_digest(states),
                        "kv_bit_exact": all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(states, expected_states, strict=True)),
                        "kv_worst_abs_error": max(float(np.max(np.abs(a-b))) for a, b in zip(states, expected_states, strict=True))})
                if pending or seen != layers:
                    raise ValueError("candidate lacks complete semantic nonlinear coverage")
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
            "screen_pass": len(positions) == 4*len(PUBLIC_HELDOUT) and all(p["selection_match"] for p in positions),
            "native_oracle_checked_elements": bridges[name].checked if name in bridges else 0}
    return {"source": fixture.lock(), "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
        "reference": "pinned compiled W8A8 clear-kernel same-token trajectory; not upstream FP32 or broad task quality",
        "summary": summary, "trials": trials, "protected_checkpoint_execution": False}


def geometry(config, cases, source):
    from pllm import lower_model
    baseline = json.loads((ROOT/CONFIGURATION["baseline"]).read_text())
    if any(baseline["source"][k] != source[k] for k in ("model", "revision", "body_fingerprint")):
        raise ValueError("historical cost reference source differs")
    result = []
    for control in baseline["cohorts"]:
        outputs = control["output_tokens"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        counts = {phase: [math.prod(op["output_shape"]) for op in plan.to_dict()[phase]["operations"] if op["operator"] == "silu"]
                  for phase in ("prefill", "decode")}
        count = sum(counts["prefill"])+(outputs-1)*sum(counts["decode"])
        rows = []
        for case in cases:
            lanes = case["lanes"]
            batches = sum(math.ceil(n/lanes) for n in counts["prefill"])+(outputs-1)*sum(math.ceil(n/lanes) for n in counts["decode"])
            keys = 2*case["party_key_payload_bytes"]*count//lanes
            peer = (case["peer_frame_bytes"]-156*case["rounds"])*count//lanes+batches*156*case["rounds"]
            rows.append({"fraction": case["fraction"], "pieces": case["pieces"], "layout": case["layout"], "batch_lanes": lanes,
                "both_party_material_payload_bytes": keys, "peer_frame_bytes": peer,
                "input_output_sharing_payload_bytes_if_not_resident": 24*count,
                "material_plus_peer_bytes": keys+peer,
                "cost_gate_pass": keys+peer < control["prepared_control"]["covered_all_link_body_bytes"],
                "issuance_seconds_linear_projection": case["median_seconds"]["issuance"]*count/lanes,
                "local_online_seconds_linear_projection": case["median_seconds"]["online"]*count/lanes})
        result.append({"input_tokens": 39, "output_tokens": outputs, "semantic_plan_digest": plan.digest, "gated_elements": count,
            "historical_prepared_covered_bytes": control["prepared_control"]["covered_all_link_body_bytes"],
            "scope": "MLP raw material/peer projection versus historical whole-response bodies; not a matched runtime ranking",
            "cases": rows, "executable_decoder": False})
    return result


def run(output):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    if host.available-host.reserve < 4*GiB:
        raise RuntimeError("piecewise checkpoint diagnostic requires 4 GiB beyond host reserve")
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-core", "--example", "piecewise_gated_probe",
                    "-p", "pllm-garble", "--example", "shared_piecewise_gated_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binaries = Path(metadata["target_directory"])/"release/examples"
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
            bridges[f"q{fraction}-{pieces}"] = NumericBridge(binaries/"piecewise_gated_probe", fraction, pieces, active)
        profiles = {k: v.profile for k, v in bridges.items()}
        cases = native_cases(binaries/"shared_piecewise_gated_probe", profiles, active, check)
        from scripts.decoder_probe_support import DecoderFixture
        fixture = DecoderFixture(inputs=CONFIGURATION["maximum_input_tokens"], outputs=CONFIGURATION["selected_positions"])
        numeric = numeric_gate(fixture, bridges, check)
        projected = geometry((fixture.source.path/"config.json").read_bytes(), cases, fixture.lock())
        check()
    finally:
        for bridge in bridges.values():
            bridge.close()
        guard.close()
    files = ["benchmarks/research/piecewise_gated_reference.py", "benchmarks/research/shared_rescale_reference.py", "scripts/decoder_probe_support.py",
        "crates/pllm-core/src/piecewise_gated_reference.rs", "crates/pllm-core/examples/piecewise_gated_probe.rs",
        "crates/pllm-garble/src/shared_affine_lookup_reference.rs", "crates/pllm-garble/src/shared_piecewise_gated_reference.rs",
        "crates/pllm-garble/src/shared_arithmetic_reference.rs", "crates/pllm-garble/src/shared_rescale_reference.rs",
        "crates/pllm-garble/src/interval_fss.rs", "crates/pllm-garble/src/compact_dcf.rs", "crates/pllm-garble/examples/shared_piecewise_gated_probe.rs", "Cargo.lock"]
    decisions = {}
    for name, profile in profiles.items():
        numeric_pass = numeric["summary"][name]["screen_pass"]
        cost_pass = all(any(c["cost_gate_pass"] and c["fraction"] == profile["fraction"] and c["pieces"] == profile["pieces"]
                            for c in projection["cases"]) for projection in projected)
        decisions[name] = {"numeric_screen_pass": numeric_pass, "cost_screen_pass": cost_pass,
                           "further_integration_justified": numeric_pass and cost_pass}
    report = {"schema": "pllm.piecewise_gated_reference.v1", "configuration": CONFIGURATION, "profiles": profiles,
        "cases": cases, "numeric_gate": numeric, "geometry_projections": projected,
        "decision": "requires further review" if any(d["further_integration_justified"] for d in decisions.values()) else "no-go",
        "candidate_decisions": decisions,
        "source_sha256": {p: sha((ROOT/p).read_bytes()) for p in files},
        "control_sha256": {p: sha((ROOT/p).read_bytes()) for p in (CONFIGURATION["baseline"], CONFIGURATION["historical_block"])},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "executable_sdk": False, "privacy_reviewed": False, "whole_generation_quality": False, "full_wire_bytes": None,
        "limitations": ["clear checkpoint numeric override; native protected block measured separately on public synthetic inputs",
            "input Q16 share domain is caller-enforced; secret float-to-fixed bridge/range proofs absent",
            "both parties and trusted dealer co-located; authenticated transport and distributed issuance absent",
            "projected material is uncompressed payload; offline framing, linear/attention/norm/feedback/model distribution extra",
            "all four input/activation/output rescalings, two triples, masks and coefficient keys are included",
            "historical Q7 construction uses different numerics; only same-profile separate/vector layouts rank together"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"numeric": numeric["summary"], "decision": report["decision"], "native_checked_outputs": sum(c["checked_outputs"] for c in cases)}))


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
