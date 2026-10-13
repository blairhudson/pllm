"""Tiny public geometry and software-AEAD control; never attests or runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "docs/evidence/research-all-link-frontier.json"
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"
CONFIRM = ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"
HEADER = struct.Struct(">4s16s32sBHQHIB")
FRAME_OVERHEAD = HEADER.size + 12 + 16  # header, explicit nonce, GCM tag
MAX_VALUES = 65536


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_bits(bound):
    if type(bound) is not int or bound < 0:
        raise ValueError("invalid public bound")
    return max(1, bound.bit_length() + 1)


def pack_signed(values, bits):
    """Fixed public width, little-endian bit stream, canonical zero tail."""
    if not 1 <= bits <= 32 or not 0 < len(values) <= MAX_VALUES:
        raise ValueError("invalid bounded frame")
    packed = bytearray()
    accumulator = used = 0
    for value in values:
        if type(value) is not int or not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
            raise ValueError("integer out of range")
        accumulator |= (value & ((1 << bits) - 1)) << used
        used += bits
        while used >= 8:
            packed.append(accumulator & 255)
            accumulator >>= 8
            used -= 8
    if used:
        packed.append(accumulator)
    return bytes(packed)


def unpack_signed(raw, count, bits):
    if not 1 <= bits <= 32 or not 0 < count <= MAX_VALUES or len(raw) != (count * bits + 7) // 8:
        raise ValueError("invalid bounded frame")
    tail = count * bits % 8
    if tail and raw[-1] >> tail:
        raise ValueError("noncanonical padding")
    value = int.from_bytes(raw, "little")
    result = []
    for index in range(count):
        item = (value >> (index * bits)) & ((1 << bits) - 1)
        result.append(item - (1 << bits) if item & (1 << (bits - 1)) else item)
    return result


class SoftwareChannel:
    """In-memory ordered transport control, NOT durable lifecycle or TEE proof."""

    def __init__(self, key, session, epoch, nonce_prefix):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(session) != 16 or len(epoch) != 32 or len(nonce_prefix) != 4:
            raise ValueError("invalid context")
        self.aead = AESGCM(key)
        self.session, self.epoch, self.prefix = session, epoch, nonce_prefix
        self.sent = self.received = 0

    def seal(self, raw, *, phase, stage, rows, bits):
        if self.aead is None or self.sent >= (1 << 64) - 1 or len(raw) > MAX_VALUES * 4:
            raise ValueError("closed, exhausted or oversized channel")
        header = HEADER.pack(b"ALB1", self.session, self.epoch, phase, stage,
                             self.sent, rows, len(raw), bits)
        nonce = self.prefix + self.sent.to_bytes(8, "big")
        body = header + nonce + self.aead.encrypt(nonce, raw, header)
        self.sent += 1
        return body

    def open(self, body, *, phase, stage, rows, bits):
        if self.aead is None or not FRAME_OVERHEAD <= len(body) <= MAX_VALUES * 4 + FRAME_OVERHEAD:
            raise ValueError("closed or invalid frame length")
        header = body[:HEADER.size]
        magic, session, epoch, actual_phase, actual_stage, counter, actual_rows, length, actual_bits = HEADER.unpack(header)
        if (magic, session, epoch, actual_phase, actual_stage, counter, actual_rows, actual_bits) != (
                b"ALB1", self.session, self.epoch, phase, stage, self.received, rows, bits):
            raise ValueError("context or replay mismatch")
        nonce = body[HEADER.size:HEADER.size + 12]
        if len(body) != length + FRAME_OVERHEAD or nonce != self.prefix + counter.to_bytes(8, "big"):
            raise ValueError("length or nonce mismatch")
        raw = self.aead.decrypt(nonce, body[HEADER.size + 12:], header)
        self.received += 1  # authentication failure never advances receive counter
        return raw

    def close(self):
        self.aead = None  # Python reference disposal is NOT secure zeroization.


def synthetic_control():
    try:
        import cryptography
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError:
        return {"executed": False, "reason": "cryptography unavailable", "tee_proof": False}
    # Public deterministic test keys ONLY. Not an attestation/key exchange.
    epoch = hashlib.sha256(b"public synthetic immutable weights/source epoch").digest()
    keys = HKDF(algorithm=hashes.SHA256(), length=64, salt=epoch,
                info=b"software-control/direction-keys").derive(b"public-test-secret")
    session = b"public-session01"
    uplink = [SoftwareChannel(keys[:32], session, epoch, b"UP00") for _ in range(2)]
    downlink = [SoftwareChannel(keys[32:], session, epoch, b"DN00") for _ in range(2)]
    rng = random.Random(6101)
    checked = charged = 0
    for trial in range(32):
        weights = [[rng.randint(-128, 127) for _ in range(11)] for _ in range(7)]
        inputs = [[rng.randint(-128, 127) for _ in range(11)] for _ in range(3)]
        if trial == 0:
            weights = [[-128] * 11 for _ in range(7)]
            inputs = [[-128] * 11, [127] * 11, [0] * 11]
        expected = [sum(w * x for w, x in zip(row, vector)) for vector in inputs for row in weights]
        raw = pack_signed([x for row in inputs for x in row], 8)
        frame = uplink[0].seal(raw, phase=1, stage=trial, rows=3, bits=8)
        decoded = unpack_signed(uplink[1].open(frame, phase=1, stage=trial, rows=3, bits=8), 33, 8)
        actual = [sum(w * decoded[r * 11 + c] for c, w in enumerate(row))
                  for r in range(3) for row in weights]
        bound = 128 * max(sum(abs(w) for w in row) for row in weights)
        bits = signed_bits(bound)
        if actual != expected or any(abs(x) > bound for x in actual):
            raise AssertionError("exact integer parity or public bound failed")
        reply = downlink[0].seal(pack_signed(actual, bits), phase=1, stage=trial, rows=3, bits=bits)
        if trial == 0:
            corrupt = reply[:-1] + bytes([reply[-1] ^ 1])
            try:
                downlink[1].open(corrupt, phase=1, stage=trial, rows=3, bits=bits)
            except InvalidTag:
                pass
            else:
                raise AssertionError("tampered tag accepted")
        restored = unpack_signed(downlink[1].open(reply, phase=1, stage=trial, rows=3, bits=bits), 21, bits)
        if restored != expected:
            raise AssertionError("AEAD round trip changed exact integers")
        try:
            downlink[1].open(reply, phase=1, stage=trial, rows=3, bits=bits)
        except ValueError:
            pass
        else:
            raise AssertionError("replay accepted")
        checked += len(actual)
        charged += len(frame) + len(reply)
    for channel in uplink + downlink:
        channel.close()
    return {"executed": True, "tee_proof": False, "cryptography_version": cryptography.__version__,
            "trials": 32, "exact_output_integers": checked, "serialized_aead_bytes": charged,
            "tamper_rejected": True, "replay_rejected": True,
            "scope": "public tiny Python integer + AES-256-GCM software control; no model execution"}


def budget(control, geometry, confirmation):
    if not control["complete"] or not geometry["complete"] or not confirmation["complete"]:
        raise ValueError("incomplete evidence")
    ledger = control["control"]
    if ledger["pair_bytes"] != sum(ledger["classes"].values()):
        raise ValueError("ledger does not conserve")
    if geometry["source"]["body_fingerprint"] != ledger["body_fingerprint"]:
        raise ValueError("source body mismatch")
    if ledger["evidence_sha256"] != digest(CONFIRM):
        raise ValueError("confirmation hash mismatch")
    reports = [r["execution"]["details"]["report"] for r in confirmation["results"]
               if r["execution"]["case"]["name"] == "prepared/digit8"]
    if len(reports) != 3:
        raise ValueError("changed control replicate count")
    if {s["stage"] for s in geometry["stages"]} != set(ledger["stages"]) or len(geometry["stages"]) != 96:
        raise ValueError("changed stage set")
    stages = []
    for stage in geometry["stages"]:
        name = stage["stage"]
        terminal = name.startswith("layers.23.") and not name.endswith("qkv_proj")
        prefill = 1 if terminal else 150
        # Cold prefill + seven decode rows; exact-repeat prefix hit + seven decode rows.
        chunks = [min(16, prefill - start) for start in range(0, prefill, 16)] + [1] * 14
        out_features, in_features = stage["shape"]
        modulus = stage["odd_radix"]["modulus"]
        if type(modulus) is not int or modulus < 3 or modulus % 2 != 1:
            raise ValueError("invalid recorded public bound")
        bound = (modulus - 1) // 2
        if bound > in_features * 128 * 128:
            raise ValueError("recorded bound exceeds conservative fan-in bound")
        bits = signed_bits(bound)
        row = {"stage": name, "weight_sha256": stage["weight_sha256"], "shape": stage["shape"],
               "terminal_prefill": terminal, "pair_rows": sum(chunks), "frames_each_direction": len(chunks),
               "public_signed_bound": bound, "public_output_bits": bits,
               "input_payload_bytes": sum(chunks) * in_features,
               "output_packed_bytes": sum((n * out_features * bits + 7) // 8 for n in chunks),
               "output_fanin_packed_bytes": sum((n * out_features * signed_bits(in_features * 16384) + 7) // 8 for n in chunks),
               "output_int32_bytes": sum(chunks) * out_features * 4,
               "aead_overhead_bytes": 2 * len(chunks) * FRAME_OVERHEAD,
               "pair_linear_macs": sum(chunks) * in_features * out_features}
        stages.append(row)
    rows = sum(s["pair_rows"] for s in stages)
    for report in reports:
        # Historical aggregate row ledger independently reconciles terminal geometry.
        if report["configuration"]["source_lock_digest"] != geometry["source"]["source_lock_digest"]:
            raise ValueError("source lock mismatch")
        if any(r["status"] != "completed" or r["model_fingerprint"] != ledger["body_fingerprint"]
               for r in report["runs"]):
            raise ValueError("incomplete or changed measured body")
        accounts = report["topology_accounting"]
        if sum(run["all_link_serialized_body_bytes"] for run in accounts["runs"]) != ledger["pair_bytes"]:
            raise ValueError("measured pair ledger mismatch")
        for run in accounts["runs"]:
            if sum(edge["serialized_body_bytes"] for edge in run["body_bytes_by_edge"]) != run["all_link_serialized_body_bytes"]:
                raise ValueError("measured directed ledger does not conserve")
        if any(not run["reconciled_with_protocol_bodies"] for run in accounts["stages"]["runs"]):
            raise ValueError("measured stages not reconciled")
        if report["prepared_material_accounting"]["counts"]["claimed"] != rows:
            raise ValueError("terminal/prefix schedule disagrees with measured row ledger")
        if [(r["tokens"]["input_tokens"], r["tokens"]["output_tokens"]) for r in report["runs"]] != [(150, 8)] * 2:
            raise ValueError("changed workload")
    totals = {key: sum(s[key] for s in stages) for key in (
        "pair_rows", "frames_each_direction", "input_payload_bytes", "output_packed_bytes",
        "output_fanin_packed_bytes", "output_int32_bytes", "aead_overhead_bytes", "pair_linear_macs")}
    public = ledger["classes"]["public_bundle"]
    recurring = ledger["pair_bytes"] - public
    common = totals["input_payload_bytes"] + totals["aead_overhead_bytes"] + ledger["classes"]["control_and_seeds"]
    alternatives = {}
    for mode, key in (("recorded_public_bound", "output_packed_bytes"),
                      ("conservative_fanin", "output_fanin_packed_bytes"), ("int32", "output_int32_bytes")):
        candidate = common + totals[key]
        saving = recurring - candidate
        horizons = []
        for h in (1, 2, 4, 8, 16, 64):
            baseline = public + h * recurring
            horizons.append({"pairs": h, "baseline_body_bytes": baseline,
                             "candidate_body_bytes_before_extra_setup": public + h * candidate,
                             "saving_fraction_before_extra_setup": h * saving / baseline,
                             "max_extra_setup_bytes_by_target": {
                                 str(t): math.floor(h * saving - (t / 100) * baseline) for t in (20, 35, 50)}})
        alternatives[mode] = {"recurring_pair_bytes": candidate,
                              "asymptotic_saving_fraction_before_extra_costs": saving / recurring,
                              "horizons": horizons,
                              "minimum_pairs_by_extra_setup_sensitivity": {
                                  str(extra): {str(t): minimum_horizon(public, recurring, candidate, extra, t / 100)
                                               for t in (20, 35, 50)}
                                  for extra in (0, 65536, 357826560, 988097824)}}
    return {"scope": "application-body projection, not measured TEE or physical wire", "stages": stages,
            "model_legal_signed_input_range": [-127, 127],
            "totals": totals, "baseline": ledger, "alternatives": alternatives,
            "raw_body_i8_weight_bytes": sum(math.prod(s["shape"]) for s in stages),
            "separate_cold_distribution_payload_scenarios": {
                "two_original_provider_checkpoints_baseline": control["publisher"]["two_provider_source_safetensors_bytes"],
                "one_original_checkpoint_candidate": control["publisher"]["two_provider_source_safetensors_bytes"] // 2,
                "role_artifacts_baseline": control["publisher"]["combined_artifact_payload_bytes"],
                "one_inference_artifact_candidate_conservative": control["publisher"]["inference_artifact_payload_bytes"],
                "scope": "raw payload scenarios outside serving ledger; not measured new bootstrap; publisher retention/publication/manifests still extra"},
            "extra_setup_sensitivities_are_not_measured": True,
            "retained_control_bytes_per_pair": ledger["classes"]["control_and_seeds"]}


def minimum_horizon(public, baseline, candidate, extra, target):
    if any(type(v) is not int or v < 0 for v in (public, baseline, candidate, extra)) or not 0 < target < 1:
        raise ValueError("invalid horizon inputs")
    margin = (1 - target) * baseline - candidate
    if margin <= 0:
        return None
    return max(1, math.ceil((extra + target * public) / margin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), CONTROL, GEOMETRY, CONFIRM,
             ROOT / "python/pllm/deployment/role_placement.py",
             ROOT / "schemas/role-deployment.schema.json",
             ROOT / "python/pllm/runtime/transformer_engine.py",
             ROOT / "python/pllm/runtime/quantization.py",
             ROOT / "tests/test_attested_linear_budget_probe.py",
             ROOT / "docs/research/attested-linear-bootstrap-reference.md",
             ROOT / "scripts/probe_network_algebra.py",
             ROOT / "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
             ROOT / "crates/pllm-compiler/tests/decoder_runtime_schedule.rs"]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result = {"schema": "pllm.attested_linear_bootstrap_budget.v1", "complete": True,
              "production_ready": False, "actual_attestation_verified": False,
              "source_identity": sources, "python_version": sys.version.split()[0],
              "frame": {"header_bytes": HEADER.size, "nonce_bytes": 12, "tag_bytes": 16,
                        "overhead_bytes": FRAME_OVERHEAD, "prefill_rows_per_frame": 16},
              "budget": budget(json.loads(CONTROL.read_text()), json.loads(GEOMETRY.read_text()),
                               json.loads(CONFIRM.read_text())), "software_transport_control": synthetic_control(),
              "unpriced": ["actual attestation/endorsement/revocation bodies and verification CPU",
                           "persistent key establishment, renewal, failure and shutdown bodies",
                           "source/quantizer/manifests and provider/client artifact distribution",
                           "host-enclave IPC, copies, paging, device transfers and public-weight validation",
                           "TCP/TLS/HTTP/ACK/retransmission bytes and failed attempts",
                           "client/enclave/host/device/cache peak resources and full decoder parity"]}
    if sources != {str(p.relative_to(ROOT)): digest(p) for p in paths}:
        raise ValueError("sources changed during probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
