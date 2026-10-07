"""Run opt-in native issuance profiling and independently check public outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


def expected_digest(lanes):
    if type(lanes) is not int or not 1 <= lanes <= 256:
        raise ValueError("expected 1–256 public fixture lanes")
    gates = [-(48 << 24), -(16 << 24), -(1 << 23), -1, 0, 1, 1 << 24, 16 << 24, 48 << 24]
    ups = [128 << 24, -4097, -1, 0, 1, 4097, -(128 << 24)]
    table = [math.floor(4096 * (i / 64) / (1 + math.exp(i / 64))) for i in range(1024)]
    encoded = bytearray()
    for i in range(lanes):
        gate, up = gates[i % len(gates)] // 4096, ups[i % len(ups)] // 4096
        value = (max(gate, 0) - table[min(abs(gate // 64), 1023)]) * up // 4096
        encoded.extend(value.to_bytes(8, "little", signed=True))
    return hashlib.sha256(encoded).hexdigest()


def validate(report):
    assert report["schema"] == "pllm.masked_wire_issuance_profile.v1"
    assert len(report["cases"]) == 18
    assert {(r["layout"], r["lanes"], r["sample"]) for r in report["cases"]} == {
        (layout, lanes, sample) for layout in ("unfused", "fused")
        for lanes in (1, 64, 256) for sample in range(3)
    }
    for case in report["cases"]:
        lanes = case["lanes"]
        micro, complete = case["microprofile"], case["complete"]
        assert complete["checked_outputs"] == lanes
        assert complete["output_sha256"] == expected_digest(lanes)
        assert set(micro["components"]) == {
            "wire_masks", "gate_gapars", "up_gapars", "index_reduce", "sign_interval",
            "clipped_index", "relu", "residual_lookup", "product", "output_gapars",
        }
        assert all(c["calls"] == lanes for c in micro["components"].values())
        micro_times = [c[key] for c in micro["components"].values()
                       for key in ("issue_ns", "encode_ns", "retire_ns")]
        complete_times = [complete[key] for key in (
            "issuance_ns", "export_ns", "import_ns", "client_input_ns",
            "worker_start_ns", "output_codec_decode_ns",
        )] + complete["advance_ns"]
        assert all(type(t) is int and t >= 0 for t in micro_times + complete_times)
        assert sum(micro_times) <= micro["wall_ns"]
        assert sum(complete_times) <= complete["wall_ns"]
        for party in (0, 1):
            assert sum(c["party_key_bits"][party] for c in micro["components"].values()) == micro["party_key_bits"]
        assert micro["party_key_bits"] == lanes * {"unfused": 5162, "fused": 9124}[case["layout"]]
        assert micro["party_key_bytes"] == (micro["party_key_bits"] + 7) // 8
        assert complete["party_packet_bytes"] == [76 + micro["party_key_bytes"]] * 2
        widths = ([2, 80, 1, 13, 2, 10, 40, 40, 40, 1, 40] if case["layout"] == "unfused"
                  else [5, 93, 10, 40, 40, 40, 1, 40])
        assert len(complete["advance_ns"]) == len(widths)
        assert complete["phase_frame_bytes"] == [[78 + (bits * lanes + 7) // 8] * 2 for bits in widths]
        assert micro["combined_allocation_estimate_bytes"] <= 64 << 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    command = ["cargo", "test", "--release", "--locked", "-p", "pllm-garble", "--lib",
               "masked_wire_reference::issuance_profile::profile_issuance_and_codecs", "--",
               "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    result = subprocess.run(command, cwd=root, env=dict(os.environ, CARGO_BUILD_JOBS="2"),
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        result.check_returncode()
    marker = "PLLM_ISSUANCE_PROFILE="
    line = next(line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line)
    report = json.loads(line)
    validate(report)
    report["checked_integer_outputs"] = sum(case["complete"]["checked_outputs"] for case in report["cases"])
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["environment"] = {"platform": platform.platform(), "machine": platform.machine(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()}
    report["command"] = command
    sources = ["benchmarks/research/masked_wire_issuance.py",
        "crates/pllm-garble/src/masked_wire_reference/issuance_profile.rs",
        "crates/pllm-garble/src/masked_wire_reference.rs",
        "crates/pllm-garble/src/masked_wire_reference/transport.rs",
        "crates/pllm-garble/src/compact_dpf.rs",
        "crates/pllm-core/src/residual_silu_reference.rs", "Cargo.lock"]
    report["source_sha256"] = {source: hashlib.sha256((root / source).read_bytes()).hexdigest() for source in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    print(f"Checked {report['checked_integer_outputs']} outputs across {len(report['cases'])} issuance/codec profiles.")


if __name__ == "__main__":
    main()
