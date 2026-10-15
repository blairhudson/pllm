#!/usr/bin/env python3
"""Replay a public row certificate and exercise its complete native codec layout.

No checkpoint values are loaded. Endpoint vectors are independent per-row
certificate-domain fixtures, not necessarily a simultaneous W*x for one input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def audit(report: dict, *, native=None) -> dict:
    """Recompute bounds/layout/costs; independently check installed native packing."""
    if report.get("schema") != "pllm.compact-private-head-scan.v1":
        raise ValueError("unsupported head scan")
    if report.get("verification") != {
        "full_passes": 2, "every_row_scalar_verified": True,
        "installed_native_every_row_verified": True,
    }:
        raise ValueError("incomplete source scan")
    pin, cert, cost = report["pin"], report["certificate"], report["costs"]
    if any(cert.get(k) != v for k, v in pin.items()):
        raise ValueError("certificate source binding")
    if sha(canonical(cert)) != report["certificate_sha256"]:
        raise ValueError("certificate digest")
    widths = bytes.fromhex(report["widths_hex"])
    bounds_raw = bytes.fromhex(report["bounds_le_u64_hex"])
    vocab, hidden = pin["vocabulary"], pin["hidden"]
    if len(widths) != vocab or len(bounds_raw) != 8 * vocab:
        raise ValueError("certificate geometry")
    if sha(widths) != cert["width_sha256"] or sha(bounds_raw) != cert["bounds_sha256"]:
        raise ValueError("certificate arrays digest")
    bounds = np.frombuffer(bounds_raw, dtype="<u8")
    expected = bytes(max(1, int(bound).bit_length() + 1) for bound in bounds)
    if (widths != expected or pin["qmax"] != 127
            or any(int(bound) % 127 or int(bound) > hidden * 128 * 127 for bound in bounds)):
        raise ValueError("signed-center widths or public i8 bounds")
    uniform = (hidden * 128 * 127).bit_length() + 1
    payload = 2 * math.ceil(sum(widths) / 8)
    baseline = 2 * math.ceil(vocab * uniform / 8)
    if (cost["reply_payload_bytes_both"] != payload
            or cost["baseline_reply_payload_bytes_both"] != baseline
            or cost["width_bit_sum_per_role"] != sum(widths)
            or cost["baseline_uniform_width"] != uniform):
        raise ValueError("payload ledger")
    histogram = {str(k): v for k, v in Counter(widths).items()}
    if {str(k): v for k, v in cost["width_histogram"].items()} != histogram:
        raise ValueError("histogram ledger")
    layout_sizes = {
        "raw": vocab + 1, "five_bit": math.ceil(5 * vocab / 8) + 1,
        "rle_u32": 1 + 5 * (1 + sum(a != b for a, b in zip(widths, widths[1:]))),
        "zlib9": 1 + len(zlib.compress(widths, 9)),
    }
    if len(histogram) == 1:
        layout_sizes["uniform"] = 2
    metadata = canonical(cert)
    metadata_sizes = {"raw": len(metadata) + 1, "zlib9": len(zlib.compress(metadata, 9)) + 1}
    if (cost["layout_encoding_bytes"] != layout_sizes
            or cost["metadata_encoding_bytes"] != metadata_sizes):
        raise ValueError("compressed metadata ledger (requires original zlib version)")
    outer = cost["outer_body_overhead_bytes_per_role_assumption"]
    http = cost["http_overhead_bytes_per_role_assumption"]
    framed = payload + 2 * (74 + 32 + outer + http)
    baseline_framed = baseline + 2 * (74 + outer + http)
    if (cost["reply_framed_bytes_both_estimate"] != framed
            or cost["baseline_reply_framed_bytes_both_estimate"] != baseline_framed):
        raise ValueError("framed ledger")
    saved = baseline_framed - framed
    if saved != cost["saved_reply_bytes_both_per_head_estimate"]:
        raise ValueError("framed ledger")
    cold = 3 * (88 + min(layout_sizes.values()) + min(metadata_sizes.values()) + outer + http)
    if cost["cold_layout_metadata_delivery_bytes_three_recipients_estimate"] != cold:
        raise ValueError("cold metadata ledger")
    net = saved * cost["head_calls"] - cold
    if net != cost["net_saved_bytes_at_head_calls_estimate"]:
        raise ValueError("cold metadata ledger")
    if native is None:
        from pllm import _native
        native = _native
    signed = bounds.astype(np.int64)
    rng = np.random.default_rng(60217)  # public test shares, not production randomness
    vectors = [np.zeros(vocab, dtype=np.int64), signed, -signed,
               np.where(np.arange(vocab) % 2, signed, -signed),
               rng.integers(-signed, signed + 1, dtype=np.int64)]
    commitments = []
    for values in vectors:
        left = rng.integers(0, 1 << 32, size=vocab, dtype=np.uint32)
        right = ((values - left.astype(np.int64)) & 0xFFFF_FFFF).astype("<u4")
        a = native.offset_pack_rows(left.astype("<u4").tobytes(), widths, 1)
        b = native.offset_pack_rows(right.tobytes(), widths, 1)
        if len(a) + len(b) != payload:
            raise ValueError("native packed length")
        restored = np.frombuffer(native.offset_reconstruct_rows(a, b, widths, 1), dtype="<i8")
        if not np.array_equal(restored, values):
            raise ValueError("native reconstruction mismatch")
        commitments.append(sha(restored.tobytes()))
    return {
        "schema": "pllm.compact-private-head-native-audit.v1",
        "certificate_sha256": report["certificate_sha256"],
        "source_pin": pin,
        "checked_integer_outputs": len(vectors) * vocab,
        "fixture": "zero, positive/negative/alternating endpoints, public-seed random per-row values",
        "output_sha256": commitments,
        "payload_bytes_both": payload,
        "baseline_payload_bytes_both": baseline,
        "payload_reduction_percent": 100 * (baseline - payload) / baseline,
        "framed_saved_bytes_both_per_head_estimate": saved,
        "cold_net_saved_bytes_estimate": net,
        "worker_40Mbps_floor_difference_seconds": (saved / 2) * 8 / 40_000_000,
        "scope": "source-bound per-row domain codec checks; no full decoder, protocol, measured latency or RSS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = audit(json.loads(raw))
    from pllm.metrics import BuildRef
    result["scan_sha256"] = sha(raw)
    result["auditor_sha256"] = sha(Path(__file__).read_bytes())
    result["native_build"] = BuildRef.capture().to_spec()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
