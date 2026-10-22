#!/usr/bin/env python3
"""Replay the frozen public-table-to-Rust encoding; no upstream code executes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIXTURE_SHA256 = "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
OUTPUT = ROOT / "crates/pllm-garble/src/fusefss_backend/composition/source_table_values.rs"
ACTIVATION_OUTPUT = ROOT / "crates/pllm-garble/src/fixed_activation_table_values.rs"
GELU_FIXTURE = HERE.parent / "gelu-table.json"
GELU_FIXTURE_SHA256 = "4d91281c274275d8b17f6711eb3f454ad25865a197570115fdede8b2db7db69b"
TABLE_SHA256 = {
    "gelu": "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b",
    "silu": "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f",
    "exp0": "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
    "exp1": "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
}
SUPPLEMENT = HERE / "source-tables-supplement.json"


def gate_fixture():
    raw = (HERE / "fixtures.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("frozen gate fixture changed")
    return json.loads(raw)


def append_table(lines, name, count, values, expected_hash, storage):
    if len(values) != count or any(type(v) is not int or not 0 <= v <= 4096 for v in values):
        raise ValueError("invalid public table domain")
    digest = hashlib.sha256(b"".join(struct.pack("<Q", v) for v in values)).hexdigest()
    if digest != expected_hash or digest != TABLE_SHA256[name]:
        raise ValueError("public table digest changed")
    lines.extend([
        f"// u64LE SHA-256: {digest}",
        "#[rustfmt::skip]",
        f"pub(super) {storage} {name.upper()}: [u64; {count}] = [",
    ])
    for start in range(0, count, 16):
        lines.append("    " + ", ".join(map(str, values[start : start + 16])) + ",")
    lines.append("];")


def generate():
    """Return the existing output path's bytes (now exp-only), as before."""
    fixture = gate_fixture()
    lines = [
        "// Exp constants retained unchanged from",
        "// paper/reproduction-contracts/fusefss/gate-set/generate-native-tables.py.",
        "// Frozen fixture SHA-256:",
        f"// {FIXTURE_SHA256}",
        "// Shared activation constants now live in crate::fixed_activation_tables.",
    ]
    for name in ("exp0", "exp1"):
        lines.append("")
        append_table(
            lines, name, 256, fixture["tables"][name], fixture["sha256_u64le"][name], "const"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def generate_activations():
    """Encode existing frozen equation fixtures, without numerical generation."""
    fixture = gate_fixture()
    raw = GELU_FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GELU_FIXTURE_SHA256:
        raise ValueError("frozen GELU fixture changed")
    gelu = json.loads(raw)
    lines = [
        "// SiLU extracted unchanged from the former FuseFSS composition table owner.",
        "// Public mathematical constants only. Frozen independent fixture SHA-256:",
        f"// {FIXTURE_SHA256}",
        "// Rational-floor certificates live in the independent gate-set reference.",
        "",
        "// GELU copied exactly from paper/reproduction-contracts/fusefss/gelu-table.json.",
    ]
    append_table(lines, "gelu", 256, gelu["values"], gelu["sha256_u64le"], "static")
    lines.append("")
    append_table(
        lines, "silu", 1024, fixture["tables"]["silu"], fixture["sha256_u64le"]["silu"], "static"
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def supplement():
    """Width-2 norm appears in original cases but had no frozen table array."""
    values = [math.isqrt(((1 << 55) * 2) // ((128 + (i >> 6)) << (i & 63))) for i in range(8192)]
    raw = b"".join(struct.pack("<Q", value) for value in values)
    obj = {
        "schema": "pllm.fusefss.table-supplement.v1",
        "gate_fixture_sha256": FIXTURE_SHA256,
        "source_lock_sha256": "8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd",
        "name": "rsqrt-k2",
        "count": 8192,
        "input_bits": 50,
        "output_bits": 50,
        "k": 2,
        "formula": "isqrt(floor((2^55*k)/((128+(index>>6))*2^(index&63))))",
        "sha256_u64le": hashlib.sha256(raw).hexdigest(),
        "values": values,
    }
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write both Rust data files and the norm supplement")
    args = parser.parse_args()
    expected = generate()
    activations = generate_activations()
    supplemental = supplement()
    for path, data in ((OUTPUT, expected), (ACTIVATION_OUTPUT, activations), (SUPPLEMENT, supplemental)):
        if args.write:
            path.write_bytes(data)
        elif path.read_bytes() != data:
            raise ValueError(f"generated native table constants differ: {path.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "file": str(OUTPUT.relative_to(ROOT)),
                "bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
                "activation_file": str(ACTIVATION_OUTPUT.relative_to(ROOT)),
                "activation_bytes": len(activations),
                "activation_sha256": hashlib.sha256(activations).hexdigest(),
                "supplement_sha256": hashlib.sha256(supplemental).hexdigest(),
                "write": args.write,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
