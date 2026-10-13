"""Public bounded cyclic-mask codec reference; no SDK or security activation.

Only standard-library arithmetic and small correctness probes. Layout is public
side information, never derived from private activations. No primality assumption
is needed by this codec; delegation protocols may independently require a field.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import struct

MAX_Q = (1 << 64) - 1
MAX_VALUES = 65536
HEADER = struct.Struct("<4sII")
MAGIC = b"CMC1"


def integer(value):
    if type(value) is not int:
        raise ValueError("integer required")
    return value


@dataclass(frozen=True)
class Geometry:
    q: int
    lo: int
    hi: int

    def __post_init__(self):
        for value in (self.q, self.lo, self.hi):
            integer(value)
        if not 2 <= self.q <= MAX_Q:
            raise ValueError("modulus outside uint64 reference domain")
        if not -(1 << 63) <= self.lo <= self.hi < (1 << 63):
            raise ValueError("interval outside signed64 reference domain")
        if self.m > self.q:
            raise ValueError("interval aliases modulo q")

    @classmethod
    def symmetric(cls, q, bound):
        integer(bound)
        if bound < 0:
            raise ValueError("negative bound")
        return cls(q, -bound, bound)

    @property
    def m(self):
        return self.hi - self.lo + 1

    @property
    def n(self):
        return self.q // self.m

    @property
    def length(self):
        return self.q // self.n

    @property
    def remainder(self):
        return self.q % self.n

    @property
    def alphabet(self):
        return self.length + bool(self.remainder)

    @property
    def bits(self):
        return (self.alphabet - 1).bit_length()

    def start(self, block):
        if not 0 <= integer(block) < self.n:
            raise ValueError("invalid block")
        return block * self.length + min(block, self.remainder)

    def block(self, value):
        if not 0 <= integer(value) < self.q:
            raise ValueError("noncanonical field value")
        cut = self.remainder * (self.length + 1)
        if value < cut:
            return value // (self.length + 1)
        return self.remainder + (value - cut) // self.length

    def encode(self, value):
        return value - self.start(self.block(value))

    def decode(self, color, mask):
        if not 0 <= integer(color) < self.alphabet:
            raise ValueError("invalid color")
        if not 0 <= integer(mask) < self.q:
            raise ValueError("noncanonical mask")
        # The interval starts at mask+lo, not at mask, for asymmetric bounds.
        anchor = (mask + self.lo) % self.q
        block = self.block(anchor)
        hits = set()
        for neighbor in {(block - 1) % self.n, block, (block + 1) % self.n}:
            size = self.length + (neighbor < self.remainder)
            if color < size:
                candidate = self.start(neighbor) + color
                offset = (candidate - anchor) % self.q
                if offset < self.m:
                    hits.add(self.lo + offset)
        if len(hits) != 1:
            raise ValueError("no unique bounded candidate")
        return hits.pop()


def oracle(g, color, mask):
    """Independent domain scan: no block, start, or optimized decoder calls."""
    if g.m > 4096:
        raise ValueError("oracle scan resource limit")
    n = g.q // (g.hi - g.lo + 1)
    length, remainder = divmod(g.q, n)
    matches = []
    for x in range(g.lo, g.hi + 1):
        y = (mask + x) % g.q
        # Independent color computation by the long-prefix boundary.
        cut = remainder * (length + 1)
        expected = y % (length + 1) if y < cut else (y - cut) % length
        if expected == color:
            matches.append(x)
    return matches


def layout_stats(layout):
    if not isinstance(layout, (list, tuple)) or not 1 <= len(layout) <= MAX_VALUES:
        raise ValueError("invalid public layout count")
    if any(not isinstance(g, Geometry) for g in layout):
        raise ValueError("invalid public geometry")
    bits = sum(g.bits for g in layout)
    return {"values": len(layout), "payload_bits": bits,
            "payload_bytes": (bits + 7) // 8, "padding_bits": (-bits) % 8,
            "frame_bytes": HEADER.size + (bits + 7) // 8,
            # Canonical externally supplied public layout representation.
            "public_layout_bytes": 24 * len(layout)}


def serialize(values, layout):
    stats = layout_stats(layout)
    if len(values) != len(layout):
        raise ValueError("value count differs")
    output = bytearray()
    buffer = used = 0
    for value, g in zip(values, layout):
        color = g.encode(value)
        buffer |= color << used
        used += g.bits
        while used >= 8:
            output.append(buffer & 255)
            buffer >>= 8
            used -= 8
    if used:
        output.append(buffer)
    return HEADER.pack(MAGIC, len(layout), stats["payload_bytes"]) + output


def parse_colors(frame, layout):
    stats = layout_stats(layout)
    if type(frame) is not bytes or len(frame) != stats["frame_bytes"]:
        raise ValueError("invalid frame length")
    if HEADER.unpack_from(frame) != (MAGIC, len(layout), stats["payload_bytes"]):
        raise ValueError("invalid public frame header")
    data = frame[HEADER.size:]
    if stats["padding_bits"] and data[-1] >> (8 - stats["padding_bits"]):
        raise ValueError("nonzero terminal padding")
    colors = []
    buffer = used = index = 0
    for g in layout:
        while used < g.bits:
            buffer |= data[index] << used
            used += 8
            index += 1
        color = buffer & ((1 << g.bits) - 1)
        buffer >>= g.bits
        used -= g.bits
        if color >= g.alphabet:
            raise ValueError("invalid packed color")
        colors.append(color)
    return colors


def deserialize(frame, masks, layout):
    colors = parse_colors(frame, layout)  # Public errors before secret-dependent work.
    if len(masks) != len(layout):
        raise ValueError("mask count differs")
    return [g.decode(color, mask) for g, color, mask in zip(layout, colors, masks)]


def pack_layout(layout):
    layout_stats(layout)
    return b"".join(struct.pack("<Qqq", g.q, g.lo, g.hi) for g in layout)


def unpack_layout(data):
    if type(data) is not bytes or not data or len(data) % 24 or len(data) // 24 > MAX_VALUES:
        raise ValueError("invalid layout storage")
    return tuple(Geometry(*row) for row in struct.iter_unpack("<Qqq", data))


def checked_decode(g, color, mask):
    expected = oracle(g, color, mask)
    try:
        actual = g.decode(color, mask)
    except ValueError:
        assert expected == []
    else:
        assert expected == [actual]


def exhaustive():
    recovered = oracle_checks = geometries = 0
    # Every width, even/composite moduli included; asymmetric and shifted intervals.
    for q in range(2, 33):
        for width in range(1, q + 1):
            for lo in {0, -(width // 2), 3}:
                g = Geometry(q, lo, lo + width - 1)
                geometries += 1
                for h in range(q):
                    for x in range(g.lo, g.hi + 1):
                        assert g.decode(g.encode((h + x) % q), h) == x
                        recovered += 1
                    for color in range(g.alphabet):
                        checked_decode(g, color, h)
                        oracle_checks += 1
    parent_cases = 0
    for q in (17, 31, 67, 127, 257):
        for b in (1, 2, 3, 5, 10):
            if 2 * b + 1 > q:
                continue
            g = Geometry.symmetric(q, b)
            for h in range(q):
                for x in range(-b, b + 1):
                    assert g.decode(g.encode((h + x) % q), h) == x
                    parent_cases += 1
    return {"geometries": geometries, "bounded_recoveries": recovered,
            "independent_oracle_color_checks": oracle_checks,
            "parent_grid_recoveries": parent_cases}


def sampled_large():
    rng = random.Random(6101)
    cases = frames = 0
    for q in (2147483659, 4294967291, (1 << 61) - 1):
        for bound in (0, 1, 7, 128 * 128 * 896, 128 * 128 * 4864,
                      128 * 128 * 16384, (q - 1) // 2):
            g = Geometry.symmetric(q, bound)
            blocks = {0, g.n - 1, g.n // 2, min(g.remainder, g.n - 1)}
            points = {0, q - 1, q // 2}
            for block in blocks:
                start = g.start(block)
                points.update({start, (start - 1) % q, (start + g.length - 1) % q})
            points.update(rng.randrange(q) for _ in range(12))
            for h in sorted(points):
                xs = {-bound, bound, 0}
                xs.update(rng.randint(-bound, bound) for _ in range(4))
                xs = sorted(xs)
                layout = [g] * len(xs)
                frame = serialize([(h + x) % q for x in xs], layout)
                assert deserialize(frame, [h] * len(xs), layout) == xs
                assert unpack_layout(pack_layout(layout)) == tuple(layout)
                frames += 1
                cases += len(xs)
                if g.m <= 4096:
                    for color in {0, g.alphabet - 1, g.encode(h)}:
                        checked_decode(g, color, h)
    return {"bounded_recoveries": cases, "serialized_frames": frames,
            "primes": [2147483659, 4294967291, (1 << 61) - 1]}


def price(q, bound, count):
    g = Geometry.symmetric(q, bound)
    stats = layout_stats([g] * count)
    residue_bits = max(1, bound.bit_length() + 1)
    return {"q": q, "bound": bound, "count": count, "alphabet": g.alphabet,
            "cyclic_bits_per_value": g.bits, "row_residue_bits_per_value": residue_bits,
            "wide_fixed_storage_payload_bytes": count * ((q.bit_length() + 7) // 8),
            "wide_bitpacked_payload_bytes": (count * q.bit_length() + 7) // 8,
            "existing_row_residue_payload_bytes": (count * residue_bits + 7) // 8,
            "retained_mask_fixed_storage_bytes": count * ((q.bit_length() + 7) // 8),
            "retained_mask_bitpacked_storage_bytes": (count * q.bit_length() + 7) // 8,
            **stats}


def run(root):
    source_paths = ["scripts/probe_cyclic_mask_codec.py", "tests/test_cyclic_mask_codec_probe.py",
                    "docs/research/cyclic-mask-codec-reference.md",
                    "docs/research/maverick-reference.md", "crates/pllm-core/src/offset_transport.rs",
                    "crates/pllm-core/src/coded_delegation_reference.rs",
                    "docs/evidence/research-all-link-frontier.json",
                    "docs/evidence/research-maverick-reference-qwen25.json"]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths}
    previous = json.loads((root / source_paths[-1]).read_text())
    stages = previous["storage_lower_bound"]["stage_dimensions"]
    shapes = sorted({(s["input_width"], s["output_width"]) for s in stages})
    prices = []
    for q in (2147483659, 4294967291, (1 << 61) - 1):
        for columns, outputs in shapes:
            entry = price(q, columns * 128 * 128, outputs)
            entry.update(input_width=columns, output_width=outputs,
                         stages=sum(s["input_width"] == columns and s["output_width"] == outputs for s in stages),
                         provenance="source-derived dimensions; conservative public signed-i8 geometry, not checkpoint row values",
                         wide_request_payload_bytes=columns * ((q.bit_length() + 7) // 8))
            prices.append(entry)
    # Tiny public synthetic row values, full-width input absolute bound 128.
    rows = [[0] * 16, [1] * 16, [-128, 127] * 8, list(range(-8, 8))]
    synthetic = [price(4294967291, 128 * sum(abs(w) for w in row), 1) for row in rows]
    public_input = [-128, 127] * 8
    for row, entry in zip(rows, synthetic):
        x = sum(w * v for w, v in zip(row, public_input))
        g = Geometry.symmetric(entry["q"], entry["bound"])
        mask = entry["q"] - 1
        frame = serialize([(mask + x) % g.q], [g])
        assert deserialize(frame, [mask], [g]) == [x]
        assert len(frame) == entry["frame_bytes"]
        entry.update(public_dot_product=x, public_mask=mask, actual_frame_hex=frame.hex(),
                     actual_frame_sha256=hashlib.sha256(frame).hexdigest())
    aggregate = []
    for q in (2147483659, 4294967291, (1 << 61) - 1):
        group = [entry for entry in prices if entry["q"] == q]
        aggregate.append({"q": q, "scope": "one row per each of 96 source-derived stages, not full response",
                          **{key: sum(entry[key] * entry["stages"] for entry in group) for key in
                             ("payload_bytes", "frame_bytes", "public_layout_bytes", "wide_request_payload_bytes",
                              "wide_fixed_storage_payload_bytes", "existing_row_residue_payload_bytes",
                              "retained_mask_fixed_storage_bytes")}})
    control = json.loads((root / source_paths[-2]).read_text())["control"]
    return {"schema": "pllm.cyclic_mask_codec_reference.v1", "complete": True,
            "scope": "standalone arithmetic and actual serializer correctness; public evidence only",
            "source_sha256": hashes, "model_metadata_from_prior_evidence": previous["model"],
            "model_config_sha256_from_prior_evidence": previous["model_config_sha256"],
            "exhaustive": exhaustive(), "sampled_large": sampled_large(),
            "source_derived_geometry_prices": prices, "public_synthetic_row_prices": synthetic,
            "source_derived_one_row_schedule_prices": aggregate,
            "synthetic_rows": rows, "synthetic_public_input": public_input,
            "control": {"pair_application_body_bytes": control["pair_bytes"],
                        "output_application_body_bytes": control["classes"]["outputs"],
                        "zero_output_body_optimistic_ceiling_fraction": control["classes"]["outputs"] / control["pair_bytes"],
                        "new_pair_bytes": None,
                        "reason": "output bodies are already row residues and include framing; actual per-row layouts and field-input expansion not replayed"},
            "counterexample": {"q": 17, "bound": 1, "mask": 0, "outside_x": 4,
                               "sent_color": Geometry.symmetric(17, 1).encode(4),
                               "decoded_x": Geometry.symmetric(17, 1).decode(Geometry.symmetric(17, 1).encode(4), 0)},
            "bit_threshold_counterexample": price(17, 3, 8),
            "secret_dependent_rejection_example": {"q": 17, "bound": 1, "color": 3,
                                                   "mask_0": "reject", "mask_3": "accept decoded x=0"},
            "resource_contract": {"max_q": MAX_Q, "max_values": MAX_VALUES,
                                  "max_payload_bytes": MAX_VALUES * 8,
                                  "header_bytes": HEADER.size,
                                  "max_public_layout_storage_bytes": MAX_VALUES * 24},
            "limitations": ["No malicious-protocol proof or cryptographic review", "No SDK activation",
                            "No exact checkpoint runtime, full-decoder field proof, latency or peak-memory measurement",
                            "No private adaptive widths; public layout storage charged separately",
                            "Constant-time processing and uniform external failure policy not implemented"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/research-cyclic-mask-codec.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"exhaustive": result["exhaustive"], "sampled_large": result["sampled_large"]}, sort_keys=True))


if __name__ == "__main__":
    main()
