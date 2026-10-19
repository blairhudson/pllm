"""Bounded independent unfused mathematics. Never imports scalar/native compiler.

Public test inputs include clear masks; this is an offline qualification interface,
not an evaluator key format or an authenticated protocol.
"""

import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent
TABLE_HASH = "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b"
LOCK_HASH = "8c42ba8f3ff7156fe87882dca089b8fb6c71b2739f8d761583e2ccc94c794609"
GELU_DIGEST = "5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3"
CONTEXT = "pllm.fusefss.independent-native-conformance.v1"
MAX_RUNS = 4096
MAX_EVALUATIONS = 100000
MAX_FIXTURE_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 32 << 20
MAX_LINE_BYTES = 2 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def digest(descriptor):
    return hashlib.sha256(b"FuseFSS-reference-descriptor-v1\0" + canonical(descriptor)).hexdigest()


def load_table():
    record = json.loads((FROZEN / "gelu-table.json").read_bytes())
    values = record["values"]
    if len(values) != 256 or any(type(v) is not int or not 0 <= v < 4096 for v in values):
        raise ValueError("invalid frozen table shape")
    actual = hashlib.sha256(b"".join(struct.pack("<Q", v) for v in values)).hexdigest()
    if actual != TABLE_HASH or record["sha256_u64le"] != TABLE_HASH:
        raise ValueError("frozen certified table hash mismatch")
    if hashlib.sha256((FROZEN / "source-lock.json").read_bytes()).hexdigest() != LOCK_HASH:
        raise ValueError("frozen source lock hash mismatch")
    return values


def signed(x, bits):
    return x - (1 << bits) if x >= (1 << (bits - 1)) else x


def gelu(word, table):
    """Exact selected integer GELU, not continuous GELU or host-libm floors.

    T[j]=floor(2048*(j/64)*erfc((j/64)/sqrt(2))) is independently
    rational-certified in frozen math_oracle.py; reuse only hash-checked bytes.
    """
    x = signed(word, 50)
    quotient, _ = divmod(x, 64)  # floor before abs, including -1 -> -1
    tail = table[min(abs(quotient), 255)]
    return ((x if x >= 0 else 0) - tail) % (1 << 50)


def predicate(expr, x, bits):
    tag, *a = expr
    if tag == "const":
        return a[0]
    if tag == "lt":
        return int(x < a[0])
    if tag == "low_lt":
        return int((x & ((1 << a[0]) - 1)) < a[1])
    if tag == "slt":
        return int(signed(x, bits) < a[0])
    if tag == "msb":
        return ((x + a[0]) % (1 << bits)) >> (bits - 1)
    if tag == "not":
        return 1 - predicate(a[0], x, bits)
    left, right = (predicate(v, x, bits) for v in a)
    if tag == "xor":
        return int(left != right)
    if tag == "and":
        return int(left == right == 1)
    if tag == "or":
        return int(left == 1 or right == 1)
    raise ValueError("unknown predicate")


def unfused(descriptor, x, table):
    n = descriptor["input"]["bits"]
    if descriptor["profile"] == "fusefss.source-derived-ring50-q12.v1":
        return [gelu(x, table), []]
    # Indicator sum over original public intervals: no translated lookup,
    # normalized predicates, Horner, or compiler evaluation supplies results.
    result, outputs = 0, [0] * descriptor["output"]["boolean_arity"]
    for lo, hi, piece in zip(descriptor["boundaries"], descriptor["boundaries"][1:], descriptor["pieces"]):
        active = int(lo <= x < hi)
        c, a = piece["coefficients"]
        result += active * (c + a * x)
        for i, expr in enumerate(piece["booleans"]):
            outputs[i] += active * predicate(expr, x, n)
    return [result % (1 << n), outputs]


def descriptor(bits, boundaries, pieces, gelu_profile=False):
    frac = 12 if gelu_profile else 0
    return {
        "schema": "pllm.fusefss.scalar-descriptor.v1",
        "profile": "fusefss.source-derived-ring50-q12.v1" if gelu_profile else "fusefss.synthetic-conformance.v1",
        "gate": "sigma-cpu-gelu.affine-q0-q12.identity.v1" if gelu_profile else "synthetic-affine",
        "source_lock_sha256": LOCK_HASH if gelu_profile else "0" * 64,
        "input": {"domain": "A", "bits": bits, "fractional": frac, "signed": True},
        "output": {"arithmetic": "same-as-input", "boolean_arity": len(pieces[0]["booleans"])},
        "external_encoding": "nearest-ties-even-input-and-weight",
        "internal_rounding": "signed-floor-only",
        "partition_order": "unsigned-canonical-half-open", "degree": 1,
        "coefficient_order": ["constant", "linear"], "coefficient_fractional": [frac, 0],
        "selector_semantics": "floor-x-div-64-before-abs-clip255" if gelu_profile else "descriptor-defined",
        "source_schedule": {
            "effective_input": [38, 12], "early_tr_output": [32, 6], "clipped_index": [8, 6],
            "lut_and_output": [50, 12], "required_signed_input_range": [-(1 << 37), 1 << 37],
            "range_status": "requires-sign-extension-provenance-not-a-model-proof",
        } if gelu_profile else {},
        "phi": "identity", "boundaries": boundaries, "pieces": pieces,
    }


def gelu_descriptor(table):
    n = 1 << 50
    # Solve floor(X/64)=q, then clip |q|. Negative bin q=-j is
    # [-64*j, -64*j+64); j>=255 coalesces into negative tail.
    cuts = [0] + [64 * j for j in range(1, 256)] + [n // 2]
    cuts += [n - 64 * j for j in range(254, 0, -1)] + [n]
    pieces = []
    for lo, hi in zip(cuts, cuts[1:]):
        x = signed(lo, 50)
        j = min(abs(x // 64), 255)
        pieces.append({"coefficients": [(-table[j]) % n, int(x >= 0)], "booleans": []})
        # Independently check affine interval against exact formula at BOTH
        # endpoints, not against scalar compiler output.
        for word in (lo, hi - 1):
            if (pieces[-1]["coefficients"][0] + pieces[-1]["coefficients"][1] * word) % n != gelu(word, table):
                raise ValueError("GELU interval endpoint contradiction")
    result = descriptor(50, cuts, pieces, True)
    if digest(result) != GELU_DIGEST or len(canonical(result)) != 32900:
        raise ValueError("independently constructed descriptor differs from frozen identity")
    return result


def query_atoms(d):
    """Structural query contract only, NEVER used for expected gate outputs.

    Public 0/N atoms cost zero slots. Every other atom costs two, including
    instance thresholds that happen to become zero. No deduplication.
    """
    n = d["input"]["bits"]

    def visit(expr):
        tag, *a = expr
        if tag == "const":
            return []
        if tag == "lt":
            return [(n, 0, a[0])]
        if tag == "low_lt":
            return [(a[0], 0, a[1])]
        if tag == "slt":
            return [(n, 1 << (n - 1), a[0] + (1 << (n - 1)))]
        if tag == "msb":
            return [(n, a[0], 1 << (n - 1))]
        return [v for e in a for v in visit(e)]

    atoms = []
    for out in range(d["output"]["boolean_arity"]):
        for i, p in enumerate(d["pieces"]):
            atoms += [(n, 0, d["boundaries"][i + 1]), (n, 0, d["boundaries"][i])]
            atoms += visit(p["booleans"][out])
    return [(k, shift, beta) for k, shift, beta in atoms if beta not in (0, 1 << k)]


def shape(d):
    m = len(d["pieces"])
    return {"source_intervals": m, "padded_intervals": min(m + 1, 1 << d["input"]["bits"]),
            "boolean_outputs": d["output"]["boolean_arity"], "queries": 2 * len(query_atoms(d)),
            "degree": 1, "coefficients_per_interval": 2, "horner_products": 1}


def nearest_even(numerator, denominator):
    # Exact rational external Q12 encoding, not an implicit tensor bridge.
    q, rem = divmod(numerator * 4096, denominator)
    return q + int(rem * 2 > denominator or (rem * 2 == denominator and q % 2 == 1))


def build_fixture():
    table = load_table()
    descriptors, runs = [], []

    def add(d, masks, words, family):
        ident = digest(d)
        descriptors.append({"id": ident, "descriptor": d, "shape": shape(d), "family": family})
        expected = [unfused(d, x, table) for x in words]
        n = d["input"]["bits"]
        for mask in masks:
            run_id = f"r{len(runs):04d}"
            runs.append({"id": run_id, "descriptor_id": ident, "mask": mask,
                         "context": f"{CONTEXT}/{run_id}", "words": words,
                         "masked_words": [(x + mask) % (1 << n) for x in words], "expected": expected})

    # ALL complete partitions for rings 2,4,8, with ALL masks and ALL inputs.
    for bits in range(1, 4):
        n = 1 << bits
        for subset in range(1 << (n - 1)):
            cuts = [0] + [v for v in range(1, n) if subset & (1 << (v - 1))] + [n]
            pieces = []
            for i in range(len(cuts) - 1):
                predicates = [["const", i % 2], ["low_lt", max(1, bits - 1), i % (1 << max(1, bits - 1))],
                              ["slt", (i % n) - n // 2],
                              ["xor", ["lt", cuts[i + 1]], ["msb", i % n]]]
                pieces.append({"coefficients": [(2 * i + 1) % n, (3 * i + 3) % n], "booleans": predicates})
            add(descriptor(bits, cuts, pieces), range(n), list(range(n)), "all-small-partitions")

    # Every unsigned/signed/low threshold including sentinels, every MSB shift;
    # nested Boolean operations add type-sensitive direct truth-table evidence.
    for bits in range(1, 5):
        n = 1 << bits
        expressions = [["lt", b] for b in range(n + 1)]
        expressions += [["slt", b] for b in range(-n // 2, n // 2 + 1)]
        expressions += [["low_lt", k, b] for k in range(1, bits + 1) for b in range((1 << k) + 1)]
        expressions += [["msb", c] for c in range(n)]
        expressions += [["not", ["const", 0]], ["and", ["lt", n // 2], ["msb", 1]],
                        ["or", ["low_lt", 1, 1], ["not", ["slt", 0]]]]
        for start in range(0, len(expressions), 4):
            piece = {"coefficients": [1, 3 % n], "booleans": expressions[start:start + 4]}
            add(descriptor(bits, [0, n], [piece]), range(n), list(range(n)), "all-small-predicates")

    d = gelu_descriptor(table)
    n = 1 << 50
    words = set()
    for boundary in d["boundaries"]:
        words.update((boundary + delta) % n for delta in (-1, 0, 1))
    for x in (-(1 << 37) - 1, -(1 << 37), -(1 << 37) + 1,
              (1 << 37) - 1, 1 << 37, (1 << 37) + 64, -65, -64, -63, -1, 0, 1, 63, 64, 65):
        words.add(x % n)
    masks = [0, 1, 63, 64, n // 2, n - 64, n - 1]
    add(d, masks, sorted(words), "gelu-all-boundaries-and-signed38-neighbors")
    # Same descriptor identity, separate central sweep (not duplicate descriptor).
    descriptors.pop()
    add(d, [n - 1], [x % n for x in range(-16385, 16386)], "gelu-central-sweep")
    # The descriptor family above describes all GELU runs, including boundaries.
    descriptors[-1]["family"] = "gelu"

    external = [{"numerator": a, "denominator": b, "signed_q12": nearest_even(a, b)}
                for b in (8192, 16384) for a in range(-9, 10)]
    evaluations = sum(len(r["words"]) for r in runs)
    family_by_id = {item["id"]: item["family"] for item in descriptors}
    by_family = {name: sum(len(r["words"]) for r in runs if family_by_id[r["descriptor_id"]] == name)
                 for name in ("all-small-partitions", "all-small-predicates", "gelu")}
    if len(runs) > MAX_RUNS or evaluations > MAX_EVALUATIONS:
        raise ValueError("fixture work budget exceeded")
    files = ["README.md", "reference.py", "source-lock.json", "vectors.json", "math_oracle.py", "gelu-table.json"]
    receipts = {name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() for name in files}
    # Verify local PDF bytes only if present. Keep fixture portable: availability
    # is not numerical specification. Receipt locks hashes, not local presence.
    source_receipts = []
    root = HERE.parents[3]
    for src in json.loads((FROZEN / "source-lock.json").read_bytes())["sources"]:
        item = {"id": src["id"], "expected_sha256": src["sha256"], "status": "hash-locked-not-executed"}
        if "path" in src and (root / src["path"]).is_file():
            actual = hashlib.sha256((root / src["path"]).read_bytes()).hexdigest()
            if actual != src["sha256"]:
                raise ValueError("local source hash mismatch")
        source_receipts.append(item)
    return {"schema": CONTEXT, "scope": "independent-algebra-fixtures-not-native-parity-or-security",
            "receipt": {"files_sha256": receipts, "table_sha256_u64le": TABLE_HASH,
                        "canonical_gelu_descriptor_sha256": GELU_DIGEST, "sources": source_receipts},
            "counts": {"descriptors": len(descriptors), "runs": len(runs), "evaluations": evaluations,
                       "small_complete_partitions": sum(item["family"] == "all-small-partitions" for item in descriptors),
                       "small_partition_evaluations": by_family["all-small-partitions"],
                       "small_predicate_evaluations": by_family["all-small-predicates"],
                       "gelu_evaluations": by_family["gelu"],
                       "external_encoding_cases": len(external), "gelu_boundary_words": len(words)},
            "descriptors": descriptors, "runs": runs, "external_encoding": external}


if __name__ == "__main__":
    import sys
    encoded = canonical(build_fixture()) + b"\n"
    if len(encoded) > MAX_FIXTURE_BYTES:
        raise SystemExit("fixture byte budget exceeded")
    sys.stdout.buffer.write(encoded)
