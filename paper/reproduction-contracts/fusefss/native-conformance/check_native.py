"""Read-only, bounded native-output checker. No native code is invoked here."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from independent import (CONTEXT, HERE, MAX_FIXTURE_BYTES, MAX_LINE_BYTES, MAX_OUTPUT_BYTES,
                         build_fixture, canonical, query_atoms)


class Mismatch(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Mismatch(message)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "duplicate JSON object key")
        result[k] = v
    return result


def no_float(_):
    raise Mismatch("floating/nonfinite JSON value forbidden")


def integer(text):
    require(len(text) <= 18, "integer token budget")
    return int(text)


def parse(raw):
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_int=integer,
                          parse_float=no_float, parse_constant=no_float)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise Mismatch(str(exc)) from exc


def equal(actual, expected, message):
    # Canonical bytes distinguish True from 1 and reject extra/missing fields.
    require(canonical(actual) == canonical(expected), message)


def load_fixture(path=HERE / "fixtures.json"):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FIXTURE_BYTES + 1)
    require(len(raw) <= MAX_FIXTURE_BYTES, "fixture byte budget")
    fixture = parse(raw)
    equal(fixture, build_fixture(), "fixture differs from independent equations/source receipt")
    require(raw == canonical(fixture) + b"\n", "fixture must use frozen canonical encoding")
    return fixture, hashlib.sha256(raw).hexdigest()


def validate_compilation(compiled, d, mask):
    require(type(compiled) is dict and set(compiled) ==
            {"boundaries", "ordinals", "payloads", "queries"}, "compiled fields")
    bits = d["input"]["bits"]
    n = 1 << bits
    boundaries, ordinals, payloads = (compiled[k] for k in ("boundaries", "ordinals", "payloads"))
    m = min(len(d["pieces"]) + 1, n)
    require(type(boundaries) is list and len(boundaries) == m + 1, "padded boundary count")
    require(all(type(v) is int and 0 <= v <= n for v in boundaries), "boundary type/range")
    require(boundaries[0] == 0 and boundaries[-1] == n, "wide 0/N sentinels")
    require(all(a < b for a, b in zip(boundaries, boundaries[1:])), "strict partition order")
    require(type(ordinals) is list and len(ordinals) == m and
            all(type(v) is int and 0 <= v < len(d["pieces"]) for v in ordinals), "ordinal shape/type")
    require(type(payloads) is list and len(payloads) == m, "payload shape")

    # Validate actual translated intervals via inverse-image endpoint equations.
    # No call to a compile routine, scalar or native, supplies expected payloads.
    for lo, hi, ordinal, payload in zip(boundaries, boundaries[1:], ordinals, payloads):
        a, b = (lo - mask) % n, (hi - 1 - mask) % n
        lower, upper = d["boundaries"][ordinal:ordinal + 2]
        require(lower <= a <= b < upper, "interval inverse image/ordinal mismatch")
        equal(payload, d["pieces"][ordinal]["coefficients"], "coefficient payload mismatch")

    natural = {0, n} | {(a + mask) % n for a in d["boundaries"][:-1]}
    actual = set(boundaries)
    require(natural <= actual, "missing true translated boundary")
    extra = actual - natural
    if len(natural) - 1 == m:
        require(not extra, "unexpected padding")
    else:
        eligible = [a + 1 for a, b in zip(sorted(natural), sorted(natural)[1:]) if b - a >= 2]
        require(bool(eligible) and extra == {min(eligible)}, "not first-eligible padding")

    expected_queries = []
    for width, shift, beta in query_atoms(d):
        modulus = 1 << width
        low_mask = mask % modulus
        expected_queries += [[width, shift, (low_mask + beta) % modulus], [width, shift, low_mask]]
    equal(compiled["queries"], expected_queries, "query order/width/shift/threshold/sentinel slots")


def check_stream(stream, fixture, fixture_sha, suite="all"):
    descriptors = {d["id"]: d for d in fixture["descriptors"]}
    require(len(descriptors) == len(fixture["descriptors"]), "duplicate fixture descriptor")
    consumed, lines = 0, 0

    def read_record():
        nonlocal consumed, lines
        raw = stream.readline(MAX_LINE_BYTES + 1)
        require(bool(raw), "missing native output record")
        consumed += len(raw)
        lines += 1
        require(len(raw) <= MAX_LINE_BYTES and consumed <= MAX_OUTPUT_BYTES, "native output byte budget")
        require(raw.endswith(b"\n"), "unterminated native output record")
        result = parse(raw)
        require(type(result) is dict, "native record must be object")
        return result

    header = read_record()
    require(set(header) == {"schema", "fixture_sha256", "execution", "implementation", "suite"}, "header fields")
    equal(header["schema"], CONTEXT + "/native-output", "output schema")
    equal(header["fixture_sha256"], fixture_sha, "wrong fixture binding")
    equal(header["suite"], suite, "wrong replay suite")
    require(header["execution"] in ("compiler-clear", "protected-local-reconstruction"), "unknown execution class")
    require(type(header["implementation"]) is str and 0 < len(header["implementation"]) <= 256,
            "bounded implementation identification required")
    sizes = {}
    selected = [r for r in fixture["runs"] if suite == "all" or r["descriptor_id"] ==
                fixture["receipt"]["canonical_gelu_descriptor_sha256"]]
    require(bool(selected), "empty replay suite")
    for run in selected:
        record = read_record()
        require(set(record) == {"id", "descriptor_id", "mask", "context", "shape", "native_bytes", "compiled", "outputs"},
                "run fields")
        for field in ("id", "descriptor_id", "mask", "context"):
            equal(record[field], run[field], f"wrong/reordered/duplicate {field}")
        entry = descriptors[run["descriptor_id"]]
        equal(record["shape"], entry["shape"], "wrong public compiler shape")
        size = record["native_bytes"]
        require(type(size) is int and 0 < size <= (16 << 20), "native diagnostic/key owner byte bound")
        require(size == sizes.setdefault(run["descriptor_id"], size), "mask-dependent native byte count")
        validate_compilation(record["compiled"], entry["descriptor"], run["mask"])
        equal(record["outputs"], run["expected"], "wrong/missing/reordered/duplicate numeric or Boolean output")

    external = read_record()
    equal(external, {"external_encoding": [c["signed_q12"] for c in fixture["external_encoding"]]},
          "wrong/missing external nearest-even outputs")
    require(stream.read(1) == b"", "extra/duplicate/trailing native output")
    return {"status": "pass", "execution_claim": header["execution"], "implementation": header["implementation"],
            "fixture_sha256": fixture_sha, "suite": suite,
            "checked": {"runs": len(selected), "evaluations": sum(len(r["words"]) for r in selected),
                        "external_encoding_cases": len(fixture["external_encoding"])}, "records": lines,
            "bytes_read": consumed, "security_qualified": False, "whole_model_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_output", help="JSONL file or '-' for stdin; never executed")
    parser.add_argument("--fixtures", type=Path, default=HERE / "fixtures.json")
    parser.add_argument("--suite", choices=("all", "gelu"), default="all")
    args = parser.parse_args()
    try:
        fixture, checksum = load_fixture(args.fixtures)
        if args.native_output == "-":
            result = check_stream(sys.stdin.buffer, fixture, checksum, args.suite)
        else:
            with Path(args.native_output).open("rb") as stream:
                result = check_stream(stream, fixture, checksum, args.suite)
        print(canonical(result).decode())
    except (Mismatch, OSError) as exc:
        print(canonical({"status": "fail", "error": str(exc)}).decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
