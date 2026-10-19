"""Read-only finite gate-output JSONL checker; never launches a native producer."""

from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys
import reference as ref

MAX_FILE = 8 << 20
MAX_LINE = 256 << 10


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def reject(_):
    raise ValueError("noninteger JSON number")


def parse_int(text):
    if len(text) > 18:
        raise ValueError("integer token bound")
    return int(text)


def parse(line):
    if len(line) > MAX_LINE or not line.endswith(b"\n") or line == b"\n":
        raise ValueError("JSONL line bound/newline")
    return json.loads(line, object_pairs_hook=unique, parse_float=reject,
                      parse_constant=reject, parse_int=parse_int)


def verify_fixture():
    frozen = (ref.HERE / "fixtures.json").read_bytes()
    if frozen != ref.canonical(ref.fixture()):
        raise ValueError("fixture substitution")
    return frozen, json.loads(frozen)


def selection(text, known):
    """Exact labels, no whitespace aliases, duplicate labels or empty tokens."""
    if type(text) is not str:
        raise ValueError("comma-delimited labels required")
    names = text.split(",")
    if not names or any(not name or name not in known for name in names) or len(set(names)) != len(names):
        raise ValueError("unknown/empty/duplicate label")
    return [name for name in known if name in names]


def check(data, gates=None):
    if type(data) is not bytes or len(data) > MAX_FILE:
        raise ValueError("output byte bound")
    frozen, fixture = verify_fixture()
    known = list(dict.fromkeys(case["gate"] for case in fixture["cases"]))
    selected = selection(gates, known) if gates is not None else None
    cases = [case for case in fixture["cases"] if selected is None or case["gate"] in selected]
    lines = data.splitlines(keepends=True)
    if len(lines) != len(cases) + 1:
        raise ValueError("missing/extra records")
    header = parse(lines[0])
    fields = {"schema", "fixture_sha256", "execution", "implementation"}
    if selected is not None:
        fields |= {"selected_gates", "selected_case_count", "source_lock_sha256", "coverage"}
    if type(header) is not dict or set(header) != fields:
        raise ValueError("header shape")
    schema = "pllm.fusefss.gate-set.v1/native-output" if selected is None else "pllm.fusefss.gate-set.v2/selected-native-output"
    if (header["schema"] != schema
            or header["fixture_sha256"] != sha256(frozen).hexdigest()
            or header["execution"] != "protected-local-reconstruction"
            or type(header["implementation"]) is not str
            or not 1 <= len(header["implementation"]) <= 256):
        raise ValueError("header binding")
    if selected is not None:
        expected_selection = {"selected_gates": selected, "selected_case_count": len(cases),
                              "source_lock_sha256": fixture["source_lock_sha256"], "coverage": "selected-gates-only"}
        if ref.canonical({key: header[key] for key in expected_selection}) != ref.canonical(expected_selection):
            raise ValueError("selected coverage binding")
    for line, case in zip(lines[1:], cases):
        obj = parse(line)
        expected = {"id": case["id"], "gate": case["gate"], "output": case["expected"]}
        # Canonical re-encoding avoids Python's True==1 and float==int equivalence.
        if ref.canonical(obj) != ref.canonical(expected):
            raise ValueError("case order/type/value mismatch: " + case["id"])
    return {"status": "pass", "cases": len(cases), "fixture_sha256": sha256(frozen).hexdigest(),
            "selected_gates": selected, "coverage": "all-frozen-cases" if selected is None else "selected-gates-only",
            "scope": "reported finite outputs only; execution provenance not established"}


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gates", help="exact comma-delimited frozen gate labels; explicit selected v2 contract")
        parser.add_argument("output", help="JSONL path or -")
        args = parser.parse_args()
        if args.output == "-":
            data = sys.stdin.buffer.read(MAX_FILE + 1)
        else:
            with Path(args.output).open("rb") as handle:
                data = handle.read(MAX_FILE + 1)
        print(json.dumps(check(data, args.gates), sort_keys=True))
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"status": "reject", "error": str(error)}), file=sys.stderr)
        sys.exit(1)
