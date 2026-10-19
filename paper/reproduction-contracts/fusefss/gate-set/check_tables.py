"""Independent bounded public-table lookup replay; not complete nonlinear gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import check_outputs as common
import reference as ref

TABLES = ("silu", "exp0", "exp1", "reciprocal-k1", "reciprocal-k2", "reciprocal-k3", "rsqrt-k4", "rsqrt-k768")
MAX_FILE = 2 << 20


def fixture():
    frozen, base = common.verify_fixture()
    tables = []
    for name in TABLES:
        values = base["tables"][name]
        if not 1 <= len(values) <= 8192:
            raise ValueError("table bound")
        tables.append({"name": name, "input_bits": 50, "output_bits": 50,
                       "start": 0, "stop": len(values), "count": len(values),
                       "sha256_u64le": base["sha256_u64le"][name]})
    return {"schema": "pllm.fusefss.table-replay.v1", "gate_fixture_sha256": sha256(frozen).hexdigest(),
            "source_lock_sha256": base["source_lock_sha256"], "profile": base["profile"],
            "coverage": "public-table-lookup-only", "table_count": len(tables),
            "case_count": sum(table["count"] for table in tables), "tables": tables}


def check(data):
    if type(data) is not bytes or len(data) > MAX_FILE:
        raise ValueError("table output bound")
    expected_fixture = fixture()
    frozen = (ref.HERE / "table-fixtures.json").read_bytes()
    if frozen != ref.canonical(expected_fixture):
        raise ValueError("table fixture substitution")
    lines = data.splitlines(keepends=True)
    if len(lines) != len(TABLES) + 1:
        raise ValueError("table record count")
    header = common.parse(lines[0])
    if type(header) is not dict:
        raise ValueError("table header shape")
    impl = header.get("implementation")
    if type(impl) is not str or not 1 <= len(impl) <= 256:
        raise ValueError("table implementation identity")
    expected_header = {"schema": "pllm.fusefss.table-replay.v1/native-output",
                       "fixture_sha256": sha256(frozen).hexdigest(),
                       "gate_fixture_sha256": expected_fixture["gate_fixture_sha256"],
                       "source_lock_sha256": expected_fixture["source_lock_sha256"],
                       "execution": "protected-local-reconstruction", "implementation": impl,
                       "coverage": "public-table-lookup-only", "table_count": len(TABLES),
                       "case_count": expected_fixture["case_count"]}
    if ref.canonical(header) != ref.canonical(expected_header):
        raise ValueError("table header binding")
    _, base = common.verify_fixture()
    for line, table in zip(lines[1:], expected_fixture["tables"]):
        obj = common.parse(line)
        expected = {"table": table["name"], "sha256_u64le": table["sha256_u64le"],
                    "outputs": base["tables"][table["name"]]}
        if ref.canonical(obj) != ref.canonical(expected):
            raise ValueError("table order/type/value mismatch: " + table["name"])
    return {"status": "pass", "tables": len(TABLES), "cases": expected_fixture["case_count"],
            "coverage": "public-table-lookup-only", "fixture_sha256": sha256(frozen).hexdigest(),
            "scope": "finite reported lookup values; no complete gates or execution provenance established"}


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--fixture", action="store_true", help="regenerate fixture to stdout only")
        parser.add_argument("output", nargs="?", help="native JSONL path or -")
        args = parser.parse_args()
        if args.fixture:
            if args.output is not None:
                raise ValueError("fixture mode takes no output path")
            sys.stdout.buffer.write(ref.canonical(fixture()))
        else:
            if args.output is None:
                raise ValueError("native JSONL path required")
            if args.output == "-":
                data = sys.stdin.buffer.read(MAX_FILE + 1)
            else:
                with Path(args.output).open("rb") as handle:
                    data = handle.read(MAX_FILE + 1)
            print(json.dumps(check(data), sort_keys=True))
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"status": "reject", "error": str(error)}), file=sys.stderr)
        sys.exit(1)
