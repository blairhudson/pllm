"""Bounded read-only archival gate replay. No builds, keys, writes or network."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIMIT = 32 << 20
LIVE_LIMIT = 64 << 20
MAX_FILES = 512
STEPS = (
    "clippy",
    "selected-native",
    "tables-native",
    "checker-selected",
    "checker-tables",
    "native-library",
    "frozen-reference",
    "independent-fixture",
    "gate-set-fixture",
    "format",
)
GATES = ["shift", "b2a", "compare", "phi-select", "external"]
PREFIX = "paper/reproduction-contracts/fusefss/gate-set/"
SUMS = (
    "fresh_issuance_count",
    "fresh_lane_issuance_count",
    "dcf_keys_per_party",
    "dcf_evaluations_per_party",
    "triples_per_party",
    "cumulative_private_material_bytes_both_parties",
    "cumulative_public_table_copy_bytes_both_parties",
    "peer_frames_both_parties",
    "peer_frame_bytes_both_parties",
)
MAXES = ("peak_logical_budget_bytes", "max_observed_live_budget_bytes")
COUNTERS = set(SUMS + MAXES + ("final_reservation_bytes",))
ACCOUNTING = {
    "material": "cumulative issued private payload layout, both parties; not allocator residency or transport",
    "budget": "peak exact reserved envelope derived from resources before each issuance, including live prior outputs/table; not RSS",
    "excluded_test_harness_owners": "fixture JSON, caller buffers, bounded descriptor parser/borrowed descriptor, export buffers, filesystem I/O",
    "freshness": "each issuance uses production OS randomness for masks, keys and correlations; input shares freshly randomized",
    "frames": "actual exported peer frame lengths; checked against resources",
}
LIMITATIONS = [
    "finite test outputs only",
    "no source/model/SDK completion claim",
    "not independent security qualification",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, lower=0, upper=(1 << 63) - 1):
    require(type(value) is int and lower <= value <= upper, "integer type/range")
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_number(_):
    raise ValueError("noninteger JSON number")


def parse_integer(text):
    require(len(text) <= 20, "JSON integer token limit")
    return int(text)


def finite_float(text):
    require(len(text) <= 64, "JSON float token limit")
    value = float(text)
    require(math.isfinite(value), "nonfinite JSON float")
    return value


def loads(data, *, allow_floats=False):
    require(type(data) is bytes and len(data) <= LIMIT, "JSON byte bound")
    return json.loads(
        data,
        object_pairs_hook=unique,
        parse_float=finite_float if allow_floats else reject_number,
        parse_constant=reject_number,
        parse_int=parse_integer,
    )


def relative(name):
    require(type(name) is str and 0 < len(name) <= 512 and "\\" not in name, "unsafe path")
    path = PurePosixPath(name)
    require(
        not path.is_absolute() and ".." not in path.parts and str(path) == name and name != ".",
        "unsafe path",
    )
    return Path(*path.parts)


def read(path, limit=LIMIT):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "file byte limit")
    return data


def safe_read(directory, name, limit=LIMIT):
    rel = relative(name)
    path = directory / rel
    require(
        not any(
            (directory / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)
        ),
        "symlink artifact",
    )
    require(path.is_file(), "missing regular file: " + name)
    return read(path, limit)


def pin(data):
    return {"bytes": len(data), "sha256": sha(data)}


def validate_pin(value):
    require(type(value) is dict and set(value) == {"bytes", "sha256"}, "pin fields")
    integer(value["bytes"], 0, LIMIT)
    digest = value["sha256"]
    require(
        type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None, "SHA256 encoding"
    )


def decode_gzip(data, limit=LIMIT):
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        result = stream.read(limit + 1)
    require(len(result) <= limit, "decoded byte bound")
    return result


def archived_inputs(evidence):
    inventory = loads(safe_read(evidence, "source-inventory.json"))
    require(
        type(inventory) is dict and type(inventory.get("files")) is dict, "source inventory fields"
    )
    pins = inventory["files"]
    require(1 <= len(pins) <= MAX_FILES, "source file count")
    for name, value in pins.items():
        relative(name)
        validate_pin(value)
    raw = decode_gzip(safe_read(evidence, "inputs.tar.gz"))
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            relative(member.name)
            require(
                member.isfile() and member.name not in sources and member.name in pins,
                "archive member/coverage",
            )
            integer(member.size, 0, LIMIT)
            handle = archive.extractfile(member)
            require(handle is not None, "missing archive entry")
            with handle:
                data = handle.read(LIMIT + 1)
            equal(pin(data), pins[member.name], "archived source mismatch: " + member.name)
            sources[member.name] = data
            require(len(sources) <= MAX_FILES, "archive member count")
    require(set(sources) == set(pins), "source inventory exact coverage")
    return sources


def manifest(evidence):
    obj = loads(safe_read(evidence, "manifest.json"))
    require(type(obj) is dict and set(obj) == {"schema", "files"}, "manifest fields")
    require(
        obj["schema"] == "pllm.fusefss.gates-evidence-files.v1" and type(obj["files"]) is dict,
        "manifest schema",
    )
    files = obj["files"]
    required = {
        "source-inventory.json",
        "inputs.tar.gz",
        "run-receipt.json",
        "selected-gates.jsonl.gz",
        "selected-gates.receipt.json",
        "tables.jsonl.gz",
        "tables.receipt.json",
    }
    require(required <= set(files) and 1 <= len(files) <= MAX_FILES, "manifest file coverage")
    total = 0
    for name, expected in files.items():
        validate_pin(expected)
        data = safe_read(evidence, name)
        total += len(data)
        require(total <= 128 << 20, "aggregate artifact bound")
        equal(pin(data), expected, "artifact mismatch: " + name)
    return files


def validation(evidence, files):
    obj = loads(safe_read(evidence, "run-receipt.json"), allow_floats=True)
    require(
        type(obj) is dict and obj.get("schema") == "pllm.fusefss.gates-validation.v1",
        "validation schema",
    )
    require(
        obj.get("status") == "pass" and obj.get("source_unchanged") is True, "validation incomplete"
    )
    steps = obj.get("steps")
    require(type(steps) is list and len(steps) == len(STEPS), "validation step count")
    seen = set()
    for step in steps:
        require(
            type(step) is dict
            and {"name", "command", "returncode", "expected_returncode", "log_sha256"} <= set(step),
            "step fields",
        )
        name = step["name"]
        require(type(name) is str and name in STEPS and name not in seen, "unknown/duplicate step")
        seen.add(name)
        integer(step["returncode"], 0, 0)
        integer(step["expected_returncode"], 0, 0)
        elapsed = step.get("elapsed_seconds")
        require(
            type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 86400,
            "validation elapsed-time field",
        )
        command = step["command"]
        require(
            (type(command) is str and 0 < len(command) <= 8192)
            or (
                type(command) is list
                and 0 < len(command) <= 128
                and all(type(part) is str and 0 < len(part) <= 8192 for part in command)
            ),
            "recorded command shape",
        )
        # Permit root or a logs/ directory, never two same-name logs/ambiguity.
        logs = [path for path in files if PurePosixPath(path).name == name + ".log"]
        require(len(logs) == 1, "missing/ambiguous step log: " + name)
        require(sha(safe_read(evidence, logs[0])) == step["log_sha256"], "step log digest: " + name)
    require(seen == set(STEPS), "validation step coverage")
    return obj


def stats(value):
    require(type(value) is dict and set(value) == COUNTERS, "missing/extra accounting counters")
    for number in value.values():
        integer(number)
    require(value["final_reservation_bytes"] == 0, "unretired reservation")
    require(
        value["max_observed_live_budget_bytes"] <= value["peak_logical_budget_bytes"] <= LIVE_LIMIT,
        "live peak/observation bound",
    )
    return value


def combined(rows, totals):
    stats(totals)
    for row in rows:
        stats(row["resources"])
    expected = {key: sum(row["resources"][key] for row in rows) for key in SUMS}
    expected.update({key: max(row["resources"][key] for row in rows) for key in MAXES})
    expected["final_reservation_bytes"] = 0
    equal(totals, expected, "per-record accounting conservation")


def base_receipt(evidence, filename, sources, scope, output_file, output_limit):
    receipt = loads(safe_read(evidence, filename))
    fields = {
        "schema",
        "coverage",
        "implementation",
        "source_sha256",
        "fixture_sha256",
        "table_fixture_sha256",
        "source_lock_sha256",
        "source_input_fingerprints_before",
        "source_input_fingerprints_after",
        "source_inputs_unchanged",
        "output_sha256",
        "totals",
        "records",
        "accounting",
        "limitations",
    }
    require(type(receipt) is dict and set(receipt) == fields, "native receipt fields")
    require(
        receipt["schema"] == "pllm.fusefss.f2.native-export-receipt.v1"
        and receipt["coverage"] == scope,
        "native receipt scope/schema",
    )
    require(receipt["source_inputs_unchanged"] is True, "changed source/input flag")
    equal(receipt["accounting"], ACCOUNTING, "accounting scope changed")
    equal(receipt["limitations"], LIMITATIONS, "qualification limitations changed")
    native = sources["crates/pllm-garble/src/fusefss_backend/gate_conformance.rs"]
    registry = native.split(b"const SOURCES:", 1)[1].split(b"\n];", 1)[0]
    names = [
        name.decode("ascii") for name in re.findall(rb'"([^"]+)"\s*,\s*include_bytes!', registry)
    ]
    require(names and len(names) == len(set(names)), "archived native source registry")
    expected_sources = {name: sha(sources[name]) for name in names}
    equal(receipt["source_sha256"], expected_sources, "native compiled source snapshot")
    implementation = "pllm-native-f2-test-sha256:" + sha(canonical(expected_sources))
    require(receipt["implementation"] == implementation, "native implementation identity")
    fingerprints = dict(expected_sources)
    for name, key in (
        ("fixtures.json", "fixture_sha256"),
        ("table-fixtures.json", "table_fixture_sha256"),
        ("source-lock.json", "source_lock_sha256"),
    ):
        digest = sha(sources[PREFIX + name])
        require(receipt[key] == digest, "native input snapshot: " + name)
        fingerprints[PREFIX + name] = digest
    equal(
        receipt["source_input_fingerprints_before"],
        fingerprints,
        "source/input before archive binding",
    )
    equal(
        receipt["source_input_fingerprints_after"],
        fingerprints,
        "source/input after archive binding",
    )
    output = decode_gzip(safe_read(evidence, output_file), output_limit)
    require(sha(output) == receipt["output_sha256"], "native decoded output digest")
    lines = output.splitlines(keepends=True)
    require(
        lines and all(line.endswith(b"\n") and len(line) <= 256 << 10 for line in lines),
        "native output JSONL bound",
    )
    header = loads(lines[0])
    require(header["implementation"] == implementation, "native output implementation binding")
    rows = receipt["records"]
    require(type(rows) is list and 1 <= len(rows) <= 275, "native receipt record count")
    combined(rows, receipt["totals"])
    return receipt, output


def selected_accounting(receipt, fixture):
    cases = [case for case in fixture["cases"] if case["gate"] in GATES]
    require(len(cases) == len(receipt["records"]) == 66, "selected case count")
    external = 0
    for row, case in zip(receipt["records"], cases):
        require(
            type(row) is dict and set(row) == {"id", "gate", "resources", "binding"},
            "selected row fields",
        )
        require(
            row["id"] == case["id"] and row["gate"] == case["gate"], "selected row order/binding"
        )
        s, gate, args = row["resources"], case["gate"], case["args"]
        expected = {key: 0 for key in SUMS}
        binding = {}
        if gate == "external":
            external += 1
            binding = {"execution": "local-exact-rational-native-frontend; no protected issuance"}
            require(
                s["peak_logical_budget_bytes"] == s["max_observed_live_budget_bytes"] == 0,
                "public external reservation",
            )
        else:
            expected.update(
                fresh_issuance_count=1, fresh_lane_issuance_count=1, peer_frames_both_parties=2
            )
            require(s["max_observed_live_budget_bytes"] > 0, "missing protected live accounting")
            if gate == "shift":
                widths = [args[2]] if args[3] == "TR" else [args[1], args[2]]
                keys = len(widths)
                expected.update(
                    dcf_keys_per_party=keys,
                    dcf_evaluations_per_party=keys,
                    cumulative_private_material_bytes_both_parties=2
                    * (16 + sum(32 + 34 * width for width in widths)),
                    peer_frame_bytes_both_parties=176,
                )
            elif gate in ("b2a", "phi-select"):
                count = 2 if gate == "phi-select" else 1
                expected.update(
                    fresh_issuance_count=count,
                    fresh_lane_issuance_count=count,
                    triples_per_party=count,
                    cumulative_private_material_bytes_both_parties=48 * count,
                    peer_frames_both_parties=2 * count,
                    peer_frame_bytes_both_parties=192 * count,
                )
            elif gate == "compare":
                width = args[4] if args[3] == "low" else args[1]
                expected.update(
                    dcf_keys_per_party=2,
                    dcf_evaluations_per_party=2,
                    triples_per_party=1,
                    cumulative_private_material_bytes_both_parties=2 * (2 * (24 + 26 * width) + 56),
                    peer_frames_both_parties=4,
                    peer_frame_bytes_both_parties=368,
                )
                require(
                    type(row["binding"]) is dict
                    and set(row["binding"])
                    == {
                        "descriptor_sha256",
                        "bit_operands_per_party",
                        "descriptor_parser_and_borrowed_owner",
                    },
                    "comparison binding fields",
                )
                integer(row["binding"]["bit_operands_per_party"], 3, 3)
                require(
                    row["binding"]["descriptor_parser_and_borrowed_owner"]
                    == "outside backend Budget; <=4096-byte test input",
                    "comparison parser ownership",
                )
                descriptor = comparison_descriptor(args, fixture["source_lock_sha256"])
                require(
                    row["binding"]["descriptor_sha256"]
                    == sha(b"FuseFSS-reference-descriptor-v1\0" + canonical(descriptor)),
                    "comparison descriptor identity",
                )
                binding = row["binding"]
        equal(
            {key: s[key] for key in SUMS},
            expected,
            "selected per-gate material/frame counts: " + case["id"],
        )
        equal(row["binding"], binding, "selected detail binding")
    require(external == 6, "public external encoding coverage")
    require(
        receipt["totals"]["fresh_issuance_count"]
        == receipt["totals"]["fresh_lane_issuance_count"]
        == 62,
        "selected issuance coverage",
    )


def comparison_descriptor(args, source):
    mode = args[3]
    expr = (
        ["low_lt", args[4], args[2]]
        if mode == "low"
        else ["slt" if mode == "signed" else "lt", args[2]]
    )
    return {
        "schema": "pllm.fusefss.scalar-descriptor.v1",
        "profile": "fusefss.synthetic-conformance.v1",
        "gate": "f2-comparison-conformance",
        "source_lock_sha256": source,
        "input": {"domain": "A", "bits": args[1], "fractional": 0, "signed": True},
        "output": {"arithmetic": "same-as-input", "boolean_arity": 1},
        "external_encoding": "nearest-ties-even-input-and-weight",
        "internal_rounding": "signed-floor-only",
        "partition_order": "unsigned-canonical-half-open",
        "degree": 1,
        "coefficient_order": ["constant", "linear"],
        "coefficient_fractional": [0, 0],
        "selector_semantics": "descriptor-defined",
        "source_schedule": {},
        "phi": "identity",
        "boundaries": [0, 1 << args[1]],
        "pieces": [{"coefficients": [0, 0], "booleans": [expr]}],
    }


def table_accounting(receipt, fixture, table_fixture):
    rows = receipt["records"]
    require(len(rows) == table_fixture["table_count"] == 8, "complete table coverage")
    for row, meta in zip(rows, table_fixture["tables"]):
        fields = {
            "table",
            "table_digest",
            "outputs",
            "chunks_per_lane",
            "chunk_entries",
            "original_public_value_bytes",
            "dcf_path_levels_per_party",
            "resources",
        }
        require(type(row) is dict and set(row) == fields, "table accounting row fields")
        name, count = meta["name"], meta["count"]
        require(row["table"] == name, "table accounting order")
        values = fixture["tables"][name]
        require(count == len(values) and count % 4 == 0 and count <= 8192, "bounded complete table")
        chunks = (count + 255) // 256
        batches = count // 4
        intervals = [min(256, count - 256 * j) + int(j != 0) + 1 for j in range(chunks)]
        evaluations = count * sum(intervals)
        keys = count * chunks
        expected = {
            "fresh_issuance_count": batches,
            "fresh_lane_issuance_count": count,
            "dcf_keys_per_party": keys,
            "dcf_evaluations_per_party": evaluations,
            "triples_per_party": 0,
            "cumulative_private_material_bytes_both_parties": 2 * keys * (24 + 26 * 50 + 24),
            "cumulative_public_table_copy_bytes_both_parties": 2
            * count
            * sum(24 * i + 8 for i in intervals),
            "peer_frames_both_parties": 2 * batches,
            "peer_frame_bytes_both_parties": 2 * batches * (80 + 8 * 4 * chunks),
        }
        equal(
            {key: row["resources"][key] for key in SUMS},
            expected,
            "table per-record accounting: " + name,
        )
        for field, value in (
            ("outputs", count),
            ("chunks_per_lane", chunks),
            ("chunk_entries", 256),
            ("original_public_value_bytes", count * 8),
            ("dcf_path_levels_per_party", evaluations * 50),
        ):
            integer(row[field], value, value)
        words = b"".join(integer(v, 0, (1 << 50) - 1).to_bytes(8, "little") for v in values)
        identity = (
            b"pllm.fusefss.zero-extended-public-table.v1\0"
            + bytes.fromhex(fixture["source_lock_sha256"])
            + bytes([50])
            + count.to_bytes(8, "little")
            + (256).to_bytes(8, "little")
            + words
        )
        require(row["table_digest"] == sha(identity), "native table plan/source binding")
        require(
            row["resources"]["max_observed_live_budget_bytes"] > 0, "missing table live reservation"
        )
    require(sum(row["outputs"] for row in rows) == 18560, "table output count")
    require(
        receipt["totals"]["fresh_issuance_count"] == 4640
        and receipt["totals"]["fresh_lane_issuance_count"] == 18560,
        "table batch/lane coverage",
    )


def live_checker_dependencies(sources):
    required = {
        PREFIX + name
        for name in (
            "check_outputs.py",
            "check_tables.py",
            "reference.py",
            "fixtures.json",
            "table-fixtures.json",
            "source-lock.json",
        )
    }
    required |= {
        "paper/reproduction-contracts/shared_profile_oracle.py",
        "paper/reproduction-contracts/fusefss/reference.py",
        "paper/reproduction-contracts/fusefss/gelu-table.json",
        "paper/reproduction-contracts/fusefss/source-lock.json",
    }
    require(required <= set(sources), "missing checker dependency snapshot")
    for name in sorted(required):
        require(
            sha(safe_read(ROOT, name)) == sha(sources[name]),
            "live checker dependency differs from archive: " + name,
        )


def independent_check(script, args, output):
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / PREFIX / script), *args, "-"],
        input=output,
        capture_output=True,
        timeout=60,
        cwd=ROOT,
        env=env,
        check=False,
    )
    require(
        result.returncode == 0 and len(result.stdout) <= 65536 and not result.stderr,
        "independent checker process rejected",
    )
    summary = loads(result.stdout)
    require(type(summary) is dict and summary.get("status") == "pass", "independent checker status")
    return summary


def main(evidence=HERE):
    evidence = Path(evidence).resolve()
    files = manifest(evidence)
    sources = archived_inputs(evidence)
    validation(evidence, files)
    fixture = loads(sources[PREFIX + "fixtures.json"])
    table_fixture = loads(sources[PREFIX + "table-fixtures.json"])
    selected, selected_output = base_receipt(
        evidence,
        "selected-gates.receipt.json",
        sources,
        "selected-gates-only",
        "selected-gates.jsonl.gz",
        8 << 20,
    )
    tables, table_output = base_receipt(
        evidence,
        "tables.receipt.json",
        sources,
        "public-table-lookup-only",
        "tables.jsonl.gz",
        2 << 20,
    )
    require(
        selected["implementation"] == tables["implementation"], "native producer identity differs"
    )
    selected_accounting(selected, fixture)
    table_accounting(tables, fixture, table_fixture)
    live_checker_dependencies(sources)
    selected_summary = independent_check(
        "check_outputs.py", ["--gates", ",".join(GATES)], selected_output
    )
    live_checker_dependencies(sources)
    table_summary = independent_check("check_tables.py", [], table_output)
    equal(
        selected_summary,
        {
            "status": "pass",
            "cases": 66,
            "fixture_sha256": sha(sources[PREFIX + "fixtures.json"]),
            "selected_gates": GATES,
            "coverage": "selected-gates-only",
            "scope": "reported finite outputs only; execution provenance not established",
        },
        "selected independent scope",
    )
    equal(
        table_summary,
        {
            "status": "pass",
            "tables": 8,
            "cases": 18560,
            "coverage": "public-table-lookup-only",
            "fixture_sha256": sha(sources[PREFIX + "table-fixtures.json"]),
            "scope": "finite reported lookup values; no complete gates or execution provenance established",
        },
        "table independent scope",
    )
    return {
        "status": "pass",
        "selected_cases": 66,
        "selected_issuances": 62,
        "public_external_encoding_cases": 6,
        "tables": 8,
        "table_outputs": 18560,
        "table_batches": 4640,
        "selected_accounting": selected["totals"],
        "table_accounting": tables["totals"],
        "scope": "archived selected gates and public table lookups only; not complete gates/model/security qualification",
    }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--evidence",
            type=Path,
            default=HERE,
            help="evidence directory (default: script directory)",
        )
        args = parser.parse_args()
        print(json.dumps(main(args.evidence), sort_keys=True, separators=(",", ":")))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        UnicodeError,
        EOFError,
        RecursionError,
        tarfile.TarError,
        zlib.error,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"status": "reject", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
