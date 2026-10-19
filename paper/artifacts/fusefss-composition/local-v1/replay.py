"""Bounded read-only composition archive replay; no builds, writes, keys or network.

Self-pinned archives establish internal consistency, not publisher authenticity,
execution provenance, independent cryptographic security, or model completion.
Only dependency-matched local PLLM checkers are executed, never archived code.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
ARCHIVE_LIMIT = 128 << 20
LIVE_LIMIT = 64 << 20
MAX_FILES = 512
PREFIX = "paper/reproduction-contracts/fusefss/gate-set/"
NATIVE = "crates/pllm-garble/src/fusefss_backend/"
FIXTURE_SHA = "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
LOCK_SHA = "8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd"
STEPS = (
    "clippy",
    "complete-native",
    "checker-complete",
    "native-library",
    "frozen-reference",
    "independent-fixture",
    "gate-set-fixture",
    "native-tables",
    "format",
)
PRIMITIVES = {"shift", "b2a", "compare", "phi-select", "external"}
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
COMPOSITION_COUNTERS = {
    "instructions",
    "registers",
    "issuances",
    "lanes",
    "dcf_evaluations_per_party",
    "dcf_keys_total",
    "dcf_evaluations_total",
    "triples_per_party",
    "triples_total",
    "private_material_bytes_total",
    "public_table_copy_bytes_total",
    "cumulative_work",
    "plan_reservation_bytes",
    "public_table_reservation_bytes",
    "retained_frame_overlap_bytes",
    "combined_reservation_bytes",
    "peak_observed_reserved_bytes",
    "peer_frames_both_parties",
    "peer_frame_bytes_both_parties",
}
ACCOUNTING = {
    "freshness": "production OS randomness per primitive; externally issued material; fresh input shares",
    "budget": "complete planned envelopes and sampled native reservations; observed samples can miss issuance scratch and are not a high-water mark, allocator RSS or process peak",
    "frames": "actual native peer frame lengths; no authenticated network transport",
    "excluded_test_owners": "fixture JSON, caller buffers, source snapshots, reconstruction, export buffers and filesystem I/O",
}
LIMITATIONS = [
    "public input ranges are caller assumptions, not private membership proofs",
    "finite protected gate conformance, not complete decoder or SDK execution",
    "local trusted-dealer feed, not distributed preprocessing or independent security qualification",
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
        parse_int=parse_integer,
        parse_float=finite_float if allow_floats else reject_number,
        parse_constant=reject_number,
    )


def relative(name):
    require(
        type(name) is str and 0 < len(name) <= 512 and "\\" not in name and "\0" not in name,
        "unsafe path",
    )
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
    require(
        not any(
            (directory / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)
        ),
        "symlink artifact",
    )
    path = directory / rel
    require(path.is_file(), "missing regular file: " + name)
    return read(path, limit)


def pin(data):
    return {"bytes": len(data), "sha256": sha(data)}


def digest(value):
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "SHA256 encoding"
    )
    return value


def validate_pin(value):
    require(type(value) is dict and set(value) == {"bytes", "sha256"}, "pin fields")
    integer(value["bytes"], 0, LIMIT)
    digest(value["sha256"])


def decode_gzip(data, limit=ARCHIVE_LIMIT):
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        result = stream.read(limit + 1)
    require(len(result) <= limit, "decoded byte bound")
    return result


def archive_files(data, pins):
    require(type(pins) is dict and 1 <= len(pins) <= MAX_FILES, "archive pin count")
    for name, value in pins.items():
        relative(name)
        validate_pin(value)
    raw = decode_gzip(data)
    sources = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            relative(member.name)
            require(
                member.isfile()
                and not member.issparse()
                and member.name not in sources
                and member.name in pins,
                "archive member/coverage",
            )
            integer(member.size, 0, LIMIT)
            total += member.size
            require(total <= ARCHIVE_LIMIT and len(sources) < MAX_FILES, "archive aggregate bound")
            handle = archive.extractfile(member)
            require(handle is not None, "missing archive entry")
            with handle:
                content = handle.read(LIMIT + 1)
            equal(pin(content), pins[member.name], "archived file mismatch: " + member.name)
            sources[member.name] = content
        # tarfile otherwise silently ignores nonzero garbage/second tar after EOF.
        require(not any(raw[archive.offset :]), "nonzero trailing tar data")
    require(set(sources) == set(pins), "archive exact coverage")
    return sources


def bundle(evidence):
    index = loads(safe_read(evidence, "artifact-index.json"))
    require(
        type(index) is dict and set(index) in ({"schema", "files"}, {"schema", "files", "archive"}),
        "artifact index fields",
    )
    require(
        index["schema"] == "pllm.fusefss.composition-evidence-files.v1", "artifact index schema"
    )
    data = safe_read(evidence, "evidence.tar.gz")
    if "archive" in index:
        validate_pin(index["archive"])
        equal(pin(data), index["archive"], "compressed evidence pin")
    files = archive_files(data, index["files"])
    require(
        {
            "inputs.tar.gz",
            "source-inventory.json",
            "run-receipt.json",
            "complete-gates.jsonl",
            "complete-gates.receipt.json",
        }
        <= set(files),
        "required artifact coverage",
    )
    return files


def archived_inputs(files):
    inventory = loads(files["source-inventory.json"])
    require(type(inventory) is dict and type(inventory.get("files")) is dict, "source inventory")
    return archive_files(files["inputs.tar.gz"], inventory["files"])


def validation(files):
    obj = loads(files["run-receipt.json"], allow_floats=True)
    require(
        type(obj) is dict and obj.get("schema") == "pllm.fusefss.composition-validation.v1",
        "validation schema",
    )
    require(
        obj.get("status") == "pass" and obj.get("source_unchanged") is True, "validation incomplete"
    )
    steps = obj.get("steps")
    require(type(steps) is list and len(steps) == len(STEPS), "validation step count")
    require(
        [step.get("name") for step in steps if type(step) is dict] == list(STEPS),
        "validation step order/coverage",
    )
    for step in steps:
        name = step["name"]
        integer(step.get("returncode"), 0, 0)
        integer(step.get("expected_returncode"), 0, 0)
        elapsed = step.get("elapsed_seconds")
        require(
            type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 86400,
            "validation elapsed field",
        )
        command = step.get("command")
        require(
            type(command) is list
            and 0 < len(command) <= 128
            and all(type(s) is str and 0 < len(s) <= 8192 for s in command),
            "command shape",
        )
        if name == "checker-complete":
            require(
                "--gates" not in command
                and not any(s.startswith("--gates=") for s in command)
                and any(s.endswith("gate-set/check_outputs.py") for s in command),
                "partial checker command",
            )
        if name == "complete-native":
            require(
                all(
                    s in command
                    for s in ("--release", "--locked", "--offline", "--ignored", "--exact")
                ),
                "native command contract",
            )
            require(
                "fusefss_backend::composition::conformance::export_complete_gates" in command
                and "--test-threads=1" in command,
                "complete native test selection",
            )
            overrides = step.get("environment_overrides")
            require(
                type(overrides) is dict and overrides.get("PLLM_FUSEFSS_EXPORT_COMPOSITION") == "1",
                "native opt-in environment",
            )
        logs = [path for path in files if PurePosixPath(path).name == name + ".log"]
        require(
            len(logs) == 1 and sha(files[logs[0]]) == step.get("log_sha256"),
            "validation log binding",
        )
    return obj


def source_registry(sources, name):
    native = sources[name]
    require(native.count(b"const SOURCES:") == 1, "native source registry")
    registry = native.split(b"const SOURCES:", 1)[1].split(b"\n];", 1)[0]
    names = [s.decode("ascii") for s in re.findall(rb'"([^"]+)"\s*,\s*include_bytes!', registry)]
    require(names and len(names) == len(set(names)), "source registry coverage")
    return {name: sha(sources[name]) for name in names}


def source_dependencies(sources):
    required = {
        PREFIX + name
        for name in (
            "check_outputs.py",
            "reference.py",
            "fixtures.json",
            "source-lock.json",
            "source-tables-supplement.json",
        )
    }
    required |= {
        "paper/reproduction-contracts/shared_profile_oracle.py",
        "paper/reproduction-contracts/fusefss/reference.py",
        "paper/reproduction-contracts/fusefss/gelu-table.json",
        "paper/reproduction-contracts/fusefss/source-lock.json",
    }
    require(required <= set(sources), "checker dependency coverage")
    require(sha(sources[PREFIX + "fixtures.json"]) == FIXTURE_SHA, "original frozen fixture")
    require(sha(sources[PREFIX + "source-lock.json"]) == LOCK_SHA, "original frozen source lock")
    lock = loads(sources[PREFIX + "source-lock.json"])
    for name, expected in lock["local_sha256"].items():
        relative(name)
        require(sha(sources[name]) == digest(expected), "source-locked oracle dependency: " + name)
        required.add(name)
    for name in sorted(required):
        require(
            sha(safe_read(ROOT, name)) == sha(sources[name]),
            "live checker dependency differs: " + name,
        )


def supplement(sources):
    obj = loads(sources[PREFIX + "source-tables-supplement.json"])
    values = [math.isqrt((1 << 56) // ((128 + (i >> 6)) << (i & 63))) for i in range(8192)]
    expected = {
        "schema": "pllm.fusefss.table-supplement.v1",
        "name": "rsqrt-k2",
        "k": 2,
        "count": 8192,
        "input_bits": 50,
        "output_bits": 50,
        "formula": "isqrt(floor((2^55*k)/((128+(index>>6))*2^(index&63))))",
        "gate_fixture_sha256": FIXTURE_SHA,
        "source_lock_sha256": LOCK_SHA,
        "sha256_u64le": sha(b"".join(v.to_bytes(8, "little") for v in values)),
        "values": values,
    }
    equal(obj, expected, "separate exact rsqrt-k2 supplement")


def primitive_stats(value):
    require(type(value) is dict and set(value) == COUNTERS, "primitive counter fields")
    for number in value.values():
        integer(number)
    require(value["final_reservation_bytes"] == 0, "unretired primitive reservation")
    require(
        value["max_observed_live_budget_bytes"] <= value["peak_logical_budget_bytes"] <= LIVE_LIMIT,
        "primitive live envelope",
    )


def comparison_descriptor(args):
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
        "source_lock_sha256": LOCK_SHA,
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


def primitive_accounting(execution, case):
    require(
        set(execution) == {"producer", "resources", "binding"}
        and execution["producer"] == "shared-selected-primitive",
        "primitive execution fields",
    )
    s, binding = execution["resources"], execution["binding"]
    primitive_stats(s)
    expected = dict.fromkeys(SUMS, 0)
    gate, args = case["gate"], case["args"]
    detail = {}
    if gate == "external":
        detail = {"execution": "local-exact-rational-native-frontend; no protected issuance"}
        require(
            s["peak_logical_budget_bytes"] == s["max_observed_live_budget_bytes"] == 0,
            "external is public encoding",
        )
    else:
        expected.update(
            fresh_issuance_count=1, fresh_lane_issuance_count=1, peer_frames_both_parties=2
        )
        require(s["max_observed_live_budget_bytes"] > 0, "protected primitive reservation")
        if gate == "shift":
            widths = [args[2]] if args[3] == "TR" else [args[1], args[2]]
            expected.update(
                dcf_keys_per_party=len(widths),
                dcf_evaluations_per_party=len(widths),
                cumulative_private_material_bytes_both_parties=2
                * (16 + sum(32 + 34 * w for w in widths)),
                peer_frame_bytes_both_parties=176,
            )
        elif gate in ("b2a", "phi-select"):
            n = 2 if gate == "phi-select" else 1
            expected.update(
                fresh_issuance_count=n,
                fresh_lane_issuance_count=n,
                triples_per_party=n,
                cumulative_private_material_bytes_both_parties=48 * n,
                peer_frames_both_parties=2 * n,
                peer_frame_bytes_both_parties=192 * n,
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
            detail = {
                "descriptor_sha256": sha(
                    b"FuseFSS-reference-descriptor-v1\0" + canonical(comparison_descriptor(args))
                ),
                "bit_operands_per_party": 3,
                "descriptor_parser_and_borrowed_owner": "outside backend Budget; <=4096-byte test input",
            }
    equal({key: s[key] for key in SUMS}, expected, "primitive per-case conservation: " + case["id"])
    equal(binding, detail, "primitive descriptor binding")


class Ledger:
    """Public schedule counts only, not gate evaluation or native provenance.

    Costs derive separately from fixed-width native payload/frame formulas. No
    private indices, output words or receipt-reported counts enter this ledger.
    """

    def __init__(self):
        self.instructions = self.steps = self.keys = self.evaluations = self.triples = 0
        self.private = self.frames = self.frame_bytes = self.work = self.overlap = 0
        self.public_table_copy_bytes_total = 0
        self.tables = []

    def local(self, count=1):
        self.instructions += count
        self.work += count

    def native(self, keys, evaluations, triples, private, frames, frame_bytes, work):
        self.steps += 1
        self.keys += keys
        self.evaluations += evaluations
        self.triples += triples
        self.private += private
        self.frames += frames
        self.frame_bytes += frame_bytes
        self.overlap = max(self.overlap, frame_bytes)
        self.work += work
        self.local()

    def conversion(self, kind, bits=50, shift=0):
        product = kind == "product"
        widths = (
            [] if product else ([shift] if kind == "tr" else [bits, *([shift] if shift else [])])
        )
        private = 48 if product else 2 * (16 + sum(32 + 34 * w for w in widths))
        self.native(
            len(widths),
            len(widths),
            int(product),
            private,
            2,
            192 if product else 176,
            len(widths) * bits + int(product),
        )

    def compare(self):
        # Two DCF queries, nine VM ops, three private Bit operands, one AND.
        self.native(2, 2, 1, 2 * (2 * (24 + 26 * 50) + 56), 4, 368, 2 * 50 + 9 + 1)
        self.conversion("product", 1)  # Fresh B2A, not arithmetic addition.

    def lookup(self, entries):
        chunks = (entries + 255) // 256
        intervals = sum(min(256, entries - 256 * j) + int(j != 0) + 1 for j in range(chunks))
        # Both parties retain each chunk's actual boundary/coefficient owners:
        # 24 bytes per source interval plus one 8-byte boundary sentinel.
        # Charge EVERY lookup, independently of reusable original Table storage.
        self.public_table_copy_bytes_total += 2 * (24 * intervals + 8 * chunks)
        self.native(
            chunks,
            intervals,
            0,
            2 * chunks * (24 + 26 * 50 + 24),
            2,
            2 * (80 + 8 * chunks),
            intervals * 50,
        )

    def positive_shift(self, shift):
        self.conversion("tr", shift=shift)
        self.conversion("extend", bits=50 - shift)

    def nexp(self):
        self.local()  # Local low-byte view.
        self.conversion("extend", bits=8)
        self.positive_shift(8)
        self.lookup(256)
        self.lookup(256)
        self.conversion("product")
        self.positive_shift(12)

    def index(self):
        self.local(4)
        for exponent in range(1, 50):
            if exponent <= 7:
                self.local()
            else:
                self.positive_shift(exponent - 7)
            self.local(2)
            self.compare()
            self.conversion("product")
            self.conversion("product")
        self.local(2)


def schedule(case):
    ledger = Ledger()
    gate, args = case["gate"], case["args"]
    inputs = 1
    if gate == "silu":
        ledger.tables = [1024]
        ledger.local()
        ledger.conversion("tr", bits=38, shift=6)
        ledger.conversion("extend", bits=32)
        ledger.compare()
        ledger.local(2)
        ledger.conversion("product")
        ledger.compare()
        ledger.local()
        ledger.conversion("product")
        ledger.lookup(1024)
        ledger.conversion("product")
        ledger.local()
    elif gate == "nexp":
        ledger.tables = [256, 256]
        ledger.nexp()
    elif gate == "reciprocal":
        ledger.tables = [1 << (6 + args[1].bit_length())]
        ledger.lookup(ledger.tables[0])
    elif gate == "softmax":
        inputs = len(args[0])
        active = [1] * inputs if args[1] is None else args[1]
        k = sum(active)
        reciprocal = 1 << (6 + k.bit_length())
        ledger.tables = [256, 256, reciprocal]
        ledger.local(2)
        for _ in range(k - 1):
            ledger.local()
            ledger.compare()
            ledger.conversion("product")
        for _ in range(k):
            ledger.local()
            ledger.compare()
            ledger.conversion("product")
            ledger.nexp()
            ledger.local()
        ledger.positive_shift(6)
        ledger.lookup(reciprocal)
        for _ in range(k):
            ledger.conversion("product")
            ledger.positive_shift(12)
    elif gate == "norm-index":
        ledger.index()
    elif gate == "norm":
        inputs = len(args[0])
        ledger.tables = [8192]
        ledger.local()
        if args[3] == "layer":
            ledger.local(inputs + 1)
            ledger.conversion("ars", shift=12)
        for _ in range(inputs):
            ledger.local(1 + int(args[3] == "layer"))
            ledger.conversion("product")
        ledger.local(2)
        ledger.index()
        ledger.lookup(8192)
        for _ in range(inputs):
            ledger.conversion("product")
            ledger.conversion("ars", shift=12)
            ledger.local()
            ledger.conversion("ars", shift=12)
            ledger.local(2)
    else:
        raise ValueError("unsupported composition gate")
    return ledger, inputs


def composition_accounting(execution, case):
    require(
        set(execution) == {"producer", "resources", "final_reserved_bytes"}
        and execution["producer"] == "composed-party-local-native",
        "composition execution fields",
    )
    integer(execution["final_reserved_bytes"], 0, 0)
    r = execution["resources"]
    require(
        type(r) is dict and set(r) == COMPOSITION_COUNTERS | {"plan_sha256"},
        "composition counter fields",
    )
    digest(r["plan_sha256"])
    for key in COMPOSITION_COUNTERS:
        integer(r[key])
    ledger, inputs = schedule(case)
    expected = {
        "instructions": ledger.instructions,
        "registers": inputs + ledger.instructions,
        "issuances": ledger.steps,
        "lanes": ledger.steps,
        "dcf_evaluations_per_party": ledger.evaluations,
        "dcf_keys_total": 2 * ledger.keys,
        "dcf_evaluations_total": 2 * ledger.evaluations,
        "triples_per_party": ledger.triples,
        "triples_total": 2 * ledger.triples,
        "private_material_bytes_total": ledger.private,
        "public_table_copy_bytes_total": ledger.public_table_copy_bytes_total,
        "cumulative_work": ledger.work,
        "retained_frame_overlap_bytes": ledger.overlap,
        "peer_frames_both_parties": ledger.frames,
        "peer_frame_bytes_both_parties": ledger.frame_bytes,
    }
    equal(
        {key: r[key] for key in expected},
        expected,
        "composition schedule/frame conservation: " + case["id"],
    )
    require(0 < r["plan_reservation_bytes"] <= 16 << 20, "plan reservation bound")
    require(
        r["plan_reservation_bytes"]
        <= r["peak_observed_reserved_bytes"]
        <= r["combined_reservation_bytes"]
        <= LIVE_LIMIT,
        "composition live envelope",
    )
    payload = sum(ledger.tables) * 8
    overhead = r["public_table_reservation_bytes"] - payload
    require(0 <= overhead <= 4096 * len(ledger.tables), "public table owner envelope")
    if not ledger.tables:
        require(overhead == 0, "unexpected table reservation")
    require(
        r["public_table_reservation_bytes"] < r["plan_reservation_bytes"], "table included in plan"
    )
    require(
        r["combined_reservation_bytes"]
        >= r["plan_reservation_bytes"] + ledger.overlap + 16 * r["registers"],
        "complete joint owner envelope",
    )
    return ledger


def public_plan_key(case):
    """Only public producer metadata, never private fixture input values."""
    gate, args = case["gate"], case["args"]
    if gate == "reciprocal":
        metadata = args[1]
    elif gate == "softmax":
        metadata = [len(args[0]), args[1] if args[1] is not None else [1] * len(args[0])]
    elif gate == "norm":
        k = len(args[0])
        metadata = [
            k,
            args[3],
            args[1] if args[1] is not None else [4096] * k,
            args[2] if args[2] is not None else [0] * k,
        ]
    else:
        metadata = None
    return canonical([gate, metadata])


def receipt(files, sources):
    obj = loads(files["complete-gates.receipt.json"])
    fields = {
        "schema",
        "coverage",
        "case_count",
        "gate_counts",
        "fixture_sha256",
        "source_lock_sha256",
        "implementation",
        "source_sha256",
        "source_input_fingerprints_before",
        "source_input_fingerprints_after",
        "source_inputs_unchanged",
        "output_sha256",
        "records",
        "final_reserved_bytes",
        "accounting",
        "limitations",
    }
    require(type(obj) is dict and set(obj) == fields, "native receipt fields")
    require(
        obj["schema"] == "pllm.fusefss.composition.native-export-receipt.v1"
        and obj["coverage"] == "all-frozen-cases",
        "complete receipt scope/schema",
    )
    integer(obj["case_count"], 275, 275)
    integer(obj["final_reserved_bytes"], 0, 0)
    require(obj["source_inputs_unchanged"] is True, "changed native inputs")
    equal(obj["accounting"], ACCOUNTING, "accounting scope")
    equal(obj["limitations"], LIMITATIONS, "qualification limits")
    expected = source_registry(sources, NATIVE + "gate_conformance.rs")
    expected.update(source_registry(sources, NATIVE + "composition/conformance.rs"))
    expected[PREFIX + "fixtures.json"] = FIXTURE_SHA
    required_native = {
        NATIVE + name
        for name in (
            "mod.rs",
            "composition.rs",
            "gate_conformance.rs",
            "lookup.rs",
            "conversions.rs",
            "predicates.rs",
            "composition/runtime.rs",
            "composition/gates.rs",
            "composition/norm.rs",
            "composition/source_tables.rs",
            "composition/source_table_values.rs",
            "composition/conformance.rs",
        )
    }
    required_native |= {
        "crates/pllm-garble/src/fusefss/typed_comparison.rs",
        PREFIX + "generate-native-tables.py",
        PREFIX + "source-tables-supplement.json",
    }
    require(required_native <= set(expected), "native implementation source coverage")
    equal(obj["source_sha256"], expected, "compiled source snapshot")
    equal(obj["source_input_fingerprints_before"], expected, "source fingerprints before")
    equal(obj["source_input_fingerprints_after"], expected, "source fingerprints after")
    require(
        obj["implementation"] == "pllm-native-composed-f2-test-sha256:" + sha(canonical(expected)),
        "implementation source binding",
    )
    require(
        obj["fixture_sha256"] == FIXTURE_SHA and obj["source_lock_sha256"] == LOCK_SHA,
        "frozen native input binding",
    )
    output = files["complete-gates.jsonl"]
    require(
        len(output) <= 8 << 20 and sha(output) == obj["output_sha256"], "native output digest/bound"
    )
    lines = output.splitlines(keepends=True)
    require(
        len(lines) == 276
        and all(line.endswith(b"\n") and len(line) <= 256 << 10 for line in lines),
        "complete JSONL framing/count",
    )
    equal(
        loads(lines[0]),
        {
            "schema": "pllm.fusefss.gate-set.v1/native-output",
            "fixture_sha256": FIXTURE_SHA,
            "execution": "protected-local-reconstruction",
            "implementation": obj["implementation"],
        },
        "complete output header",
    )
    return obj, output


def independent_check(output):
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / PREFIX / "check_outputs.py"), "-"],
        input=output,
        capture_output=True,
        timeout=60,
        cwd=ROOT,
        env=env,
        check=False,
    )
    require(
        result.returncode == 0 and len(result.stdout) <= 65536 and not result.stderr,
        "independent complete checker rejected",
    )
    summary = loads(result.stdout)
    equal(
        summary,
        {
            "status": "pass",
            "cases": 275,
            "fixture_sha256": FIXTURE_SHA,
            "selected_gates": None,
            "coverage": "all-frozen-cases",
            "scope": "reported finite outputs only; execution provenance not established",
        },
        "independent complete checker scope",
    )
    return summary


def main(evidence=HERE):
    files = bundle(Path(evidence).resolve())
    sources = archived_inputs(files)
    validation(files)
    source_dependencies(sources)
    supplement(sources)
    native, output = receipt(files, sources)
    fixture = loads(sources[PREFIX + "fixtures.json"])
    cases, rows = fixture["cases"], native["records"]
    require(type(rows) is list and len(rows) == len(cases) == 275, "complete native records")
    equal(native["gate_counts"], dict(Counter(c["gate"] for c in cases)), "gate coverage")
    primitives, compositions = [], []
    plans = {}
    public_plans = {}
    digest_contracts = {}
    table_owner_sizes = set()
    plan_bases = set()
    for case, row in zip(cases, rows):
        require(type(row) is dict and set(row) == {"id", "gate", "execution"}, "native row fields")
        require(row["id"] == case["id"] and row["gate"] == case["gate"], "native row order/binding")
        execution = row["execution"]
        require(type(execution) is dict, "execution fields")
        if case["gate"] in PRIMITIVES:
            primitive_accounting(execution, case)
            primitives.append(execution["resources"])
        else:
            ledger = composition_accounting(execution, case)
            r = execution["resources"]
            compositions.append(r)
            plan_bases.add(r["plan_reservation_bytes"] - r["public_table_reservation_bytes"])
            if ledger.tables:
                overhead = r["public_table_reservation_bytes"] - 8 * sum(ledger.tables)
                require(overhead % len(ledger.tables) == 0, "table owner size conservation")
                table_owner_sizes.add(overhead // len(ledger.tables))
            stable = {k: v for k, v in r.items() if k != "peak_observed_reserved_bytes"}
            old = plans.setdefault(r["plan_sha256"], stable)
            equal(stable, old, "same-plan resource consistency")
            key = public_plan_key(case)
            require(
                public_plans.setdefault(key, r["plan_sha256"]) == r["plan_sha256"],
                "same public contract changed plan identity",
            )
            require(
                digest_contracts.setdefault(r["plan_sha256"], key) == key,
                "different public contracts alias one plan identity",
            )
    require(len(primitives) == 66 and len(compositions) == 209, "distinct producer coverage")
    require(
        len(table_owner_sizes) == len(plan_bases) == 1,
        "native public owner reservation conservation",
    )
    primitive_totals = {key: sum(r[key] for r in primitives) for key in SUMS}
    primitive_totals.update({key: max(r[key] for r in primitives) for key in MAXES})
    primitive_totals["final_reservation_bytes"] = 0
    sum_keys = COMPOSITION_COUNTERS - {
        "plan_reservation_bytes",
        "public_table_reservation_bytes",
        "retained_frame_overlap_bytes",
        "combined_reservation_bytes",
        "peak_observed_reserved_bytes",
    }
    totals = {key: sum(r[key] for r in compositions) for key in sorted(sum_keys)}
    for key in COMPOSITION_COUNTERS - sum_keys:
        totals["max_" + key] = max(r[key] for r in compositions)
    independent_check(output)
    source_dependencies(sources)
    return {
        "status": "pass",
        "cases": 275,
        "primitive_cases": 66,
        "composition_cases": 209,
        "public_external_encoding_cases": 6,
        "primitive_accounting": primitive_totals,
        "composition_accounting": totals,
        "accounting_scope": "primitive counters cover 66 fixture cases; composition sums cover 209 fixture executions; public_table_copy_bytes_total is cumulative both-party lookup copies, separate from original table reservation; max_ fields are maxima of per-case planned envelopes or sampled reservations; observed samples can miss issuance scratch and are not high-water marks, simultaneous aggregate residency or full-model estimates",
        "final_reserved_bytes": 0,
        "scope": "archived finite complete gate set only; self-pins do not establish execution provenance, model/SDK completion or security qualification",
    }


ERRORS = (
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
)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--evidence", type=Path, default=HERE)
        args = parser.parse_args()
        print(json.dumps(main(args.evidence), sort_keys=True, separators=(",", ":")))
    except ERRORS as error:
        print(json.dumps({"status": "reject", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
