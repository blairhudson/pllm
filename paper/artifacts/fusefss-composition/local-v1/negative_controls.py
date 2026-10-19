"""Malformed archive controls and optional repinned mutations of real evidence.

Without --evidence, no native evidence is assumed or fabricated. With evidence,
require real replay first; mutations use temporary copies, never the archive.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import replay


def encoded(obj):
    return replay.canonical(obj) + b"\n"


def temp_parent():
    approved = Path("/private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode")
    return str(approved) if approved.is_dir() else None


def packed(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), mtime=0)


def write_bundle(directory, files):
    data = packed(files)
    (directory / "evidence.tar.gz").write_bytes(data)
    (directory / "artifact-index.json").write_bytes(
        encoded(
            {
                "schema": "pllm.fusefss.composition-evidence-files.v1",
                "archive": replay.pin(data),
                "files": {name: replay.pin(value) for name, value in files.items()},
            }
        )
    )


def mutate_json(files, name, action):
    obj = replay.loads(files[name], allow_floats=name == "run-receipt.json")
    action(obj)
    files[name] = encoded(obj)


def mutate_output(files, action):
    rows = [replay.loads(line) for line in files["complete-gates.jsonl"].splitlines()]
    action(rows)
    output = b"".join(encoded(row) for row in rows)
    files["complete-gates.jsonl"] = output
    mutate_json(
        files,
        "complete-gates.receipt.json",
        lambda obj: obj.update(output_sha256=replay.sha(output)),
    )


def mutate_source(files, name):
    sources = replay.archived_inputs(files)
    sources[name] += b"\n"
    files["inputs.tar.gz"] = packed(sources)
    mutate_json(
        files,
        "source-inventory.json",
        lambda obj: obj["files"].update({name: replay.pin(sources[name])}),
    )


class BasicMalformedControls(unittest.TestCase):
    def test_scripts_parse_without_bytecode(self):
        for name in ("replay.py", "negative_controls.py"):
            ast.parse((Path(__file__).parent / name).read_text(), filename=name)

    def test_duplicate_noninteger_and_wide_json(self):
        for data in (
            b'{"x":1,"x":2}',
            b'{"x":1.0}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":123456789012345678901}',
            b'{"x":',
            b"[] trailing",
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                replay.loads(data)
        self.assertEqual(
            replay.loads(b'{"elapsed_seconds":0.25}', allow_floats=True), {"elapsed_seconds": 0.25}
        )
        for data in (b'{"x":1e1000}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                replay.loads(data, allow_floats=True)

    def test_paths_and_pin_types(self):
        for name in ("", ".", "../x", "/x", "x/../y", "x//y", "./x", "x\\y", "x\0y", True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replay.relative(name)
        for pin in (
            {},
            {"bytes": True, "sha256": "0" * 64},
            {"bytes": 0, "sha256": "Z" * 64},
            {"bytes": 0, "sha256": "0" * 64, "extra": 1},
        ):
            with self.assertRaises(ValueError):
                replay.validate_pin(pin)

    def test_small_decompression_bound(self):
        with self.assertRaises(ValueError):
            replay.decode_gzip(gzip.compress(b"x" * 1025), 1024)

    def test_primitive_counter_shapes_and_bounds(self):
        baseline = dict.fromkeys(replay.COUNTERS, 0)
        replay.primitive_stats(baseline)
        for key, value in (
            ("dcf_keys_per_party", True),
            ("final_reservation_bytes", 1),
            ("peak_logical_budget_bytes", replay.LIVE_LIMIT + 1),
            ("max_observed_live_budget_bytes", 1),
            ("peer_frame_bytes_both_parties", -1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.primitive_stats({**baseline, key: value})
        for stats in (
            {},
            {**baseline, "extra": 0},
            {k: v for k, v in baseline.items() if k != "triples_per_party"},
        ):
            with self.assertRaises(ValueError):
                replay.primitive_stats(stats)

    def test_archive_exact_coverage_and_pin(self):
        data = b"toy loader bytes, not native evidence\n"
        pins = {"toy.txt": replay.pin(data)}
        self.assertEqual(replay.archive_files(packed({"toy.txt": data}), pins), {"toy.txt": data})
        for files in ({}, {"toy.txt": data, "extra.txt": b"x"}, {"toy.txt": data + b"x"}):
            with self.assertRaises(ValueError):
                replay.archive_files(packed(files), pins)

    def test_unsafe_duplicate_and_nonregular_tar_members(self):
        for kind in ("duplicate", "symlink", "hardlink", "directory", "traversal", "absolute"):
            with self.subTest(kind=kind):
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:") as archive:
                    for _ in range(2 if kind == "duplicate" else 1):
                        name = {"traversal": "../toy.txt", "absolute": "/toy.txt"}.get(
                            kind, "toy.txt"
                        )
                        member = tarfile.TarInfo(name)
                        if kind in ("symlink", "hardlink", "directory"):
                            member.type = {
                                "symlink": tarfile.SYMTYPE,
                                "hardlink": tarfile.LNKTYPE,
                                "directory": tarfile.DIRTYPE,
                            }[kind]
                            member.linkname = "../escape"
                            archive.addfile(member)
                        else:
                            member.size = 1
                            archive.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    replay.archive_files(
                        gzip.compress(buffer.getvalue()), {"toy.txt": replay.pin(b"x")}
                    )

    def test_corrupt_gzip_and_trailing_tar_data(self):
        valid = packed({"toy.txt": b"x"})
        pins = {"toy.txt": replay.pin(b"x")}
        for bad in (
            b"not gzip",
            valid[:-8],
            gzip.compress(gzip.decompress(valid) + b"nonzero trailing data"),
        ):
            with self.subTest(bad=bad[:8]), self.assertRaises(replay.ERRORS):
                replay.archive_files(bad, pins)

    def test_missing_duplicate_and_symlink_index(self):
        with tempfile.TemporaryDirectory(
            prefix="fusefss-composition-basic-", dir=temp_parent()
        ) as directory:
            path = Path(directory)
            with self.assertRaises(replay.ERRORS):
                replay.main(path)
            (path / "artifact-index.json").write_bytes(b'{"schema":0,"schema":1,"files":{}}')
            with self.assertRaises(ValueError):
                replay.main(path)
            (path / "artifact-index.json").unlink()
            (path / "artifact-index.json").symlink_to(Path(__file__).resolve())
            with self.assertRaises(ValueError):
                replay.main(path)

    def test_repinned_outer_archive_still_checks_inner_pins(self):
        with tempfile.TemporaryDirectory(
            prefix="fusefss-composition-pins-", dir=temp_parent()
        ) as directory:
            path = Path(directory)
            files = {
                name: b"x"
                for name in (
                    "inputs.tar.gz",
                    "source-inventory.json",
                    "run-receipt.json",
                    "complete-gates.jsonl",
                    "complete-gates.receipt.json",
                )
            }
            write_bundle(path, files)
            self.assertEqual(replay.bundle(path), files)
            index = replay.loads((path / "artifact-index.json").read_bytes())
            files["run-receipt.json"] = b"changed"
            data = packed(files)
            (path / "evidence.tar.gz").write_bytes(data)
            index["archive"] = replay.pin(data)
            (path / "artifact-index.json").write_bytes(encoded(index))
            with self.assertRaises(ValueError):
                replay.bundle(path)

    def test_public_schedule_integer_controls(self):
        ledger, inputs = replay.schedule({"gate": "reciprocal", "args": [192, 3]})
        self.assertEqual(
            (inputs, ledger.instructions, ledger.steps, ledger.keys, ledger.evaluations),
            (1, 1, 1, 1, 257),
        )
        self.assertEqual((ledger.frames, ledger.frame_bytes, ledger.private), (2, 176, 2696))
        ledger, _ = replay.schedule({"gate": "norm-index", "args": [1]})
        self.assertEqual((ledger.instructions, ledger.steps), (391, 280))
        for key in ("silu", "nexp"):
            ledger, _ = replay.schedule({"gate": key, "args": [-1 if key == "silu" else 257]})
            self.assertGreater(ledger.steps, 0)
            self.assertGreater(ledger.keys, 0)

    def test_exact_rsqr_k2_supplement_numeric_check(self):
        name = replay.PREFIX + "source-tables-supplement.json"
        data = replay.safe_read(replay.ROOT, name)
        replay.supplement({name: data})
        obj = replay.loads(data)
        obj["values"][0] += 1
        obj["sha256_u64le"] = replay.sha(b"".join(v.to_bytes(8, "little") for v in obj["values"]))
        with self.assertRaises(ValueError):
            replay.supplement({name: encoded(obj)})

    def test_public_table_copy_controls_distinct_from_original_reservations(self):
        cases = [
            ({"gate": "silu", "args": [-1]}, 49_552),
            ({"gate": "nexp", "args": [257]}, 24_704),
            ({"gate": "reciprocal", "args": [64, 1]}, 6_208),
            ({"gate": "reciprocal", "args": [64, 768]}, 3_174_352),
            ({"gate": "reciprocal", "args": [64, 1024]}, 6_348_752),
            ({"gate": "softmax", "args": [[0, 0, 0], None]}, 86_464),
            ({"gate": "softmax", "args": [[0, 99999, -4096], [1, 0, 1]]}, 61_760),
            ({"gate": "norm", "args": [[0] * 4, None, None, "layer"]}, 396_752),
            ({"gate": "norm", "args": [[0] * 768, None, None, "rms"]}, 396_752),
            ({"gate": "norm-index", "args": [1]}, 0),
        ]
        for case, expected in cases:
            with self.subTest(gate=case["gate"], expected=expected):
                ledger, _ = replay.schedule(case)
                self.assertEqual(ledger.public_table_copy_bytes_total, expected)


def real_negative_controls(evidence):
    baseline = replay.main(evidence)
    original = replay.bundle(evidence)
    controls = []
    controls.append(
        ("missing repinned required artifact", lambda files: files.pop("complete-gates.jsonl"))
    )
    controls.append(
        (
            "duplicate receipt JSON key",
            lambda files: files.__setitem__(
                "complete-gates.receipt.json", b'{"schema":0,"schema":1}'
            ),
        )
    )

    def native(label, action):
        controls.append(
            (label, lambda files: mutate_json(files, "complete-gates.receipt.json", action))
        )

    def validation(label, action):
        controls.append((label, lambda files: mutate_json(files, "run-receipt.json", action)))

    validation("missing validation step", lambda obj: obj["steps"].pop())
    validation(
        "duplicate validation step",
        lambda obj: obj["steps"].__setitem__(1, copy.deepcopy(obj["steps"][0])),
    )
    validation("nonzero validation rc", lambda obj: obj["steps"][0].update(returncode=1))
    validation("false rc is not zero", lambda obj: obj["steps"][0].update(returncode=False))
    validation("invalid elapsed field", lambda obj: obj["steps"][0].update(elapsed_seconds=False))
    validation("changed validation input flag", lambda obj: obj.update(source_unchanged=False))
    validation("wrong log digest", lambda obj: obj["steps"][0].update(log_sha256="0" * 64))
    validation(
        "partial independent checker command",
        lambda obj: obj["steps"][2]["command"].extend(["--gates", "silu"]),
    )
    native("selected-only receipt", lambda obj: obj.update(coverage="selected-gates-only"))
    native("missing native row", lambda obj: obj["records"].pop())
    native(
        "duplicate native row",
        lambda obj: obj["records"].__setitem__(1, copy.deepcopy(obj["records"][0])),
    )
    native("wrong native implementation", lambda obj: obj.update(implementation="unbound"))
    native("wrong native output digest", lambda obj: obj.update(output_sha256="0" * 64))
    native("unretired final reservation", lambda obj: obj.update(final_reserved_bytes=1))
    native("false final is not integer zero", lambda obj: obj.update(final_reserved_bytes=False))
    native(
        "changed source before",
        lambda obj: obj["source_input_fingerprints_before"].update(
            {next(iter(obj["source_sha256"])): "0" * 64}
        ),
    )
    native(
        "changed source after",
        lambda obj: obj["source_input_fingerprints_after"].update(
            {next(iter(obj["source_sha256"])): "0" * 64}
        ),
    )
    native(
        "changed accounting scope",
        lambda obj: obj["accounting"].update(frames="authenticated transport"),
    )
    native("missing qualification limitation", lambda obj: obj["limitations"].pop())
    native("wrong gate counts", lambda obj: obj["gate_counts"].update(silu=23))
    native(
        "composition falsely labelled primitive",
        lambda obj: obj["records"][0]["execution"].update(producer="shared-selected-primitive"),
    )
    native(
        "missing composition counter",
        lambda obj: obj["records"][0]["execution"]["resources"].pop("triples_total"),
    )
    native(
        "unretired per-case reservation",
        lambda obj: obj["records"][0]["execution"].update(final_reserved_bytes=1),
    )
    native(
        "live peak above envelope",
        lambda obj: obj["records"][0]["execution"]["resources"].update(
            peak_observed_reserved_bytes=replay.LIVE_LIMIT + 1
        ),
    )
    for key in (
        "issuances",
        "lanes",
        "dcf_keys_total",
        "dcf_evaluations_total",
        "private_material_bytes_total",
        "public_table_copy_bytes_total",
        "peer_frames_both_parties",
        "peer_frame_bytes_both_parties",
        "retained_frame_overlap_bytes",
    ):
        native(
            "repinned wrong composition " + key,
            lambda obj, key=key: obj["records"][0]["execution"]["resources"].update(
                {key: obj["records"][0]["execution"]["resources"][key] + 1}
            ),
        )

    def conserved_triples(obj):
        resources = obj["records"][0]["execution"]["resources"]
        resources["triples_per_party"] += 1
        resources["triples_total"] += 2

    native("conserved but wrong composition triples", conserved_triples)

    def conserved_public_copies(obj):
        # Preserve exported aggregate exactly, but falsify two per-case counts.
        # Neither original table reservation nor private payload is modified.
        for index, delta in ((0, 8), (1, -8)):
            obj["records"][index]["execution"]["resources"]["public_table_copy_bytes_total"] += (
                delta
            )

    native("conserved but wrong per-case public table copies", conserved_public_copies)
    native(
        "missing required public table copy counter",
        lambda obj: obj["records"][0]["execution"]["resources"].pop(
            "public_table_copy_bytes_total"
        ),
    )

    def wrong_primitive(obj):
        row = next(row for row in obj["records"] if row["gate"] == "compare")
        row["execution"]["resources"]["peer_frame_bytes_both_parties"] += 1

    native("repinned wrong primitive frames", wrong_primitive)
    controls.extend(
        [
            (
                "changed locked shared oracle",
                lambda files: mutate_source(
                    files, "paper/reproduction-contracts/shared_profile_oracle.py"
                ),
            ),
            (
                "changed archived native source",
                lambda files: mutate_source(files, replay.NATIVE + "composition/runtime.rs"),
            ),
            (
                "changed archived checker",
                lambda files: mutate_source(files, replay.PREFIX + "check_outputs.py"),
            ),
            (
                "wrong computed output with updated digest",
                lambda files: mutate_output(
                    files,
                    lambda rows: rows[1]["output"].update(output=rows[1]["output"]["output"] ^ 1),
                ),
            ),
            (
                "selected output header",
                lambda files: mutate_output(
                    files,
                    lambda rows: rows[0].update(
                        schema="pllm.fusefss.gate-set.v2/selected-native-output"
                    ),
                ),
            ),
            ("missing computed row", lambda files: mutate_output(files, lambda rows: rows.pop())),
            (
                "reordered computed rows",
                lambda files: mutate_output(
                    files, lambda rows: rows.__setitem__(slice(1, 3), [rows[2], rows[1]])
                ),
            ),
        ]
    )
    checked = []
    for label, action in controls:
        files = dict(original)
        action(files)
        with tempfile.TemporaryDirectory(
            prefix="fusefss-composition-negative-", dir=temp_parent()
        ) as temporary:
            directory = Path(temporary)
            # Repin all changed archive members AND compressed archive. Rejection
            # must come from semantic validation, not stale outer byte hashes.
            write_bundle(directory, files)
            try:
                replay.main(directory)
            except replay.ERRORS:
                checked.append(label)
            else:
                raise AssertionError("mutation accepted: " + label)
    return {
        "status": "pass",
        "real_baseline": baseline["status"],
        "mutations_rejected": len(checked),
        "controls": checked,
        "scope": "temporary repinned mutations of supplied evidence; not new native proof or cryptographic authenticity",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BasicMalformedControls)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.evidence is None:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "basic_controls": result.testsRun,
                    "real_replay": "not-run",
                    "reason": "no --evidence supplied; no native artifacts assumed",
                },
                sort_keys=True,
            )
        )
    else:
        try:
            print(json.dumps(real_negative_controls(args.evidence.resolve()), sort_keys=True))
        except (*replay.ERRORS, AssertionError) as error:
            print(json.dumps({"status": "reject", "reason": str(error)}))
            raise SystemExit(1) from None
