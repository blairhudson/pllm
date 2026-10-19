"""Independent replay mutations. Basic controls need no production artifacts.

With --evidence DIR, first require real archival replay to pass, then mutate
temporary copies and require rejection. Never creates synthetic native proof.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import tarfile
import unittest

import replay


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def temp_parent():
    approved = Path("/private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode")
    return str(approved) if approved.is_dir() else None


class BasicMalformedControls(unittest.TestCase):
    def test_scripts_parse_without_writes(self):
        for name in ("replay.py", "negative_controls.py"):
            ast.parse((Path(__file__).parent / name).read_text(), filename=name)

    def test_duplicate_noninteger_and_wide_tokens(self):
        for data in (
            b'{"x":1,"x":2}',
            b'{"x":1.0}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":123456789012345678901}',
            b'{"x":',
            b"[] trailing",
        ):
            with self.assertRaises(ValueError):
                replay.loads(data)

    def test_paths_and_pins(self):
        for name in ("", ".", "../x", "/x", "x/../y", "x//y", "./x", "x\\y", True):
            with self.assertRaises(ValueError):
                replay.relative(name)
        for pin in (
            {},
            {"bytes": True, "sha256": "0" * 64},
            {"bytes": 0, "sha256": "Z" * 64},
            {"bytes": 0, "sha256": "0" * 64, "extra": 1},
        ):
            with self.assertRaises(ValueError):
                replay.validate_pin(pin)

    def test_only_receipts_allow_bounded_finite_timing_numbers(self):
        self.assertEqual(
            replay.loads(b'{"elapsed_seconds":0.25}', allow_floats=True), {"elapsed_seconds": 0.25}
        )
        for data in (b'{"x":1e1000}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                replay.loads(data, allow_floats=True)

    def test_counter_shapes_types_and_peaks(self):
        baseline = {key: 0 for key in replay.COUNTERS}
        for field, value in (
            ("dcf_keys_per_party", True),
            ("final_reservation_bytes", 1),
            ("peak_logical_budget_bytes", (64 << 20) + 1),
            ("max_observed_live_budget_bytes", 1),
            ("peer_frame_bytes_both_parties", -1),
        ):
            stats = dict(baseline)
            stats[field] = value
            with self.assertRaises(ValueError):
                replay.stats(stats)
        for stats in (
            {},
            {**baseline, "extra": 0},
            {k: v for k, v in baseline.items() if k != "triples_per_party"},
        ):
            with self.assertRaises(ValueError):
                replay.stats(stats)

    def test_small_decompression_bound(self):
        with self.assertRaises(ValueError):
            replay.decode_gzip(gzip.compress(b"x" * 1025), 1024)

    def test_missing_manifest_and_duplicate_manifest(self):
        with tempfile.TemporaryDirectory(
            prefix="fusefss-replay-basic-", dir=temp_parent()
        ) as directory:
            path = Path(directory)
            with self.assertRaises((OSError, ValueError)):
                replay.main(path)
            (path / "manifest.json").write_bytes(b'{"schema":"a","schema":"b","files":{}}')
            with self.assertRaises(ValueError):
                replay.main(path)

    def test_duplicate_missing_and_link_archive_members(self):
        # Loader-only toy bytes, not a native evidence bundle or producer proof.
        data = b"toy source\n"
        for kind in ("duplicate", "missing", "link", "wrong digest"):
            with tempfile.TemporaryDirectory(
                prefix="fusefss-archive-basic-", dir=temp_parent()
            ) as directory:
                path = Path(directory)
                inventory = {
                    "files": {
                        "toy.txt": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    }
                }
                if kind == "wrong digest":
                    inventory["files"]["toy.txt"]["sha256"] = "0" * 64
                (path / "source-inventory.json").write_bytes(encoded(inventory))
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:") as archive:
                    if kind != "missing":
                        for _ in range(2 if kind == "duplicate" else 1):
                            member = tarfile.TarInfo("toy.txt")
                            if kind == "link":
                                member.type = tarfile.SYMTYPE
                                member.linkname = "../escape"
                                archive.addfile(member)
                            else:
                                member.size = len(data)
                                archive.addfile(member, io.BytesIO(data))
                (path / "inputs.tar.gz").write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
                with self.assertRaises(ValueError):
                    replay.archived_inputs(path)


def refresh_manifest(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    for name in manifest["files"]:
        data = (directory / name).read_bytes()
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path.write_bytes(encoded(manifest))


def mutate_json(directory, name, action):
    path = directory / name
    obj = json.loads(path.read_bytes())
    action(obj)
    path.write_bytes(encoded(obj))


def mutate_output(directory, suite, action):
    path = directory / (suite + ".jsonl.gz")
    lines = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    action(lines)
    decoded = b"".join(encoded(line) for line in lines)
    path.write_bytes(gzip.compress(decoded, mtime=0))
    mutate_json(
        directory,
        suite + ".receipt.json",
        lambda obj: obj.update(output_sha256=hashlib.sha256(decoded).hexdigest()),
    )


def mutate_snapshot(directory, name):
    source_bytes = replay.archived_inputs(directory)
    source_bytes[name] += b"\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for path, data in source_bytes.items():
            member = tarfile.TarInfo(path)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    (directory / "inputs.tar.gz").write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    mutate_json(
        directory,
        "source-inventory.json",
        lambda obj: obj["files"].update(
            {
                name: {
                    "bytes": len(source_bytes[name]),
                    "sha256": hashlib.sha256(source_bytes[name]).hexdigest(),
                }
            }
        ),
    )


def real_negative_controls(evidence):
    # A real baseline is mandatory. This driver never fabricates native records.
    baseline = replay.main(evidence)
    manifest = json.loads((evidence / "manifest.json").read_bytes())
    controls = []

    def native(label, suite, action):
        controls.append(
            (label, lambda directory: mutate_json(directory, suite + ".receipt.json", action))
        )

    controls.append(("missing artifact", lambda d: (d / "tables.jsonl.gz").unlink()))
    controls.append(
        (
            "duplicate manifest key",
            lambda d: (d / "manifest.json").write_bytes(b'{"schema":0,"schema":1,"files":{}}'),
        )
    )
    controls.append(
        (
            "missing validation step",
            lambda d: mutate_json(d, "run-receipt.json", lambda o: o["steps"].pop()),
        )
    )
    controls.append(
        (
            "duplicate validation step",
            lambda d: mutate_json(
                d,
                "run-receipt.json",
                lambda o: o["steps"].__setitem__(1, copy.deepcopy(o["steps"][0])),
            ),
        )
    )
    controls.append(
        (
            "nonzero validation rc",
            lambda d: mutate_json(
                d, "run-receipt.json", lambda o: o["steps"][0].update(returncode=1)
            ),
        )
    )
    controls.append(
        (
            "false-is-not-zero rc",
            lambda d: mutate_json(
                d, "run-receipt.json", lambda o: o["steps"][0].update(returncode=False)
            ),
        )
    )
    controls.append(
        (
            "changed validation source flag",
            lambda d: mutate_json(
                d, "run-receipt.json", lambda o: o.update(source_unchanged=False)
            ),
        )
    )
    controls.append(
        (
            "invalid elapsed field",
            lambda d: mutate_json(
                d, "run-receipt.json", lambda o: o["steps"][0].update(elapsed_seconds=False)
            ),
        )
    )
    controls.append(
        (
            "changed log digest",
            lambda d: mutate_json(
                d, "run-receipt.json", lambda o: o["steps"][0].update(log_sha256="0" * 64)
            ),
        )
    )
    controls.append(
        (
            "changed archived checker dependency",
            lambda d: mutate_snapshot(d, replay.PREFIX + "check_outputs.py"),
        )
    )
    controls.append(
        (
            "changed archived native source",
            lambda d: mutate_snapshot(d, "crates/pllm-garble/src/fusefss_backend/lookup.rs"),
        )
    )
    native("wrong implementation", "selected-gates", lambda o: o.update(implementation="unbound"))
    native(
        "changed source before",
        "selected-gates",
        lambda o: o["source_input_fingerprints_before"].update(
            {next(iter(o["source_input_fingerprints_before"])): "0" * 64}
        ),
    )
    native(
        "changed source after",
        "tables",
        lambda o: o["source_input_fingerprints_after"].update(
            {next(iter(o["source_input_fingerprints_after"])): "0" * 64}
        ),
    )
    native("wrong native output digest", "tables", lambda o: o.update(output_sha256="0" * 64))
    native("full-gate coverage claim", "tables", lambda o: o.update(coverage="complete-gates"))
    native(
        "wrong accounting scope",
        "tables",
        lambda o: o["accounting"].update(material="transport bytes"),
    )
    native("missing native record", "selected-gates", lambda o: o["records"].pop())
    native(
        "duplicate native record",
        "tables",
        lambda o: o["records"].__setitem__(1, copy.deepcopy(o["records"][0])),
    )
    native(
        "missing native counter",
        "selected-gates",
        lambda o: o["records"][0]["resources"].pop("triples_per_party"),
    )
    native(
        "unretired native reservation",
        "tables",
        lambda o: o["records"][0]["resources"].update(final_reservation_bytes=1),
    )
    native(
        "native peak over limit",
        "tables",
        lambda o: o["records"][0]["resources"].update(peak_logical_budget_bytes=(64 << 20) + 1),
    )
    native(
        "native total mismatch",
        "tables",
        lambda o: o["totals"].update(
            peer_frame_bytes_both_parties=o["totals"]["peer_frame_bytes_both_parties"] + 1
        ),
    )

    def conserved_counter(obj, key):
        obj["records"][0]["resources"][key] += 1
        obj["totals"][key] += 1

    for counter in (
        "cumulative_private_material_bytes_both_parties",
        "cumulative_public_table_copy_bytes_both_parties",
        "peer_frame_bytes_both_parties",
    ):
        native(
            "conserved but wrong " + counter,
            "tables",
            lambda o, key=counter: conserved_counter(o, key),
        )
    controls.append(
        (
            "wrong selected computed output",
            lambda d: mutate_output(
                d,
                "selected-gates",
                lambda rows: rows[1]["output"].update(word=rows[1]["output"]["word"] ^ 1),
            ),
        )
    )
    controls.append(
        (
            "wrong table computed output",
            lambda d: mutate_output(
                d,
                "tables",
                lambda rows: rows[1]["outputs"].__setitem__(0, rows[1]["outputs"][0] ^ 1),
            ),
        )
    )
    controls.append(
        (
            "wrong selected header scope",
            lambda d: mutate_output(
                d, "selected-gates", lambda rows: rows[0].update(coverage="all-frozen-cases")
            ),
        )
    )
    controls.append(
        (
            "missing selected computed row",
            lambda d: mutate_output(d, "selected-gates", lambda rows: rows.pop()),
        )
    )
    controls.append(
        (
            "reordered table output rows",
            lambda d: mutate_output(
                d, "tables", lambda rows: rows.__setitem__(slice(1, 3), [rows[2], rows[1]])
            ),
        )
    )

    checked = []
    for label, action in controls:
        with tempfile.TemporaryDirectory(
            prefix="fusefss-replay-negative-", dir=temp_parent()
        ) as temporary:
            directory = Path(temporary)
            for name in ["manifest.json", *manifest["files"]]:
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(evidence / name, target)
            action(directory)
            # Re-pin artifacts to test inner validation, not only outer hashes.
            if label not in ("missing artifact", "duplicate manifest key"):
                refresh_manifest(directory)
            try:
                replay.main(directory)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
                checked.append(label)
            else:
                raise AssertionError("mutation accepted: " + label)
    return {
        "status": "pass",
        "real_baseline": baseline["status"],
        "mutations_rejected": len(checked),
        "controls": checked,
        "scope": "temporary mutations of existing evidence; not new native proof",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="optional real evidence directory; requires real baseline replay first",
    )
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
                    "reason": "no --evidence supplied; production artifacts not assumed",
                },
                sort_keys=True,
            )
        )
    else:
        try:
            print(json.dumps(real_negative_controls(args.evidence.resolve()), sort_keys=True))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            AssertionError,
        ) as error:
            print(json.dumps({"status": "reject", "reason": str(error)}))
            raise SystemExit(1) from None
