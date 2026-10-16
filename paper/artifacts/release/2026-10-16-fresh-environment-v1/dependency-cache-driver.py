"""Rewrap existing extracted uv wheel-cache payloads for offline installation.

This does not build dependency source distributions or retrieve original wheel
ZIP bytes. Every payload is validated against its existing wheel RECORD before
rewrapping; the new ZIP and original extracted files receive separate hashes.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wheelhouse = output.parent / ".work" / ("rewrapped-wheels-" + uuid.uuid4().hex)
    wheelhouse.mkdir()
    shutil.copyfile(__file__, output / "driver.py")
    shutil.copyfile(args.pins, output / "pins.json")
    pins = json.loads(args.pins.read_text())["packages"]
    priorities = {tag: index for index, tag in enumerate(sys_tags())}
    rows = {}
    cache = Path.home() / ".cache/uv/wheels-v6/pypi"
    for name, version in pins.items():
        candidates = []
        for directory in (cache / name).glob(version + "-*"):
            if not directory.is_dir():
                continue
            filename = name.replace("-", "_") + "-" + directory.name + ".whl"
            tags = parse_wheel_filename(filename)[3]
            compatible = tags & priorities.keys()
            if compatible:
                candidates.append((min(priorities[tag] for tag in compatible), directory, filename))
        if not candidates:
            raise RuntimeError(f"Missing compatible extracted wheel cache: {name}=={version}")
        _, directory, filename = min(candidates)
        records = list(directory.glob("*.dist-info/RECORD"))
        assert len(records) == 1
        record = records[0]
        before = {}
        for relative, encoded, size in csv.reader(record.open()):
            path = directory / relative
            assert path.resolve().is_relative_to(directory.resolve()), relative
            value = sha(path)
            before[relative] = value
            if encoded:
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(value)).decode().rstrip("=")
                assert expected == encoded and int(size) == path.stat().st_size, (name, relative)
            else:
                assert path == record, (name, relative)
        wheel = wheelhouse / filename
        with zipfile.ZipFile(wheel, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for relative in sorted(before):
                zipped.write(directory / relative, relative)
        with zipfile.ZipFile(wheel) as zipped:
            payload = {relative: hashlib.sha256(zipped.read(relative)).hexdigest() for relative in zipped.namelist()}
        after = {relative: sha(directory / relative) for relative in before}
        assert payload == before == after
        rows[name] = {"version": version, "extracted_cache_path": str(directory),
                      "resolved_cache_path": str(directory.resolve()), "files_before": before,
                      "files_after": after, "files_digest": digest(before), "record_validated": True,
                      "wheel": str(wheel), "wheel_sha256": sha(wheel), "wheel_bytes": wheel.stat().st_size}
    write(output / "receipt.json", {"schema": "pllm.offline_dependency_rewrap.v1", "status": "passed",
          "argv": sys.argv, "python": sys.executable, "python_sha256": sha(sys.executable),
          "wheelhouse": str(wheelhouse), "packages": rows, "packages_digest": digest(rows),
          "scope": "Existing extracted prebuilt wheel payloads, RECORD-verified, byte-preserving ZIP rewrap; not original downloaded wheel ZIP identity, no dependency source build, no network"})
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "manifest.json", {"files": files, "digest": digest(files)})
    print(json.dumps({"status": "passed", "rewrapped_packages": len(rows), "wheelhouse": str(wheelhouse)}))


if __name__ == "__main__":
    main()
