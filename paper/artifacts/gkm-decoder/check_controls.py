"""Replay bounded corruption controls on temporary copies of the v3 archive."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reindex(archive, relative):
    path = archive / "index.json"
    index = json.loads(path.read_bytes())
    row = next(item for item in index["files"] if item["path"] == relative)
    row.update(bytes=(archive / relative).stat().st_size, sha256=digest(archive / relative))
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update the control receipt")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("gkm_archive_verifier", BASE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    module.verify(module.FINAL_ARCHIVE)
    controls = [
        ("payload_corruption", "archive size mismatch"),
        ("extra_file", "archive manifest incomplete"),
        ("rehashed_source_substitution", "frozen source differs from built source"),
        ("rehashed_test_failure", "final installed-wheel checks incomplete"),
        ("rehashed_native_source_substitution", "native checks differ from built source"),
        ("rehashed_test_snapshot_substitution", "test snapshot differs from executed input"),
    ]
    checks = []
    # Resolve only our trusted scratch root; archive validation keeps no-follow reads.
    with tempfile.TemporaryDirectory(prefix="pllm-gkm-archive-controls-") as temporary:
        scratch = Path(temporary).resolve()
        for name, expected in controls:
            archive = scratch / name
            shutil.copytree(module.FINAL_ARCHIVE, archive)
            relative = None
            if name == "payload_corruption":
                with (archive / "build/receipt.json").open("ab") as handle:
                    handle.write(b"\n")
            elif name == "extra_file":
                (archive / "extra.txt").write_text("unlisted\n")
            elif name == "rehashed_test_failure":
                relative = "build/installed-tests.xml"
                root = ET.fromstring((archive / relative).read_bytes())
                case = next(root.iter("testcase"))
                ET.SubElement(case, "failure", message="injected control")
                (archive / relative).write_bytes(ET.tostring(root))
            elif name == "rehashed_native_source_substitution":
                relative = "native/sources-before.json"
                path = archive / relative
                document = json.loads(path.read_bytes())
                document["files"]["crates/pllm-compiler/src/delegated_decoder.rs"] = "0" * 64
                path.write_text(json.dumps(document))
            else:
                relative = (
                    "sources/python/pllm/runtime/delegated_decoder.py"
                    if name == "rehashed_source_substitution"
                    else "tests/tests/test_delegated_decoder.py"
                )
                with (archive / relative).open("ab") as handle:
                    handle.write(b"\n")
            if relative is not None:
                reindex(archive, relative)
            try:
                module.verify(archive)
            except ValueError as error:
                if str(error) != expected:
                    raise AssertionError(f"{name}: unexpected rejection {error}") from error
            else:
                raise AssertionError(f"{name}: mutated archive accepted")
            checks.append({"control": name, "reason": expected, "status": "rejected"})
    result = {
        "archive_index_sha256": digest(module.FINAL_ARCHIVE / "index.json"),
        "checks": checks,
        "scope": "read-only archive gate corruption controls; no model execution",
        "verifier_sha256": digest(BASE / "verify.py"),
        "controls_runner_sha256": digest(Path(__file__)),
    }
    receipt = BASE / "verification-controls.json"
    if args.write:
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif json.loads(receipt.read_bytes()) != result:
        raise AssertionError("control receipt differs from replay")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
