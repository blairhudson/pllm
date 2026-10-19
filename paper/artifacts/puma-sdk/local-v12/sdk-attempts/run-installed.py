"""Installed-wheel validation; raw artifacts remain immutable per invocation."""
import base64
import csv
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import sysconfig
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent / "installed-only"
OUT.mkdir(exist_ok=False)
WHEEL = ROOT / "target/puma-checked-streams-build-v1/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl"
EXPECTED = "c599f0781f1c7fdc0697b67118c3f0b897ae504d2f771c0a506ce8af350de73f"
SITE = pathlib.Path(sysconfig.get_paths()["platlib"])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(phase):
    import pllm
    from pllm import _native
    assert SITE in pathlib.Path(pllm.__file__).resolve().parents
    assert SITE in pathlib.Path(_native.__file__).resolve().parents
    rows = []
    with zipfile.ZipFile(WHEEL) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            file = SITE / name
            actual = sha(file.read_bytes()) if file.is_file() else None
            expected = sha(archive.read(name))
            rows.append({"name": name, "wheel_sha256": expected, "installed_sha256": actual,
                         "equal": actual == expected, "installer_mutable_record": name.endswith(".dist-info/RECORD")})
    distribution = importlib.metadata.distribution("pllm-run")
    record = next(path for path in distribution.files if str(path).endswith(".dist-info/RECORD"))
    record_path = pathlib.Path(distribution.locate_file(record))
    record_rows = []
    for name, digest, size in csv.reader(record_path.read_text().splitlines()):
        path = pathlib.Path(distribution.locate_file(name))
        data = path.read_bytes() if path.is_file() else None
        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode() if data is not None else None
        record_rows.append({"name": name, "exists": data is not None,
            "hash_matches": None if not digest else digest == "sha256=" + str(actual),
            "size_matches": None if not size else data is not None and int(size) == len(data)})
    installed = {}
    for directory in (SITE / "pllm", record_path.parent):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                installed[str(path.relative_to(SITE))] = sha(path.read_bytes())
    report = {"phase": phase, "python": sys.executable, "sys_path": sys.path,
        "pllm": pllm.__file__, "native": _native.__file__, "wheel": str(WHEEL),
        "wheel_sha256": sha(WHEEL.read_bytes()), "wheel_files": rows,
        "record_entries": record_rows, "all_installed_files": installed,
        "payload_equal": all(row["equal"] or row["installer_mutable_record"] for row in rows),
        "record_valid": all(row["exists"] and row["hash_matches"] is not False and row["size_matches"] is not False for row in record_rows)}
    (OUT / f"installed-{phase}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    assert report["wheel_sha256"] == EXPECTED
    assert report["payload_equal"] and report["record_valid"], f"installed audit failed: {phase}"
    return report


before = audit("before")
focused = ["tests/test_replicated_policy_quality_failures.py", "tests/test_replicated_execution_policy.py",
    "tests/test_replicated_peer.py", "tests/test_replicated_peer_handshake.py", "tests/test_replicated_transport.py",
    "tests/test_component_families.py", "tests/test_replicated_worker.py::test_cancelled_native_operation_drains_before_owner_refund"]
results = []
try:
    for label, tests in (("native-original", ["tests/test_replicated_policy_sdk.py"]), ("focused-original", focused)):
        command = [sys.executable, "-I", "-m", "pytest", "-o", "pythonpath=" + str(SITE), *tests, "-vv", "--tb=long",
                   "--junitxml=" + str(OUT / f"{label}.xml")]
        (OUT / f"{label}-command.json").write_text(json.dumps(command, indent=2))
        start = time.monotonic()
        with (OUT / f"{label}.log").open("wb") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        result = {"label": label, "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - start}
        results.append(result)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
finally:
    after = audit("after")
    changed = {name: {"before": before["all_installed_files"].get(name), "after": after["all_installed_files"].get(name)}
        for name in before["all_installed_files"].keys() | after["all_installed_files"].keys()
        if before["all_installed_files"].get(name) != after["all_installed_files"].get(name)}
    (OUT / "installed-changes.json").write_text(json.dumps(changed, indent=2, sort_keys=True))
print(str(OUT), flush=True)
