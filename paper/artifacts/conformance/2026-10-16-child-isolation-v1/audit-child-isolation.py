"""Focused subprocess/framing checks only: no providers, model/operator execution.

Run with an existing installed wheel interpreter and a matching wheel archive.
Records import identities before/after three tiny captured Python processes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "paper/artifacts/conformance"
POLICY = runpy.run_path(str(ARCHIVE / "python-child-isolation.py"))
FRAMING = runpy.run_path(str(ARCHIVE / "equalization-wire-accounting.py"))
OUTPUT = None


def save(path, value):
    POLICY["save"](path, value)


def raw_command(argv, environment, name, *, cwd=None):
    process = POLICY["_POPEN"](argv, env=environment, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(timeout=30)
    (OUTPUT / f"{name}.stdout.txt").write_text(stdout)
    (OUTPUT / f"{name}.stderr.txt").write_text(stderr)
    save(OUTPUT / f"{name}.command.json", {"argv": argv, "exit_code": process.returncode,
                                         "scope": "non-provider subprocess probe"})
    return process.returncode, stdout, stderr


class FocusedChecks(unittest.TestCase):
    def test_01_unisolated_control_imports_pythonpath_canary(self):
        environment = POLICY["clean_environment"](os.environ)
        environment["PYTHONPATH"] = str(OUTPUT / "shadow")
        code, _, stderr = raw_command([sys.executable, "-c", "import pllm"], environment, "unisolated-control")
        self.assertNotEqual(code, 0)
        self.assertIn("PYTHONPATH_CANARY_EXECUTED", stderr)

    def test_02_actual_role_spawn_boundary_enforces_isolation(self):
        from pllm.runtime.servers import _spawn_role_process

        environment = {**os.environ, "PYTHONPATH": str(OUTPUT / "shadow"), "PYTHONHOME": "/unusable-home",
                       "PYTHONUSERBASE": str(OUTPUT / "shadow"), "PYTHONSTARTUP": "/unusable-startup"}
        snippet = "import os,sys,json,pllm; assert sys.flags.isolated; assert not any(k.startswith('PYTHON') for k in os.environ); print(json.dumps({'pllm':pllm.__file__,'pid':os.getpid()}))"
        process = _spawn_role_process([sys.executable, "-c", snippet], environment=environment,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=30)
        (OUTPUT / "role-boundary.stdout.txt").write_bytes(stdout)
        (OUTPUT / "role-boundary.stderr.txt").write_bytes(stderr)
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertNotIn("shadow", json.loads(stdout)["pllm"])

    def test_03_nested_launch_keeps_policy_after_environment_replacement(self):
        inner = "import os,sys,pllm; assert sys.flags.isolated; assert 'PYTHONPATH' not in os.environ; print('NESTED_ISOLATED')"
        outer = f"import os,subprocess,sys; p=subprocess.run([sys.executable,'-c',{inner!r}],env={{'PATH':os.environ['PATH'],'PYTHONPATH':{str(OUTPUT / 'shadow')!r}}},capture_output=True,text=True,check=True); print(p.stdout,end='')"
        result = subprocess.run([sys.executable, "-c", outer], cwd=OUTPUT / "shadow",
                                capture_output=True, text=True, timeout=30, check=True)
        (OUTPUT / "nested.stdout.txt").write_text(result.stdout)
        self.assertEqual(result.stdout, "NESTED_ISOLATED\n")

    def test_04_wrong_native_identity_rejects_before_entry(self):
        negative = OUTPUT / "rejected-native"
        negative.mkdir()
        expected = json.loads((OUTPUT / "child-identities/expected.json").read_text())
        expected["native_sha256"] = "0" * 64
        save(negative / "expected.json", expected)
        helper = OUTPUT / "child-identities/python-child-isolation.py"
        marker = negative / "entry-ran.txt"
        envelope = {"directory": str(negative), "expected_path": str(negative / "expected.json"),
                    "helper": str(helper), "key": "rejected", "mode": "c", "arguments": [],
                    "entry": f"from pathlib import Path; Path({str(marker)!r}).write_text('incorrect')"}
        snippet = f"import runpy; runpy.run_path({str(helper)!r})['run_child']({envelope!r})"
        code, _, stderr = raw_command([sys.executable, "-I", "-c", snippet],
                                      POLICY["clean_environment"](os.environ), "rejected-native")
        self.assertNotEqual(code, 0)
        self.assertIn("child installed identity mismatch before entry point", stderr)
        self.assertFalse(marker.exists())
        record = json.loads((negative / "rejected.start.json").read_text())
        self.assertEqual(record["errors"], ["loaded native binary differs"])

    def test_05_missing_exit_receipt_keeps_identity_gate_failed(self):
        directory = OUTPUT / "missing-receipt-control"
        directory.mkdir()
        positive = OUTPUT / "child-identities"
        launch = sorted(positive.glob("*.launch.json"))[0]
        key = launch.name.removesuffix(".launch.json")
        for name in ("expected.json", f"{key}.launch.json", f"{key}.spawn.json", f"{key}.start.json"):
            (directory / name).write_bytes((positive / name).read_bytes())
        summary = POLICY["summarize"](directory)
        save(directory / "summary.json", summary)
        self.assertEqual(summary["status"], "failed")
        self.assertIn(f"missing {key}.end receipt", summary["failures"])

    def test_06_ambiguous_shell_launch_rejected_before_spawn(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous Python subprocess"):
            subprocess.Popen([sys.executable, "-c", "raise AssertionError('must not execute')"], shell=True)

    def test_07_accounting_ignores_only_timing_value_width(self):
        import msgpack
        import numpy as np
        from pllm.runtime.stage_protocol import PreparedStageBatchResponse

        def packet(clock=65535, *, columns=32, bits=16, identifier="0" * 32):
            return PreparedStageBatchResponse(identifier, np.zeros((33, columns), dtype=np.uint32), bits, clock).pack()

        first, second = packet(), packet(65536)
        left = FRAMING["account_response"](first, stage_id="public-fixed-stage")
        right = FRAMING["account_response"](second, stage_id="public-fixed-stage")
        FRAMING["assert_equal_static_contract"]([left], [right])
        self.assertEqual(right["full_frame_bytes"] - left["full_frame_bytes"], 2)
        self.assertEqual(left["static_contract"]["arithmetic_payload_bytes"], 33 * 32 * 2)
        controls = []
        for label, value in (("geometry", packet(columns=33)), ("wire-width", packet(bits=24)),
                             ("non-timing-metadata-length", packet(identifier="0" * 33))):
            different = FRAMING["account_response"](value, stage_id="public-fixed-stage")
            with self.assertRaises(AssertionError):
                FRAMING["assert_equal_static_contract"]([left], [different])
            controls.append({"case": label, "outcome": "rejected", "accounting": different})
        truncated = msgpack.unpackb(first, raw=False)
        truncated["data"] = truncated["data"][:-1]
        with self.assertRaises(ValueError):
            FRAMING["account_response"](msgpack.packb(truncated, use_bin_type=True), stage_id="public-fixed-stage")
        (OUTPUT / "framing-65535.bin").write_bytes(first)
        (OUTPUT / "framing-65536.bin").write_bytes(second)
        save(OUTPUT / "framing-controls.json", {"left": left, "right": right, "negative_controls": controls,
                                                "truncated_arithmetic_payload": "rejected",
                                                "historical_failure": "unresolved; original frames absent"})


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in ("audit-child-isolation.py", "equalization-wire-accounting.py", "python-child-isolation.py"):
        (OUTPUT / name).write_bytes((ARCHIVE / name).read_bytes())
    (OUTPUT / "prior-archive-index.json").write_bytes((ARCHIVE / "index.json").read_bytes())
    shadow = OUTPUT / "shadow/pllm"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('PYTHONPATH_CANARY_EXECUTED')\n")
    import pllm
    import pllm._native

    package = Path(pllm.__file__).resolve().parent
    with zipfile.ZipFile(args.wheel) as archive:
        files = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                 if name.startswith("pllm/") and not name.endswith("/")}
    if any(POLICY["sha"](package.parent / name) != digest for name, digest in files.items()):
        raise RuntimeError("installed package differs from supplied wheel")
    save(OUTPUT / "provenance.json", {
        "python": sys.executable, "package_path": str(package), "wheel_sha256": POLICY["sha"](args.wheel),
        "native_sha256": POLICY["sha"](pllm._native.__file__), "wheel_package_files": files,
        "wheel": str(args.wheel.absolute()), "distribution_version": metadata.version("pllm.run"),
    })
    POLICY["activate"](OUTPUT / "child-identities", OUTPUT / "provenance.json")
    with (OUTPUT / "focused-tests.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FocusedChecks))
    identities = POLICY["finish"](OUTPUT / "child-identities")
    passed = result.wasSuccessful() and identities["status"] == "passed" and identities["captured_python_processes"] == 3
    receipt = {"schema": "pllm.paper_child_isolation_checks.v1", "status": "passed" if passed else "failed",
               "argv": [sys.executable, "-I", *sys.argv], "tests_run": result.testsRun,
               "failed": [test.id() for test, _ in result.failures], "errors": [test.id() for test, _ in result.errors],
               "captured_python_processes": identities["captured_python_processes"],
               "provider_model_operator_executions": 0, "scope": "subprocess provenance and fixed-message accounting only",
               "historical_provider_child_identity": "unknown; original receipts absent",
               "artifacts": {p.relative_to(OUTPUT).as_posix(): POLICY["sha"](p)
                             for p in sorted(OUTPUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}}
    save(OUTPUT / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "tests_run", "failed", "errors", "captured_python_processes")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
