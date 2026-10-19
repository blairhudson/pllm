"""Explicit generated-fixture audit; receipts contain public metadata only."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = (
    "crates/pllm-compiler/src/replicated_decoder/oracle_trace.rs",
    "tests/replicated_decoder_oracle.py",
    "tests/test_replicated_decoder_oracle.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def other_sources():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for directory in (ROOT / "crates", ROOT / "python")
        for suffix in ("*.rs", "*.py")
        for p in directory.rglob(suffix)
        if str(p.relative_to(ROOT)) not in SOURCES
    }


def expected_cases():
    result = set()
    for family in ("qwen2", "qwen3", "llama"):
        for mode in ("static", "checked_valid", "checked_invalid"):
            for lookup in (False, True):
                for selection in (False, True):
                    result.add((family, mode, lookup, selection, 1, 2, 2, 2, 3))
        for mode in ("checked_valid", "checked_invalid"):
            for lookup in (False, True):
                for selection in (False, True):
                    result.add((family, mode, lookup, selection, 2, 72, 131, 18, 5))
    result.add(("llama", "static", True, True, 1, 2, 130, 2, 3))
    result.add(("llama", "static", True, True, 1, 2, 2, 2, 151936))
    for mode in ("checked_valid", "checked_invalid"):
        result.add(("qwen3", mode, True, True, 2, 128, 133, 32, 5))
    return result


def case_key(case):
    return tuple(case[k] for k in ("family", "mode", "lookup_tiled", "selection_tiled",
                                    "layers", "hidden", "intermediate", "head_dim", "vocab"))


def main():
    before, others_before = hashes(), other_sources()
    env = dict(os.environ)
    env.pop("RUST_MIN_STACK", None)
    env.pop("RUST_TEST_THREADS", None)
    receipt = {"source_sha256": before, "RUST_MIN_STACK": None,
               "stack_rlimit": resource.getrlimit(resource.RLIMIT_STACK),
               "commands": [], "cases": [], "status": "running"}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = OUT / f"audit-{stamp}.log"
    receipt_path = OUT / f"receipt-{stamp}.json"
    commands = [
        ["cargo", "test", "-p", "pllm-compiler", "--lib", test, "--", "--nocapture", "--test-threads=1"]
        for test in (
            "native_multilayer_gqa_guard_tiles_and_tensor_tails",
            "native_generated_capture_has_complete_checkpoints_and_actual_truncation_events",
            "native_full_vocabulary_narrow_fixture_admission_and_audit",
        )
    ] + [
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "tests/test_replicated_decoder_oracle.py", "-q"],
        [str(ROOT / ".venv/bin/python"), "-m", "ruff", "check",
         "tests/replicated_decoder_oracle.py", "tests/test_replicated_decoder_oracle.py"],
    ]
    try:
        with log_path.open("w") as log:
            for command in commands:
                started = time.monotonic()
                log.write("COMMAND " + json.dumps(command) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if "ORACLE_CASE " in line:
                        receipt["cases"].append(json.loads(line.split("ORACLE_CASE ", 1)[1]))
                status = process.wait()
                receipt["commands"].append({"argv": command, "exit_code": status,
                                             "seconds": round(time.monotonic()-started, 3)})
                print(f"finished exit={status}: {' '.join(command)}", flush=True)
                assert status == 0, f"command failed; inspect {log_path}"
        keys = [case_key(case) for case in receipt["cases"]]
        assert len(keys) == len(set(keys)) == 64
        assert set(keys) == expected_cases()
        totals = {}
        for case in receipt["cases"]:
            for name, count in case["negative_controls"].items():
                totals[name] = totals.get(name, 0) + count
        assert totals["wrong_kv_group_routing"] == 416
        assert totals["guard_tile_sticky_mutation"] == 26
        assert totals["head_dense_middle_corruption"] == 1
        assert totals["head_rounding_tape_rotation"] == 1
        assert sum(totals.values()) == 824
        receipt.update(status="passed", native_program_cases=64,
                       negative_controls=totals, negative_control_total=sum(totals.values()))
    except BaseException as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt["source_sha256_after"] = hashes()
        others_after = other_sources()
        receipt["other_source_changes_observed"] = sorted(
            name for name in set(others_before) | set(others_after)
            if others_before.get(name) != others_after.get(name))
        receipt["source_unchanged_during_run"] = receipt["source_sha256_after"] == before
        if not receipt["source_unchanged_during_run"]:
            receipt["status"] = "failed"
        receipt["log"] = str(log_path.relative_to(ROOT))
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f"receipt: {receipt_path.relative_to(ROOT)}", flush=True)
    assert receipt["status"] == "passed"
    print(f"PASS: 64 native cases; 824 negative controls; source_sha256={json.dumps(before, sort_keys=True)}")


if __name__ == "__main__":
    main()
