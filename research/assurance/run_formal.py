#!/usr/bin/env python3
"""Run finite SMT-LIB specifications through system libz3."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import sys


class SolverUnavailable(RuntimeError):
    """Raised when system Z3 shared library cannot be loaded."""


def run_formal(formal_dir: Path) -> dict[str, object]:
    name = ctypes.util.find_library("z3")
    if not name:
        raise SolverUnavailable(
            "Native libz3 unavailable; install Z3 to run finite specification checks. "
            "No pass inferred."
        )

    lib = ctypes.CDLL(name)
    lib.Z3_get_full_version.restype = ctypes.c_char_p
    lib.Z3_mk_config.restype = ctypes.c_void_p
    lib.Z3_mk_context.argtypes = [ctypes.c_void_p]
    lib.Z3_mk_context.restype = ctypes.c_void_p
    lib.Z3_del_config.argtypes = [ctypes.c_void_p]
    lib.Z3_del_context.argtypes = [ctypes.c_void_p]
    lib.Z3_eval_smtlib2_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.Z3_eval_smtlib2_string.restype = ctypes.c_char_p

    manifest = json.loads((formal_dir / "manifest.json").read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for check in manifest["checks"]:
        script = (formal_dir / check["path"]).read_bytes()
        config = lib.Z3_mk_config()
        context = lib.Z3_mk_context(config)
        lib.Z3_del_config(config)
        try:
            output = lib.Z3_eval_smtlib2_string(context, script).decode("utf-8").strip()
        finally:
            lib.Z3_del_context(context)
        actual = output.splitlines()[0] if output else "missing"
        results.append(
            {
                **check,
                "solver_result": actual,
                "expectation_met": actual == check["expected"],
                "script_sha256": hashlib.sha256(script).hexdigest(),
                "solver_output": output,
            }
        )

    return {
        "schema_version": "pllm.formal_result.v1",
        "origin": "native_z3_executed",
        "solver_version": lib.Z3_get_full_version().decode(),
        "checks": results,
        "limitations": [
            "finite specification scope only",
            "implementation refinement not established",
            "whole-protocol privacy not established",
        ],
    }


def main() -> int:
    formal_dir = Path(__file__).resolve().parent / "formal"
    try:
        result = run_formal(formal_dir)
    except (OSError, SolverUnavailable) as exc:
        print(f"formal checks unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    checks = result["checks"]
    assert isinstance(checks, list)
    return 0 if all(check["expectation_met"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
