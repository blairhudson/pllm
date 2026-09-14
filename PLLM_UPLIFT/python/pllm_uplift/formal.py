"""Invoke the native Z3 SMT-LIB engine. This is not an inference implementation."""
from __future__ import annotations
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path

class SolverUnavailable(RuntimeError):
    pass

def run_formal(root: Path) -> dict:
    name = ctypes.util.find_library("z3")
    if not name:
        raise SolverUnavailable("Native libz3 is unavailable; install Z3 to run formal specification checks. No pass is inferred.")
    lib = ctypes.CDLL(name)
    lib.Z3_get_full_version.restype = ctypes.c_char_p
    lib.Z3_mk_config.restype = ctypes.c_void_p
    lib.Z3_mk_context.argtypes = [ctypes.c_void_p]
    lib.Z3_mk_context.restype = ctypes.c_void_p
    lib.Z3_del_config.argtypes = [ctypes.c_void_p]
    lib.Z3_del_context.argtypes = [ctypes.c_void_p]
    lib.Z3_eval_smtlib2_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.Z3_eval_smtlib2_string.restype = ctypes.c_char_p
    manifest = json.loads((root / "security/formal/manifest.json").read_text())
    results = []
    for check in manifest["checks"]:
        script = (root / check["path"]).read_bytes()
        cfg = lib.Z3_mk_config()
        ctx = lib.Z3_mk_context(cfg)
        lib.Z3_del_config(cfg)
        try:
            answer = lib.Z3_eval_smtlib2_string(ctx, script).decode("utf-8")
        finally:
            lib.Z3_del_context(ctx)
        actual = answer.strip().splitlines()[0]
        results.append({**check, "solver_result": actual,
                        "expectation_met": actual == check["expected"],
                        "script_sha256": hashlib.sha256(script).hexdigest(),
                        "solver_output": answer.strip()})
    return {"schema_version": "pllm.formal_result.v1", "origin": "native_z3_executed",
            "solver_version": lib.Z3_get_full_version().decode(), "checks": results,
            "whole_protocol_privacy_proven": False,
            "rust_implementation_verified": False}
