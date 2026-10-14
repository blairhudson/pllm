"""Standalone test-only Accelerate FFI. No build, registration, or SDK edits.

Uses the legacy LP64 CBLAS ABI, matching existing digit_gemm.rs SGEMM FFI.
This fixture verifies observed parity, not Apple's undocumented reduction DAG.
"""
import ctypes
import platform

import numpy as np


def load():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("fixture requires Apple Silicon Accelerate")
    lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    fn = lib.cblas_dgemm
    integer, double = ctypes.c_int, ctypes.c_double
    pointer = ctypes.POINTER(double)
    fn.argtypes = [integer] * 6 + [double, pointer, integer, pointer,
                                  integer, double, pointer, integer]
    fn.restype = None
    return lib, fn


def product(a, b, *, handle=None):
    """Compute contiguous float64 [batch,k] @ [out,k].T, alpha=1 beta=0."""
    if (a.dtype != np.float64 or b.dtype != np.float64 or a.ndim != 2
            or b.ndim != 2 or not a.flags.c_contiguous or not b.flags.c_contiguous
            or a.shape[1] != b.shape[1]):
        raise ValueError("DGEMM requires matching contiguous rank-2 float64 arrays")
    m, k = a.shape
    n = b.shape[0]
    if not 0 < min(m, n, k) <= max(m, n, k) < (1 << 31):
        raise ValueError("DGEMM dimension exceeds positive LP64 range")
    lib, fn = handle if handle is not None else load()
    c = np.empty((m, n), np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    fn(101, 111, 112, m, n, k, 1.0, a.ctypes.data_as(ptr), k,
       b.ctypes.data_as(ptr), k, 0.0, c.ctypes.data_as(ptr), n)
    return c
