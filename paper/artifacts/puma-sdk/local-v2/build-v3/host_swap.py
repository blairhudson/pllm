"""Read actual host swap activity, separately from Darwin file pageouts.

Darwin psutil 7.2.2 reports vm_statistics64.pageouts as `swap_memory().sout`.
The zero-growth research guard instead needs vm_statistics64.swapouts, which
counts pages swapped out via compressor segments. These are page-equivalent
bytes, not measured physical disk I/O. Missing or incompatible telemetry raises.
"""

import ctypes
from functools import lru_cache
import os
import sys


class _VmStatisticsRev1(ctypes.Structure):
    # Public mach/vm_statistics.h revision-one ABI, through byte 152. Requesting
    # its 38 integer_t words avoids depending on later SDK tail additions.
    _fields_ = [
        *[(name, ctypes.c_uint32) for name in
          ("free_count", "active_count", "inactive_count", "wire_count")],
        *[(name, ctypes.c_uint64) for name in
          ("zero_fill_count", "reactivations", "pageins", "pageouts", "faults",
           "cow_faults", "lookups", "hits", "purges")],
        ("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32),
        *[(name, ctypes.c_uint64) for name in
          ("decompressions", "compressions", "swapins", "swapouts")],
        *[(name, ctypes.c_uint32) for name in
          ("compressor_page_count", "throttled_count", "external_page_count", "internal_page_count")],
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


@lru_cache(maxsize=1)
def _libsystem():
    if sys.platform != "darwin" or ctypes.sizeof(_VmStatisticsRev1) != 152:
        raise RuntimeError("unsupported Darwin VM statistics ABI")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.mach_host_self.argtypes = []
    library.mach_host_self.restype = ctypes.c_uint32
    library.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                         ctypes.POINTER(_VmStatisticsRev1), ctypes.POINTER(ctypes.c_uint32)]
    library.host_statistics64.restype = ctypes.c_int
    library.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.mach_port_deallocate.restype = ctypes.c_int
    task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
    return library, task


def _darwin_vm_statistics():
    library, task = _libsystem()
    host = library.mach_host_self()
    if not host:
        raise RuntimeError("mach_host_self failed")
    statistics, count = _VmStatisticsRev1(), ctypes.c_uint32(38)
    try:
        status = library.host_statistics64(host, 4, ctypes.byref(statistics), ctypes.byref(count))
        if status != 0 or count.value != 38:
            raise RuntimeError(f"host_statistics64 failed: status={status}, count={count.value}")
        return statistics
    finally:
        if library.mach_port_deallocate(task, host) != 0:
            raise RuntimeError("mach_port_deallocate failed")


def swap_snapshot():
    import psutil

    swap = psutil.swap_memory()
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    if page_bytes not in (4096, 16384, 65536):
        raise RuntimeError("unsupported OS page size")
    row = {"schema": "pllm.host_swap_observation.v1", "swap_used_bytes": swap.used,
           "page_bytes": page_bytes, "psutil_sout_bytes_diagnostic": swap.sout,
           "psutil_version": psutil.__version__}
    if sys.platform == "darwin":
        stats = _darwin_vm_statistics()
        row.update(counter_source="darwin.HOST_VM_INFO64.rev1.swapouts",
                   swapout_pages=stats.swapouts, swapout_bytes=stats.swapouts * page_bytes,
                   pageout_bytes_diagnostic=stats.pageouts * page_bytes)
    elif sys.platform == "linux":
        row.update(counter_source="linux.psutil.proc_vmstat.pswpout", swapout_bytes=swap.sout)
    else:
        raise RuntimeError("unsupported host swap telemetry platform")
    if any(type(value) is not int or value < 0 for key, value in row.items()
           if key.endswith("_bytes") or key == "swapout_pages"):
        raise RuntimeError("invalid host swap counters")
    return row


def validate_counter_identity(current, baseline):
    """Reject counter substitution/reset; never treat unknown activity as zero."""
    if (baseline.get("schema") != "pllm.host_swap_observation.v1"
            or baseline.get("counter_source") not in
            ("darwin.HOST_VM_INFO64.rev1.swapouts", "linux.psutil.proc_vmstat.pswpout")
            or baseline.get("page_bytes") not in (4096, 16384, 65536)):
        raise RuntimeError("unknown host swap counter identity")
    if any(current.get(key) != baseline.get(key) for key in ("schema", "counter_source", "page_bytes")):
        raise RuntimeError("host swap counter identity changed")
    if any(type(row.get(key)) is not int or row[key] < 0
           for row in (current, baseline) for key in ("swapout_bytes", "swap_used_bytes")):
        raise RuntimeError("invalid host swap counters")
    if current["swapout_bytes"] < baseline["swapout_bytes"]:
        raise RuntimeError("host swap counter regressed")
