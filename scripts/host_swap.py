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

AVAILABLE_SOURCE = "pllm.darwin.HOST_VM_INFO64.rev1.free_count+inactive_count.v1"
FREE_SOURCE = "darwin.HOST_VM_INFO64.rev1.free_count-speculative_count"
PAGE_SOURCE = "darwin.host_page_size"


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
    library.host_page_size.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    library.host_page_size.restype = ctypes.c_int
    library.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    library.sysctlbyname.restype = ctypes.c_int
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


def _darwin_kernel_page_size():
    """Mach counters use kernel pages, not necessarily process SC_PAGE_SIZE."""
    library, task = _libsystem()
    host = library.mach_host_self()
    if not host:
        raise RuntimeError("mach_host_self failed")
    page = ctypes.c_uint32()
    try:
        status = library.host_page_size(host, ctypes.byref(page))
        if status != 0:
            raise RuntimeError(f"host_page_size failed: status={status}")
        return _checked_page_size(page.value)
    finally:
        if library.mach_port_deallocate(task, host) != 0:
            raise RuntimeError("mach_port_deallocate failed")


def _checked_page_size(page):
    if type(page) is not int or page not in (4096, 16384, 65536):
        raise RuntimeError("unsupported OS page size")
    return page


@lru_cache(maxsize=1)
def _darwin_physical_total_bytes():
    """Checked immutable hw.memsize; failed reads are never cached."""
    library, _ = _libsystem()
    total = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(total))
    status = library.sysctlbyname(b"hw.memsize", ctypes.byref(total), ctypes.byref(size), None, 0)
    if status != 0 or size.value != ctypes.sizeof(total) or total.value <= 0:
        raise RuntimeError("invalid/missing hw.memsize")
    return total.value


def _uint(value, bits):
    return type(value) is int and 0 <= value < 1 << bits


def validate_headroom_observation(row):
    """Validate owned Darwin estimator, raw diagnostic, ABI and physical bounds."""
    if (row.get("schema") != "pllm.host_swap_observation.v1"
            or row.get("counter_source") != "darwin.HOST_VM_INFO64.rev1.swapouts"
            or row.get("headroom_metric") != "available_estimate_bytes"
            or row.get("available_estimate_source") != AVAILABLE_SOURCE
            or row.get("free_source") != FREE_SOURCE or row.get("page_size_source") != PAGE_SOURCE):
        raise RuntimeError("unknown Darwin headroom identity")
    page = _checked_page_size(row.get("page_bytes"))
    free, spec, inactive = (row.get(k) for k in
                            ("free_count_including_speculative", "speculative_count", "inactive_count"))
    if not all(_uint(v, 32) for v in (free, spec, inactive)) or spec > free:
        raise RuntimeError("inconsistent free/speculative/inactive VM counters")
    raw, available, total = (row.get(k) for k in
                             ("free_bytes", "available_estimate_bytes", "physical_total_bytes"))
    if (not all(_uint(v, 64) for v in (raw, available, total)) or total <= 0
            or raw != (free - spec) * page or available != (free + inactive) * page
            or not raw <= available <= total):
        raise RuntimeError("invalid Darwin headroom formula/physical bounds")
    return available


def _snapshot(include_free=False):
    import psutil

    swap = psutil.swap_memory()
    page_bytes = _checked_page_size(_darwin_kernel_page_size() if sys.platform == "darwin"
                                    else os.sysconf("SC_PAGE_SIZE"))
    row = {"schema": "pllm.host_swap_observation.v1", "swap_used_bytes": swap.used,
           "page_bytes": page_bytes, "psutil_sout_bytes_diagnostic": swap.sout,
           "psutil_version": psutil.__version__}
    if sys.platform == "darwin":
        stats = _darwin_vm_statistics()
        if any(not _uint(getattr(stats, key, None), 64)
               for key in ("swapouts", "pageouts")):
            raise RuntimeError("invalid host swap counters")
        row.update(counter_source="darwin.HOST_VM_INFO64.rev1.swapouts",
                   swapout_pages=stats.swapouts, swapout_bytes=stats.swapouts * page_bytes,
                   pageout_bytes_diagnostic=stats.pageouts * page_bytes)
        if include_free:
            free = getattr(stats, "free_count", None)
            speculative = getattr(stats, "speculative_count", None)
            inactive = getattr(stats, "inactive_count", None)
            if (not all(_uint(v, 32) for v in (free, speculative))
                    or speculative > free):
                raise RuntimeError("inconsistent free/speculative VM counters")
            # XNU free_count includes speculative; vm_stat Pages free subtracts
            # it already. Never subtract speculative from displayed Pages free.
            row.update(free_bytes=(free - speculative) * page_bytes,
                       free_count_including_speculative=free, speculative_count=speculative,
                       inactive_count=inactive, free_source=FREE_SOURCE, page_size_source=PAGE_SOURCE,
                       physical_total_bytes=_darwin_physical_total_bytes(),
                       headroom_metric="available_estimate_bytes",
                       available_estimate_source=AVAILABLE_SOURCE)
            if not _uint(inactive, 32):
                raise RuntimeError("invalid inactive VM counter")
            # Same formula as psutil Darwin available, but OUR estimator from
            # this Mach sample, not psutil.virtual_memory(). Spec is in free
            # already. Purgeable/file-backed/compressor pages are not additions.
            row["available_estimate_bytes"] = (free + inactive) * page_bytes
            validate_headroom_observation(row)
    elif sys.platform == "linux":
        row.update(counter_source="linux.psutil.proc_vmstat.pswpout", swapout_bytes=swap.sout)
    else:
        raise RuntimeError("unsupported host swap telemetry platform")
    if any(type(value) is not int or value < 0 for key, value in row.items()
           if key.endswith("_bytes") or key == "swapout_pages"):
        raise RuntimeError("invalid host swap counters")
    return row


def swap_snapshot():
    """Backward-compatible swap observation; swap used is separately read by psutil."""
    return _snapshot()


def host_snapshot():
    """Raw free, available estimate and swapouts from ONE Mach statistics sample.

    Swap-used bytes remain a separate psutil observation, not an atomic Mach read.
    psutil_version describes swap accounting only. Estimate is not a reservation
    or a promise against swapping.
    """
    return _snapshot(include_free=True)


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
