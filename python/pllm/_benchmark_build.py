"""Build inspection in a separate process, outside target measurement owners."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def file_digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1 << 20):
            state.update(block)
    return state.hexdigest()


def hardware_identity():
    """Public resource identity; never hostnames, serials or device identifiers."""
    observed = {"logical_cpus": os.cpu_count(), "cpu_affinity": None, "details": {}}
    if hasattr(os, "sched_getaffinity"):
        try:
            observed["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    if sys.platform == "darwin":
        for name in ("machdep.cpu.brand_string", "hw.model", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"):
            try:
                value = subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", name], timeout=5, stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                value = None
            observed["details"][name] = value
    elif sys.platform.startswith("linux"):
        # Bounded, allowlisted fields cover x86 and ARM without retaining IDs.
        fields = {"model name", "vendor_id", "cpu family", "model", "stepping",
                  "CPU implementer", "CPU architecture", "CPU variant", "CPU part", "CPU revision"}
        try:
            with Path("/proc/cpuinfo").open() as source:
                text = source.read(1 << 20)
            observed["details"]["cpu_models"] = sorted({
                (name.strip(), value.strip()) for line in text.splitlines() if ":" in line
                for name, value in [line.split(":", 1)] if name.strip() in fields
            })
        except OSError:
            observed["details"]["cpu_models"] = None
        try:
            observed["details"]["physical_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            observed["details"]["physical_memory_bytes"] = None
    else:
        observed["details"]["processor"] = platform.processor() or None
    return observed


def inspect():
    from pllm._benchmark_contract import digest

    root = Path(__file__).parent
    sources = {p.relative_to(root).as_posix(): file_digest(p)
               for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".py", ".json"}}
    native = importlib.util.find_spec("pllm._native")
    dependencies = {}
    for name in ("numpy", "httpx", "cryptography", "psutil", "msgpack", "safetensors", "tokenizers",
                 "fastapi", "starlette", "uvicorn", "websockets", "pydantic", "pydantic-core",
                 "huggingface-hub", "tenseal", "mlx"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
            continue
        files = {}
        for entry in dist.files or ():
            if str(entry).endswith((".py", ".so", ".dylib", ".pyd")):
                path = dist.locate_file(entry)
                if path.is_file():
                    files[str(entry)] = file_digest(path)
        dependencies[name] = {"version": dist.version, "files_digest": digest(files)}
    attributes = {"python": platform.python_version(), "implementation": platform.python_implementation(),
        "python_binary": file_digest(sys.executable), "sdk_sources": sources,
        "native_sha256": None if native is None or native.origin is None else file_digest(native.origin),
        "dependencies": dependencies, "worker_protocol": "pllm.benchmark_worker.v1",
        "os_release": platform.release(), "os_version": platform.version(),
        "machine": platform.machine(), "hardware": hardware_identity()}
    import numpy as np
    attributes["numeric_backend"] = np.show_config(mode="dicts")
    return {"digest": digest(attributes), "attributes": attributes}


if __name__ == "__main__":
    print(json.dumps(inspect(), sort_keys=True, allow_nan=False))
