"""Isolated provider serialization control over one immutable public i8 matrix.

Compares materialize/pack/unpack/repack with the segmented MessagePack path.
This is a bounded serialization mechanism probe, not decoder peak memory,
client import, provider deployment, network performance, or a quality cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def worker(mode, width):
    import msgpack
    import numpy as np
    import psutil
    from pllm.runtime.bundle_document import BundleDocument
    from pllm.runtime.native import MaskedGEMM

    source = np.random.default_rng(710).integers(-127, 128, (width, width), dtype=np.int8)
    matrix = MaskedGEMM(threads=1).compile(source)
    del source
    weight = matrix.weight_view()
    start, cpu = time.perf_counter(), time.process_time()
    value = {"v": 2, "weights": {"dtype": "i1", "shape": [width, width], "data": memoryview(weight)}}
    if mode == "materialized":
        # The old provider boundaries copied binary leaves, packed a contiguous
        # bundle, and parsed/repacked it for canonical artifact export.
        value["weights"]["data"] = weight.tobytes()
        payload = msgpack.packb(value, use_bin_type=True)
        del value
        parsed = msgpack.unpackb(payload, raw=False)
        assert msgpack.packb(parsed, use_bin_type=True) == payload
        fingerprint = hashlib.sha256(payload).hexdigest()
        size = len(payload)
        extra_retained = size + len(parsed["weights"]["data"])
    else:
        document = BundleDocument(value)
        fingerprint = document.descriptor["sha256"]
        digest = hashlib.sha256()
        for chunk in document.chunks():
            digest.update(chunk)
        assert digest.hexdigest() == fingerprint
        size, extra_retained = len(document), 0
    elapsed, used_cpu = time.perf_counter() - start, time.process_time() - cpu
    return {"mode": mode, "size": size, "sha256": fingerprint,
            "matrix_bytes": width * width, "retained_serialized_copy_bytes": extra_retained,
            "end_rss_bytes": psutil.Process().memory_info().rss,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if platform.system() == "Darwin" else 1024),
            "serialization_seconds": elapsed, "serialization_cpu_seconds": used_cpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("materialized", "segmented"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 16 <= args.width <= 8192:
        parser.error("width must be between 16 and 8192")
    if args.worker:
        print(json.dumps(worker(args.worker, args.width)))
        return
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    if host.available - host.reserve < 2 * GiB:
        raise RuntimeError("probe needs 2 GiB of admitted RAM")
    active = []
    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                process.terminate()
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        samples = []
        for mode in ("materialized", "segmented"):
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            process = subprocess.Popen([sys.executable, __file__, "--worker", mode, "--width", str(args.width)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            active.append(process)
            try:
                stdout, stderr = process.communicate(timeout=90)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                active.remove(process)
            if process.returncode or guard.error:
                raise RuntimeError(guard.error or stderr)
            samples.append(json.loads(stdout))
        assert samples[0]["sha256"] == samples[1]["sha256"]
        assert samples[0]["size"] == samples[1]["size"]
        report = {"schema": "pllm.bundle_memory_probe.v1", "scope": __doc__, "samples": samples,
                  "host": asdict(host), "max_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                  "measured_at": datetime.now(timezone.utc).isoformat(), "parity": True,
                  "python": platform.python_version(), "platform": platform.platform()}
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
