"""Isolated client-mask residency/CPU, using semantic geometry and public test seeds.

This does not load checkpoint tensors, contact providers or measure a decoder.
Rings use conservative public signed-i8 bounds, not weight-derived live bounds.
Both modes execute identical 64-row prefill and seven one-row decode claims.
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


def worker(config_path: Path, mode: str) -> dict:
    import numpy as np
    import psutil
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime.preparation_protocol import PreparationRequest, seeded_ring_profile, expand_preparation_mask, expand_output_mask
    from pllm.runtime.semantic_stages import scheduled_stage_specs, provider_owns_linear
    from pllm.runtime.transformer_client import PreparedInventory, PreparedStageRows

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
    stages = [stage for stage in scheduled_stage_specs(plan, MaskedLinearCpu(Model.path(str(config_path.parent))))
              if provider_owns_linear(stage)]
    if len(stages) > 512 or 71 * sum(s.in_features + s.out_features for s in stages) * 4 > 1 << 30:
        raise ValueError("probe inventory exceeds its 1 GiB material/512-stage bound")
    rows = {}
    started = time.process_time()
    for index, stage in enumerate(stages):
        profile = seeded_ring_profile(127 * 127 * stage.in_features)
        req = PreparationRequest(f"{index:032x}", "public-memory-fixture", "shape-only", plan.digest,
            stage.id, "no-weight-values-loaded", 71, stage.in_features, stage.out_features, 8, 8,
            profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
            hashlib.sha256(f"public mask fixture {index}".encode()).digest())
        rows[stage.id] = (PreparedStageRows(req, expand_preparation_mask(req), expand_output_mask(req))
                          if mode == "eager" else PreparedStageRows(req))
    issue_cpu = time.process_time() - started
    retained = sum(stage.retained_mask_bytes for stage in rows.values())
    inventory = PreparedInventory("public-memory-fixture", 71, rows)
    lease = inventory.reserve(71)
    digest = hashlib.sha256()
    started = time.process_time()
    for count in (64, 1, 1, 1, 1, 1, 1, 1):
        for stage in stages:
            r, s, attempts = lease.take(stage.id, count)
            digest.update(memoryview(np.ascontiguousarray(r)).cast("B"))
            digest.update(memoryview(np.ascontiguousarray(s)).cast("B"))
            digest.update("".join(attempts).encode())
    online_cpu = time.process_time() - started
    lease.close()
    inventory.cancel()
    return {"mode": mode, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "model_plan_digest": plan.digest, "stage_count": len(stages), "rows": 71,
            "mask_and_ticket_digest": digest.hexdigest(), "retained_mask_bytes": retained,
            "issue_cpu_seconds": issue_cpu, "claim_cpu_seconds": online_cpu,
            "combined_cpu_seconds": issue_cpu + online_cpu,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
            "end_rss_bytes": psutil.Process().memory_info().rss}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("eager", "stream"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config.stat().st_size > 1 << 20:
        raise ValueError("configuration exceeds 1 MiB")
    if args.worker:
        print(json.dumps(worker(args.config, args.worker)))
        return
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    if host.available - host.reserve < 2 * GiB:
        raise RuntimeError("mask probe needs 2 GiB of admitted RAM")
    active = []
    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                process.terminate()
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        samples = []
        for mode in ("eager", "stream"):
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            child = subprocess.Popen([sys.executable, __file__, "--config", str(args.config), "--worker", mode],
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            active.append(child)
            try:
                stdout, stderr = child.communicate(timeout=180)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                active.remove(child)
            if child.returncode or guard.error:
                raise RuntimeError(guard.error or stderr)
            samples.append(json.loads(stdout))
        assert samples[0]["mask_and_ticket_digest"] == samples[1]["mask_and_ticket_digest"]
        report = {"schema": "pllm.client_mask_memory.v1", "scope": __doc__, "samples": samples,
                  "host": asdict(host), "max_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                  "measured_at": datetime.now(timezone.utc).isoformat(), "parity": True,
                  "python": platform.python_version(), "platform": platform.platform()}
        if args.output:
            with args.output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        print(json.dumps(report, indent=2))
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
