"""Isolated authenticated client import, mask inventory and token/head operations.

Uses an actual checkpoint's public bundle and compiled binding. The bounded
publisher discards remote weights after quantization/metadata validation. It is
not a provider: no body stage, attention, KV cache or complete response executes.
Control uses the previous eager mask arrays and resident boundary weights;
candidate uses the ordinary streamed/paged SDK importer and native mask cursors.
Both consume identical public fixture masks and execute identical local boundaries.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time


def publish(source: Path, root: Path):
    import asyncio
    import pllm
    from pllm.model_loader import resolve_model
    from pllm.runtime.bundle_artifacts import export_bundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1,
                                    weight_residency="provider_and_bundle")
    load_stage = engine._load_stage

    def metadata_and_boundary(store, spec, manifest, *, retain_weight=True):
        # Probe-only bounded publisher. Ordinary role admission is not changed;
        # every omitted remote weight is still quantized and committed normally.
        return load_stage(store, spec, manifest, retain_weight=(
            retain_weight and spec.role in {"token_lookup", "lm_head"}))

    engine._load_stage = metadata_and_boundary
    model = pllm.Model.path(str(source), model_id="client-storage-probe")
    resolved = resolve_model(model)
    asyncio.run(engine.load(resolved.manifest))
    document = engine.client_bundle_document("client-storage-probe")
    artifacts = export_bundle(document)
    (root / "manifest").write_bytes(artifacts.manifest)
    for key, value in artifacts.objects.items():
        with (root / key).open("xb") as output:
            view = memoryview(value).cast("B")
            for start in range(0, len(view), 1 << 20):
                output.write(view[start:start + (1 << 20)])
    descriptor = {"source": model.to_spec(), "raw": dict(document.descriptor),
                  "model_id": "client-storage-probe",
                  "config": json.loads((source / "config.json").read_bytes())}
    (root / "descriptor.json").write_text(json.dumps(descriptor))
    return {"bundle_sha256": document.descriptor["sha256"],
            "bundle_bytes": len(document), "object_count": len(artifacts.objects),
            "publisher_peak_rss_bytes": peak_rss()}


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)


def worker(mode, root):
    import httpx
    import numpy as np
    import psutil
    import pllm
    from pllm.kernels import Cpu
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import ClientBundleTransport
    from pllm.runtime.bundle_artifacts import ENCODING
    from pllm.runtime.client import RuntimeClient
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.preparation_protocol import PreparationRequest, expand_preparation_mask, expand_output_mask
    from pllm.runtime.transformer_client import PreparedInventory, PreparedStageRows

    descriptor = json.loads((root / "descriptor.json").read_bytes())
    storage = "memory" if mode == "eager-resident" else "paged"
    pipeline = MaskedLinearCpu(pllm.Model.from_spec(descriptor["source"]), kernels=Cpu(threads=1),
                              delivery=ClientBundleTransport("artifacts", storage=storage))
    experiment = pllm.Experiment("client-storage", pipeline, pllm.Deployment.local(root=str(root)),
                                pllm.ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=8))

    class FileStream(httpx.SyncByteStream):
        def __init__(self, path):
            self.path = path

        def __iter__(self):
            with self.path.open("rb") as source:
                while chunk := source.read(65536):
                    yield chunk

    def handler(request):
        assert request.url.host == "public-memory-fixture.test"
        assert request.headers["Authorization"] == "Bearer public-fixture-key"
        fingerprint = descriptor["raw"]["sha256"]
        if request.url.path.endswith("/client-bundle"):
            return httpx.Response(200, stream=FileStream(root / "manifest"), headers={
                "X-PLLM-Bundle-SHA256": fingerprint, "X-PLLM-Bundle-Encoding": ENCODING})
        key = request.url.path.rsplit("/", 1)[-1]
        assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
        return httpx.Response(200, stream=FileStream(root / key), headers={
            "X-PLLM-Bundle-SHA256": fingerprint, "X-PLLM-Object-SHA256": key})

    started, cpu = time.perf_counter(), time.process_time()
    with closing(RuntimeClient(base_url="https://public-memory-fixture.test", api_key="public-fixture-key",
        experiment=experiment, bundle_cache_dir=root / f"cache-{mode}",
        http_client=httpx.Client(base_url="https://public-memory-fixture.test",
                                 transport=httpx.MockTransport(handler)))) as client:
        bundle = client._load_artifact_bundle(descriptor["model_id"], schema=2,
            fingerprint=descriptor["raw"]["sha256"], size=descriptor["raw"]["size"])
        plan = pllm.lower_model(descriptor["config"], batch=1, max_input_tokens=64, max_new_tokens=8)
        compiled = compile_runtime_model(plan, bundle, composition=pipeline)
        # Pin the resident comparator's boundary executor too (the default head
        # lazily creates a default-sized pool); paged import shares Cpu(1).
        bundle._local_kernel = MaskedGEMM(threads=1)
        stages = [s for s in bundle.stages.values() if s.client_weight is None]
        inventory_rows = {}
        for index, stage in enumerate(stages):
            profile = stage.seeded_profile
            assert profile is not None
            request = PreparationRequest(f"{index:032x}", bundle.model_id, "public-memory-fixture",
                bundle.manifest["metadata"]["body_fingerprint"], stage.id, stage.weight_digest, 71,
                stage.in_features, stage.out_features, stage.weight_bits, stage.activation_bits,
                profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
                hashlib.sha256(f"public fixture {index}".encode()).digest())
            inventory_rows[stage.id] = (PreparedStageRows(request,
                expand_preparation_mask(request), expand_output_mask(request)) if mode == "eager-resident"
                else PreparedStageRows(request))
        inventory = PreparedInventory("public-memory-fixture", 71, inventory_rows)
        lease = inventory.reserve(71)
        retained_masks = sum(s.retained_mask_bytes for s in inventory_rows.values())
        setup_cpu = time.process_time() - cpu
        mask_digest, output_digest = hashlib.sha256(), hashlib.sha256()
        execute_cpu = time.process_time()
        for index, rows in enumerate((64, 1, 1, 1, 1, 1, 1, 1)):
            for stage in stages:
                r, s, attempts = lease.take(stage.id, rows)
                mask_digest.update(memoryview(r).cast("B"))
                mask_digest.update(memoryview(s).cast("B"))
                mask_digest.update("".join(attempts).encode())
            tokens = bundle.local_token_lookup(np.arange(index + 2, index + 2 + rows, dtype=np.int64))
            logits = bundle.local_linear("lm_head", tokens[-1:])
            output_digest.update(memoryview(tokens).cast("B"))
            output_digest.update(memoryview(logits).cast("B"))
        execution_cpu = time.process_time() - execute_cpu
        lease.close()
        inventory.cancel()
        return {"mode": mode, "bundle_sha256": descriptor["raw"]["sha256"],
                "model_plan_digest": plan.digest, "compiled_binding_digest": compiled.digest,
                "pipeline_digest": pipeline.digest(), "retained_mask_bytes": retained_masks,
                "mask_digest": mask_digest.hexdigest(), "boundary_digest": output_digest.hexdigest(),
                "body_stage_count": len(stages), "boundary_integer_mac_count": 8 * bundle.stages["lm_head"].in_features
                    * bundle.stages["lm_head"].out_features,
                "artifact_stats": client.bundle_artifact_cache.stats.to_dict(),
                "import_and_inventory_cpu_seconds": setup_cpu, "execution_cpu_seconds": execution_cpu,
                "combined_cpu_seconds": time.process_time() - cpu,
                "elapsed_seconds": time.perf_counter() - started,
                "end_rss_bytes": psutil.Process().memory_info().rss, "peak_rss_bytes": peak_rss()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("publish", "eager-resident", "stream-paged"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = publish(args.source, args.directory) if args.worker == "publish" else worker(args.worker, args.directory)
        print(json.dumps(result))
        return
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.modeling import lower_model
    from pllm.runtime.semantic_stages import scheduled_stage_specs
    from pllm.profiles import MaskedLinearCpu
    from pllm import Model

    # Check geometry without tensors, cap the probe itself, then price its bounded
    # publisher and isolated eager client. Swap is never admitted as capacity.
    config = json.loads((args.source / "config.json").read_bytes())
    plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
    stages = scheduled_stage_specs(plan, MaskedLinearCpu(Model.path(str(args.source))))
    boundary = sum(s.in_features * s.out_features for s in stages if s.role in {"token_lookup", "lm_head"})
    if config.get("tie_word_embeddings"):
        boundary //= 2
    masks = 71 * sum(s.in_features + s.out_features for s in stages if s.role not in {"token_lookup", "lm_head"}) * 4
    if boundary > GiB or masks > GiB:
        raise ValueError("probe exceeds its 1 GiB boundary or mask storage bound")
    required = 2 * GiB + 4 * boundary + masks
    host = host_memory()
    if host.available - host.reserve < required or host.disk_free < 4 * GiB + 5 * boundary:
        raise RuntimeError(f"probe needs {required / GiB:.2f} GiB RAM headroom and bounded disk storage")
    active = []
    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        with tempfile.TemporaryDirectory(prefix="pllm-client-storage-") as directory:
            samples = []
            for mode in ("publish", "eager-resident", "stream-paged"):
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                print(f"client storage probe: {mode}", file=sys.stderr, flush=True)
                child = subprocess.Popen([sys.executable, __file__, "--source", str(args.source),
                    "--worker", mode, "--directory", directory], text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                active.append(child)
                try:
                    stdout, stderr = child.communicate(timeout=600)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
                    active.remove(child)
                if child.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                samples.append(json.loads(stdout))
        for key in ("bundle_sha256", "model_plan_digest", "mask_digest", "boundary_digest",
                    "boundary_integer_mac_count", "body_stage_count"):
            assert samples[1][key] == samples[2][key], key
        report = {"schema": "pllm.client_storage_probe.v1", "scope": __doc__, "publisher": samples[0],
                  "samples": samples[1:], "parity": True, "host": asdict(host),
                  "required_headroom_bytes": required, "max_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                  "measured_at": datetime.now(timezone.utc).isoformat(),
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
