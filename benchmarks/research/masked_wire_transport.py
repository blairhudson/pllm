"""Bounded, synthetic-input transport diagnosis at a pinned decoder's MLP width.

Runs one row, one region and fresh material per sample. Real checkpoint values,
whole-decoder costs and model quality are outside this diagnostic.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import platform
import secrets
import statistics
import tempfile
import time

import httpx
import numpy as np

import pllm
from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm.deployment import WanConditions
from pllm.nonlinear import MaskedWireGatedMultiplyQ12
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.runtime.benchmark_memory import MemoryWatchdog, benchmark_memory, require_admission
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.masked_wire import ClientRegions, WorkerRegions
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_reference import TwoOnlineOffsetTransport
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def fixture(width):
    gate = np.resize(np.array([-48, -16, -1.1, -1 / 2**24, 0, 0.7, 16, 48], np.float32), (1, width))
    up = np.resize(np.array([-128, 128, -0.4, 0.7], np.float32), gate.shape)
    table = [math.floor(4096 * (i / 64) / (1 + math.exp(i / 64))) for i in range(1024)]
    expected = []
    for g, u in zip(gate.flat, up.flat, strict=True):
        g, u = math.floor(float(g) * 2**24) // 4096, math.floor(float(u) * 2**24) // 4096
        expected.append((max(g, 0) - table[min(abs(g // 64), 1023)]) * u // 4096)
    return gate, up, (np.array(expected, np.float32) / np.float32(4096)).reshape(gate.shape)


def measured_row(client, step, gate, up, expected):
    original = client.post
    actions = {action: {"requests": 0, "seconds": 0.0, "upload_bytes": 0, "download_bytes": 0}
               for action in ("start", "advance")}
    deadline = time.monotonic() + 60

    def post(index, path, **kwargs):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded region exceeded 60 seconds")
        action = actions[path.rsplit("/", 1)[1]]
        started = time.perf_counter()
        status, body = original(index, path, **kwargs)
        action["seconds"] += time.perf_counter() - started
        action["requests"] += 1
        action["upload_bytes"] += len(kwargs["content"])
        action["download_bytes"] += len(body)
        return status, body

    client.post = post
    started, cpu = time.perf_counter(), time.process_time()
    try:
        actual = client.run("prefill", step, gate, up)
        wall, cpu = time.perf_counter() - started, time.process_time() - cpu
    finally:
        client.post = original
    np.testing.assert_array_equal(actual, expected)
    return {"wall_seconds": wall, "process_cpu_seconds": cpu,
            "exact_outputs": gate.size, "stats": dict(client.stats), "actions": actions,
            "application_body_bytes": sum(x["upload_bytes"] + x["download_bytes"] for x in actions.values())}


def local_case(schedule, gate, up, expected):
    document = schedule.to_dict()
    step = next(s for s in document["prefill"]["steps"] if s["executor"] == "remote_gated_block")
    sessions = tuple(secrets.token_hex(16) for _ in range(2))
    workers = [WorkerRegions(document, schedule.digest, i, 2) for i in range(2)]

    def post(index, path, **kwargs):
        if path.endswith("/arm"):
            return 200, json.dumps(workers[index].arm(sessions[index], kwargs["json"])).encode()
        region, action = path.split("/")[-2:]
        return 200, getattr(workers[index], action)(region, kwargs["content"])

    client = ClientRegions(document, schedule.digest, sessions, post)
    try:
        return measured_row(client, step, gate, up, expected)
    finally:
        client.cancel()
        for worker in workers:
            worker.cancel()


def http_cases(experiment, compiled, step, gate, up, expected, *, paced, samples):
    admission = benchmark_memory(experiment, backend="native", enforced_wan=paced)
    require_admission(admission)
    results = []
    with build_roles(experiment, engine_threads=1, startup_timeout=60,
                     wan=WanConditions() if paced else None) as topology, ExitStack() as stack:
        watchdog = MemoryWatchdog(admission, lambda _: topology.close())
        watchdog.start()
        stack.callback(watchdog.close)
        connections = topology.worker_connections()
        clients, keys = [], []
        for role in ("worker_a", "worker_b"):
            url, key = connections[role]
            clients.append(stack.enter_context(httpx.Client(base_url=url, timeout=5)))
            keys.append(key)

        def cpu_samples():
            return [client.get("/v1/offset-reference/metrics", headers={"authorization": f"Bearer {key}"})
                    .raise_for_status().json()["cpu_ns"] for client, key in zip(clients, keys, strict=True)]

        for _ in range(samples):
            watchdog.check()
            with TwoOnlineOffsetTransport(compiled, model_id=experiment.resolve().model,
                    worker_a=clients[0], worker_b=clients[1], api_key_a=keys[0], api_key_b=keys[1]) as remote:
                before_cpu = cpu_samples()
                before = topology.resource_samples()["wan_emulation"] if paced else None
                row = measured_row(remote._nonlinear, step, gate, up, expected)
                after = topology.resource_samples()["wan_emulation"] if paced else None
                after_cpu = cpu_samples()
                row["worker_cpu_seconds"] = [(b - a) / 1e9 for a, b in zip(before_cpu, after_cpu, strict=True)]
                row["aggregate_process_cpu_seconds"] = row["process_cpu_seconds"] + sum(row["worker_cpu_seconds"])
                row["directed_tcp_stream_bytes"] = None if not paced else {
                    link: value - before["directed_stream_bytes"].get(link, 0)
                    for link, value in after["directed_stream_bytes"].items()}
                body = sum(x["online_upload_bytes"] + x["online_download_bytes"]
                           for x in remote.http_body_costs.values())
                assert body == row["application_body_bytes"]
                remote.complete()
                results.append(row)
        watchdog.check()
        return {"samples": results, "memory_admission": admission,
                "maximum_swap_growth_bytes": watchdog.maximum_swap_growth_bytes}


def run(samples=3, *, record=lambda document: None):
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True)).read_bytes()
    config = json.loads(source)
    target_plan = pllm.lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=2)
    target_pipeline = TwoOnlineOffsetCpu(Model.hf(MODEL, revision=REVISION), nonlinear=MaskedWireGatedMultiplyQ12())
    target_regions = [s for s in target_plan.runtime_schedule(target_pipeline).to_dict()["prefill"]["steps"]
                      if s["executor"] == "remote_gated_block"]
    widths = {s["outputs"][0]["stage_width"] for s in target_regions}
    if len(widths) != 1 or not 1 <= next(iter(widths)) <= 16384:
        raise ValueError("diagnostic requires one bounded uniform semantic gated width")
    width = widths.pop()
    gate, up, expected = fixture(width)
    report = {"schema": "pllm.masked_wire_transport_diagnostic.v1", "status": "running",
        "scope": "one synthetic row/region; compiler-bound ordinary worker transport; not checkpoint inference",
        "source": {"model": MODEL, "revision": REVISION, "config_sha256": hashlib.sha256(source).hexdigest(),
                   "plan_digest": target_plan.digest, "gated_regions": len(target_regions), "width": width},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "wan_conditions": WanConditions().to_spec(), "cases": [], "active_case": None,
        "exclusions": ["checkpoint inference and quality", "whole-response costs", "startup/session setup and teardown",
                       "physical wire", "independent operators", "peak memory comparison"],
        "cpu_scope": "process deltas include telemetry/idle overhead; in-process mode includes both workers in parent CPU"}
    record(report)
    with tempfile.TemporaryDirectory(prefix="pllm-region-") as temporary:
        root = create_tiny_llama_checkpoint(Path(temporary) / "checkpoint", num_hidden_layers=1, intermediate_size=width)
        model = Model.path(str(root), model_id="masked-wire-transport-diagnostic")
        tiny_config = json.loads((root / "config.json").read_text())
        plan = pllm.lower_model(tiny_config, batch=1, max_input_tokens=1, max_new_tokens=2)
        for layout in ("unfused", "fused"):
            pipeline = TwoOnlineOffsetCpu(model, nonlinear=MaskedWireGatedMultiplyQ12(layout=layout))
            experiment = Experiment(name=f"transport-{layout}", pipeline=pipeline,
                deployment=Deployment.local(root=temporary),
                budget=ExecutionBudget(max_input_tokens=1, max_new_tokens=2, requests=samples))
            admission = benchmark_memory(experiment, backend="native")
            require_admission(admission)
            schedule = plan.runtime_schedule(pipeline)
            report["active_case"] = {"layout": layout, "transport": "in-process"}
            record(report)
            case = {"layout": layout, "transport": "in-process",
                    "samples": [local_case(schedule, gate, up, expected) for _ in range(samples)]}
            report["cases"].append(case)
            report["active_case"] = None
            record(report)
            engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
            asyncio.run(engine.load(load_hf_directory(root, model_id=model.model_id)))
            try:
                bundle = ClientBundle.unpack(engine.client_bundle(model.model_id, placement="offset"))
                compiled = compile_runtime_model(plan, bundle, composition=pipeline)
                step = next(s for s in schedule.to_dict()["prefill"]["steps"] if s["executor"] == "remote_gated_block")
                for paced in (False, True):
                    report["active_case"] = {"layout": layout, "transport": "native-100/40" if paced else "http-loopback"}
                    record(report)
                    case = {"layout": layout, "transport": "native-100/40" if paced else "http-loopback",
                        **http_cases(experiment, compiled, step, gate, up, expected, paced=paced, samples=samples)}
                    report["cases"].append(case)
                    report["active_case"] = None
                    record(report)
            finally:
                engine.models.clear()
    for case in report["cases"]:
        rows = case["samples"]
        assert all(row["stats"]["elements"] == width for row in rows)
        case["median_wall_seconds"] = statistics.median(r["wall_seconds"] for r in rows)
        case["median_issuance_seconds"] = statistics.median(r["stats"]["client_issuance_ns"] / 1e9 for r in rows)
        case["native_resources"] = json.loads(_native.masked_wire_resources(256, case["layout"]))
    report["status"] = "completed"
    return report


def main():
    from pllm._cli.benchmark_output import write_atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latest = None

    def record(document):
        nonlocal latest
        write_atomic_json(args.output, document, replace=latest is not None)
        latest = document

    try:
        report = run(args.samples, record=record)
    except BaseException as error:
        if latest is not None:
            latest["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            latest["failure"] = {"type": type(error).__name__}
            record(latest)
        raise
    record(report)
    for case in report["cases"]:
        print(f"{case['layout']} {case['transport']}: {case['median_wall_seconds']:.3f}s; "
              f"issuance {case['median_issuance_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
