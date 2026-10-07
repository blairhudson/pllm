"""Compiler-bound nonlinear regions, independent numerics and worker lifecycle."""
import asyncio
import json
import math
import struct
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import pllm
from pllm import _native
from pllm.nonlinear import MaskedWireGatedMultiplyQ12
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.masked_wire import ClientRegions, WorkerRegions, encode_input, operation_digest
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_reference import TwoOnlineOffsetTransport
from pllm.runtime.offset_worker import create_offset_worker_app
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _fixture(tmp_path, layout="unfused", family="qwen2", *, width=32, dispatch="serial"):
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1,
        model_type=family, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3",
        intermediate_size=width)
    config = json.loads((root / "config.json").read_text())
    pipeline = TwoOnlineOffsetCpu(pllm.Model.path(str(root), model_id="nonlinear-test"),
        nonlinear=MaskedWireGatedMultiplyQ12(layout=layout, dispatch=dispatch))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=3)
    return root, pipeline, plan


def _oracle(gate, up):
    table = [math.floor(4096 * (i / 64) / (1 + math.exp(i / 64))) for i in range(1024)]
    output = []
    for x, y in zip(gate.flat, up.flat, strict=True):
        g, u = math.floor(float(x) * 2**24) // 4096, math.floor(float(y) * 2**24) // 4096
        output.append(((max(g, 0) - table[min(abs(g // 64), 1023)]) * u) // 4096)
    return (np.array(output, dtype=np.float32) / np.float32(4096)).reshape(gate.shape)


@pytest.mark.parametrize("changed", ["layout", "dispatch", "cache", "kernels", "protocol"])
def test_native_compiler_rejects_unimplemented_nonlinear_compositions(tmp_path, changed):
    from pllm.configuration import Pipeline
    from pllm.kernels import AppleMetal
    from pllm.protocols import MaskedLinear
    from pllm.state import ClientPrefixReuse

    _, pipeline, plan = _fixture(tmp_path)
    spec = pipeline.to_spec()
    if changed == "layout":
        spec["components"]["nonlinear"]["params"] = {"layout": "other"}
    elif changed == "dispatch":
        spec["components"]["nonlinear"]["params"] = {"dispatch": "unbounded"}
    elif changed == "cache":
        spec["components"]["cache"] = ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=4).to_spec()
    elif changed == "kernels":
        spec["components"]["kernels"] = AppleMetal().to_spec()
    else:
        spec["components"]["linear"] = MaskedLinear().to_spec()
    with pytest.raises(ValueError):
        plan.runtime_schedule(Pipeline.from_spec(spec))


@pytest.mark.parametrize("layout", ["unfused", "fused"])
@pytest.mark.parametrize("dispatch", ["serial", "overlap"])
def test_bound_transport_matches_integer_oracle_and_burns_replay(tmp_path, layout, dispatch):
    _, pipeline, plan = _fixture(tmp_path, layout, width=129, dispatch=dispatch)
    native = plan.runtime_schedule(pipeline)
    schedule = native.to_dict()
    assert schedule["nonlinear"]["input_boundary"] == "client.float32-to-signed-q24.floor.reject.v1"
    if dispatch == "serial":
        assert "dispatch" not in schedule["nonlinear"]
    else:
        assert schedule["nonlinear"]["dispatch"] == dispatch
        assert native.digest != plan.runtime_schedule(pipeline.with_params(nonlinear__dispatch="serial")).digest
    step = next(s for s in schedule["prefill"]["steps"] if s["executor"] == "remote_gated_block")
    sessions = ("ab" * 16, "cd" * 16)
    workers = [WorkerRegions(schedule, native.digest, p, 3) for p in range(2)]
    requests = []

    def post(index, path, *, phase, **kwargs):
        requests.append((index, path, kwargs))
        if path.endswith("/arm"):
            return 200, json.dumps(workers[index].arm(sessions[index], kwargs["json"])).encode()
        region, action = path.split("/")[-2:]
        return 200, getattr(workers[index], action)(region, kwargs["content"])

    client = ClientRegions(schedule, native.digest, sessions, post)
    g = np.resize(np.array([-48, -16, -1/2**25, 0, 1/2**25, 1, 16, 48], dtype=np.float32), (4, 129))
    u = np.resize(np.array([-128, 128, -0.4, 0.7], dtype=np.float32), g.shape)
    np.testing.assert_array_equal(client.run("prefill", step, g, u), _oracle(g, u))
    assert client.stats["elements"] == 516 and client.stats["chunks"] == 3
    assert all(worker.pending is None for worker in workers)
    index, path, kwargs = next(item for item in requests if item[1].endswith("/start"))
    with pytest.raises(ValueError, match="replayed"):
        workers[index].start(path.split("/")[-2], kwargs["content"])
    with pytest.raises(ValueError, match="public range"):
        encode_input(np.array([49], dtype=np.float32), gate=True)
    client.cancel()
    with pytest.raises(ValueError):
        client.run("prefill", step, g, u)


@pytest.mark.parametrize("layout", ["unfused", "fused"])
@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("dispatch", ["serial", "overlap"])
def test_whole_decoder_http_matches_independent_nonlinear_oracle(tmp_path, monkeypatch, layout, family, dispatch):
    width = 257 if dispatch == "overlap" else 32
    root, pipeline, plan = _fixture(tmp_path, layout, family, width=width, dispatch=dispatch)
    manifest = load_hf_directory(root, model_id="nonlinear-test")
    engines = [MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8) for _ in range(2)]
    for engine in engines:
        asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engines[0].client_bundle("nonlinear-test", placement="offset"))
    compiled = compile_runtime_model(plan, bundle, composition=pipeline)
    assert len(compiled.to_spec()["nonlinear_operations"]) == 4
    transports, adapters = [], []
    keys = ("a" * 32, "b" * 32)
    for index, engine in enumerate(engines):
        app = create_offset_worker_app(engine, model_id="nonlinear-test",
            role_id=("worker_a", "worker_b")[index], api_key=keys[index], composition=pipeline)
        adapter = TestClient(app)
        adapters.append(adapter)

        def send(request, adapter=adapter):
            response = adapter.request(request.method, request.url.path,
                content=request.content, headers=dict(request.headers))
            return httpx.Response(response.status_code, headers=response.headers, content=response.content)

        transports.append(httpx.Client(base_url="http://worker.test", transport=httpx.MockTransport(send)))
    try:
        snapshots = []
        for reference in (False, True):
            with TwoOnlineOffsetTransport(compiled, model_id="nonlinear-test",
                worker_a=transports[0], worker_b=transports[1], api_key_a=keys[0], api_key_b=keys[1]) as remote:
                with monkeypatch.context() as m:
                    if reference:
                        m.setattr(remote._nonlinear, "run", lambda phase, step, g, u: _oracle(g, u))
                    session = compiled.session(remote)
                    session.prefill_ids([0, 2])
                    prefill = session.logits.copy()
                    session.select_next()
                    session.decode_selected()
                    snapshots.append((prefill, session.logits.copy(), [
                        (c.key[:c.length].copy(), c.value[:c.length].copy()) for c in session._runtime.caches]))
                    session.finish()
                    remote.complete()
                    if not reference:
                        assert remote.nonlinear_costs["elements"] == 3 * width
                        bodies = sum(v["online_upload_bytes"] + v["online_download_bytes"]
                            for v in remote.http_body_costs.values())
                        assert bodies > remote.costs.total_stage_body_bytes + remote.nonlinear_costs["party_key_bytes"]
        for i in (0, 1):
            np.testing.assert_array_equal(snapshots[0][i], snapshots[1][i])
        for actual, expected in zip(snapshots[0][2], snapshots[1][2], strict=True):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
    finally:
        for client in transports + adapters:
            client.close()


@pytest.mark.parametrize("corruption", ["context", "party", "frame", "cancel", "replay"])
def test_native_party_failure_burns_handle(corruption):
    context = (b"p" * 32, b"s" * 32, b"o" * 32, 0, 1, "unfused")
    client, left, right = _native._masked_wire_issue(*context)
    if corruption == "context":
        with pytest.raises(ValueError):
            _native._MaskedWireParty(left, b"x" * 32, *context[1:], 0)
        return
    if corruption == "party":
        with pytest.raises(ValueError):
            _native._MaskedWireParty(right, *context, 0)
        return
    a, b = (_native._MaskedWireParty(key, *context, p) for p, key in enumerate((left, right)))
    masked = client.encode(struct.pack("<q", 1), struct.pack("<q", -1))
    opening = b.start(masked)
    a.start(masked)
    if corruption == "frame":
        with pytest.raises(ValueError):
            a.advance(opening[:-1])
    elif corruption == "cancel":
        a.cancel()
    else:
        with pytest.raises(ValueError):
            a.start(masked)
    with pytest.raises(ValueError):
        a.advance(opening)
    b.cancel()
    client.cancel()


@pytest.mark.parametrize("layout,family", [("unfused", "qwen2"), ("fused", "qwen3")])
@pytest.mark.parametrize("dispatch", ["serial", "overlap"])
def test_sdk_and_gateway_execute_separate_party_processes(tmp_path, layout, family, dispatch):
    from pllm.runtime.servers import build_roles

    root, pipeline, _ = _fixture(tmp_path, layout, family, dispatch=dispatch)
    experiment = pllm.Experiment(name="nonlinear-test", pipeline=pipeline,
        deployment=pllm.Deployment.local(root=str(tmp_path)),
        budget=pllm.ExecutionBudget(max_input_tokens=32, max_new_tokens=2, requests=1))
    assert tuple(role.id for role in pllm.Experiment.from_spec(experiment.to_spec()).resolve().role_graph.roles) == (
        "client", "worker_a", "worker_b")
    with build_roles(experiment, engine_threads=1) as topology:
        assert len({status.pid for status in topology.statuses}) == 2
        with topology.client() as client:
            response = client.responses.create(model="nonlinear-test", input="A", max_output_tokens=2)
            assert response.usage.output_tokens > 0
            audit = client.privacy_audit.to_dict()
            assert audit["nonlinear_elements"] >= response.usage.input_tokens * 32
            assert audit["nonlinear_party_key_bytes"] > 0
            assert audit["nonlinear_client_issuance_ns"] > 0
            assert audit["nonlinear_client_issuance_cpu_ns"] > 0
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        with TestClient(topology.gateway_app(local_api_key="local")) as gateway:
            answer = gateway.post("/v1/responses", headers={"Authorization": "Bearer local"},
                json={"model": "nonlinear-test", "input": "A", "max_output_tokens": 2})
            assert answer.status_code == 200, answer.text
            assert answer.json()["output"]


def test_standard_benchmark_accounts_issuance_and_relay_bodies(tmp_path):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    root, pipeline, _ = _fixture(tmp_path, "fused")
    experiment = pllm.Experiment(name="nonlinear-test", pipeline=pipeline,
        deployment=pllm.Deployment.local(root=str(tmp_path)),
        budget=pllm.ExecutionBudget(max_input_tokens=32, max_new_tokens=2, requests=1))
    report = run_loopback_benchmark(model=str(root), model_id="nonlinear-test", tiny=False,
        prompt="A", max_output_tokens=2, warmups=0, repetitions=1, timeout_seconds=120,
        experiment=experiment, temperature=0)
    assert report["checks"]["passed"]
    privacy = report["runs"][0]["privacy"]
    assert privacy["nonlinear_elements"] > 0
    assert privacy["nonlinear_party_key_bytes"] > 0
    bodies = report["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"]
    assert bodies > privacy["nonlinear_party_key_bytes"] + 2 * privacy["nonlinear_peer_frame_bytes"]
    assert report["topology_accounting"]["runs"][0]["aggregate_run_window_cpu_seconds"] is not None


@pytest.mark.parametrize("failure", ["cancel", "peer", "issuance", "input"])
def test_overlap_burns_prefetched_handles_and_joins_threads_on_failure(tmp_path, monkeypatch, failure):
    _, pipeline, plan = _fixture(tmp_path, width=129, dispatch="overlap")
    native = plan.runtime_schedule(pipeline)
    schedule = native.to_dict()
    step = next(s for s in schedule["prefill"]["steps"] if s["executor"] == "remote_gated_block")
    sessions = ("ab" * 16, "cd" * 16)
    workers = [WorkerRegions(schedule, native.digest, p, 3) for p in range(2)]
    ready, release = threading.Event(), threading.Event()
    issued, first_indices, calls = [], [], []
    original = _native._masked_wire_issue

    def issue(*args):
        first_indices.append(args[3])
        if args[3] and failure == "issuance":
            ready.set()
            raise ValueError("injected issuance failure")
        result = original(*args)
        issued.append((result[0], args[4]))
        if args[3]:
            ready.set()
            assert release.wait(5), "prefetched issuance was not released"
        return result

    def post(index, path, *, phase, **kwargs):
        if path.endswith("/arm"):
            return 200, json.dumps(workers[index].arm(sessions[index], kwargs["json"])).encode()
        region, action = path.split("/")[-2:]
        calls.append((index, action))
        assert ready.wait(5), "worker dispatch did not overlap issuance"
        release.set()
        if failure == "cancel":
            client.cancel()
        if failure in {"cancel", "peer"}:
            return 409, b"declined"
        return 200, getattr(workers[index], action)(region, kwargs["content"])

    monkeypatch.setattr(_native, "_masked_wire_issue", issue)
    client = ClientRegions(schedule, native.digest, sessions, post)
    gate, up = np.zeros((4, 129), dtype=np.float32), np.ones((4, 129), dtype=np.float32)
    if failure == "input":
        gate.flat[256] = 49
    try:
        with pytest.raises(ValueError):
            client.run("prefill", step, gate, up)
        assert client.closed and client.current is None
        assert first_indices == [0, 256]
        assert calls
        for handle, lanes in issued:
            with pytest.raises(ValueError, match="Consumed"):
                handle.encode(bytes(lanes * 8), bytes(lanes * 8))
        assert not any(t.name.startswith("pllm-nonlinear") for t in threading.enumerate())
        with pytest.raises(ValueError):
            client.run("prefill", step, gate, up)
    finally:
        release.set()
        client.cancel()
        for worker in workers:
            worker.cancel()


def test_overlap_pairs_workers_before_waiting_for_either_response(tmp_path):
    _, pipeline, plan = _fixture(tmp_path, dispatch="overlap")
    native = plan.runtime_schedule(pipeline)
    schedule = native.to_dict()
    step = next(s for s in schedule["prefill"]["steps"] if s["executor"] == "remote_gated_block")
    sessions = ("ab" * 16, "cd" * 16)
    workers = [WorkerRegions(schedule, native.digest, p, 3) for p in range(2)]
    barrier = threading.Barrier(2, timeout=5)

    def post(index, path, *, phase, **kwargs):
        if path.endswith("/arm"):
            return 200, json.dumps(workers[index].arm(sessions[index], kwargs["json"])).encode()
        barrier.wait()
        region, action = path.split("/")[-2:]
        return 200, getattr(workers[index], action)(region, kwargs["content"])

    client = ClientRegions(schedule, native.digest, sessions, post)
    gate = np.full((1, 32), 0.7, dtype=np.float32)
    try:
        np.testing.assert_array_equal(client.run("prefill", step, gate, gate), _oracle(gate, gate))
    finally:
        client.cancel()
        for worker in workers:
            worker.cancel()
