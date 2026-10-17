"""Installed full-share GKM: real worker processes and complete decoder oracles."""

import asyncio
import json
import os
from pathlib import Path
import threading

import httpx
import pytest

from pllm import AsyncOpenAI, Model, OpenAI, _native
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime import delegated_transport as transport
from pllm.runtime.delegated_source import AdmittedSource
from pllm.runtime.client import ProtocolError
from pllm.runtime.responses import normalize_input, prompt_text

from test_delegated_decoder import clear_oracle
from test_delegated_source import write_source_fixture


def experiment_fixture(path, *, family="qwen2", tied=True, requests=4,
                       capacity="capsule", vocabulary=32, rows=8):
    cfg, _, arrays, _ = write_source_fixture(path / "source", family=family, tied=tied,
                                           capacity=capacity, vocabulary=vocabulary)
    experiment = Experiment(
        "gkm-sdk", OpenWeightVerifiableCpu(Model.path(str(path / "source")),
            linear=OpenWeightVerifiableOutsourcing(capacity=capacity)),
        Deployment.local(root=str(path / "roles")),
        ExecutionBudget(requests=requests, max_input_tokens=rows, max_new_tokens=3),
    )
    return Experiment.from_spec(json.loads(experiment.canonical_bytes())), cfg, arrays


def active_sessions(core):
    result = []
    for url, key in core.role_connections.values():
        origin = url.removesuffix("/v1")
        response = httpx.get(origin + "/v1/delegation/metrics", headers={"authorization": f"Bearer {key}"},
                             timeout=5, trust_env=False)
        response.raise_for_status()
        result.append(response.json()["active_sessions"])
    return result


@pytest.mark.parametrize("tokens", [(7,), (7, 9, 1), [7, 9], (True, 9), (32, 9)])
def test_quality_continuation_rejects_before_transport_material(tmp_path, monkeypatch, tokens):
    from pllm.runtime.delegated_source import admit_source

    experiment, _, _ = experiment_fixture(tmp_path)
    source = admit_source(experiment)

    def forbidden(*args, **kwargs):
        raise AssertionError("native material loaded before continuation admission")

    monkeypatch.setattr(AdmittedSource, "load_transport", forbidden)
    with pytest.raises(ValueError, match="continuation"):
        transport.execute_request(experiment, source, (3, 4), 3, {}, 60,
                                  lambda: False, decode_tokens=tokens)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_sdk_two_workers_complete_state_oracle_and_retirement(tmp_path, family, tied, record_property):
    experiment, cfg, arrays = experiment_fixture(tmp_path, family=family, tied=tied)
    captured = []
    with OpenAI(experiment=experiment) as client:
        topology = client._owned_topology
        client._core._result_hook = captured.append
        roles = client._owned_topology.statuses
        assert {item.role for item in roles} == {"worker_a", "worker_b"}
        assert all(item.running and item.pid != os.getpid() for item in roles)
        assert len({item.pid for item in roles}) == 2
        record_property("worker_pids", [item.pid for item in roles])
        record_property("native", str(Path(_native.__file__).resolve()))
        response = client.responses.create(input="hello world", max_output_tokens=3)
        source = client._core._ensure_source()
        ids = source.tokenizer().encode(prompt_text(normalize_input("hello world")), add_bos=False)
        assert response.usage.input_tokens == len(ids)
        assert response.usage.output_tokens == 3
        assert response.status == "incomplete"
        result, = captured
        assert (result.logits, result.kv, result.selected, result.executed, result.position) == clear_oracle(
            cfg, arrays, tuple(ids), 2,
        )
        record_property("checked_float32_values", (
            sum(map(len, result.logits)) + sum(len(k) + len(v) for _, k, v in result.kv)) // 4)
        assert json.loads(result.acceptance)["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
        assert response.output_text == source.tokenizer().decode(list(result.selected))
        assert active_sessions(client._core) == [0, 0]
        audit = client._core.audit.to_dict()
        online = sum(audit[f"role_link.{role}.online_{direction}_bytes"]
                     for role in ("worker_a", "worker_b") for direction in ("upload", "download"))
        assert online == audit["inference_upload_bytes"] + audit["inference_download_bytes"]
        assert online == audit["generation_prefill_online_body_bytes"] + audit["generation_decode_online_body_bytes"]
        assert audit["generation_prefill_online_body_bytes"] > audit["generation_decode_online_body_bytes"] > 0
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        assert not client._core._transformer_states
    assert all(not item.running for item in topology.statuses)


def test_stream_and_failed_final_barrier_expose_no_events(tmp_path, monkeypatch):
    experiment, _, _ = experiment_fixture(tmp_path)
    with OpenAI(experiment=experiment) as client:
        load_transport = AdmittedSource.load_transport
        checks = []

        class FinalBarrier:
            def __init__(self, native):
                self.native = native

            def __getattr__(self, name):
                return getattr(self.native, name)

            def finish(self):
                self.native.finish()
                checks.append("finished")
                raise ValueError("injected final barrier failure")

        def failed_barrier(source, *args, **kwargs):
            native = load_transport(source, *args, **kwargs)
            return FinalBarrier(native) if kwargs.get("worker") is None else native

        monkeypatch.setattr(AdmittedSource, "load_transport", failed_barrier)
        observed = []
        with pytest.raises(ValueError, match="final barrier"):
            with client.responses.create(input="hello world", max_output_tokens=2, stream=True) as stream:
                for event in stream:
                    observed.append(event)
        assert checks == ["finished"] and not observed
        assert not client._core.records and client._core._active is None
        assert active_sessions(client._core) == [0, 0]
        monkeypatch.setattr(AdmittedSource, "load_transport", load_transport)
        with client.responses.create(input="hello world", max_output_tokens=2, stream=True) as stream:
            events = list(stream)
        assert events and events[-1].type == "response.incomplete"
        assert active_sessions(client._core) == [0, 0]


@pytest.mark.parametrize("failure", ["cancel", "malformed", "replay", "partial_admission"])
def test_transport_failure_retires_both_real_roles(tmp_path, monkeypatch, failure):
    experiment, _, _ = experiment_fixture(tmp_path)
    original = transport._Peers.post
    injected = []

    def fail(self, role, action, *args, **kwargs):
        if not injected and ((failure == "partial_admission" and role == 1 and action == "admit")
                             or (failure != "partial_admission" and role == 0 and action == "step")):
            injected.append(True)
            if failure == "cancel":
                # Client cancellation can land after one native role consumed its frame.
                original(self, role, action, *args, **kwargs)
                raise asyncio.CancelledError()
            if failure == "malformed":
                original(self, role, action, *args, **kwargs)
                return b"bad native reply"
            if failure == "replay":
                original(self, role, action, *args, **kwargs)
                return original(self, role, action, *args, **kwargs)
            raise ProtocolError("injected second admission failure")
        return original(self, role, action, *args, **kwargs)

    with OpenAI(experiment=experiment) as client:
        monkeypatch.setattr(transport._Peers, "post", fail)
        with pytest.raises((ValueError, ProtocolError, asyncio.CancelledError)):
            client.responses.create(input="hello world", max_output_tokens=2)
        assert injected and not client._core.records and client._core._active is None
        assert active_sessions(client._core) == [0, 0]
        monkeypatch.setattr(transport._Peers, "post", original)
        assert client.responses.create(input="hello world", max_output_tokens=2).usage.output_tokens == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["task", "concurrent_stream_close"])
async def test_async_cancellation_drains_real_workers_before_return(tmp_path, monkeypatch, mode):
    experiment, _, _ = experiment_fixture(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = transport._Peers.post

    def pause(self, role, action, *args, **kwargs):
        result = original(self, role, action, *args, **kwargs)
        if role == 0 and action == "step" and not entered.is_set():
            entered.set()
            if not release.wait(10):
                raise AssertionError("cancellation test did not release its worker")
        return result

    monkeypatch.setattr(transport._Peers, "post", pause)
    async with AsyncOpenAI(experiment=experiment) as client:
        core = client.sync._core
        stream = None
        if mode == "task":
            task = asyncio.create_task(client.responses.create(input="hello world", max_output_tokens=2))
        else:
            stream = await client.responses.create(input="hello world", max_output_tokens=2, stream=True)
            task = asyncio.create_task(anext(stream))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if stream is None:
                task.cancel()
                asyncio.get_running_loop().call_soon(release.set)
            else:
                closing = asyncio.create_task(stream.aclose())
                assert await asyncio.to_thread(core._active.cancelled.wait, 10)
                release.set()
                await closing
            with pytest.raises(asyncio.CancelledError if stream is None else ProtocolError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            if stream is not None:
                await stream.aclose()
        assert core._active is None and not core.records
        assert await asyncio.to_thread(active_sessions, core) == [0, 0]
        monkeypatch.setattr(transport._Peers, "post", original)
        assert (await client.responses.create(input="hello world", max_output_tokens=2)).usage.output_tokens == 2


@pytest.mark.parametrize("origin", ["http://example.com", "http://192.0.2.1", "https://a:secret@example.com",
                                   "https://example.com?credential=x", "https://example.com/redirect"])
def test_remote_channels_require_confidential_fixed_origins(origin):
    with pytest.raises(ValueError):
        transport._connections({"worker_a": (origin, "a" * 32),
                                "worker_b": ("http://127.0.0.1:1234", "b" * 32)})
