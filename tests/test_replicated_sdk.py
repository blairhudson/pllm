"""Generated-source three-party decoder and ordinary SDK ownership controls."""

import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pllm import AsyncOpenAI, Model, OpenAI, _native
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import PumaThreePartyCpu
from pllm.runtime import replicated_transport as transport
from pllm.runtime.replicated_protocol import ROLES, input_frame
from pllm.runtime.replicated_source import admit_source


def experiment_fixture(path, *, family="qwen2", outputs=2, requests=8, active=False):
    root = path / "source"
    root.mkdir(parents=True)
    config = {
        "model_type": family, "hidden_size": 2, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": 5, "tie_word_embeddings": False,
        "hidden_act": "silu", "attention_bias": family == "qwen2",
        "max_position_embeddings": 64, "rope_theta": 10000,
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": False, "use_cache": True, "max_window_layers": 1,
        "bos_token_id": 0, "eos_token_id": 1,
    }
    (root / "config.json").write_text(json.dumps(config))
    vocabulary = {"<bos>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<bos>", "<eos>", "<unk>"])
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<bos>", "eos_token": "<eos>", "add_bos_token": False,
        "model_max_length": 64,
    }))
    publisher = _native.ReplicatedPublisher(json.dumps(config).encode(), 2, outputs, b"s" * 32, b"c" * 32)
    arrays = {}
    for key, shape in publisher.shapes:
        if active and key == "model.embed_tokens.weight":
            values = [[0, 0], [5 / 32, 5 / 32], [0, 0], [5 / 32, 5 / 32], [-5 / 32, -5 / 32]]
        elif key == "model.embed_tokens.weight":
            values = [[1, 0], [-1, 0], [1, 0], [-1, 0], [-1, 0]]
        elif active and key == "lm_head.weight":
            values = [[0, 0], [1, 0], [0, 0], [0, 0], [-1, 0]]
        elif key == "lm_head.weight":
            values = [[0, 0], [1, 0], [-1, 0], [0, 0], [0, 0]]
        elif "norm" in key:
            values = np.ones(shape)
        elif active and key.endswith("v_proj.weight"):
            values = np.eye(2)
        elif active and key.endswith("o_proj.weight"):
            values = -np.eye(2) / 2
        elif active and "mlp." in key:
            values = np.full(shape, 1 / 64)
        else:
            values = np.zeros(shape)
        arrays[key] = np.array(values, dtype=np.float32)
    publisher.close()
    save_file(arrays, root / "model.safetensors", metadata={"format": "pt"})
    experiment = Experiment("puma-sdk", PumaThreePartyCpu(Model.path(str(root))),
        Deployment.local(root=str(path / "roles")),
        ExecutionBudget(requests=requests, max_input_tokens=8, max_new_tokens=outputs))
    return Experiment.from_spec(json.loads(experiment.canonical_bytes()))


def active_sessions(core):
    result = []
    for url, key in core.role_connections.values():
        response = httpx.get(url.removesuffix("/v1") + "/v1/replicated/metrics",
                            headers={"authorization": f"Bearer {key}"}, timeout=5, trust_env=False)
        response.raise_for_status()
        result.append(response.json()["active_sessions"])
    return result


def native_execution(source, experiment, ids, count):
    publisher, manifest = source.publisher(experiment, ids, count, lambda: False)
    workers = [_native.ReplicatedWorker(manifest, index) for index in range(3)]
    try:
        offers = [worker.call("offer", b"", index) for index, worker in enumerate(workers)]
        for index, (complete, frames) in enumerate(offers):
            assert not complete and len(frames) == 1 and frames[0][0] == index
            for peer in range(3):
                if peer != index:
                    workers[peer].call("public", frames[0][1], index)
        seeds = [worker.call("seed", b"", index) for index, worker in enumerate(workers)]
        for index, (complete, frames) in enumerate(seeds):
            assert not complete and len(frames) == 1
            peer, frame = frames[0]
            workers[peer].call("receive_seed", frame, index)
        for worker in workers:
            worker.activate()
        while (chunk := publisher.next_chunk()) is not None:
            key, offset, frames = chunk
            for index, worker in enumerate(workers):
                worker.call("import", input_frame({"key": key, "offset": offset, "frame": frames[index]}), index)
        complete = [False] * 3
        for _ in range(1 << 20):
            queue = []
            for index, worker in enumerate(workers):
                complete[index], frames = worker.call("advance", b"", index)
                queue.extend((index, peer, frame) for peer, frame in frames)
            if all(complete):
                assert not queue
                return tuple(publisher.finish(*(worker.finish() for worker in workers)))
            assert not any(complete)
            while queue:
                batch, queue = queue, []
                for sender, recipient, frame in batch:
                    _, replies = workers[recipient].call("peer", frame, sender)
                    queue.extend((recipient, peer, reply) for peer, reply in replies)
        raise AssertionError("native public schedule did not complete")
    finally:
        publisher.close()
        for worker in workers:
            worker.close()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_native_source_publisher_and_complete_private_feedback(tmp_path, family):
    experiment = experiment_fixture(tmp_path, family=family)
    source = admit_source(experiment)
    # Zero body projections leave each original embedding intact. RMSNorm
    # preserves its sign. The complete head picks +x or -x; feedback alternates.
    assert native_execution(source, experiment, (3, 4), 2) == (2, 1)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_nonzero_body_private_prefix_feedback_through_three_processes(tmp_path, family):
    experiment = experiment_fixture(tmp_path, family=family, active=True)
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as client:
        client._core._result_hook = captured.append
        # Rendering prepends user:, encoded as two zero-embedding unknowns.
        # Q=K=0, V=I, O=-I/2: [0,0,+,+] -> negative; [0,0,-,+] -> positive.
        # Embeddings ±5/32 keep a >1/32 margin, including the second selection.
        # Tiny positive SiLU/MLP terms and stochastic error cannot cross it.
        # Recomputed private feedback preserves each branch on the second step.
        for prompt, expected in [("hello hello", (4, 4)), ("world hello", (1, 1))]:
            response = client.responses.create(input=prompt, max_output_tokens=2)
            assert captured[-1].selected == expected
            assert response.usage.input_tokens == 4
            assert response.usage.output_tokens == 2
            assert active_sessions(client._core) == [0, 0, 0]


@pytest.mark.parametrize("method,args", [("call", ("offer", b"", True)),
    ("call", ("offer", b"", 3)), ("activate", (1,)), ("finish", (1,))])
def test_native_signature_failures_consume_worker(tmp_path, method, args):
    experiment = experiment_fixture(tmp_path)
    publisher, manifest = admit_source(experiment).publisher(experiment, (3, 4), 2, lambda: False)
    worker = _native.ReplicatedWorker(manifest, 0)
    try:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            getattr(worker, method)(*args)
        with pytest.raises(ValueError, match="retired"):
            worker.call("offer", b"", 0)
    finally:
        publisher.close()
        worker.close()


def test_sdk_three_processes_final_only_ids_and_retirement(tmp_path, record_property):
    experiment = experiment_fixture(tmp_path)
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as client:
        topology = client._owned_topology
        statuses = topology.statuses
        assert {item.role for item in statuses} == set(ROLES)
        assert len({item.pid for item in statuses}) == 3
        assert all(item.running and item.pid != os.getpid() for item in statuses)
        record_property("native", str(Path(_native.__file__).resolve()))
        record_property("worker_pids", [item.pid for item in statuses])
        client._core._result_hook = captured.append
        response = client.responses.create(input="hello", max_output_tokens=2)
        result, = captured
        assert len(result.selected) == response.usage.output_tokens == 2
        assert result.selected in {(1, 2), (2, 1)}
        assert not hasattr(result, "logits") and not hasattr(result, "kv")
        assert active_sessions(client._core) == [0, 0, 0]
        audit = client._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        assert "generation_decode_online_body_bytes" not in audit
    assert all(not item.running for item in topology.statuses)


@pytest.mark.parametrize("failure", ["cancel", "malformed", "replay", "partial_admission"])
def test_failure_retires_all_three_roles_before_next_request(tmp_path, monkeypatch, failure):
    from pllm.runtime.client import ProtocolError

    experiment = experiment_fixture(tmp_path)
    original = transport._Peers.post
    injected = []

    def fail(self, role, action, *args, **kwargs):
        if not injected and ((failure == "partial_admission" and role == 1 and action == "admit")
                             or (failure != "partial_admission" and role == 0 and action == "input")):
            injected.append(True)
            if failure == "partial_admission":
                raise ProtocolError("injected partial admission")
            original(self, role, action, *args, **kwargs)
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "malformed":
                return b"invalid acknowledgement"
            return original(self, role, action, *args, **kwargs)
        return original(self, role, action, *args, **kwargs)

    with OpenAI(experiment=experiment) as client:
        monkeypatch.setattr(transport._Peers, "post", fail)
        with pytest.raises((ValueError, ProtocolError, asyncio.CancelledError)):
            with client.responses.create(input="hello", max_output_tokens=1, stream=True) as stream:
                for _ in stream:
                    pytest.fail("failed request exposed an event")
        assert injected and not client._core.records and client._core._active is None
        assert active_sessions(client._core) == [0, 0, 0]
        monkeypatch.setattr(transport._Peers, "post", original)
        assert client.responses.create(input="hello", max_output_tokens=1).usage.output_tokens == 1


@pytest.mark.asyncio
async def test_async_cancellation_drains_active_peer_execution(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    original = transport._Peers.post

    def observe(self, role, action, *args, **kwargs):
        if role == 0 and action == "execute":
            entered.set()
        return original(self, role, action, *args, **kwargs)

    monkeypatch.setattr(transport._Peers, "post", observe)
    async with AsyncOpenAI(experiment=experiment_fixture(tmp_path)) as client:
        core = client.sync._core
        task = asyncio.create_task(client.responses.create(input="hello", max_output_tokens=2))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert core._active is None and not core.records
            assert await asyncio.to_thread(active_sessions, core) == [0, 0, 0]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
