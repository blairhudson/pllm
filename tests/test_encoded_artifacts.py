"""Provider publication is bounded, source-bound and retired on every exit."""
import asyncio
import hashlib
import json
import os

import pytest

from pllm.runtime.bundle_compression import decode_bundle_frames, encode_bundle_frames
from pllm.runtime.encoded_artifacts import EncodedArtifactCache, MAX_READERS


def stream(cache, payload, bundle="a" * 64):
    return cache.stream(payload, bundle_digest=bundle, object_digest=hashlib.sha256(payload).hexdigest())


def decode(chunks, payload):
    result = decode_bundle_frames(chunks, expected_size=len(payload), on_wire_bytes=lambda _: None)
    assert result == payload


def test_reuse_original_frames_repair_and_source_binding():
    payload = b"public weights" * 90000
    cache = EncodedArtifactCache(4 << 20)
    root = cache.directory
    try:
        expected = b"".join(encode_bundle_frames(payload))
        assert b"".join(stream(cache, payload)) == expected
        published = cache.snapshot()["published_bytes"]
        decode(stream(cache, payload), payload)
        assert cache.snapshot()["hits"] == 1
        assert cache.snapshot()["published_bytes"] == published
        entry = next(iter(cache._entries.values()))
        entry.path.write_bytes(b"bad encoding")
        decode(stream(cache, payload), payload)
        assert cache.snapshot()["repairs"] == 1
        assert cache.snapshot()["active_readers"] == 0
        with pytest.raises(ValueError, match="source changed"):
            cache.stream(b"forged", bundle_digest="a" * 64,
                         object_digest=hashlib.sha256(payload).hexdigest())
        decode(stream(cache, payload, "b" * 64), payload)
        assert cache.snapshot()["entries"] == 2
    finally:
        cache.close()
    assert not root.exists()
    with pytest.raises(ValueError, match="closed"):
        stream(cache, payload)


def test_capacity_pins_bypass_eviction_and_close_unstarted_readers():
    first, second = os.urandom(600000), os.urandom(600000)
    cache = EncodedArtifactCache(1 << 20)
    try:
        pinned = stream(cache, first)
        decode(stream(cache, second), second)
        assert cache.snapshot()["bypasses"] == 1
        assert cache.snapshot()["evictions"] == 0
        pinned.close()  # no first next(): still release its pin and fd
        decode(stream(cache, second), second)
        assert cache.snapshot()["evictions"] == 1
        assert cache.snapshot()["peak_disk_bytes"] <= cache.max_bytes
        readers = [stream(cache, second) for _ in range(MAX_READERS)]
        with pytest.raises(ValueError, match="reader capacity"):
            stream(cache, second)
        root = cache.directory
        cache.close()
        assert not root.exists()
        assert cache.snapshot()["active_readers"] == cache.snapshot()["resident_bytes"] == 0
        assert all(list(reader) == [] for reader in readers)
    finally:
        cache.close()


def test_failed_publication_removes_staging_and_can_retry(monkeypatch):
    from pllm.runtime import encoded_artifacts as module

    payload = b"public" * 1024
    cache = EncodedArtifactCache(1 << 20)
    def broken(_):
        yield b"unfinished"
        raise OSError("publication interrupted")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, "encode_bundle_frames", broken)
            with pytest.raises(OSError, match="interrupted"):
                stream(cache, payload)
        assert list(cache.directory.iterdir()) == []
        assert cache.snapshot()["resident_bytes"] == cache.snapshot()["active_readers"] == 0
        decode(stream(cache, payload), payload)
    finally:
        cache.close()


def test_asgi_send_failure_retires_active_file():
    from pllm.runtime.bundle_artifacts import object_response
    from pllm.runtime.bundle_compression import ENCODING

    cache = EncodedArtifactCache(1 << 20)
    async def run():
        response = await object_response(b"public", bundle_digest="a" * 64,
            object_digest="b" * 64, encoding=ENCODING, encoded_cache=cache)
        assert cache.snapshot()["active_readers"] == 1
        async def send(_):
            raise OSError("disconnected")
        async def receive():
            return {"type": "http.disconnect"}
        with pytest.raises(Exception):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert cache.snapshot()["active_readers"] == 0
    try:
        asyncio.run(run())
    finally:
        cache.close()


def test_cancel_during_publication_retires_unclaimed_reader(monkeypatch):
    import threading
    from pllm.runtime import bundle_artifacts
    from pllm.runtime.bundle_compression import ENCODING

    cache = EncodedArtifactCache(1 << 20)
    release = threading.Event()

    async def run():
        loop = asyncio.get_running_loop()
        started, retired = asyncio.Event(), asyncio.Event()
        original = cache.stream

        def publish(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            reader = original(*args, **kwargs)
            close = reader.close
            def retire():
                close()
                if not loop.is_closed():
                    loop.call_soon_threadsafe(retired.set)
            reader.close = retire
            return reader

        monkeypatch.setattr(cache, "stream", publish)
        task = asyncio.create_task(bundle_artifacts.object_response(b"public",
            bundle_digest="a" * 64, object_digest="b" * 64,
            encoding=ENCODING, encoded_cache=cache))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.wait_for(retired.wait(), 5)
        assert cache.snapshot()["active_readers"] == 0
    try:
        asyncio.run(run())
    finally:
        release.set()
        cache.close()


def test_component_and_native_admission_bind_capacity():
    from pllm import Experiment, Model, lower_model
    from pllm.configuration import ComponentRef, ConfigurationError
    from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
    from pllm.protocols import ClientBundleTransport
    from pllm.sources import _tiny_model_config

    base = ClientBundleTransport("artifacts", compression="zlib", storage="paged", batch_objects=64)
    assert "provider_cache_bytes" not in base.params
    cached = base.with_params(provider_cache_bytes=32 << 20)
    assert cached.with_params(provider_cache_bytes=0).to_spec() == base.to_spec()
    pipeline = MaskedLinearCpu(Model.tiny(), delivery=cached)
    assert resolve_runtime_composition(pipeline).provider_artifact_cache_bytes == 32 << 20
    plan = lower_model(_tiny_model_config(), batch=1, max_input_tokens=32, max_new_tokens=2)
    assert plan.runtime_schedule(pipeline).complete
    for value in (True, -1, 1, (2 << 30) + 1):
        with pytest.raises(ConfigurationError):
            base.with_params(provider_cache_bytes=value)
        forged = ComponentRef(base.component, {**cached.params, "provider_cache_bytes": value})
        with pytest.raises(ValueError):
            plan.runtime_schedule(MaskedLinearCpu(Model.tiny(), delivery=forged))
    with pytest.raises(ConfigurationError):
        ClientBundleTransport("artifacts", provider_cache_bytes=1 << 20)


@pytest.mark.integration
def test_public_example_reuses_provider_files_across_fresh_sdk_and_gateway_clients(tmp_path):
    import runpy
    from pathlib import Path
    import httpx
    from fastapi.testclient import TestClient
    import pllm
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.encoded_artifacts import HEADER

    build = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/prepared_sdk.py"))["build_experiment"]
    experiment = build(root=tmp_path).with_params(budget__max_new_tokens=2)
    restored = pllm.Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    assert restored.configuration_digest() == experiment.configuration_digest()
    with pllm.serve_local(restored) as roles:
        headers = {"Authorization": f"Bearer {roles._inference_key}"}
        endpoint = roles.inference_url + "/v1/runtime/artifact-cache/metrics"
        assert httpx.get(endpoint).status_code == 401
        states, outputs = [], []
        for i in range(2):
            with roles.client(bundle_cache_dir=tmp_path / f"fresh-{i}") as client:
                with client.tokenizer_scope():
                    client.prepare_response("hello", max_output_tokens=2)
                    response = client.responses.create(input="hello", max_output_tokens=2, temperature=0)
                outputs.append(response.output_text)
                audit = client.privacy_audit.to_dict()
                assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
                assert audit["inference_stage_calls"] > 0
            result = httpx.get(endpoint, headers=headers)
            result.raise_for_status()
            states.append(result.json())
        assert outputs[0] == outputs[1]
        assert states[0]["published_bytes"] > 0
        assert states[1]["published_bytes"] == states[0]["published_bytes"]
        assert states[1]["hits"] > states[0]["hits"]
        assert states[1]["active_readers"] == 0
        assert states[1]["peak_disk_bytes"] <= states[1]["capacity_bytes"]
        with roles.client(bundle_cache_dir=tmp_path / "bad-ack") as client:
            client._core.http.event_hooks["response"].append(lambda response: response.headers.pop(HEADER, None))
            with pytest.raises(ProtocolError, match="acknowledgement differs"):
                client.prepare_response("hello", max_output_tokens=2)
        with TestClient(roles.gateway_app(local_api_key="local-example", bundle_cache_dir=tmp_path / "gateway")) as api:
            reply = api.post("/v1/responses", headers={"Authorization": "Bearer local-example"},
                json={"model": roles.model_id, "input": "hello", "max_output_tokens": 2, "temperature": 0})
            assert reply.status_code == 200, reply.text
            assert reply.json()["usage"]["output_tokens"] == 2
        assert httpx.get(endpoint, headers=headers).json()["hits"] > states[1]["hits"]
